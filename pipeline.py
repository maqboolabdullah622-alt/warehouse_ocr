"""
Warehouse OCR Pipeline V7
=========================
Video → Katna Keyframes → YOLO Detect → NAFNet Enhance → OCR → JSON

Pipeline Flow:
  STEP 1: Katna extracts keyframes from video (smart scene-based selection)
  STEP 2: YOLO detects objects on keyframes (crops saved object-wise)
  STEP 3: NAFNet enhances crop images (replaces Real-ESRGAN)
  STEP 4: OCR with multi-rotation on enhanced crops

Folder Structure:
  jobs/{job_id}/
    keyframes/             ← Katna-extracted keyframes
    detected_objects/      ← YOLO crops (object-wise)
    enhanced_crops/        ← NAFNet enhanced
    annotated_frames/      ← Keyframes with YOLO bounding boxes drawn
    results.json

Install:
    pip install ultralytics opencv-python-headless numpy easyocr pytesseract
    pip install katna scikit-image basicsr nafnet
    # GPU: pip install torch torchvision
"""

import argparse, cv2, json, logging, numpy as np, os, re, time, shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

try: import easyocr
except ImportError: easyocr = None

try: import pytesseract
except ImportError: pytesseract = None

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError: HAS_YOLO = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False; torch = None

try:
    from Katna.video import Video as KatnaVideo
    HAS_KATNA = True
except ImportError:
    HAS_KATNA = False

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError: HAS_SSIM = False

# NAFNet import
try:
    from basicsr.models import create_model
    from basicsr.utils import img2tensor, tensor2img
    from basicsr.utils.options import parse as parse_options
    HAS_NAFNET = True
except ImportError:
    HAS_NAFNET = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

# Box colors for annotations (BGR)
COLORS = [
    (255, 120, 50), (50, 255, 120), (120, 50, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (200, 100, 50), (50, 200, 100),
]


@dataclass
class Config:
    input_path: str = ""
    output_path: str = "results.json"
    device: str = "cpu"
    yolo_model: str = "best_25_epoch.pt"
    yolo_conf: float = 0.50
    # Katna settings
    num_keyframes: int = 15
    # Folder structure
    keyframes_dir: str = "keyframes"
    detected_dir: str = "detected_objects"
    enhanced_dir: str = "enhanced_crops"
    annotated_dir: str = "annotated_frames"
    # Enhancement
    enhance: str = "nafnet"        # nafnet | adaptive | none
    upscale: int = 2
    min_ocr_dim: int = 2000
    nafnet_model: str = "NAFNet-REDS-width64"
    # OCR
    langs: List[str] = field(default_factory=lambda: ["en"])
    ocr_conf: float = 0.4
    min_text_len: int = 2
    best_k: int = 3                # best crops per object for OCR


# ═══════════════════════════════════════════════════════════════
# STEP 1: KATNA KEYFRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_keyframes_katna(video_path: str, output_dir: str, num_frames: int = 15) -> List[str]:
    """Extract keyframes using Katna's scene-based algorithm."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if HAS_KATNA:
        try:
            vd = KatnaVideo()
            imgs = vd.extract_video_keyframes(
                no_of_frames=num_frames,
                file_path=video_path
            )
            paths = []
            for i, img in enumerate(imgs):
                fp = str(out / f"keyframe_{i+1:04d}.png")
                cv2.imwrite(fp, img)
                paths.append(fp)
                log.info(f"  Keyframe {i+1}/{len(imgs)} saved")
            log.info(f"  Katna extracted {len(paths)} keyframes")
            return sorted(paths)
        except Exception as e:
            log.warning(f"  Katna failed: {e}, falling back to uniform sampling")

    # Fallback: uniform + sharpness-based sampling
    return extract_keyframes_fallback(video_path, output_dir, num_frames)


def extract_keyframes_fallback(video_path: str, output_dir: str, num_frames: int) -> List[str]:
    """Fallback: sample uniformly, score by sharpness, keep best."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Sample 3x more frames than needed, pick sharpest
    sample_count = min(total, num_frames * 3)
    indices = np.linspace(0, total - 1, sample_count, dtype=int)

    candidates = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        candidates.append((sharp, idx, frame))

    cap.release()

    # Sort by sharpness, keep top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    kept = candidates[:num_frames]

    # Sort by frame index for chronological order
    kept.sort(key=lambda x: x[1])

    paths = []
    for i, (sharp, idx, frame) in enumerate(kept):
        fp = str(out / f"keyframe_{i+1:04d}_f{idx:06d}.png")
        cv2.imwrite(fp, frame)
        paths.append(fp)

    log.info(f"  Fallback extracted {len(paths)} keyframes (sharpness-ranked)")
    return paths


# ═══════════════════════════════════════════════════════════════
# STEP 2: YOLO DETECTION ON KEYFRAMES
# ═══════════════════════════════════════════════════════════════

def run_yolo_on_keyframes(keyframe_paths: List[str], cfg: Config) -> Dict:
    """
    Run YOLO on each keyframe, save crops object-wise.
    Also save annotated frames with bounding boxes.
    Returns: {object_id: [crop_paths]}
    """
    if not HAS_YOLO:
        raise ImportError("pip install ultralytics")

    log.info(f"  Loading YOLO: {cfg.yolo_model}")
    model = YOLO(cfg.yolo_model)

    det_dir = Path(cfg.detected_dir)
    det_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = Path(cfg.annotated_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)

    obj_crops = {}  # {obj_id: [(crop_path, sharpness)]}
    detection_count = 0

    for ki, kf_path in enumerate(keyframe_paths):
        frame = cv2.imread(kf_path)
        if frame is None: continue
        h, w = frame.shape[:2]

        results = model(frame, conf=cfg.yolo_conf, iou=0.45, verbose=False)

        # Draw annotated frame
        annotated = frame.copy()
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))

            for di, (box, conf, cls) in enumerate(zip(xyxy, confs, classes)):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                obj_id = f"object_{int(cls)}_{di+1}"
                # Use a more stable ID based on position region
                region_key = f"cls{int(cls)}_r{y1//100}_{x1//100}"

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                # Compute sharpness for ranking
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(gray, cv2.CV_64F).var()

                if region_key not in obj_crops:
                    obj_crops[region_key] = []

                crop_path = det_dir / region_key
                crop_path.mkdir(exist_ok=True)
                fname = f"kf{ki+1:03d}_conf{conf:.2f}_sharp{sharp:.0f}.png"
                cv2.imwrite(str(crop_path / fname), crop)
                obj_crops[region_key].append((str(crop_path / fname), sharp, conf))
                detection_count += 1

                # Draw box on annotated frame
                color = COLORS[di % len(COLORS)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label = f"{region_key} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
                cv2.putText(annotated, label, (x1+4, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Save annotated
        ann_name = f"annotated_kf{ki+1:03d}_{Path(kf_path).stem}.jpg"
        cv2.imwrite(str(ann_dir / ann_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.info(f"  Keyframe {ki+1}/{len(keyframe_paths)}: "
                 f"{len(boxes) if boxes is not None else 0} detections")

    # Select best crops per object (top-K by sharpness)
    best_crops = {}
    for obj_id, crops in obj_crops.items():
        crops.sort(key=lambda x: (x[1], x[2]), reverse=True)  # sharpness, then conf
        best_crops[obj_id] = [p for p, _, _ in crops[:cfg.best_k]]

    log.info(f"  YOLO: {detection_count} total detections → {len(best_crops)} unique objects")
    return best_crops


# ═══════════════════════════════════════════════════════════════
# STEP 3: NAFNet ENHANCEMENT
# ═══════════════════════════════════════════════════════════════

_nafnet_model = [None]

def _load_nafnet(cfg: Config):
    """Load NAFNet deblurring/denoising model."""
    if _nafnet_model[0] is not None:
        return _nafnet_model[0]

    if not HAS_NAFNET:
        log.warning("NAFNet not available, using adaptive enhancement")
        return None

    try:
        # NAFNet config — using basicsr framework
        opt_path = None
        model_urls = {
            "NAFNet-REDS-width64": "https://github.com/megvii-research/NAFNet/releases/download/v0.1.0/NAFNet-REDS-width64.pth",
            "NAFNet-GoPro-width64": "https://github.com/megvii-research/NAFNet/releases/download/v0.1.0/NAFNet-GoPro-width64.pth",
        }

        model_name = cfg.nafnet_model
        if model_name in model_urls:
            from basicsr.archs.nafnet_arch import NAFNet as NAFNetArch
            model = NAFNetArch(img_channel=3, width=64, middle_blk_num=12,
                              enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2])

            model_path = Path("models") / f"{model_name}.pth"
            model_path.parent.mkdir(exist_ok=True)

            if not model_path.exists():
                log.info(f"  Downloading {model_name}...")
                import urllib.request
                urllib.request.urlretrieve(model_urls[model_name], str(model_path))

            state = torch.load(str(model_path), map_location='cpu')
            if 'params' in state: state = state['params']
            elif 'state_dict' in state: state = state['state_dict']
            model.load_state_dict(state, strict=True)

            device = 'cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            model = model.to(device).eval()
            _nafnet_model[0] = (model, device)
            log.info(f"  NAFNet loaded on {device}")
            return _nafnet_model[0]

    except Exception as e:
        log.warning(f"  NAFNet load failed: {e}")
        return None


def enhance_with_nafnet(img: np.ndarray, model_tuple) -> np.ndarray:
    """Run NAFNet inference on a single image."""
    if model_tuple is None:
        return img

    model, device = model_tuple
    try:
        inp = img.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Pad to multiple of 64
        _, _, h, w = inp.shape
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            out = model(inp)

        out = out[:, :, :h, :w]
        out = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return out
    except Exception as e:
        log.warning(f"    NAFNet inference failed: {e}")
        return img


# ── Adaptive enhancement fallback ──

def reduce_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = np.median(np.abs(cv2.Laplacian(gray, cv2.CV_64F))) / 0.6745
    if noise < 5: return img
    sc = min(int(noise * 3), 75)
    result = cv2.bilateralFilter(img, d=7, sigmaColor=sc, sigmaSpace=min(int(noise * 2), 75))
    if noise > 15:
        h_val = min(int(noise * 0.6), 12)
        result = cv2.fastNlMeansDenoisingColored(result, None, h_val, h_val, 7, 21)
    return result

def refine_details(img):
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    detail = cv2.subtract(img, smooth)
    boosted = np.clip(cv2.multiply(detail, np.array([1.8], dtype=np.float64)), 0, 255).astype(np.uint8)
    result = cv2.add(smooth, boosted)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    em = (edges / max(edges.max(), 1) * 255).astype(np.uint8)
    em = cv2.GaussianBlur(em, (3, 3), 0)
    sharp = cv2.addWeighted(result, 1.5, cv2.GaussianBlur(result, (0, 0), 2), -0.5, 0)
    m3 = cv2.cvtColor(em, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    return np.clip(sharp.astype(np.float32) * m3 + result.astype(np.float32) * (1 - m3),
                   0, 255).astype(np.uint8)

def increase_quality(img):
    rf = img.astype(np.float32)
    avgs = [rf[:, :, i].mean() for i in range(3)]
    avg_all = sum(avgs) / 3
    for i in range(3):
        if avgs[i] > 0: rf[:, :, i] *= avg_all / avgs[i]
    img = np.clip(rf, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 2.0, blur, -1.0, 0)

def boost_text(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(g) <= 150: return img
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sharp = cv2.addWeighted(img, 2.5, cv2.GaussianBlur(img, (0, 0), 3), -1.5, 0)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.where(mask3 == 0, sharp, img)


def enhance_crop(img: np.ndarray, cfg: Config, nafnet=None) -> np.ndarray:
    """Full crop enhancement pipeline."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Upscale first
    h, w = img.shape[:2]
    if cfg.upscale > 1:
        img = cv2.resize(img, None, fx=cfg.upscale, fy=cfg.upscale, interpolation=cv2.INTER_CUBIC)

    # Ensure min OCR dimension
    h, w = img.shape[:2]
    if max(h, w) < cfg.min_ocr_dim:
        s = cfg.min_ocr_dim / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    if cfg.enhance == "none":
        return img

    # NAFNet enhancement
    if cfg.enhance == "nafnet" and nafnet is not None:
        img = enhance_with_nafnet(img, nafnet)

    # Always run adaptive pipeline after
    img = reduce_noise(img)
    img = refine_details(img)
    img = increase_quality(img)
    img = boost_text(img)
    return img


def run_enhancement(obj_crops: Dict[str, List[str]], cfg: Config) -> Dict[str, List[np.ndarray]]:
    """
    STEP 3: Enhance all selected crops.
    Returns: {obj_id: [enhanced_images]}
    """
    enh_dir = Path(cfg.enhanced_dir)
    enh_dir.mkdir(parents=True, exist_ok=True)

    nafnet = None
    if cfg.enhance == "nafnet":
        nafnet = _load_nafnet(cfg)

    enhanced = {}
    total_crops = sum(len(v) for v in obj_crops.values())
    done = 0

    for obj_id, paths in obj_crops.items():
        obj_enh_dir = enh_dir / obj_id
        obj_enh_dir.mkdir(parents=True, exist_ok=True)
        imgs = []

        for i, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None: continue

            orig_h, orig_w = img.shape[:2]
            enh = enhance_crop(img, cfg, nafnet)
            enh_h, enh_w = enh.shape[:2]

            fname = f"enhanced_{i+1}_{Path(path).name}"
            cv2.imwrite(str(obj_enh_dir / fname), enh)
            imgs.append(enh)
            done += 1
            log.info(f"    [{done}/{total_crops}] {obj_id}: {orig_w}x{orig_h} → {enh_w}x{enh_h}")

        if imgs:
            enhanced[obj_id] = imgs

    return enhanced


# ═══════════════════════════════════════════════════════════════
# ROTATION FIX
# ═══════════════════════════════════════════════════════════════

def fix_rotation(img: np.ndarray) -> np.ndarray:
    if pytesseract is None: return img
    h, w = img.shape[:2]
    if h < 50 or w < 50: return img
    try:
        work = img
        if max(h, w) < 300:
            s = 300 / max(h, w)
            work = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle == 0: return img
        log.info(f"    Rotation fix: {angle}°")
        rots = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
        if angle in rots: return cv2.rotate(img, rots[angle])
    except: pass
    return img


# ═══════════════════════════════════════════════════════════════
# STEP 4: OCR
# ═══════════════════════════════════════════════════════════════

_ocr = [None]

def _get_reader(cfg):
    if _ocr[0] is None:
        if easyocr is None: raise ImportError("pip install easyocr")
        gpu = cfg.device == "cuda" and HAS_TORCH and torch.cuda.is_available()
        log.info(f"  EasyOCR init (gpu={gpu})")
        _ocr[0] = easyocr.Reader(cfg.langs, gpu=gpu, verbose=False)
    return _ocr[0]


def run_ocr(frames: List[np.ndarray], cfg: Config) -> Dict:
    reader = _get_reader(cfg)
    all_lines, seen = [], set()

    all_rotations = [(0, None), (90, cv2.ROTATE_90_COUNTERCLOCKWISE),
                     (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_CLOCKWISE)]
    no_rotation = [(0, None)]

    for fi, img in enumerate(frames):
        img = fix_rotation(img)
        rots = all_rotations if fi == 0 else no_rotation

        for angle, code in rots:
            rot = cv2.rotate(img, code) if code else img
            try:
                results = reader.readtext(rot, detail=1, paragraph=False,
                                          contrast_ths=0.05, adjust_contrast=0.8,
                                          text_threshold=0.4, low_text=0.2,
                                          width_ths=0.8, mag_ratio=1.0)
            except: continue

            for entry in results:
                if len(entry) >= 3: _, text, conf = entry[:3]
                elif len(entry) == 2: _, text = entry; conf = 0.5
                else: continue
                t = text.strip()
                norm = re.sub(r"[^a-z0-9]", "", t.lower())
                if len(norm) < cfg.min_text_len or conf < cfg.ocr_conf: continue
                if norm not in seen:
                    seen.add(norm)
                    all_lines.append({"text": t, "confidence": round(float(conf), 3), "rotation": angle})

    all_lines.sort(key=lambda x: (len(x["text"]), x["confidence"]), reverse=True)
    return {"lines": all_lines, "full_text": "\n".join(l["text"] for l in all_lines), "total_lines": len(all_lines)}


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(cfg: Config) -> Dict:
    t0 = time.time()
    p = Path(cfg.input_path)

    # ── STEP 1: Katna keyframe extraction ──
    log.info("=" * 60)
    log.info("STEP 1: Katna Keyframe Extraction")
    log.info(f"  Video: {cfg.input_path}")

    if p.is_file() and p.suffix.lower() in VID_EXTS:
        keyframe_paths = extract_keyframes_katna(
            str(p), cfg.keyframes_dir, cfg.num_keyframes
        )
    elif p.is_dir():
        keyframe_paths = sorted(
            str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXTS
        )
        log.info(f"  Using {len(keyframe_paths)} images from directory")
    elif p.is_file() and p.suffix.lower() in IMG_EXTS:
        keyframe_paths = [str(p)]
        log.info(f"  Single image input")
    else:
        raise ValueError(f"Unsupported input: {p}")

    step1_time = time.time() - t0

    # ── STEP 2: YOLO detection on keyframes ──
    t1 = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"STEP 2: YOLO Detection on {len(keyframe_paths)} keyframes")

    obj_crops = run_yolo_on_keyframes(keyframe_paths, cfg)
    step2_time = time.time() - t1

    if not obj_crops:
        log.warning("  No objects detected!")

    # ── STEP 3: NAFNet Enhancement ──
    t2 = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"STEP 3: Enhancement ({cfg.enhance}) — {len(obj_crops)} objects")

    enhanced = run_enhancement(obj_crops, cfg)
    step3_time = time.time() - t2

    # ── STEP 4: OCR ──
    t3 = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"STEP 4: OCR — {len(enhanced)} objects")

    results = []
    for obj_id, imgs in enhanced.items():
        log.info(f"\n  Object: {obj_id} ({len(imgs)} enhanced frames)")
        ocr = run_ocr(imgs, cfg)
        log.info(f"  → {ocr['total_lines']} lines extracted")
        results.append({
            "object_id": obj_id,
            "frames_analyzed": len(imgs),
            "crop_paths": [str(p) for p in (Path(cfg.detected_dir) / obj_id).glob("*.png")] if Path(cfg.detected_dir, obj_id).exists() else [],
            "total_lines": ocr["total_lines"],
            "full_text": ocr["full_text"],
            "lines": ocr["lines"],
        })

    step4_time = time.time() - t3
    elapsed = round(time.time() - t0, 2)

    # Collect annotated frame paths
    ann_dir = Path(cfg.annotated_dir)
    annotated_paths = sorted(str(f) for f in ann_dir.glob("*.jpg")) if ann_dir.exists() else []

    output = {
        "objects": results,
        "metadata": {
            "total_objects": len(results),
            "total_keyframes": len(keyframe_paths),
            "enhancement": cfg.enhance,
            "upscale": cfg.upscale,
            "device": cfg.device,
            "time_seconds": elapsed,
            "step_times": {
                "katna_keyframes": round(step1_time, 2),
                "yolo_detection": round(step2_time, 2),
                "enhancement": round(step3_time, 2),
                "ocr": round(step4_time, 2),
            },
            "folders": {
                "keyframes": cfg.keyframes_dir,
                "detected_objects": cfg.detected_dir,
                "enhanced_crops": cfg.enhanced_dir,
                "annotated_frames": cfg.annotated_dir,
            },
            "keyframe_paths": [str(p) for p in keyframe_paths],
            "annotated_paths": annotated_paths,
        },
    }

    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"\nSaved: {cfg.output_path} ({elapsed}s)")
    return output


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Warehouse OCR Pipeline V7")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", default="results.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--yolo-model", default="best_25_epoch.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.50)
    ap.add_argument("--num-keyframes", type=int, default=15)
    ap.add_argument("--keyframes-dir", default="keyframes")
    ap.add_argument("--detected-dir", default="detected_objects")
    ap.add_argument("--enhanced-dir", default="enhanced_crops")
    ap.add_argument("--annotated-dir", default="annotated_frames")
    ap.add_argument("--enhance", default="nafnet", choices=["nafnet", "adaptive", "none"])
    ap.add_argument("--upscale", type=int, default=2, choices=[1, 2, 4])
    ap.add_argument("--best-k", type=int, default=3)
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--ocr-conf", type=float, default=0.4)
    a = ap.parse_args()

    run_pipeline(Config(
        input_path=a.input, output_path=a.output, device=a.device,
        yolo_model=a.yolo_model, yolo_conf=a.yolo_conf,
        num_keyframes=a.num_keyframes, keyframes_dir=a.keyframes_dir,
        detected_dir=a.detected_dir, enhanced_dir=a.enhanced_dir,
        annotated_dir=a.annotated_dir, enhance=a.enhance, upscale=a.upscale,
        best_k=a.best_k,
        langs=a.langs, ocr_conf=a.ocr_conf,
    ))

if __name__ == "__main__":
    main()