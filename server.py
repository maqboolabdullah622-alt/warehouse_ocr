"""
Warehouse OCR Pipeline V7 — FastAPI Backend
=============================================
Endpoints:
  POST /api/upload          → Upload video, start pipeline
  WS   /ws/{job_id}         → Real-time progress + live frames
  GET  /api/status/{id}     → Poll status
  GET  /api/result/{id}     → Final JSON
  GET  /api/video/{id}      → Stream uploaded video (Range-request aware)
  GET  /api/files/{id}/**   → Serve images (keyframes, crops, annotated, enhanced)
  GET  /api/browse/{id}/{f} → List folder contents
  GET  /                    → Frontend

Run:
  pip install fastapi uvicorn python-multipart websockets
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio, json, logging, os, time, uuid, mimetypes
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from pipeline import Config, run_pipeline

app = FastAPI(title="Warehouse OCR Pipeline V7")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR = Path("jobs"); JOBS_DIR.mkdir(exist_ok=True)

executor = ThreadPoolExecutor(max_workers=2)
jobs: Dict[str, dict] = {}


def send_progress(job_id: str, step: str, progress: float, detail: str = ""):
    if job_id in jobs:
        jobs[job_id].update({"step": step, "progress": round(progress, 1), "detail": detail})
        jobs[job_id]["messages"].append({
            "step": step, "progress": round(progress, 1),
            "detail": detail, "ts": time.time()
        })


def _push_live_frame(job_id, url, num, total, ftype, pct, detail=""):
    """Queue a live_frame message for the WebSocket to deliver."""
    jobs[job_id]["messages"].append({
        "step": "live_frame",
        "progress": round(pct, 1),
        "detail": detail,
        "frame_url": url,
        "frame_num": num,
        "frame_total": total,
        "frame_type": ftype,   # "katna" | "yolo"
        "ts": time.time(),
    })


def run_job(job_id: str, video_path: str, device: str, enhance: str, upscale: int, num_kf: int):
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        jobs[job_id]["status"] = "running"
        send_progress(job_id, "init", 0, "Initializing pipeline...")

        cfg = Config(
            input_path=video_path,
            output_path=str(job_dir / "results.json"),
            device=device,
            yolo_model="best_25_epoch.pt",
            yolo_conf=0.50,
            num_keyframes=num_kf,
            keyframes_dir=str(job_dir / "keyframes"),
            detected_dir=str(job_dir / "detected_objects"),
            enhanced_dir=str(job_dir / "enhanced_crops"),
            annotated_dir=str(job_dir / "annotated_frames"),
            enhance=enhance,
            upscale=upscale,
            best_k=2,
            langs=["en"],
            ocr_conf=0.2,
        )

        # ── Monkey-patch pipeline logger to emit progress/live-frame messages ──
        import pipeline as pl
        orig_info = pl.log.info

        def patched_info(msg, *a, **kw):
            orig_info(msg, *a, **kw)
            m = str(msg)

            # ── STEP 1: Katna ──
            if "STEP 1" in m:
                send_progress(job_id, "katna", 5, "Extracting keyframes with Katna...")

            # Katna logs "  Keyframe 1/15 saved" for each frame as it's written
            elif "Keyframe " in m and " saved" in m:
                try:
                    part = m.strip().split("Keyframe ")[1]
                    c = int(part.split("/")[0])
                    t = int(part.split("/")[1].split(" ")[0])
                    pct = 5 + (c / t) * 20
                    send_progress(job_id, "katna", pct, f"Katna extracting frame {c}/{t}...")
                    kf_dir = JOBS_DIR / job_id / "keyframes"
                    kf_name = f"keyframe_{c:04d}.png"
                    kf_path = kf_dir / kf_name
                    if kf_path.exists():
                        _push_live_frame(
                            job_id,
                            url=f"/api/files/{job_id}/keyframes/{kf_name}",
                            num=c, total=t, ftype="katna", pct=pct,
                            detail=f"Katna frame {c}/{t}",
                        )
                except: pass

            # "Katna extracted N keyframes" — also handles fallback mode (no per-frame logs)
            elif "Katna extracted" in m or "Fallback extracted" in m:
                send_progress(job_id, "katna", 25, m.strip())
                # Fallback sampler doesn't emit per-frame logs, so bulk-emit now
                already = any(
                    mm.get("frame_type") == "katna"
                    for mm in jobs[job_id]["messages"]
                    if isinstance(mm, dict)
                )
                if not already:
                    kf_dir = JOBS_DIR / job_id / "keyframes"
                    if kf_dir.exists():
                        kf_files = sorted(
                            list(kf_dir.glob("*.jpg")) +
                            list(kf_dir.glob("*.jpeg")) +
                            list(kf_dir.glob("*.png"))
                        )
                        for i, kf in enumerate(kf_files):
                            _push_live_frame(
                                job_id,
                                url=f"/api/files/{job_id}/keyframes/{kf.name}",
                                num=i + 1, total=len(kf_files), ftype="katna",
                                pct=5 + (i / max(len(kf_files), 1)) * 20,
                                detail=f"Keyframe {i+1}/{len(kf_files)}",
                            )

            # ── STEP 2: YOLO ──
            elif "STEP 2" in m:
                send_progress(job_id, "yolo", 28, "Running YOLO detection...")

            # YOLO logs "  Keyframe 1/15: 3 detections" — NOTE: must check BEFORE generic "Keyframe" match
            elif "Keyframe " in m and "detections" in m:
                try:
                    part = m.strip().split("Keyframe ")[1]
                    c = int(part.split("/")[0])
                    t = int(part.split("/")[1].split(":")[0])
                    pct = 28 + (c / t) * 27
                    send_progress(job_id, "yolo", pct, m.strip())
                    # Pipeline writes annotated frame BEFORE logging, so file is on disk now
                    ann_dir = JOBS_DIR / job_id / "annotated_frames"
                    if ann_dir.exists():
                        ann_files = sorted(ann_dir.glob("*.jpg"))
                        if ann_files:
                            latest = ann_files[-1]
                            _push_live_frame(
                                job_id,
                                url=f"/api/files/{job_id}/annotated_frames/{latest.name}",
                                num=c, total=t, ftype="yolo", pct=pct,
                                detail=m.strip(),
                            )
                except: pass

            elif "YOLO:" in m:
                send_progress(job_id, "yolo", 55, m.strip())

            # ── STEP 3: Enhancement ──
            elif "STEP 3" in m:
                send_progress(job_id, "enhance", 58, f"Enhancing crops ({enhance})...")
            elif "NAFNet loaded" in m:
                send_progress(job_id, "enhance", 62, m.strip())
            elif m.strip().startswith("[") and "/" in m and "]" in m:
                try:
                    inner = m.split("[")[1].split("]")[0]
                    c, t = inner.split("/")
                    pct = 60 + (int(c) / int(t)) * 20
                    send_progress(job_id, "enhance", pct, m.strip())
                except: pass

            # ── STEP 4: OCR ──
            elif "STEP 4" in m:
                send_progress(job_id, "ocr", 82, "Running OCR...")
            elif "lines extracted" in m:
                send_progress(job_id, "ocr", 90, m.strip())
            elif "Saved:" in m:
                send_progress(job_id, "done", 100, "Pipeline complete!")

        pl.log.info = patched_info
        result = run_pipeline(cfg)
        pl.log.info = orig_info

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        send_progress(job_id, "done", 100,
                      f"Done! {result['metadata']['total_objects']} objects, "
                      f"{result['metadata']['total_keyframes']} keyframes")

    except Exception as e:
        log.exception(f"Job {job_id} failed")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        send_progress(job_id, "error", -1, str(e))


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    device: str = Query("cpu"),
    enhance: str = Query("nafnet"),
    upscale: int = Query(2),
    num_keyframes: int = Query(10),
):
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix
    video_path = str(UPLOAD_DIR / f"{job_id}{ext}")

    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "id": job_id, "filename": file.filename,
        "video_path": video_path,
        "file_size_mb": round(len(content) / (1024 * 1024), 2),
        "status": "queued", "step": "queued", "progress": 0,
        "detail": "Queued...", "result": None, "error": None,
        "messages": [], "created": time.time(),
    }

    executor.submit(run_job, job_id, video_path, device, enhance, upscale, num_keyframes)
    return {"job_id": job_id, "status": "queued", "filename": file.filename}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs: return JSONResponse({"error": "Not found"}, status_code=404)
    j = jobs[job_id]
    return {k: j[k] for k in ("id", "status", "step", "progress", "detail", "filename", "error")}


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs: return JSONResponse({"error": "Not found"}, status_code=404)
    j = jobs[job_id]
    if j["status"] != "done": return JSONResponse({"error": "Not ready"}, status_code=202)
    return j["result"]


@app.get("/api/video/{job_id}")
async def stream_video(job_id: str, request: Request):
    """
    Video streaming endpoint with HTTP Range support.
    Browsers require partial-content (206) responses for <video> seek/play to work.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Not found"}, status_code=404)
    vp = Path(jobs[job_id]["video_path"])
    if not vp.exists():
        return JSONResponse({"error": "Video file not found"}, status_code=404)

    file_size = vp.stat().st_size
    mt = mimetypes.guess_type(str(vp))[0] or "video/mp4"
    range_header = request.headers.get("range")

    CHUNK = 1024 * 256  # 256 KB per chunk

    if range_header:
        try:
            rng = range_header.strip().replace("bytes=", "")
            start_s, end_s = rng.split("-")
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
        except Exception:
            start, end = 0, file_size - 1

        end = min(end, file_size - 1)
        length = end - start + 1

        def iter_range():
            with open(vp, "rb") as f:
                f.seek(start)
                rem = length
                while rem > 0:
                    data = f.read(min(CHUNK, rem))
                    if not data:
                        break
                    rem -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=mt,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Cache-Control": "no-cache",
            },
        )
    else:
        def iter_full():
            with open(vp, "rb") as f:
                while True:
                    data = f.read(CHUNK)
                    if not data:
                        break
                    yield data

        return StreamingResponse(
            iter_full(),
            media_type=mt,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "no-cache",
            },
        )


@app.get("/api/files/{job_id}/{path:path}")
async def serve_file(job_id: str, path: str):
    fp = JOBS_DIR / job_id / path
    if not fp.exists(): return JSONResponse({"error": "Not found"}, status_code=404)
    mt = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
    return FileResponse(str(fp), media_type=mt)


@app.get("/api/browse/{job_id}/{folder}")
async def browse(job_id: str, folder: str):
    base = JOBS_DIR / job_id / folder
    if not base.exists(): return {"objects": {}, "files": []}
    result = {"objects": {}, "files": []}

    # Root-level image files
    for f in sorted(base.iterdir()):
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            result["files"].append({
                "name": f.name,
                "url": f"/api/files/{job_id}/{folder}/{f.name}",
            })

    # Sub-directories (object folders)
    for d in sorted(base.iterdir()):
        if d.is_dir():
            files = []
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    files.append({
                        "name": f.name,
                        "url": f"/api/files/{job_id}/{folder}/{d.name}/{f.name}",
                    })
            result["objects"][d.name] = {"count": len(files), "files": files}

    return result


@app.get("/api/logs/{job_id}")
async def get_logs(job_id: str):
    if job_id not in jobs: return JSONResponse({"error": "Not found"}, status_code=404)
    return {"messages": jobs[job_id]["messages"]}


@app.websocket("/ws/{job_id}")
async def ws_endpoint(ws: WebSocket, job_id: str):
    await ws.accept()
    last_idx = 0
    try:
        while True:
            if job_id in jobs:
                msgs = jobs[job_id]["messages"]
                for m in msgs[last_idx:]:
                    await ws.send_json(m)
                last_idx = len(msgs)
                if jobs[job_id]["status"] in ("done", "error"):
                    await ws.send_json({
                        "step": jobs[job_id]["status"],
                        "progress": 100 if jobs[job_id]["status"] == "done" else -1,
                        "detail": ("Complete!" if jobs[job_id]["status"] == "done"
                                   else jobs[job_id].get("error", "")),
                        "final": True,
                    })
                    break
            await asyncio.sleep(0.25)   # 250ms poll — responsive but not hammering
    except WebSocketDisconnect:
        pass


@app.get("/")
async def serve_frontend():
    return FileResponse("frontend.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
