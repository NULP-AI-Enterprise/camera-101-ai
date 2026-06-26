"""Media endpoints: MJPEG live stream, SSE log tail, snapshot/video file serving."""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .auth import require_auth
from .deps import BASE_DATA, LOGS_DIR, PREVIEW_PATH

router = APIRouter()


# ── MJPEG live stream ─────────────────────────────────────────────────────────

@router.get("/stream")
async def mjpeg_stream(_=Depends(require_auth)):
    async def _frames():
        while True:
            if os.path.exists(PREVIEW_PATH):
                try:
                    with open(PREVIEW_PATH, "rb") as f:
                        data = f.read()
                    if data:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + data + b"\r\n")
                except OSError:
                    pass
            await asyncio.sleep(0.1)   # ~10 fps

    return StreamingResponse(
        _frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── SSE log tail ──────────────────────────────────────────────────────────────

_LOG_FILES = {
    "stream":   "stream.log",
    "analyser": "analyser.log",
}


@router.get("/logs/stream")
async def log_sse(file: str = "stream", _=Depends(require_auth)):
    fname = _LOG_FILES.get(file)
    if not fname:
        raise HTTPException(400, "unknown log file")
    log_path = os.path.join(LOGS_DIR, fname)

    async def _lines():
        pos = 0
        # send last 150 lines on connect
        if os.path.exists(log_path):
            with open(log_path, "r", errors="replace") as f:
                lines = f.readlines()[-150:]
                pos = f.tell()
            for line in lines:
                yield f"data: {json.dumps(line.rstrip())}\n\n"
        # tail forever
        while True:
            if os.path.exists(log_path):
                try:
                    if os.path.getsize(log_path) < pos:
                        pos = 0  # log rotated / cleared
                    with open(log_path, "r", errors="replace") as f:
                        f.seek(pos)
                        new = f.readlines()
                        pos = f.tell()
                    for line in new:
                        yield f"data: {json.dumps(line.rstrip())}\n\n"
                except OSError:
                    pass
            await asyncio.sleep(0.4)

    return StreamingResponse(
        _lines(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/logs/{file}")
def clear_log(file: str, _=Depends(require_auth)):
    fname = _LOG_FILES.get(file)
    if not fname:
        raise HTTPException(400, "unknown log file")
    log_path = os.path.join(LOGS_DIR, fname)
    if os.path.exists(log_path):
        open(log_path, "w").close()
    return {"ok": True}


# ── static file serving ───────────────────────────────────────────────────────

def _safe(subpath: str) -> str:
    """Resolve a user-supplied path under BASE_DATA; reject traversal."""
    resolved = os.path.realpath(os.path.join(BASE_DATA, subpath.lstrip("/")))
    if not resolved.startswith(os.path.realpath(BASE_DATA) + os.sep):
        raise HTTPException(403, "Forbidden path")
    return resolved


@router.get("/snapshot/{path:path}")
def serve_snapshot(path: str, _=Depends(require_auth)):
    full = _safe(path)
    if not os.path.isfile(full):
        raise HTTPException(404)
    return FileResponse(full, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


@router.get("/video/{path:path}")
def serve_video(path: str, _=Depends(require_auth)):
    full = _safe(path)
    if not os.path.isfile(full):
        raise HTTPException(404)
    return FileResponse(full, media_type="video/mp4")
