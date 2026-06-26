"""Shared dependencies: config constants, auth token, lazy embedder."""
from __future__ import annotations

import hashlib
import os
import threading

AUTH_USER = os.environ.get("AUTH_USERNAME", "")
AUTH_PASS = os.environ.get("AUTH_PASSWORD", "")

SESSION_TOKEN: str | None = (
    hashlib.sha256(f"{AUTH_USER}:{AUTH_PASS}:cam101".encode()).hexdigest()[:32]
    if AUTH_USER else None
)

# ── paths ────────────────────────────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA    = "/data"
PREVIEW_PATH = "stream_preview.jpg"   # relative to CWD=/data — written atomically by stream.py
PID_FILE     = "stream.pid"
ANALYSER_PID = "analyser.pid"
STATE_FILE   = "module_state.json"
LOGS_DIR     = "logs"
STREAM_SCRIPT   = os.path.join(APP_DIR, "stream.py")
ANALYSER_SCRIPT = os.path.join(APP_DIR, "post_analyser.py")

# ── lazy FaceEmbedder singleton ───────────────────────────────────────────────
_embedder      = None
_embedder_lock = threading.Lock()


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from embeddings import FaceEmbedder
                _embedder = FaceEmbedder()
    return _embedder
