"""
FastAPI entry point — assembles all API routers.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8501
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.auth import is_authed, router as auth_router
from api.control import router as control_router, watchdog_start
from api.deps import APP_DIR, limiter
from api.media import router as media_router
from api.recordings import router as recordings_router
from api.users import router as users_router
from log_setup import get_logger

log = get_logger("server", "server.log")

app = FastAPI(docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(auth_router)
app.include_router(control_router, prefix="/api")
app.include_router(users_router,   prefix="/api")
app.include_router(recordings_router, prefix="/api")
app.include_router(media_router)


@app.on_event("startup")
async def startup() -> None:
    log.info("server starting (pid=%d)", os.getpid())
    watchdog_start()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def root(authed: bool = Depends(is_authed)):
    html_path = os.path.join(APP_DIR, "static", "app.html")
    with open(html_path, "r") as f:
        html = f.read()
    return HTMLResponse(html.replace("__AUTHED__", "true" if authed else "false"))
