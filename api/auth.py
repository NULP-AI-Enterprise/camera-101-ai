"""Cookie-based session auth: /login and /logout endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Cookie, HTTPException, Form, Response
from fastapi.responses import JSONResponse

from .deps import AUTH_USER, AUTH_PASS, SESSION_TOKEN

router = APIRouter()


def require_auth(session: str = Cookie(default="")) -> None:
    if SESSION_TOKEN and session != SESSION_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def is_authed(session: str = Cookie(default="")) -> bool:
    return not SESSION_TOKEN or session == SESSION_TOKEN


@router.post("/login")
async def login(username: str = Form(""), password: str = Form("")):
    if not AUTH_USER or (username == AUTH_USER and password == AUTH_PASS):
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            "session", SESSION_TOKEN or "anon",
            httponly=True, samesite="lax", max_age=86400 * 30, path="/",
        )
        return resp
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return Response(status_code=302, headers={"Location": "/"})
