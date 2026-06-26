"""User CRUD and face embedding management."""
from __future__ import annotations

import datetime

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from .auth import require_auth
from .deps import get_embedder
from db import User, UserFeature, bytes_to_emb, emb_to_bytes, get_session

router = APIRouter()


@router.get("/users")
def list_users(_=Depends(require_auth)):
    with get_session() as db:
        users = db.query(User).order_by(User.created_at.desc()).all()
        return [
            {
                "id": u.id,
                "name": u.name,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "embedding_count": db.query(UserFeature).filter_by(user_id=u.id).count(),
            }
            for u in users
        ]


@router.post("/users")
async def create_user(user_id: str = Form(""), name: str = Form(""),
                      _=Depends(require_auth)):
    uid, nm = user_id.strip(), name.strip()
    if not uid or not nm:
        raise HTTPException(400, "user_id and name are required")
    with get_session() as db:
        if db.get(User, uid):
            raise HTTPException(409, "User ID already exists")
        db.add(User(id=uid, name=nm, created_at=datetime.datetime.now(datetime.timezone.utc)))
    return {"ok": True}


@router.delete("/users/{user_id}")
def delete_user(user_id: str, _=Depends(require_auth)):
    with get_session() as db:
        u = db.get(User, user_id)
        if u:
            db.delete(u)
    return {"ok": True}


@router.post("/users/{user_id}/embeddings")
async def add_embedding(user_id: str, file: UploadFile = File(...),
                        _=Depends(require_auth)):
    data = await file.read()
    img  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")

    try:
        emb = get_embedder().get_embedding(img)
    except Exception as e:
        raise HTTPException(500, f"Embedder error: {e}")

    if emb is None:
        raise HTTPException(422, "No face detected in image")

    with get_session() as db:
        existing = [
            bytes_to_emb(r.embedding)
            for r in db.query(UserFeature).filter_by(user_id=user_id).all()
        ]

    if existing:
        max_sim = max(float(np.dot(emb, e)) for e in existing)
        if max_sim > 0.82:
            raise HTTPException(409,
                f"Angle too similar to existing ({max_sim:.0%}), try a different pose")

    with get_session() as db:
        db.add(UserFeature(user_id=user_id, embedding=emb_to_bytes(emb)))

    return {"ok": True, "embedding_count": len(existing) + 1}


@router.post("/face-check")
async def face_check(file: UploadFile = File(...), _=Depends(require_auth)):
    """Check whether an image contains a detectable face (used during Add User scan)."""
    data = await file.read()
    img  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"face_detected": False}
    try:
        emb = get_embedder().get_embedding(img)
        return {"face_detected": emb is not None}
    except Exception:
        return {"face_detected": False}
