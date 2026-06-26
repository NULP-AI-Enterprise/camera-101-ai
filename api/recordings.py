"""Recordings and person-event management."""
from __future__ import annotations

import datetime
import os

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException

from .auth import require_auth
from .deps import get_embedder
from db import PersonEvent, Recording, User, UserFeature, bytes_to_emb, emb_to_bytes, get_session

router = APIRouter()


@router.get("/recordings")
def list_recordings(_=Depends(require_auth)):
    with get_session() as db:
        recs  = db.query(Recording).order_by(Recording.start_time.desc()).limit(50).all()
        umap  = {u.id: u.name for u in db.query(User).all()}
        return [
            {
                "id":           r.id,
                "start_time":   r.start_time.isoformat() if r.start_time else None,
                "end_time":     r.end_time.isoformat()   if r.end_time   else None,
                "video_path":   r.video_path,
                "video_exists": bool(r.video_path and os.path.exists(r.video_path)),
                "events": [
                    {
                        "id":              e.id,
                        "user_id":         e.user_id,
                        "user_name":       umap.get(e.user_id) if e.user_id else None,
                        "track_id":        e.track_id,
                        "first_seen":      e.first_seen.isoformat()  if e.first_seen  else None,
                        "last_seen":       e.last_seen.isoformat()   if e.last_seen   else None,
                        "snapshot_path":   e.snapshot_path,
                        "snapshot_exists": bool(e.snapshot_path and os.path.exists(e.snapshot_path)),
                    }
                    for e in db.query(PersonEvent).filter_by(recording_id=r.id).all()
                ],
            }
            for r in recs
        ]


@router.post("/events/{event_id}/reanalyse")
def reanalyse_event(event_id: int, _=Depends(require_auth)):
    from post_analyser import PostAnalyser
    with get_session() as db:
        ev  = db.get(PersonEvent, event_id)
        if not ev:
            raise HTTPException(404, "Event not found")
        rec = db.get(Recording, ev.recording_id)
        if not rec or not rec.video_path or not os.path.exists(rec.video_path):
            raise HTTPException(404, "Video file not found")
        video_path  = rec.video_path
        scene_start = rec.start_time
        first_seen  = ev.first_seen
        last_seen   = ev.last_seen or datetime.datetime.now()

    with get_session() as db:
        db_embeddings = [
            (r.user_id, bytes_to_emb(r.embedding))
            for r in db.query(UserFeature).all()
        ]

    analyser = PostAnalyser(get_embedder(), db_embeddings)
    uid = analyser.analyse_event_sync(event_id, video_path, scene_start, first_seen, last_seen)

    name = None
    if uid:
        with get_session() as db:
            u = db.get(User, uid)
            name = u.name if u else uid

    return {"user_id": uid, "user_name": name}


@router.post("/events/{event_id}/assign-user")
def assign_event_user(event_id: int, user_id: str = Form(""), _=Depends(require_auth)):
    """Link an unrecognized event to an existing user and save their face embedding."""
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(400, "user_id is required")

    embedded   = False
    user_name  = user_id   # fallback before session opens
    snap_path  = None

    with get_session() as db:
        ev = db.get(PersonEvent, event_id)
        if not ev:
            raise HTTPException(404, "Event not found")
        u = db.get(User, user_id)
        if not u:
            raise HTTPException(404, f"User '{user_id}' not found")
        user_name  = u.name          # read while session is open
        snap_path  = ev.snapshot_path
        ev.user_id = user_id

    # Extract face embedding outside the DB session to avoid holding the
    # connection open during the (potentially slow) InsightFace call.
    if snap_path and os.path.exists(snap_path):
        try:
            img = cv2.imread(snap_path)
            if img is not None:
                emb = get_embedder().get_embedding(img)
                if emb is not None:
                    with get_session() as db:
                        db.add(UserFeature(user_id=user_id,
                                           embedding=emb_to_bytes(emb)))
                    embedded = True
        except Exception:
            pass

    return {"ok": True, "user_name": user_name, "embedding_added": embedded}


@router.delete("/recordings/all")
def clear_recordings(_=Depends(require_auth)):
    with get_session() as db:
        db.query(PersonEvent).delete()
        db.query(Recording).delete()
    return {"ok": True}
