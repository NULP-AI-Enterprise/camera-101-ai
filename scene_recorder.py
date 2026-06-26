"""
Scene recorder — one continuous video per occupied scene, N people per recording.

  SCENE ABSENT ──first person──► SCENE ACTIVE (open shared AsyncVideoWriter)
  SCENE ACTIVE ──all left + timeout──► SCENE ABSENT (flush, submit to analyser)

  Per-person:
    PERSON ABSENT ──detected──► PERSON ACTIVE (create PersonEvent, save snapshot)
    PERSON ACTIVE ──lost──► PERSON COOLDOWN (short grace period)
    PERSON COOLDOWN ──detected──► PERSON ACTIVE (cancel timer, resume)
    PERSON COOLDOWN ──timeout──► PERSON GONE (update last_seen)

  When the last person goes GONE the scene cooldown timer starts.
  Any new detection cancels the scene cooldown and starts a new person if needed.
"""
from __future__ import annotations

import datetime
import os
import threading
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

from db import get_session, Recording, PersonEvent
from log_setup import get_logger
from video_writer import AsyncVideoWriter

if TYPE_CHECKING:
    from video_analyser import VideoAnalyser

log = get_logger("scene_recorder", "scene_recorder.log")

RECORDINGS_DIR   = "recordings"
SNAPSHOTS_DIR    = "snapshots"
PERSON_COOLDOWN  = 3.0    # seconds: brief occlusion grace period (spec: 2–3 s)
SCENE_COOLDOWN   = 5.0    # seconds: after last person leaves before closing file
MIN_SCENE_SEC    = 2.0    # discard recordings shorter than this


# ── per-person state ──────────────────────────────────────────────────────────

@dataclass
class _PersonState:
    event_id:       int
    first_seen:     datetime.datetime
    last_seen:      Optional[datetime.datetime]  = None
    active:         bool                         = True
    snapshot_saved: bool                         = False
    snapshot_path:  Optional[str]               = None
    cooldown_timer: Optional[threading.Timer]    = None


# ── scene recorder ────────────────────────────────────────────────────────────

class SceneRecorder:
    """
    on_detected(track_id, frame, bbox) — call every frame a confirmed track is visible
    on_lost(track_id)                  — call when tracker drops a person
    set_analyser(analyser)             — wire up post-scene recognition
    release_all()                      — flush on shutdown
    """

    def __init__(self, fps: float = 25.0):
        self._fps     = fps
        self._lock    = threading.Lock()
        self._analyser: Optional[VideoAnalyser] = None

        self._writer:       Optional[AsyncVideoWriter]  = None
        self._recording_id: Optional[int]               = None
        self._scene_start:  Optional[datetime.datetime] = None
        self._video_path:   Optional[str]               = None
        self._scene_timer:  Optional[threading.Timer]   = None

        self._persons: dict[int, _PersonState] = {}

        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    def set_analyser(self, analyser: "VideoAnalyser") -> None:
        self._analyser = analyser

    # ── public ────────────────────────────────────────────────────────────────

    def on_detected(self, track_id: int, frame: np.ndarray, bbox: list[int]) -> None:
        with self._lock:
            # Open a new scene if none is active
            if self._writer is None:
                self._open_scene(frame)

            # Cancel scene cooldown — someone is still present
            if self._scene_timer:
                self._scene_timer.cancel()
                self._scene_timer = None

            # Write frame into the shared recording
            if self._writer:
                self._writer.write(frame)

            now = datetime.datetime.now()

            if track_id not in self._persons:
                # New person entering this scene
                self._add_person(track_id, frame, bbox, now)
            else:
                ps = self._persons[track_id]
                if not ps.active:
                    # Returned during cooldown — resume
                    if ps.cooldown_timer:
                        ps.cooldown_timer.cancel()
                        ps.cooldown_timer = None
                    ps.active = True
                ps.last_seen = now
                if not ps.snapshot_saved:
                    self._save_snapshot(ps, frame, bbox, track_id)

    def on_lost(self, track_id: int) -> None:
        with self._lock:
            if track_id not in self._persons:
                return
            ps = self._persons[track_id]
            if not ps.active:
                return
            timer = threading.Timer(PERSON_COOLDOWN, self._expire_person, args=(track_id,))
            timer.daemon = True
            timer.start()
            ps.cooldown_timer = timer

    def release_all(self) -> None:
        with self._lock:
            if self._scene_timer:
                self._scene_timer.cancel()
                self._scene_timer = None
            for ps in self._persons.values():
                if ps.cooldown_timer:
                    ps.cooldown_timer.cancel()
                    ps.cooldown_timer = None
            self._close_scene()

    # ── internals ─────────────────────────────────────────────────────────────

    def _open_scene(self, frame: np.ndarray) -> None:
        now  = datetime.datetime.now()
        path = os.path.join(RECORDINGS_DIR, f"scene_{now.strftime('%Y%m%d_%H%M%S')}.mp4")
        h, w = frame.shape[:2]
        self._writer      = AsyncVideoWriter(path, self._fps, w, h)
        self._video_path  = path
        self._scene_start = now
        try:
            with get_session() as db:
                rec = Recording(start_time=now, video_path=path)
                db.add(rec)
                db.flush()
                self._recording_id = rec.id
        except Exception as e:
            log.error("DB error opening scene: %s", e)
        log.info("SCENE START → %s", path)

    def _add_person(self, track_id: int, frame: np.ndarray,
                    bbox: list[int], now: datetime.datetime) -> None:
        event_id = None
        try:
            with get_session() as db:
                ev = PersonEvent(
                    recording_id = self._recording_id,
                    track_id     = track_id,
                    first_seen   = now,
                    last_seen    = now,
                )
                db.add(ev)
                db.flush()
                event_id = ev.id
        except Exception as e:
            log.error("DB error adding person: %s", e)
            return
        ps = _PersonState(event_id=event_id, first_seen=now, last_seen=now)
        self._persons[track_id] = ps
        self._save_snapshot(ps, frame, bbox, track_id)
        log.info("person tid=%d → event #%d", track_id, event_id)

    def _save_snapshot(self, ps: _PersonState, frame: np.ndarray,
                       bbox: list[int], track_id: int) -> None:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad  = max(4, int((y2 - y1) * 0.08))
        crop = frame[max(0, y1-pad):min(h, y2+pad),
                     max(0, x1-pad):min(w, x2+pad)]
        if crop.size == 0:
            return
        path = os.path.join(
            SNAPSHOTS_DIR,
            f"snap_tid{track_id}_{ps.first_seen.strftime('%Y%m%d_%H%M%S')}.jpg",
        )
        if not cv2.imwrite(path, crop):
            return
        ps.snapshot_saved = True
        ps.snapshot_path  = path
        try:
            with get_session() as db:
                ev = db.get(PersonEvent, ps.event_id)
                if ev:
                    ev.snapshot_path = path
        except Exception as e:
            log.error("snapshot DB error: %s", e)

    def _expire_person(self, track_id: int) -> None:
        with self._lock:
            if track_id not in self._persons:
                return
            ps = self._persons[track_id]
            ps.active         = False
            ps.cooldown_timer = None
            now               = datetime.datetime.now()
            ps.last_seen      = now
            try:
                with get_session() as db:
                    ev = db.get(PersonEvent, ps.event_id)
                    if ev:
                        ev.last_seen = now
            except Exception as e:
                log.error("DB error expiring person: %s", e)
            log.info("person tid=%d left", track_id)

            # If no active persons remain, start the scene cooldown
            if not any(p.active for p in self._persons.values()):
                timer = threading.Timer(SCENE_COOLDOWN, self._expire_scene)
                timer.daemon = True
                timer.start()
                self._scene_timer = timer

    def _expire_scene(self) -> None:
        with self._lock:
            self._close_scene()

    def _close_scene(self) -> None:
        if self._writer is None:
            return

        now      = datetime.datetime.now()
        duration = (now - self._scene_start).total_seconds() if self._scene_start else 0

        self._writer.close()
        if self._writer.dropped:
            log.warning("dropped %d encoded frames", self._writer.dropped)
        self._writer = None

        if duration < MIN_SCENE_SEC:
            self._discard_scene()
            log.info("SCENE DISCARDED (%.1fs too short)", duration)
        else:
            self._finalise_scene(now)
            log.info("SCENE STOP → %s (%.1fs, %d person(s))",
                     self._video_path, duration, len(self._persons))

        self._recording_id = None
        self._scene_start  = None
        self._video_path   = None
        self._persons      = {}

    def _discard_scene(self) -> None:
        try:
            with get_session() as db:
                for ps in self._persons.values():
                    ev = db.get(PersonEvent, ps.event_id)
                    if ev:
                        db.delete(ev)
                if self._recording_id:
                    rec = db.get(Recording, self._recording_id)
                    if rec:
                        db.delete(rec)
        except Exception as e:
            log.error("discard DB error: %s", e)
        if self._video_path and os.path.exists(self._video_path):
            try:
                os.remove(self._video_path)
            except OSError:
                pass
        for ps in self._persons.values():
            if ps.snapshot_path and os.path.exists(ps.snapshot_path):
                try:
                    os.remove(ps.snapshot_path)
                except OSError:
                    pass

    def _finalise_scene(self, now: datetime.datetime) -> None:
        try:
            with get_session() as db:
                if self._recording_id:
                    rec = db.get(Recording, self._recording_id)
                    if rec:
                        rec.end_time = now
                for ps in self._persons.values():
                    ev = db.get(PersonEvent, ps.event_id)
                    if ev and not ev.last_seen:
                        ev.last_seen = now
        except Exception as e:
            log.error("finalise DB error: %s", e)

        if self._analyser and self._recording_id and self._video_path:
            events_payload = [
                {
                    "event_id":   ps.event_id,
                    "track_id":   tid,
                    "first_seen": ps.first_seen,
                    "last_seen":  ps.last_seen or now,
                }
                for tid, ps in self._persons.items()
            ]
            self._analyser.submit(
                self._recording_id,
                self._video_path,
                self._scene_start,
                events_payload,
            )
