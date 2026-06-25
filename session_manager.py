"""
Session manager — state machine per track_id.

    ABSENT ──detect──► ACTIVE   (open AsyncVideoWriter + DB session)
    ACTIVE ──lost───► COOLDOWN  (start cooldown timer)
    COOLDOWN ──detect► ACTIVE   (cancel timer, resume)
    COOLDOWN ──timeout► ABSENT  (flush writer → submit to VideoAnalyser → close DB)

Live stream passes NO user_id — recognition happens after the session ends
via VideoAnalyser (post-session, on the saved video file).
"""
from __future__ import annotations

import datetime
import os
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

import av
import numpy as np

from db import get_session, UserSession

if TYPE_CHECKING:
    from video_analyser import VideoAnalyser

RECORDINGS_DIR    = "recordings"
COOLDOWN_SECONDS  = 8.0    # wait longer before closing — reduces fragmentation
MIN_SESSION_SEC   = 3.0    # discard sessions shorter than this (no face possible)
ENCODE_QUEUE_MAX  = 150    # ~6 s buffer at 25 fps before frames are dropped


# ── async H.264 writer ────────────────────────────────────────────────────────

class AsyncVideoWriter:
    """Non-blocking H.264 encoder in a dedicated thread."""

    def __init__(self, filename: str, fps: float, w: int, h: int):
        self.filename = filename
        self.dropped  = 0
        self._q: queue.Queue = queue.Queue(maxsize=ENCODE_QUEUE_MAX)
        self._done    = threading.Event()
        self._t = threading.Thread(
            target=self._encode_loop, args=(filename, fps, w, h),
            daemon=True, name=f"enc-{os.path.basename(filename)}",
        )
        self._t.start()

    def write(self, frame_bgr: np.ndarray) -> None:
        try:
            self._q.put_nowait(frame_bgr)
        except queue.Full:
            self.dropped += 1

    def close(self, timeout: float = 20.0) -> None:
        self._q.put(None)
        self._done.wait(timeout=timeout)

    def _encode_loop(self, filename: str, fps: float, w: int, h: int) -> None:
        try:
            out    = av.open(filename, mode="w")
            stream = out.add_stream("h264", rate=int(fps))
            stream.width   = w
            stream.height  = h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "26", "preset": "ultrafast", "tune": "zerolatency"}

            while True:
                item = self._q.get()
                if item is None:
                    break
                rgb = item[:, :, ::-1]
                vf  = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                vf  = vf.reformat(format="yuv420p")
                for pkt in stream.encode(vf):
                    out.mux(pkt)

            for pkt in stream.encode():
                out.mux(pkt)
            out.close()
        except Exception as e:
            print(f"[AsyncVideoWriter] error: {e}")
        finally:
            self._done.set()


# ── track state ───────────────────────────────────────────────────────────────

class State(Enum):
    ABSENT   = auto()
    ACTIVE   = auto()
    COOLDOWN = auto()


@dataclass
class _TrackState:
    state:          State                      = State.ABSENT
    db_session_id:  Optional[int]              = None
    writer:         Optional[AsyncVideoWriter]  = None
    video_path:     Optional[str]              = None
    cooldown_timer: Optional[threading.Timer]  = None
    start_time:     Optional[datetime.datetime] = None


# ── session manager ───────────────────────────────────────────────────────────

class SessionManager:
    """
    on_detected(track_id, frame) — call every frame a person is visible
    on_lost(track_id)            — call when tracker drops the person
    set_analyser(analyser)       — wire up post-session recognition
    release_all()                — call on shutdown
    """

    def __init__(self, fps: float = 25.0):
        self._fps      = fps
        self._tracks: dict[int, _TrackState] = {}
        self._lock     = threading.Lock()
        self._analyser: Optional[VideoAnalyser] = None
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    def set_analyser(self, analyser: "VideoAnalyser") -> None:
        self._analyser = analyser

    # ── public ────────────────────────────────────────────────────────────────

    def on_detected(self, track_id: int, frame: np.ndarray) -> None:
        with self._lock:
            ts = self._get(track_id)

            if ts.state == State.ABSENT:
                self._start(ts, track_id, frame)

            elif ts.state == State.COOLDOWN:
                if ts.cooldown_timer:
                    ts.cooldown_timer.cancel()
                    ts.cooldown_timer = None
                ts.state = State.ACTIVE

            if ts.writer:
                ts.writer.write(frame)

    def on_lost(self, track_id: int) -> None:
        with self._lock:
            if track_id not in self._tracks:
                return
            ts = self._tracks[track_id]
            if ts.state == State.ACTIVE:
                ts.state          = State.COOLDOWN
                ts.cooldown_timer = threading.Timer(
                    COOLDOWN_SECONDS, self._expire, args=(track_id,)
                )
                ts.cooldown_timer.daemon = True
                ts.cooldown_timer.start()

    def active_sessions(self) -> list[dict]:
        with self._lock:
            return [
                {"track_id": tid, "state": ts.state.name}
                for tid, ts in self._tracks.items()
                if ts.state != State.ABSENT
            ]

    def release_all(self) -> None:
        with self._lock:
            for ts in self._tracks.values():
                if ts.cooldown_timer:
                    ts.cooldown_timer.cancel()
                self._close(ts)
            self._tracks.clear()

    # ── internals ─────────────────────────────────────────────────────────────

    def _get(self, track_id: int) -> _TrackState:
        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackState()
        return self._tracks[track_id]

    def _start(self, ts: _TrackState, track_id: int, frame: np.ndarray) -> None:
        ts.state      = State.ACTIVE
        ts.start_time = datetime.datetime.now()

        now  = ts.start_time
        path = os.path.join(
            RECORDINGS_DIR,
            f"tid{track_id}_{now.strftime('%Y%m%d_%H%M%S')}.mp4",
        )
        ts.video_path = path
        h, w          = frame.shape[:2]
        ts.writer     = AsyncVideoWriter(path, self._fps, w, h)
        print(f"[SessionManager] REC START → {path}")

        try:
            with get_session() as db:
                sess = UserSession(
                    user_id    = None,     # filled in later by VideoAnalyser
                    track_id   = track_id,
                    start_time = now,
                    video_path = path,
                )
                db.add(sess)
                db.flush()
                ts.db_session_id = sess.id
        except Exception as e:
            print(f"[SessionManager] DB error on start: {e}")

    def _close(self, ts: _TrackState) -> None:
        now = datetime.datetime.now()

        # Check duration before doing anything expensive
        too_short = (
            ts.start_time is not None
            and (now - ts.start_time).total_seconds() < MIN_SESSION_SEC
        )

        if ts.writer:
            ts.writer.close()
            if ts.writer.dropped:
                print(f"[SessionManager] dropped {ts.writer.dropped} frame(s)")
            ts.writer = None

        if ts.db_session_id:
            if too_short:
                # Discard — not worth analysing, clean up DB row and video file
                try:
                    with get_session() as db:
                        row = db.get(UserSession, ts.db_session_id)
                        if row:
                            db.delete(row)
                except Exception as e:
                    print(f"[SessionManager] DB cleanup error: {e}")
                if ts.video_path and os.path.exists(ts.video_path):
                    try:
                        os.remove(ts.video_path)
                    except OSError:
                        pass
                print(f"[SessionManager] DISCARDED short session → {ts.video_path}")
            else:
                try:
                    with get_session() as db:
                        row = db.get(UserSession, ts.db_session_id)
                        if row:
                            row.end_time = now
                except Exception as e:
                    print(f"[SessionManager] DB error on close: {e}")

                if self._analyser and ts.video_path:
                    self._analyser.submit(ts.db_session_id, ts.video_path)

                print(f"[SessionManager] REC STOP  → {ts.video_path}")

            ts.db_session_id = None

        ts.state      = State.ABSENT
        ts.video_path = None
        ts.start_time = None

    def _expire(self, track_id: int) -> None:
        with self._lock:
            if track_id not in self._tracks:
                return
            ts = self._tracks[track_id]
            if ts.state == State.COOLDOWN:
                self._close(ts)
