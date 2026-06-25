"""
Post-scene face recognition.

After a scene recording closes, VideoAnalyser receives the video path and a list
of PersonEvents (each with a time window inside the video).  For each person it
seeks to their window, samples face embeddings, and writes the identified user_id
back to the DB.

Runs in a single background thread — never touches the live stream.
"""
from __future__ import annotations

import datetime
import queue
import threading
from typing import Optional

import av
import numpy as np

from db import get_session, PersonEvent
from embeddings import FaceEmbedder

SAMPLE_FRAMES       = 15   # face samples per person per pass
SAMPLE_FRAMES_RETRY = 40   # denser retry when first pass finds nothing
ID_THRESHOLD        = 0.48


class VideoAnalyser:
    def __init__(self, embedder: FaceEmbedder, db_embeddings: list,
                 threshold: float = ID_THRESHOLD):
        self._embedder  = embedder
        self._threshold = threshold
        self._db        = db_embeddings
        self._db_lock   = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="analyser"
        )
        self._thread.start()

    # ── public ────────────────────────────────────────────────────────────────

    def submit(self, recording_id: int, video_path: str,
               scene_start: datetime.datetime, events: list[dict]) -> None:
        """Queue a completed recording for background recognition."""
        self._q.put((recording_id, video_path, scene_start, events))
        print(f"[Analyser] queued recording #{recording_id} "
              f"({len(events)} person(s)) → {video_path}")

    def reload_db(self, embeddings: list) -> None:
        with self._db_lock:
            self._db = embeddings

    def stop(self) -> None:
        """Graceful stop — lets the current analysis job finish (up to 60 s)."""
        self._q.put(None)
        self._thread.join(timeout=60)

    def stop_now(self) -> None:
        """Fast stop — drain the queue without processing, exit in < 3 s.

        Pending recordings keep their DB rows (user_id = NULL / Unknown).
        Use admin 'Re-analyze all unknowns' to process them later.
        """
        # Drain everything pending so the worker thread doesn't start a new job
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        # Send the sentinel so the worker loop exits
        self._q.put(None)
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            print("[Analyser] stop_now: worker still alive after 3 s (daemon — will die with process)")

    # ── synchronous API for admin re-analysis ─────────────────────────────────

    def analyse_event_sync(self, event_id: int, video_path: str,
                            scene_start: datetime.datetime,
                            first_seen: datetime.datetime,
                            last_seen: datetime.datetime) -> Optional[str]:
        """Re-identify a single PersonEvent synchronously (for admin use)."""
        t_in  = (first_seen - scene_start).total_seconds()
        t_out = (last_seen  - scene_start).total_seconds()
        embeddings = self._sample_window(video_path, t_in, t_out, SAMPLE_FRAMES)
        if not embeddings:
            embeddings = self._sample_window(video_path, t_in, t_out, SAMPLE_FRAMES_RETRY)
        if not embeddings:
            self._write_result(event_id, None)
            return None
        uid = self._identify(embeddings)
        self._write_result(event_id, uid)
        return uid

    # ── worker ────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            recording_id, video_path, scene_start, events = item
            try:
                self._analyse_recording(recording_id, video_path, scene_start, events)
            except Exception as e:
                print(f"[Analyser] error on recording #{recording_id}: {e}")

    def _analyse_recording(self, recording_id: int, video_path: str,
                            scene_start: datetime.datetime,
                            events: list[dict]) -> None:
        print(f"[Analyser] analysing recording #{recording_id} "
              f"({len(events)} person(s))…")

        with self._db_lock:
            db_copy = list(self._db)

        if not db_copy:
            print(f"[Analyser] no registered users — marking all unknown")
            for ev in events:
                self._write_result(ev["event_id"], None)
            return

        for ev in events:
            t_in  = (ev["first_seen"] - scene_start).total_seconds()
            t_out = (ev["last_seen"]  - scene_start).total_seconds()

            embeddings = self._sample_window(video_path, t_in, t_out, SAMPLE_FRAMES)
            if not embeddings:
                print(f"[Analyser] event #{ev['event_id']} tid={ev['track_id']}: "
                      f"no faces on first pass, retrying…")
                embeddings = self._sample_window(
                    video_path, t_in, t_out, SAMPLE_FRAMES_RETRY)

            if not embeddings:
                print(f"[Analyser] event #{ev['event_id']} tid={ev['track_id']}: no faces")
                self._write_result(ev["event_id"], None)
                continue

            uid = self._identify(embeddings, db_copy)
            print(f"[Analyser] event #{ev['event_id']} tid={ev['track_id']} "
                  f"→ {uid}  ({len(embeddings)} faces)")
            self._write_result(ev["event_id"], uid)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sample_window(self, video_path: str, t_in: float, t_out: float,
                       n: int) -> list[np.ndarray]:
        """Seek to [t_in, t_out] seconds in the video and sample up to n face embeddings."""
        embeddings: list[np.ndarray] = []
        t_in  = max(0.0, t_in)
        t_out = max(t_in + 0.5, t_out)

        try:
            container = av.open(video_path)
            vid       = container.streams.video[0]

            # Seek to the start of the person's window (microseconds)
            container.seek(int(t_in * 1_000_000), backward=True)

            duration    = t_out - t_in
            step        = duration / n
            next_sample = t_in

            for frame in container.decode(vid):
                if frame.pts is None:
                    continue
                t = float(frame.pts) * float(vid.time_base)
                if t < t_in:
                    continue
                if t > t_out:
                    break
                if t >= next_sample:
                    img = frame.to_ndarray(format="bgr24")
                    emb = self._embedder.get_embedding(img)
                    if emb is not None:
                        embeddings.append(emb)
                        if len(embeddings) >= n:
                            break
                    next_sample = t + step

            container.close()
        except Exception as e:
            print(f"[Analyser] window sample error: {e}")

        return embeddings

    def _identify(self, embeddings: list[np.ndarray],
                  db_copy: Optional[list] = None) -> Optional[str]:
        if db_copy is None:
            with self._db_lock:
                db_copy = list(self._db)
        if not db_copy:
            return None
        avg = np.mean(embeddings, axis=0).astype(np.float32)
        avg = avg / np.linalg.norm(avg)
        uid, sim = FaceEmbedder.best_match(avg, db_copy, self._threshold)
        return uid

    def _write_result(self, event_id: int, user_id: Optional[str]) -> None:
        try:
            with get_session() as db:
                ev = db.get(PersonEvent, event_id)
                if ev:
                    ev.user_id = user_id
        except Exception as e:
            print(f"[Analyser] DB write error: {e}")
