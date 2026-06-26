"""
Module B — Post-event analysis: person tracking + face recognition.

Watches raw_events/ for .mp4.ready marker files produced by Module A (stream.py).
For each raw video:
  1. Decode frames via PyAV
  2. Apple Vision body detection (VNDetectHumanRectanglesRequest) — same model used
     in Module A, no extra dependencies
  3. 4-stage IOUTracker with reid_window=45 s (equivalent to ByteTrack track_buffer≈5s)
  4. Per-track face sampling — "Lock" logic: first frame identified with
     confidence >= LOCK_THRESHOLD marks the entire track as that user_id
  5. False-positive filter: tracks < MIN_TRACK_FRAMES with no face → discard
  6. Empty-video filter: no persons detected at all → delete video
  7. Write Recording + PersonEvent rows to DB

If ultralytics + torch are installed, YOLO + ByteTrack is used instead of Vision.
Falls back to Apple Vision automatically if torch is unavailable.

Also used by admin_app.py for synchronous per-event re-identification (no threads).

Run standalone:
    python post_analyser.py
"""
from __future__ import annotations

import datetime
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import av
import cv2
import numpy as np

from db import get_session, Recording, PersonEvent, UserFeature, bytes_to_emb
from embeddings import FaceEmbedder
from log_setup import get_logger
from tracker import IOUTracker

log = get_logger("analyser", "analyser.log")

# ── tuneable ──────────────────────────────────────────────────────────────────
RAW_EVENTS_DIR   = "raw_events"
MIN_WORKERS        = int(os.environ.get("ANALYSER_MIN_WORKERS", "1"))
MAX_WORKERS        = int(os.environ.get("ANALYSER_WORKERS",     "3"))
SCALE_UP_THRESHOLD = int(os.environ.get("SCALE_UP_THRESHOLD",  "2"))
WORKER_IDLE_SECS   = float(os.environ.get("WORKER_IDLE_SECS",  "180"))
SNAPSHOTS_DIR    = "snapshots"
PID_FILE         = "analyser.pid"
POLL_INTERVAL    = 2.0      # seconds between folder scans
MIN_TRACK_FRAMES = 20       # tracks shorter than this with no face → false positive
LOCK_THRESHOLD   = 0.65     # similarity to "lock" a track to a specific user
FACE_SAMPLE_RATE = 5        # try face recognition every N frames of a track
FACE_UPPER_FRAC  = 0.60     # crop upper 60% of bbox (head + shoulders area)
DETECT_WIDTH     = 640      # resize for detection (Vision or YOLO)
BODY_CONF_THRESH = 0.40     # Vision body confidence gate
DETECT_EVERY     = 5        # run body detector every N frames (interpolate in between)
RAW_EVENTS_MAX_AGE_HOURS  = float(os.environ.get("RAW_EVENTS_MAX_AGE_HOURS",  "24"))
RECORDINGS_MAX_AGE_HOURS  = float(os.environ.get("RECORDINGS_MAX_AGE_HOURS", "168"))  # 7 days
RECORDINGS_DIR            = "recordings"

_TRACKER_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bytetrack_custom.yaml")

# ── per-thread model cache ────────────────────────────────────────────────────
# Each worker thread keeps its own FaceEmbedder + YOLO so they never share
# mutable inference state.  Models are loaded lazily on first use.
_tl = threading.local()


def _thread_embedder() -> FaceEmbedder:
    if not hasattr(_tl, "embedder"):
        log.info("worker %s: loading embedder…", threading.current_thread().name)
        _tl.embedder = FaceEmbedder()
    return _tl.embedder


def _thread_yolo():
    if not hasattr(_tl, "yolo"):
        from ultralytics import YOLO
        log.info("worker %s: loading YOLO…", threading.current_thread().name)
        _tl.yolo = YOLO("yolov8n.pt")
    else:
        # Reset predictor so ByteTrack tracker state doesn't bleed between videos
        _tl.yolo.predictor = None
    return _tl.yolo


def _thread_analyser() -> "PostAnalyser":
    """Return a per-thread PostAnalyser; reload DB embeddings on each call."""
    if not hasattr(_tl, "analyser"):
        embedder = _thread_embedder()
        with get_session() as db:
            rows = db.query(UserFeature).all()
            db_embs = [(r.user_id, bytes_to_emb(r.embedding)) for r in rows]
        _tl.analyser = PostAnalyser(embedder, db_embs)
        log.info("worker %s: ready", threading.current_thread().name)
    else:
        _tl.analyser.reload_db()
    return _tl.analyser


# ── per-track accumulator ─────────────────────────────────────────────────────

_TRACK_COLORS = [
    (  0, 230,   0),   # green
    (  0, 180, 230),   # cyan
    (230, 130,   0),   # orange
    (200,   0, 200),   # magenta
    (  0, 100, 230),   # blue
    (230, 200,   0),   # yellow
    (  0, 200, 130),   # teal
    (230,   0, 100),   # pink
]


@dataclass
class _Track:
    track_id:    int
    frame_count: int               = 0
    first_frame: int               = 0
    last_frame:  int               = 0
    bbox_first:  list              = field(default_factory=list)
    frame_bboxes: dict             = field(default_factory=dict)   # {frame_idx: [x1,y1,x2,y2]}
    embeddings:  list              = field(default_factory=list)
    locked_uid:  Optional[str]     = None
    best_sim:    float             = 0.0
    snap_path:   Optional[str]     = None   # saved during tracking pass (no extra video open)


# ── Apple Vision detector (lazy singleton) ───────────────────────────────────

_vision_lock    = threading.Lock()
_vision_request = None
_vision_cs      = None


def _vision_detect(img_bgr: np.ndarray,
                   orig_w: int, orig_h: int) -> list[list[int]]:
    """Run VNDetectHumanRectanglesRequest on img_bgr. Returns [[x1,y1,x2,y2],…]."""
    global _vision_request, _vision_cs
    with _vision_lock:
        if _vision_request is None:
            log.debug("initialising Apple Vision detector…")
            import Quartz
            import Vision as VN
            _vision_cs      = Quartz.CGColorSpaceCreateDeviceRGB()
            _vision_request = VN.VNDetectHumanRectanglesRequest.alloc().init()

        import Quartz
        import Vision as VN

        rgb  = np.ascontiguousarray(img_bgr[:, :, ::-1])
        data = rgb.tobytes()
        prov = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
        h_img, w_img = img_bgr.shape[:2]
        cgimg = Quartz.CGImageCreate(
            w_img, h_img, 8, 24, w_img * 3, _vision_cs,
            Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaNone,
            prov, None, False, Quartz.kCGRenderingIntentDefault,
        )
        handler = VN.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimg, {})
        handler.performRequests_error_([_vision_request], None)

        out = []
        for obs in (_vision_request.results() or []):
            if float(obs.confidence()) < BODY_CONF_THRESH:
                continue
            bb = obs.boundingBox()
            x1 = int(bb.origin.x * orig_w)
            y1 = int((1.0 - bb.origin.y - bb.size.height) * orig_h)
            x2 = int((bb.origin.x + bb.size.width) * orig_w)
            y2 = int((1.0 - bb.origin.y) * orig_h)
            out.append([max(0, x1), max(0, y1),
                        min(orig_w, x2), min(orig_h, y2)])
        return out


def _analyse_with_vision(video_path: str,
                          embedder: FaceEmbedder,
                          db_embeddings: list,
                          lock_threshold: float,
                          db_lock: threading.Lock) -> tuple[dict, int]:
    """
    Decode video, run Vision detection + IOUTracker, collect face embeddings.
    Returns (tracks dict, total frame count).
    """
    tracker = IOUTracker(
        iou_threshold        = 0.25,
        max_missed           = 90,
        min_age              = 2,
        max_centroid_dist    = 200.0,
        appearance_threshold = 0.60,
        reid_window          = 45.0,
    )
    tracks:    dict[int, _Track] = {}
    frame_idx: int               = 0

    container = av.open(video_path)
    vid       = container.streams.video[0]

    for av_frame in container.decode(vid):
        img      = av_frame.to_ndarray(format="bgr24")
        orig_h, orig_w = img.shape[:2]

        if frame_idx % DETECT_EVERY == 0:
            scale = DETECT_WIDTH / orig_w
            small = cv2.resize(img, (DETECT_WIDTH, int(orig_h * scale)),
                               interpolation=cv2.INTER_LINEAR)
            dets = _vision_detect(small, orig_w, orig_h)
        else:
            dets = []

        active = tracker.update(dets, img)

        for tid, bbox in active:
            if tid not in tracks:
                tracks[tid] = _Track(
                    track_id    = tid,
                    first_frame = frame_idx,
                    bbox_first  = list(bbox),
                    snap_path   = _save_snapshot_inline(img, list(bbox), tid),
                )
            tr = tracks[tid]
            tr.frame_count += 1
            tr.last_frame   = frame_idx
            tr.frame_bboxes[frame_idx] = list(bbox)

            if tr.locked_uid is None and tr.frame_count % FACE_SAMPLE_RATE == 0:
                emb, sim, uid = _try_face(img, list(bbox), embedder,
                                          db_embeddings, db_lock)
                if emb is not None:
                    tr.embeddings.append(emb)
                    if uid and sim >= lock_threshold:
                        tr.locked_uid = uid
                        tr.best_sim   = sim
                        log.info("LOCKED track %d → %s  sim=%.2f", tid, uid, sim)

        frame_idx += 1

    container.close()
    return tracks, frame_idx


def _analyse_with_yolo(video_path: str,
                        embedder: FaceEmbedder,
                        db_embeddings: list,
                        lock_threshold: float,
                        db_lock: threading.Lock) -> tuple[dict, int]:
    """
    YOLO v8n + ByteTrack tracking (used when torch/ultralytics is available).
    """
    model      = _thread_yolo()
    tracks:    dict[int, _Track] = {}
    frame_idx: int               = 0
    cfg = _TRACKER_CFG if os.path.exists(_TRACKER_CFG) else "bytetrack.yaml"

    for result in model.track(
        source  = video_path,
        stream  = True,
        persist = True,
        tracker = cfg,
        classes = [0],
        conf    = 0.4,
        verbose = False,
    ):
        if result.boxes is None or result.boxes.id is None:
            frame_idx += 1
            continue

        img    = result.orig_img
        ids    = result.boxes.id.cpu().numpy().astype(int)
        bboxes = result.boxes.xyxy.cpu().numpy().astype(int)

        for track_id, bbox in zip(ids, bboxes):
            tid = int(track_id)
            if tid not in tracks:
                tracks[tid] = _Track(
                    track_id    = tid,
                    first_frame = frame_idx,
                    bbox_first  = bbox.tolist(),
                    snap_path   = _save_snapshot_inline(img, bbox.tolist(), tid),
                )
            tr = tracks[tid]
            tr.frame_count += 1
            tr.last_frame   = frame_idx
            tr.frame_bboxes[frame_idx] = bbox.tolist()

            if tr.locked_uid is None and tr.frame_count % FACE_SAMPLE_RATE == 0:
                emb, sim, uid = _try_face(img, bbox.tolist(), embedder,
                                          db_embeddings, db_lock)
                if emb is not None:
                    tr.embeddings.append(emb)
                    if uid and sim >= lock_threshold:
                        tr.locked_uid = uid
                        tr.best_sim   = sim
                        log.info("LOCKED track %d → %s  sim=%.2f", tid, uid, sim)

        frame_idx += 1

    return tracks, frame_idx


# ── analysis engine ───────────────────────────────────────────────────────────

class PostAnalyser:
    """
    Stateless analysis engine — no background threads.
    Thread-safe (DB and embedding access guarded by _db_lock).
    """

    def __init__(self, embedder: FaceEmbedder, db_embeddings: list,
                 lock_threshold: float = LOCK_THRESHOLD):
        self._embedder  = embedder
        self._threshold = lock_threshold
        self._db        = list(db_embeddings)
        self._db_lock   = threading.Lock()

        self._use_yolo = False
        try:
            import torch          # noqa: F401
            import ultralytics    # noqa: F401
            self._use_yolo = True
            log.info("backend: YOLO v8n + ByteTrack")
        except ImportError:
            log.info("backend: Apple Vision + IOUTracker")

    def reload_db(self) -> None:
        with get_session() as db:
            rows = db.query(UserFeature).all()
            fresh = [(r.user_id, bytes_to_emb(r.embedding)) for r in rows]
        with self._db_lock:
            self._db = fresh
        log.debug("DB reloaded — %d embedding(s)", len(fresh))

    # ── full-video analysis (Module B main path) ──────────────────────────────

    def analyse(self, video_path: str) -> bool:
        """
        Run detection + tracking + face recognition on a raw video file.
        Returns True  — valid persons found (Recording saved to DB).
        Returns False — false positive (file deleted).
        """
        basename  = os.path.basename(video_path)
        file_mb   = os.path.getsize(video_path) / 1_048_576 if os.path.exists(video_path) else 0
        worker    = threading.current_thread().name
        t0        = time.monotonic()
        log.info("[%s] analysing %s  (%.1f MB)", worker, basename, file_mb)

        with self._db_lock:
            db_copy = list(self._db)

        try:
            if self._use_yolo:
                tracks, frame_idx = _analyse_with_yolo(
                    video_path, self._embedder, db_copy,
                    self._threshold, self._db_lock)
            else:
                tracks, frame_idx = _analyse_with_vision(
                    video_path, self._embedder, db_copy,
                    self._threshold, self._db_lock)
        except Exception as e:
            log.error("tracking error on %s: %s", basename, e)
            return False

        elapsed  = time.monotonic() - t0
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0
        log.debug("[%s] tracking done — %d track(s)  %d frames  %.1fs  (%.0f fps)",
                  worker, len(tracks), frame_idx, elapsed, proc_fps)

        # ── no persons at all → false positive ───────────────────────────────
        if not tracks:
            log.warning("[%s] no persons detected in %s — deleting  (%.1fs)",
                        worker, basename, time.monotonic() - t0)
            _remove_file(video_path)
            return False

        # ── resolve unlocked tracks; filter noise ─────────────────────────────
        valid: list[_Track] = []
        with self._db_lock:
            db_copy = list(self._db)

        for tr in tracks.values():
            if tr.locked_uid is None:
                if tr.frame_count < MIN_TRACK_FRAMES and not tr.embeddings:
                    log.debug("skip track %d (%d frames, no face) — noise",
                              tr.track_id, tr.frame_count)
                    continue
                if tr.embeddings:
                    # Strategy: check every individual frame embedding AND the
                    # averaged embedding; keep the highest-confidence match.
                    # Individual frames: a single clear frontal shot beats a
                    # blurred average across many profile-view frames.
                    best_uid, best_sim = None, 0.0
                    for emb in tr.embeddings:
                        uid, sim = FaceEmbedder.best_match(emb, db_copy, threshold=0.45)
                        if uid and sim > best_sim:
                            best_uid, best_sim = uid, sim
                    # Averaged embedding (more robust when many similar angles)
                    avg = np.mean(tr.embeddings, axis=0).astype(np.float32)
                    avg /= np.linalg.norm(avg) + 1e-8
                    uid, sim = FaceEmbedder.best_match(avg, db_copy, threshold=0.48)
                    if uid and sim > best_sim:
                        best_uid, best_sim = uid, sim
                    if best_uid:
                        tr.locked_uid = best_uid
                        tr.best_sim   = best_sim
                        log.debug("resolved track %d → %s (best sim=%.0f%%  frames=%d)",
                                  tr.track_id, best_uid, best_sim * 100,
                                  len(tr.embeddings))
            valid.append(tr)

        if not valid:
            log.warning("[%s] all tracks filtered as noise — deleting %s  (%.1fs)",
                        worker, basename, time.monotonic() - t0)
            _remove_file(video_path)
            return False

        fps   = _get_video_fps(video_path)
        start = _parse_start_time(video_path)
        end   = start + datetime.timedelta(seconds=frame_idx / fps)

        _annotate_video(video_path, valid, fps)
        self._write_db(video_path, valid, start, end, fps)
        names_str = ", ".join(
            f"{tr.locked_uid} ({tr.best_sim:.0%})" if tr.locked_uid
            else f"unknown#{tr.track_id}"
            for tr in valid
        )
        log.info("[%s] ✓ %s  %d person(s): %s  %.1fs",
                 worker, basename, len(valid), names_str, time.monotonic() - t0)
        return True

    # ── sync per-event re-analysis (admin "Re-analyze" button) ───────────────

    def analyse_event_sync(
        self,
        event_id:    int,
        video_path:  str,
        scene_start: datetime.datetime,
        first_seen:  datetime.datetime,
        last_seen:   datetime.datetime,
    ) -> Optional[str]:
        """
        Re-identify a single PersonEvent from its time window in the video.
        Samples face embeddings between first_seen and last_seen, updates DB.
        """
        t_in  = max(0.0, (first_seen  - scene_start).total_seconds())
        t_out = max(t_in + 0.5, (last_seen - scene_start).total_seconds())

        log.debug("re-analyse event %d  t=%.1f–%.1f", event_id, t_in, t_out)
        embeddings = _sample_faces(video_path, t_in, t_out, self._embedder, n=15)
        if not embeddings:
            embeddings = _sample_faces(video_path, t_in, t_out, self._embedder, n=40)

        with self._db_lock:
            db_copy = list(self._db)

        uid = None
        if embeddings:
            avg = np.mean(embeddings, axis=0).astype(np.float32)
            avg /= np.linalg.norm(avg) + 1e-8
            uid, sim = FaceEmbedder.best_match(avg, db_copy, threshold=0.48)
            if uid:
                log.info("re-analyse event %d → %s (sim=%.2f)", event_id, uid, sim)
            else:
                log.info("re-analyse event %d → unknown", event_id)
        else:
            log.warning("re-analyse event %d — no face embeddings found", event_id)

        try:
            with get_session() as db:
                ev = db.get(PersonEvent, event_id)
                if ev:
                    ev.user_id = uid
        except Exception as e:
            log.error("DB write error (event %d): %s", event_id, e)

        return uid

    # ── internals ─────────────────────────────────────────────────────────────

    def _write_db(self, video_path: str, tracks: list[_Track],
                  start: datetime.datetime, end: datetime.datetime,
                  fps: float) -> None:
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        try:
            with get_session() as db:
                rec = Recording(start_time=start, end_time=end, video_path=video_path)
                db.add(rec)
                db.flush()
                rec_id = rec.id
                for tr in tracks:
                    first_seen = start + datetime.timedelta(
                        seconds=tr.first_frame / fps)
                    last_seen  = start + datetime.timedelta(
                        seconds=tr.last_frame  / fps)
                    # Snapshot already saved during tracking pass — no extra video open needed
                    snap_path = tr.snap_path
                    db.add(PersonEvent(
                        recording_id  = rec_id,
                        user_id       = tr.locked_uid,
                        track_id      = tr.track_id,
                        first_seen    = first_seen,
                        last_seen     = last_seen,
                        snapshot_path = snap_path,
                    ))
            names = [tr.locked_uid or f"unknown#{tr.track_id}" for tr in tracks]
            log.info("DB recording #%d saved — %d person(s): %s  ← %s",
                     rec_id, len(tracks), ", ".join(names), os.path.basename(video_path))
        except Exception as e:
            log.error("DB error: %s", e)


# ── video annotation ──────────────────────────────────────────────────────────

def _annotate_video(video_path: str, tracks: list[_Track], fps: float) -> None:
    """
    Second pass: decode raw video, draw person rectangles + name labels,
    overwrite the file in place.
    """
    uid_to_name: dict[str, str] = {}
    try:
        from db import User as _User
        with get_session() as db:
            for u in db.query(_User).all():
                uid_to_name[u.id] = u.name
    except Exception:
        pass

    tmp_path = video_path + ".ann_tmp.mp4"
    basename = os.path.basename(video_path)
    try:
        in_c   = av.open(video_path)
        in_vid = in_c.streams.video[0]
        w = in_vid.codec_context.width
        h = in_vid.codec_context.height

        out_c      = av.open(tmp_path, mode="w")
        out_stream = out_c.add_stream("h264", rate=int(fps))
        out_stream.width, out_stream.height, out_stream.pix_fmt = w, h, "yuv420p"
        out_stream.options = {"crf": "22", "preset": "fast"}

        last_bbox: dict[int, list[int]] = {}

        for fidx, av_frame in enumerate(in_c.decode(in_vid)):
            img = av_frame.to_ndarray(format="bgr24")

            for tr in tracks:
                if fidx in tr.frame_bboxes:
                    last_bbox[tr.track_id] = tr.frame_bboxes[fidx]
                if not (tr.first_frame <= fidx <= tr.last_frame):
                    continue
                bbox = last_bbox.get(tr.track_id)
                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox
                color = _TRACK_COLORS[tr.track_id % len(_TRACK_COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                if tr.locked_uid:
                    label = uid_to_name.get(tr.locked_uid, tr.locked_uid)
                else:
                    label = f"Unknown #{tr.track_id}"

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                lx, ly = x1, max(y1 - 4, th + 8)
                cv2.rectangle(img,
                              (lx, ly - th - 8), (lx + tw + 8, ly),
                              color, cv2.FILLED)
                cv2.putText(img, label, (lx + 4, ly - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2, cv2.LINE_AA)

            vf = av.VideoFrame.from_ndarray(img[:, :, ::-1], format="rgb24")
            for pkt in out_stream.encode(vf.reformat(format="yuv420p")):
                out_c.mux(pkt)

        for pkt in out_stream.encode():
            out_c.mux(pkt)

        out_c.close()
        in_c.close()
        os.replace(tmp_path, video_path)
        log.info("annotated %s", basename)

    except Exception as e:
        log.error("annotation error on %s: %s", basename, e)
        try:
            in_c.close()
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── stateless helpers ─────────────────────────────────────────────────────────

def _try_face(img: np.ndarray, bbox: list,
              embedder: FaceEmbedder,
              db_embeddings: list,
              db_lock: threading.Lock) -> tuple:
    """Crop upper body area, run InsightFace, return (emb, sim, uid)."""
    x1, y1, x2, y2 = bbox
    h_box = y2 - y1
    crop  = img[max(0, y1) : max(0, y1 + int(h_box * FACE_UPPER_FRAC)),
                max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return None, 0.0, None
    emb = embedder.get_embedding(crop)
    if emb is None:
        return None, 0.0, None
    with db_lock:
        db_copy = list(db_embeddings)
    uid, sim = FaceEmbedder.best_match(emb, db_copy, threshold=0.48)
    return emb, sim, uid


def _save_snapshot_inline(img: np.ndarray, bbox: list, track_id: int) -> Optional[str]:
    """Save a person crop during the tracking pass — avoids a third video open."""
    try:
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        pad  = max(4, int((y2 - y1) * 0.08))
        crop = img[max(0, y1 - pad):min(h, y2 + pad),
                   max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            return None
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        path = os.path.join(SNAPSHOTS_DIR, f"snap_tid{track_id}_inline.jpg")
        return path if cv2.imwrite(path, crop) else None
    except Exception as e:
        log.debug("inline snapshot error: %s", e)
        return None


def _remove_file(path: str) -> None:
    for p in (path, path + ".ready"):
        try:
            os.remove(p)
        except OSError:
            pass


def _get_video_fps(path: str) -> float:
    try:
        c   = av.open(path)
        fps = float(c.streams.video[0].average_rate or 25)
        c.close()
        return fps
    except Exception:
        return 25.0


def _parse_start_time(video_path: str) -> datetime.datetime:
    name = os.path.basename(video_path)
    if name.startswith("raw_event_"):
        ts_str = name[len("raw_event_"):].replace(".mp4", "")
        try:
            return datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.datetime.now()


def _sample_faces(video_path: str, t_in: float, t_out: float,
                  embedder: FaceEmbedder, n: int) -> list[np.ndarray]:
    """Sample up to n face embeddings from the [t_in, t_out] second window."""
    embeddings: list[np.ndarray] = []
    try:
        container = av.open(video_path)
        vid       = container.streams.video[0]
        container.seek(int(t_in * 1_000_000), backward=True)
        step   = (t_out - t_in) / max(n, 1)
        next_t = t_in
        for frame in container.decode(vid):
            if frame.pts is None:
                continue
            t = float(frame.pts) * float(vid.time_base)
            if t < t_in:
                continue
            if t > t_out:
                break
            if t >= next_t:
                img = frame.to_ndarray(format="bgr24")
                emb = embedder.get_embedding(img)
                if emb is not None:
                    embeddings.append(emb)
                    if len(embeddings) >= n:
                        break
                next_t = t + step
        container.close()
    except Exception as e:
        log.error("_sample_faces: %s", e)
    return embeddings


def _extract_snapshot(video_path: str, tr: _Track, rec_id: int) -> Optional[str]:
    """Grab a snapshot of the person from their first frame in the video."""
    try:
        container = av.open(video_path)
        vid       = container.streams.video[0]
        fps_av    = float(vid.average_rate or 25)
        t_target  = tr.first_frame / fps_av
        container.seek(int(t_target * 1_000_000), backward=True)
        for frame in container.decode(vid):
            if frame.pts is None:
                continue
            t = float(frame.pts) * float(vid.time_base)
            if t >= t_target - 0.5:
                img = frame.to_ndarray(format="bgr24")
                if tr.bbox_first:
                    x1, y1, x2, y2 = tr.bbox_first
                    h, w = img.shape[:2]
                    pad  = max(4, int((y2 - y1) * 0.08))
                    crop = img[max(0, y1-pad):min(h, y2+pad),
                               max(0, x1-pad):min(w, x2+pad)]
                    if crop.size > 0:
                        path = os.path.join(
                            SNAPSHOTS_DIR,
                            f"snap_rec{rec_id}_tid{tr.track_id}.jpg",
                        )
                        cv2.imwrite(path, crop)
                        container.close()
                        return path
                break
        container.close()
    except Exception as e:
        log.error("snapshot error: %s", e)
    return None


# ── auto-scaling thread pool ──────────────────────────────────────────────────

class _DynamicPool:
    """
    Thread pool that starts with min_workers and scales up to max_workers
    when the pending queue depth reaches scale_threshold.
    Excess workers self-terminate after idle_secs of inactivity.
    """

    def __init__(self, min_workers: int, max_workers: int,
                 scale_threshold: int, idle_secs: float) -> None:
        self._min       = min_workers
        self._max       = max_workers
        self._threshold = scale_threshold
        self._idle      = idle_secs
        self._q: queue.Queue   = queue.Queue()
        self._lock              = threading.Lock()
        self._n_workers: int    = 0
        for _ in range(min_workers):
            self._spawn()

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def submit(self, fn, *args) -> None:
        self._q.put((fn, args))
        with self._lock:
            if self._q.qsize() >= self._threshold and self._n_workers < self._max:
                self._spawn()
                log.info("analyser: queue depth %d → scaling up to %d/%d workers",
                         self._q.qsize(), self._n_workers, self._max)

    def shutdown(self, **_) -> None:
        pass  # worker threads are daemon threads — exit with the process

    def _spawn(self) -> None:
        idx = self._n_workers
        self._n_workers += 1
        name = f"analyser_{idx}"
        threading.Thread(target=self._run, args=(name,),
                         daemon=True, name=name).start()

    def _run(self, name: str) -> None:
        while True:
            try:
                fn, args = self._q.get(timeout=self._idle)
            except queue.Empty:
                with self._lock:
                    if self._n_workers > self._min:
                        self._n_workers -= 1
                        log.info("analyser: worker %s idle → scaled down to %d/%d workers",
                                 name, self._n_workers, self._max)
                        return
                continue  # minimum worker — keep waiting
            try:
                fn(*args)
            except Exception as e:
                log.error("pool worker %s: %s", name, e)
            finally:
                self._q.task_done()


# ── daemon watcher ────────────────────────────────────────────────────────────

class AnalyserDaemon:
    """
    Polls raw_events/ every POLL_INTERVAL seconds and dispatches videos to a
    thread pool that auto-scales from MIN_WORKERS to MAX_WORKERS.
    Each worker thread owns its own FaceEmbedder + YOLO instance.
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._processing: set[str] = set()
        self._lock       = threading.Lock()
        self._pool       = _DynamicPool(
            min_workers=MIN_WORKERS,
            max_workers=MAX_WORKERS,
            scale_threshold=SCALE_UP_THRESHOLD,
            idle_secs=WORKER_IDLE_SECS,
        )

    def run_forever(self) -> None:
        os.makedirs(RAW_EVENTS_DIR, exist_ok=True)
        log.info("watching %s/  (poll %.1fs  workers=%d..%d  scale_at=%d  max_age=%.0fh)",
                 RAW_EVENTS_DIR, POLL_INTERVAL, MIN_WORKERS, MAX_WORKERS,
                 SCALE_UP_THRESHOLD, RAW_EVENTS_MAX_AGE_HOURS)
        while not self._stop_event.is_set():
            self._scan()
            self._cleanup_old_files()
            self._stop_event.wait(POLL_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()
        self._pool.shutdown(wait=False, cancel_futures=False)

    def _scan(self) -> None:
        try:
            entries = sorted(os.listdir(RAW_EVENTS_DIR))
        except FileNotFoundError:
            return

        for name in entries:
            if not name.endswith(".mp4.ready"):
                continue
            marker = os.path.join(RAW_EVENTS_DIR, name)
            video  = marker[: -len(".ready")]

            with self._lock:
                if video in self._processing:
                    continue
                if not os.path.exists(video):
                    try:
                        os.remove(marker)
                    except OSError:
                        pass
                    continue
                try:
                    os.remove(marker)
                except OSError:
                    continue
                self._processing.add(video)

            n = len(self._processing)
            log.info("▶ %s  (queued: %d  workers: %d/%d)",
                     os.path.basename(video), n,
                     self._pool.n_workers, MAX_WORKERS)
            self._pool.submit(self._worker, video)

    def _cleanup_old_files(self) -> None:
        """Remove old raw_events, recordings, snapshots and orphaned DB rows."""
        self._cleanup_dir(RAW_EVENTS_DIR,  RAW_EVENTS_MAX_AGE_HOURS,  skip=self._processing)
        self._cleanup_recordings(RECORDINGS_MAX_AGE_HOURS)

    def _cleanup_dir(self, directory: str, max_age_hours: float,
                     skip: set | None = None) -> None:
        cutoff = time.time() - max_age_hours * 3600
        try:
            entries = os.listdir(directory)
        except FileNotFoundError:
            return
        for name in entries:
            path = os.path.join(directory, name)
            if skip and (path in skip or path.removesuffix(".ready") in skip):
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    log.info("retention: removed %s", path)
            except OSError:
                pass

    def _cleanup_recordings(self, max_age_hours: float) -> None:
        """Delete old recording videos + their snapshots + DB rows."""
        cutoff_ts = time.time() - max_age_hours * 3600
        import datetime as _dt
        cutoff_dt = _dt.datetime.fromtimestamp(cutoff_ts)
        try:
            with get_session() as db:
                old_recs = (db.query(Recording)
                            .filter(Recording.start_time < cutoff_dt)
                            .all())
                for rec in old_recs:
                    events = db.query(PersonEvent).filter_by(recording_id=rec.id).all()
                    for ev in events:
                        if ev.snapshot_path and os.path.exists(ev.snapshot_path):
                            try:
                                os.remove(ev.snapshot_path)
                            except OSError:
                                pass
                        db.delete(ev)
                    if rec.video_path and os.path.exists(rec.video_path):
                        try:
                            os.remove(rec.video_path)
                        except OSError:
                            pass
                    db.delete(rec)
                if old_recs:
                    log.info("retention: removed %d recording(s) older than %.0fh",
                             len(old_recs), max_age_hours)
        except Exception as e:
            log.error("retention cleanup error: %s", e)

    def _worker(self, video_path: str) -> None:
        try:
            ok = _thread_analyser().analyse(video_path)
            # analyse() returns False and deletes the file for false-positives;
            # if the file is still here, it was a transient open/read error.
            # Retry once after a short delay (race with encoder finishing the file).
            if not ok and os.path.exists(video_path):
                log.info("[retry] video still on disk after failure — retrying in 5s: %s",
                         os.path.basename(video_path))
                time.sleep(5)
                _thread_analyser().analyse(video_path)
        except Exception as e:
            log.error("worker error on %s: %s", os.path.basename(video_path), e)
        finally:
            with self._lock:
                self._processing.discard(video_path)


# ── standalone entry point ────────────────────────────────────────────────────

def main():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    _shutdown = threading.Event()

    def _sig(sig, frame):          # noqa: ARG001
        log.info("signal %d — stopping…", sig)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    log.info("startup  pid=%d  workers=%d..%d  scale_at=%d",
             os.getpid(), MIN_WORKERS, MAX_WORKERS, SCALE_UP_THRESHOLD)
    # Models are loaded lazily by each worker thread on first video.

    daemon = AnalyserDaemon()
    t = threading.Thread(target=daemon.run_forever, daemon=True)
    t.start()

    _shutdown.wait()
    log.info("stopping…")
    daemon.stop()
    t.join(timeout=5)
    try:
        os.remove(PID_FILE)
    except OSError:
        pass
    log.info("exit")


if __name__ == "__main__":
    main()
