"""
Module A — Motion-triggered recorder.

Uses MOG2 background subtraction to detect motion (~5% CPU).
When motion is detected, records raw_events/raw_event_TIMESTAMP.mp4.
After COOLDOWN_SECS of no motion, closes the file and drops a .ready
marker so Module B (post_analyser.py) can pick it up.

Run:
    python stream.py
"""
from __future__ import annotations

import datetime
import os
import queue
import signal
import threading
import time

import av
import cv2
import numpy as np

from log_setup import get_logger
from video_writer import AsyncVideoWriter

log = get_logger("recorder", "stream.log")

# ── tuneable ──────────────────────────────────────────────────────────────────
RTSP_URL         = os.environ["RTSP_URL"]   # required — set in k8s secret or .env

# Mask credentials in log output: rtsp://user:pass@host → rtsp://***@host
def _masked_url(url: str) -> str:
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url)
        if p.username or p.password:
            host = p.hostname + (f":{p.port}" if p.port else "")
            return urlunparse(p._replace(netloc=f"***@{host}"))
    except Exception:
        pass
    return url
STREAM_FPS        = 25.0
DETECT_WIDTH      = 480          # width for MOG2 processing (saves CPU)
MIN_MOTION_FRAC        = 0.009   # area of the largest blob as fraction of MOG2-frame area
MOTION_CONFIRM_FRAMES  = 5       # consecutive frames the blob must persist before recording
MIN_BLOB_HEIGHT_FRAC   = float(os.environ.get("MIN_BLOB_HEIGHT_PCT", "8")) / 100
                                  # blob bounding-box height ≥ N % of frame height (default 8 %)
                                  # rejects small repetitive sources (3D printer, fan)
                                  # tune via env var: MIN_BLOB_HEIGHT_PCT=10 for stricter filter
RECORD_EVERY           = 2       # write every Nth frame to H.264 — halves encoder load
                                  # at 25 fps → effective 12.5 fps recording (fine for ID)
COOLDOWN_SECS     = 5.0          # silence before closing a recording
MIN_RECORD_SECS  = 1.0          # discard clips shorter than this
PREVIEW_EVERY    = 5            # submit preview JPEG every N frames
RAW_EVENTS_DIR   = "raw_events"
PREVIEW_PATH     = "stream_preview.jpg"
PREVIEW_TMP      = "stream_preview.tmp.jpg"   # must end in .jpg for OpenCV codec detection
PID_FILE         = "stream.pid"
LOG_FPS_EVERY    = 300          # log live FPS every N frames

# ── shutdown event — set by SIGTERM / SIGINT ──────────────────────────────────
_shutdown = threading.Event()

def _handle_stop(sig, frame):          # noqa: ARG001
    log.info("signal %d — shutting down", sig)
    _shutdown.set()

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT,  _handle_stop)


# ── RTSP decoder — drops stale frames, respects _shutdown ────────────────────

class _FrameDecoder:
    def __init__(self, container, vid_stream):
        self._q: queue.Queue = queue.Queue(maxsize=2)
        threading.Thread(
            target=self._run, args=(container, vid_stream),
            daemon=True, name="decoder",
        ).start()

    def get(self) -> np.ndarray | None:
        while not _shutdown.is_set():
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                continue
        return None

    def _run(self, container, vid_stream) -> None:
        try:
            for av_frame in container.decode(vid_stream):
                if _shutdown.is_set():
                    break
                img = av_frame.to_ndarray(format="bgr24")
                if self._q.full():
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                self._q.put(img)
        except Exception as e:
            if not _shutdown.is_set():
                log.warning("decoder stopped: %s", e)
        finally:
            self._q.put(None)


# ── non-blocking preview writer ───────────────────────────────────────────────

class _PreviewWriter:
    """
    Writes JPEG preview frames in a daemon thread so cv2.imwrite never
    blocks the main decode loop.  Excess frames are dropped (queue size=1).
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=1)
        threading.Thread(target=self._run, daemon=True, name="preview").start()

    def submit(self, frame: np.ndarray, label: str, fps: float) -> None:
        disp  = frame.copy()
        color = (0, 0, 220) if label == "REC" else (180, 180, 0)
        cv2.putText(disp, f"{label}  {fps:.0f} fps",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        try:
            self._q.put_nowait(disp)
        except queue.Full:
            pass  # previous write still pending — skip this frame

    def _run(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                break
            try:
                if cv2.imwrite(PREVIEW_TMP, frame):
                    os.replace(PREVIEW_TMP, PREVIEW_PATH)
            except Exception:
                pass


# ── motion-triggered recorder ─────────────────────────────────────────────────

class MotionRecorder:
    """
    MOG2-based motion detection.
    Writes raw_event_TIMESTAMP.mp4 + a .ready marker when done.
    """

    def __init__(self, fps: float):
        self._fps      = fps
        self._mog2     = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False
        )
        # 7×7 kernel: larger than 5×5 — removes more speckle noise from
        # illumination changes while keeping human-sized motion blobs intact.
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._writer:    AsyncVideoWriter | None  = None
        self._rec_start: datetime.datetime | None = None
        self._rec_path:  str | None               = None
        self._last_motion:   float = 0.0
        self._motion_streak: int   = 0   # consecutive frames with motion >= threshold
        self._rec_frame_idx: int   = 0   # frame counter inside current recording
        self._recording:     bool  = False
        os.makedirs(RAW_EVENTS_DIR, exist_ok=True)

    def process(self, frame: np.ndarray, small: np.ndarray) -> bool:
        """
        Feed full-resolution frame (recorded) + downscaled frame (for MOG2).
        Returns True while recording.
        """
        mask = self._mog2.apply(small)
        # OPEN removes isolated speckles (glare, reflections).
        # CLOSE fills gaps inside real motion blobs.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

        # Use the LARGEST connected blob, not total changed pixels.
        # Distributed small-motion sources (3D printer nozzle, fan, flickering
        # LED) produce many tiny blobs; a person produces one large region.
        # Require both sufficient AREA and minimum HEIGHT (person is tall).
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_motion   = False
        if contours:
            largest   = max(contours, key=cv2.contourArea)
            blob_area = cv2.contourArea(largest)
            _, _, _, blob_h = cv2.boundingRect(largest)
            area_ok   = blob_area >= small.shape[0] * small.shape[1] * MIN_MOTION_FRAC
            height_ok = blob_h    >= small.shape[0] * MIN_BLOB_HEIGHT_FRAC
            is_motion = area_ok and height_ok
            if area_ok and not height_ok:
                log.debug("motion suppressed: blob_h=%dpx (%.1f%% of %dpx frame) — raise MIN_BLOB_HEIGHT_PCT to detect larger objects only",
                          blob_h, blob_h / small.shape[0] * 100, small.shape[0])

        now_mono = time.monotonic()
        now_wall = datetime.datetime.now()

        if is_motion:
            self._motion_streak += 1
            self._last_motion = now_mono
            if not self._recording and self._motion_streak >= MOTION_CONFIRM_FRAMES:
                self._start(frame, now_wall)
        else:
            self._motion_streak = 0

        if self._recording:
            if now_mono - self._last_motion > COOLDOWN_SECS:
                self._stop(now_wall)
            else:
                if self._rec_frame_idx % RECORD_EVERY == 0:
                    self._writer.write(frame)
                self._rec_frame_idx += 1

        return self._recording

    def flush(self) -> None:
        """Force-close any open recording on shutdown (no .ready marker)."""
        if self._recording and self._writer:
            self._recording = False
            self._writer.close(timeout=10.0)
            log.info("flushed partial clip → %s", self._rec_path)
        self._writer   = None
        self._rec_path = None

    # ── internals ────────────────────────────────────────────────────────────

    def _start(self, frame: np.ndarray, now: datetime.datetime) -> None:
        ts   = now.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RAW_EVENTS_DIR, f"raw_event_{ts}.mp4")
        h, w = frame.shape[:2]
        self._writer        = AsyncVideoWriter(path, self._fps, w, h)
        self._rec_path      = path
        self._rec_start     = now
        self._rec_frame_idx = 0
        self._recording     = True
        log.info("MOTION START → %s  res=%dx%d", os.path.basename(path), w, h)

    def _stop(self, now: datetime.datetime) -> None:
        self._recording = False
        if not self._writer:
            return
        dur = (now - self._rec_start).total_seconds() if self._rec_start else 0
        flushed = self._writer.close(timeout=120)   # 1080p on busy box can take >30s
        if not flushed:
            log.error("encoder timed out for %s — deleting incomplete file",
                      os.path.basename(self._rec_path or ""))
            try:
                os.remove(self._rec_path)
            except OSError:
                pass
            self._writer = None
            self._rec_path = None
            self._rec_start = None
            return
        if self._writer.dropped:
            log.warning("%d frames dropped — %s",
                        self._writer.dropped, os.path.basename(self._rec_path or ""))
        self._writer = None
        path = self._rec_path
        self._rec_path  = None
        self._rec_start = None

        if dur < MIN_RECORD_SECS:
            try:
                os.remove(path)
            except OSError:
                pass
            log.debug("clip discarded (%.1fs < %.1fs) — %s",
                      dur, MIN_RECORD_SECS, os.path.basename(path))
            return

        # Signal Module B
        open(path + ".ready", "w").close()
        log.info("SAVED %s (%.1fs) → queued for analysis", os.path.basename(path), dur)


# ── main ──────────────────────────────────────────────────────────────────────

CONNECT_RETRY_SECS = 5


def main():
    log.info(
        "startup  pid=%d  blob_height_min=%.0f%%  confirm_frames=%d  motion_frac=%.3f",
        os.getpid(),
        MIN_BLOB_HEIGHT_FRAC * 100,
        MOTION_CONFIRM_FRAMES,
        MIN_MOTION_FRAC,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        _run()
    finally:
        for p in (PID_FILE, PREVIEW_PATH, PREVIEW_TMP):
            try:
                os.remove(p)
            except OSError:
                pass
        log.info("exit")


def _run() -> None:
    opts = {"rtsp_transport": "tcp", "stimeout": "8000000", "max_delay": "500000"}

    # ── connect with retry ────────────────────────────────────────────────────
    container = None
    while not _shutdown.is_set():
        try:
            log.info("connecting to %s …", _masked_url(RTSP_URL))
            container = av.open(RTSP_URL, options=opts)
            break
        except Exception as e:
            log.error("cannot connect (%s) — retry in %ds", e, CONNECT_RETRY_SECS)
            _shutdown.wait(CONNECT_RETRY_SECS)

    if container is None or _shutdown.is_set():
        return

    vid = container.streams.video[0]
    vid.thread_type = "AUTO"
    w = vid.codec_context.width or 0
    h = vid.codec_context.height or 0
    log.info("connected  %dx%d @ %.0ffps", w, h, STREAM_FPS)

    decoder  = _FrameDecoder(container, vid)
    recorder = MotionRecorder(fps=STREAM_FPS)
    preview  = _PreviewWriter()

    frame_idx = 0
    fps_t0, fps_count, fps_disp = time.time(), 0, 0.0

    try:
        while not _shutdown.is_set():
            img = decoder.get()
            if img is None:
                if _shutdown.is_set():
                    break
                log.warning("stream lost — reconnecting…")
                try:
                    container.close()
                except Exception:
                    pass
                _shutdown.wait(CONNECT_RETRY_SECS)
                if _shutdown.is_set():
                    break
                try:
                    container = av.open(RTSP_URL, options=opts)
                    vid       = container.streams.video[0]
                    vid.thread_type = "AUTO"
                    decoder   = _FrameDecoder(container, vid)
                    log.info("reconnected")
                except Exception as e:
                    log.error("reconnect failed: %s", e)
                continue

            orig_h, orig_w = img.shape[:2]
            frame_idx += 1

            scale = DETECT_WIDTH / orig_w
            small = cv2.resize(img, (DETECT_WIDTH, int(orig_h * scale)),
                               interpolation=cv2.INTER_LINEAR)
            is_rec = recorder.process(img, small)

            fps_count += 1
            t = time.time()
            if t - fps_t0 >= 1.0:
                fps_disp  = fps_count / (t - fps_t0)
                fps_count, fps_t0 = 0, t

            if frame_idx % LOG_FPS_EVERY == 0:
                state = "REC" if is_rec else "IDLE"
                log.debug("stream  %.1f fps  frame=%d  state=%s  preview_ok=%s",
                          fps_disp, frame_idx, state,
                          os.path.exists(PREVIEW_PATH))

            if frame_idx % PREVIEW_EVERY == 0:
                preview.submit(img, "REC" if is_rec else "LIVE", fps_disp)

    except Exception as e:
        if not _shutdown.is_set():
            log.error("stream error: %s", e)
    finally:
        log.info("shutting down…")
        _shutdown.set()
        recorder.flush()
        try:
            container.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
