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
STREAM_FPS       = 25.0
DETECT_WIDTH     = 480          # width for MOG2 processing (saves CPU)
MIN_MOTION_PX    = 2000         # changed-pixel threshold (~human-sized motion)
COOLDOWN_SECS    = 5.0          # silence before closing a recording
MIN_RECORD_SECS  = 1.0          # discard clips shorter than this
PREVIEW_EVERY    = 5            # write preview JPEG every N frames
RAW_EVENTS_DIR   = "raw_events"
PREVIEW_PATH     = "stream_preview.jpg"
PREVIEW_TMP      = "stream_preview.tmp.jpg"   # must end in .jpg for OpenCV codec detection
PID_FILE         = "stream.pid"
ENCODE_QUEUE_MAX = 150
LOG_FPS_EVERY    = 300          # log live FPS every N frames

# ── shutdown event — set by SIGTERM / SIGINT ──────────────────────────────────
_shutdown = threading.Event()

def _handle_stop(sig, frame):          # noqa: ARG001
    log.info("signal %d — shutting down", sig)
    _shutdown.set()

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT,  _handle_stop)


# ── async H.264 writer ────────────────────────────────────────────────────────

class AsyncVideoWriter:
    """Encodes BGR frames to H.264/MP4 in a background thread."""

    def __init__(self, filename: str, fps: float, w: int, h: int):
        self.filename = filename
        self.dropped  = 0
        self._q: queue.Queue = queue.Queue(maxsize=ENCODE_QUEUE_MAX)
        self._done    = threading.Event()
        threading.Thread(
            target=self._run, args=(filename, fps, w, h),
            daemon=True, name=f"enc-{os.path.basename(filename)}",
        ).start()

    def write(self, frame_bgr: np.ndarray) -> None:
        try:
            self._q.put_nowait(frame_bgr)
        except queue.Full:
            self.dropped += 1

    def close(self, timeout: float = 30.0) -> bool:
        """Signal end-of-stream. Returns True if flushed cleanly, False if timed out."""
        self._q.put(None)
        return self._done.wait(timeout=timeout)

    def _run(self, filename: str, fps: float, w: int, h: int) -> None:
        try:
            out    = av.open(filename, mode="w")
            stream = out.add_stream("h264", rate=int(fps))
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            stream.options = {"crf": "23", "preset": "ultrafast", "tune": "zerolatency"}
            while True:
                item = self._q.get()
                if item is None:
                    break
                vf = av.VideoFrame.from_ndarray(item[:, :, ::-1], format="rgb24")
                for pkt in stream.encode(vf.reformat(format="yuv420p")):
                    out.mux(pkt)
            for pkt in stream.encode():
                out.mux(pkt)
            out.close()
        except Exception as e:
            log.error("encoder: %s", e)
        finally:
            self._done.set()


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


# ── motion-triggered recorder ─────────────────────────────────────────────────

class MotionRecorder:
    """
    MOG2-based motion detection.
    Writes raw_event_TIMESTAMP.mp4 + a .ready marker when done.
    """

    def __init__(self, fps: float):
        self._fps      = fps
        self._mog2     = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        self._writer:    AsyncVideoWriter | None  = None
        self._rec_start: datetime.datetime | None = None
        self._rec_path:  str | None               = None
        self._last_motion: float = 0.0
        self._recording:   bool  = False
        os.makedirs(RAW_EVENTS_DIR, exist_ok=True)

    def process(self, frame: np.ndarray, small: np.ndarray) -> bool:
        """
        Feed full-resolution frame (recorded) + downscaled frame (for MOG2).
        Returns True while recording.
        """
        mask      = self._mog2.apply(small)
        motion_px = cv2.countNonZero(mask)
        now_mono  = time.monotonic()
        now_wall  = datetime.datetime.now()

        if motion_px >= MIN_MOTION_PX:
            self._last_motion = now_mono
            if not self._recording:
                self._start(frame, now_wall)

        if self._recording:
            if now_mono - self._last_motion > COOLDOWN_SECS:
                self._stop(now_wall)
            else:
                self._writer.write(frame)

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
        self._writer    = AsyncVideoWriter(path, self._fps, w, h)
        self._rec_path  = path
        self._rec_start = now
        self._recording = True
        log.info("MOTION START → %s  res=%dx%d", os.path.basename(path), w, h)

    def _stop(self, now: datetime.datetime) -> None:
        self._recording = False
        if not self._writer:
            return
        dur = (now - self._rec_start).total_seconds() if self._rec_start else 0
        flushed = self._writer.close()
        if not flushed:
            log.error("encoder timed out for %s — skipping analysis (incomplete file)",
                      os.path.basename(self._rec_path or ""))
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
    log.info("startup  pid=%d", os.getpid())
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
                label = "REC" if is_rec else "LIVE"
                color = (0, 0, 220) if is_rec else (180, 180, 0)
                disp  = img.copy()
                cv2.putText(disp, f"{label}  {fps_disp:.0f} fps",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                if cv2.imwrite(PREVIEW_TMP, disp):
                    os.replace(PREVIEW_TMP, PREVIEW_PATH)

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
