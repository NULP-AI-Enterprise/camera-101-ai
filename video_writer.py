"""
Shared async H.264/MP4 encoder used by stream.py, scene_recorder.py, session_manager.py.

Encodes BGR frames in a dedicated background thread so the caller never blocks.
"""
from __future__ import annotations

import logging
import os
import queue
import threading

import av
import numpy as np

log = logging.getLogger("video_writer")

ENCODE_QUEUE_MAX = 150


class AsyncVideoWriter:
    """Non-blocking H.264 encoder. Frames are queued and encoded in a background thread."""

    def __init__(self, filename: str, fps: float, w: int, h: int, crf: str = "23"):
        self.filename = filename
        self.dropped  = 0
        self._q: queue.Queue = queue.Queue(maxsize=ENCODE_QUEUE_MAX)
        self._done    = threading.Event()
        threading.Thread(
            target=self._run, args=(filename, fps, w, h, crf),
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

    def _run(self, filename: str, fps: float, w: int, h: int, crf: str) -> None:
        try:
            out    = av.open(filename, mode="w")
            stream = out.add_stream("h264", rate=int(fps))
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            stream.options = {"crf": crf, "preset": "ultrafast", "tune": "zerolatency"}
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
            log.error("encode error in %s: %s", os.path.basename(filename), e)
        finally:
            self._done.set()
