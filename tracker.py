"""
Multi-stage IOU tracker with appearance Re-ID.

Matching pipeline per frame
───────────────────────────
 Stage 1  IOU matching           — standard position overlap
 Stage 2  Centroid distance      — handles fast movement between detection cycles
 Stage 3  Appearance histogram   — handles larger position jumps (same clothing)
 Stage 4  Dead-pool Re-ID        — handles true absences (left frame, came back)

Stages 1–3 all operate on ACTIVE tracks, so the same person never gets a new
track ID just because they moved faster than IOU can follow.
Stage 4 covers the case where the track actually expired (> max_missed frames).
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── appearance descriptor ─────────────────────────────────────────────────────

_HIST_BINS = 16          # per-channel bins in the HSV clothing histogram


def _body_hist(frame_bgr: np.ndarray, bbox: list) -> Optional[np.ndarray]:
    """L2-normalised HSV histogram of the lower 2/3 of a bounding box (clothing)."""
    x1, y1, x2, y2 = bbox
    h    = max(1, y2 - y1)
    crop = frame_bgr[y1 + h // 3 : y2, max(0, x1) : x2]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None,
                        [_HIST_BINS, _HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
    return hist.flatten()


def _hist_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def _centroid(bbox: list) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _dist(a: list, b: list) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _iou(a: list, b: list) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── track ─────────────────────────────────────────────────────────────────────

class Track:
    _counter = 0

    def __init__(self, bbox: list):
        Track._counter += 1
        self.id     = Track._counter
        self.bbox   = list(bbox)
        self.missed = 0
        self.age    = 0
        self.hist: Optional[np.ndarray] = None  # clothing appearance

    def update(self, bbox: list, hist: Optional[np.ndarray] = None) -> None:
        self.bbox   = list(bbox)
        self.missed = 0
        self.age   += 1
        if hist is not None:
            self.hist = hist


# ── tracker ───────────────────────────────────────────────────────────────────

class IOUTracker:
    """
    Parameters
    ----------
    iou_threshold       : IOU overlap needed for Stage 1 match
    max_missed          : frames a track survives with no match before expiry
    min_age             : frames before a new track is reported (noise gate)
    max_centroid_dist   : pixel distance limit for Stage 2 centroid matching
    appearance_threshold: histogram cosine similarity for Stage 3 / Stage 4
    reid_window         : seconds dead-pool entries stay alive
    """

    def __init__(self,
                 iou_threshold:        float = 0.25,
                 max_missed:           int   = 90,
                 min_age:              int   = 2,
                 max_centroid_dist:    float = 200.0,
                 appearance_threshold: float = 0.60,
                 reid_window:          float = 45.0):
        self.iou_threshold        = iou_threshold
        self.max_missed           = max_missed
        self.min_age              = min_age
        self.max_centroid_dist    = max_centroid_dist
        self.appearance_threshold = appearance_threshold
        self.reid_window          = reid_window
        self.tracks: List[Track]  = []
        self._dead_pool: List[dict] = []

    # ── public ────────────────────────────────────────────────────────────────

    def update(self,
               detections: List[list],
               frame: Optional[np.ndarray] = None
               ) -> List[Tuple[int, list]]:
        now = time.time()

        # Expire old dead-pool entries
        self._dead_pool = [d for d in self._dead_pool
                           if now - d["died_at"] < self.reid_window]

        # Refresh appearance descriptors for currently visible tracks
        if frame is not None:
            for t in self.tracks:
                if t.missed == 0:
                    h = _body_hist(frame, t.bbox)
                    if h is not None:
                        t.hist = h

        used_tracks: set = set()
        used_dets:   set = set()

        # ── Stage 1: IOU matching ─────────────────────────────────────────────
        for d_idx, det in enumerate(detections):
            best_iou, best_t = self.iou_threshold, None
            for t in self.tracks:
                if id(t) in used_tracks:
                    continue
                score = _iou(det, t.bbox)
                if score > best_iou:
                    best_iou, best_t = score, t
            if best_t is not None:
                best_t.update(det)
                used_tracks.add(id(best_t))
                used_dets.add(d_idx)

        # ── Stage 2: Centroid distance (fast movers) ──────────────────────────
        active_unmatched = [t for t in self.tracks
                            if id(t) not in used_tracks and t.age >= self.min_age]

        for d_idx in range(len(detections)):
            if d_idx in used_dets:
                continue
            det = detections[d_idx]
            best_dist, best_t = self.max_centroid_dist, None
            for t in active_unmatched:
                if id(t) in used_tracks:
                    continue
                d = _dist(det, t.bbox)
                if d < best_dist:
                    best_dist, best_t = d, t
            if best_t is not None:
                h = _body_hist(frame, det) if frame is not None else None
                best_t.update(det, h)
                used_tracks.add(id(best_t))
                used_dets.add(d_idx)

        # ── Stage 3: Appearance histogram (same person, different position) ───
        active_unmatched = [t for t in self.tracks
                            if id(t) not in used_tracks
                            and t.age >= self.min_age
                            and t.hist is not None]

        if frame is not None:
            for d_idx in range(len(detections)):
                if d_idx in used_dets:
                    continue
                det      = detections[d_idx]
                det_hist = _body_hist(frame, det)
                if det_hist is None:
                    continue
                best_sim, best_t = self.appearance_threshold, None
                for t in active_unmatched:
                    if id(t) in used_tracks:
                        continue
                    sim = _hist_sim(det_hist, t.hist)
                    if sim > best_sim:
                        best_sim, best_t = sim, t
                if best_t is not None:
                    best_t.update(det, det_hist)
                    used_tracks.add(id(best_t))
                    used_dets.add(d_idx)

        # ── increment missed for unmatched active tracks ───────────────────────
        for t in self.tracks:
            if id(t) not in used_tracks:
                t.missed += 1

        # ── move dying tracks to dead pool ────────────────────────────────────
        for t in self.tracks:
            if t.missed > self.max_missed:
                self._dead_pool.append({
                    "id":      t.id,
                    "bbox":    list(t.bbox),
                    "hist":    t.hist,
                    "died_at": now,
                })

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        # ── Stage 4: spawn new tracks, dead-pool Re-ID first ─────────────────
        for d_idx, det in enumerate(detections):
            if d_idx in used_dets:
                continue
            t = Track(det)
            if frame is not None and self._dead_pool:
                det_hist  = _body_hist(frame, det)
                best_dead = self._match_dead(det_hist, det)
                if best_dead is not None:
                    t.id   = best_dead["id"]
                    t.age  = self.min_age
                    t.hist = best_dead["hist"]
                    self._dead_pool.remove(best_dead)
                    print(f"[Tracker] Stage4 dead-pool resume → track {t.id}")
            self.tracks.append(t)

        return [
            (t.id, t.bbox)
            for t in self.tracks
            if t.missed == 0 and t.age >= self.min_age
        ]

    def active_ids(self) -> set:
        return {t.id for t in self.tracks if t.missed == 0}

    # ── dead-pool matching ────────────────────────────────────────────────────

    def _match_dead(self, hist: Optional[np.ndarray],
                    bbox: list) -> Optional[dict]:
        if hist is None or not self._dead_pool:
            return None
        cx, cy = _centroid(bbox)
        best_entry, best_score = None, self.appearance_threshold
        for entry in self._dead_pool:
            sim = _hist_sim(hist, entry["hist"])
            if sim < self.appearance_threshold:
                continue
            lx, ly = _centroid(entry["bbox"])
            dist   = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if dist <= 500 and sim > best_score:
                best_score, best_entry = sim, entry
        return best_entry
