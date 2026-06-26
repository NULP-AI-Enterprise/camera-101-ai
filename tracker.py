"""
Multi-stage IOU tracker with appearance Re-ID.

Matching pipeline per frame
───────────────────────────
 Stage 1  IOU matching           — Hungarian optimal assignment
 Stage 2  Centroid distance      — handles fast movement between detection cycles
 Stage 3  Appearance histogram   — handles larger position jumps (same clothing)
 Stage 4  Dead-pool Re-ID        — handles true absences (left frame, came back)

Stages 1–3 all operate on ACTIVE tracks, so the same person never gets a new
track ID just because they moved faster than IOU can follow.
Stage 4 covers the case where the track actually expired (> max_missed frames).
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("tracker")

_INF = 1e9   # cost sentinel meaning "no valid match"

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
    def __init__(self, bbox: list, track_id: int):
        self.id     = track_id
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
        self._next_id: int = 0   # per-tracker counter — no shared class state

    def _new_track(self, bbox: list) -> Track:
        self._next_id += 1
        return Track(bbox, self._next_id)

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

        # ── Stage 1: IOU matching (Hungarian) ────────────────────────────────
        if detections and self.tracks:
            nd, nt = len(detections), len(self.tracks)
            cost = np.full((nd, nt), _INF)
            for i, det in enumerate(detections):
                for j, t in enumerate(self.tracks):
                    iou = _iou(det, t.bbox)
                    if iou >= self.iou_threshold:
                        cost[i, j] = 1.0 - iou
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= _INF:
                    continue
                self.tracks[c].update(detections[r])
                used_tracks.add(id(self.tracks[c]))
                used_dets.add(r)

        # ── Stage 2: Centroid distance (Hungarian, fast movers) ──────────────
        unmatched_tracks = [t for t in self.tracks
                            if id(t) not in used_tracks and t.age >= self.min_age]
        unmatched_dets   = [i for i in range(len(detections)) if i not in used_dets]

        if unmatched_dets and unmatched_tracks:
            nd, nt = len(unmatched_dets), len(unmatched_tracks)
            cost = np.full((nd, nt), _INF)
            for i, d_idx in enumerate(unmatched_dets):
                for j, t in enumerate(unmatched_tracks):
                    d = _dist(detections[d_idx], t.bbox)
                    if d < self.max_centroid_dist:
                        cost[i, j] = d
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= _INF:
                    continue
                d_idx = unmatched_dets[r]
                t     = unmatched_tracks[c]
                h = _body_hist(frame, detections[d_idx]) if frame is not None else None
                t.update(detections[d_idx], h)
                used_tracks.add(id(t))
                used_dets.add(d_idx)

        # ── Stage 3: Appearance histogram (Hungarian) ────────────────────────
        unmatched_tracks = [t for t in self.tracks
                            if id(t) not in used_tracks
                            and t.age >= self.min_age
                            and t.hist is not None]
        unmatched_dets   = [i for i in range(len(detections)) if i not in used_dets]

        if frame is not None and unmatched_dets and unmatched_tracks:
            det_hists = [_body_hist(frame, detections[i]) for i in unmatched_dets]
            nd, nt = len(unmatched_dets), len(unmatched_tracks)
            cost = np.full((nd, nt), _INF)
            for i, dh in enumerate(det_hists):
                if dh is None:
                    continue
                for j, t in enumerate(unmatched_tracks):
                    sim = _hist_sim(dh, t.hist)
                    if sim >= self.appearance_threshold:
                        cost[i, j] = 1.0 - sim
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= _INF:
                    continue
                d_idx = unmatched_dets[r]
                t     = unmatched_tracks[c]
                t.update(detections[d_idx], det_hists[r])
                used_tracks.add(id(t))
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
            t = self._new_track(det)
            if frame is not None and self._dead_pool:
                det_hist  = _body_hist(frame, det)
                best_dead = self._match_dead(det_hist, det)
                if best_dead is not None:
                    t.id   = best_dead["id"]
                    t.age  = self.min_age
                    t.hist = best_dead["hist"]
                    self._dead_pool.remove(best_dead)
                    log.debug("Stage4 dead-pool resume → track %d", t.id)
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
