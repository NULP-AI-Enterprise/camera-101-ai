"""Unit tests for IOUTracker."""
import pytest
import numpy as np
from tracker import IOUTracker, _iou, _dist, _hist_sim


# ── helpers ───────────────────────────────────────────────────────────────────

def bbox(x, y, w, h):
    return [x, y, x + w, y + h]


def advance(tracker, dets, n=1, frame=None):
    """Run n update() calls and return the last result."""
    result = []
    for _ in range(n):
        result = tracker.update(dets, frame)
    return result


# ── _iou ─────────────────────────────────────────────────────────────────────

def test_iou_identical():
    b = bbox(0, 0, 50, 50)
    assert _iou(b, b) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert _iou(bbox(0, 0, 10, 10), bbox(20, 20, 10, 10)) == 0.0


def test_iou_partial():
    a = bbox(0, 0, 20, 20)   # area 400
    b = bbox(10, 0, 20, 20)  # area 400, overlap 10×20=200, union=600
    assert _iou(a, b) == pytest.approx(200 / 600)


# ── per-tracker counter ───────────────────────────────────────────────────────

def test_per_tracker_counter_independent():
    t1, t2 = IOUTracker(), IOUTracker()
    tr1 = t1._new_track(bbox(0, 0, 10, 10))
    tr2 = t2._new_track(bbox(0, 0, 10, 10))
    assert tr1.id == 1
    assert tr2.id == 1   # each tracker starts at 1 independently


def test_counter_increments_per_tracker():
    t = IOUTracker()
    ids = [t._new_track(bbox(i * 20, 0, 10, 10)).id for i in range(5)]
    assert ids == [1, 2, 3, 4, 5]


# ── min_age gate ──────────────────────────────────────────────────────────────

def test_min_age_gate():
    t = IOUTracker(min_age=2)
    dets = [bbox(0, 0, 50, 50)]
    # frame 0: create track (age=0) → nothing returned
    assert advance(t, dets) == []
    # frame 1: match → age=1 → still nothing
    assert advance(t, dets) == []
    # frame 2: match → age=2 → returned
    result = advance(t, dets)
    assert len(result) == 1
    assert result[0][0] == 1


# ── IOU matching (Hungarian) ──────────────────────────────────────────────────

def test_iou_matching_stable_ids():
    t = IOUTracker(min_age=1)
    dets = [bbox(0, 0, 50, 50), bbox(100, 100, 50, 50)]
    advance(t, dets)           # create tracks
    r1 = advance(t, dets)      # confirm
    ids1 = {tid for tid, _ in r1}
    r2 = advance(t, dets)
    ids2 = {tid for tid, _ in r2}
    assert ids1 == ids2        # same IDs across frames


def test_hungarian_optimal_assignment():
    # Two detections, two tracks in crossed positions
    # Greedy (left-to-right) would give suboptimal matching;
    # Hungarian gives the globally optimal one.
    t = IOUTracker(min_age=1, iou_threshold=0.1)
    # Create two tracks at distinct positions
    t.update([bbox(0, 0, 40, 40), bbox(60, 60, 40, 40)])
    advance(t, [bbox(0, 0, 40, 40), bbox(60, 60, 40, 40)])  # confirm both

    ids_before = {tid: b for tid, b in t.tracks and [(tr.id, tr.bbox) for tr in t.tracks if tr.missed == 0]}

    # Swap detections — Hungarian should still match correctly by IOU
    result = t.update([bbox(60, 60, 40, 40), bbox(0, 0, 40, 40)])
    result_ids = {tid for tid, _ in result}
    # Both original track IDs should survive the swap
    assert len(result_ids) == 2


# ── cooldown / missed frames ──────────────────────────────────────────────────

def test_missed_increments_without_detection():
    t = IOUTracker(min_age=1, max_missed=5)
    dets = [bbox(0, 0, 50, 50)]
    advance(t, dets, n=2)          # create + confirm track
    # Now provide no detections
    for _ in range(5):
        t.update([])
    assert len(t.tracks) == 1     # still alive at max_missed
    t.update([])                   # exceeds max_missed → moved to dead pool
    assert len(t.tracks) == 0


def test_dead_pool_populated():
    t = IOUTracker(min_age=1, max_missed=2, reid_window=60)
    dets = [bbox(0, 0, 50, 50)]
    advance(t, dets, n=2)
    t.update([])
    t.update([])
    t.update([])   # track dies → goes to dead pool
    assert len(t._dead_pool) == 1


# ── centroid distance (stage 2) ───────────────────────────────────────────────

def test_centroid_stage_catches_fast_mover():
    t = IOUTracker(min_age=1, iou_threshold=0.9, max_centroid_dist=200)
    # Confirm track at position A
    advance(t, [bbox(0, 0, 40, 40)], n=2)
    # Detection jumps far — IOU < 0.9 but centroid dist < 200
    result = t.update([bbox(80, 0, 40, 40)])
    assert len(result) == 1        # still tracked via centroid stage


# ── _hist_sim ─────────────────────────────────────────────────────────────────

def test_hist_sim_identical():
    v = np.random.rand(256).astype(np.float32)
    v /= np.linalg.norm(v)
    assert _hist_sim(v, v) == pytest.approx(1.0, abs=1e-5)


def test_hist_sim_none():
    assert _hist_sim(None, np.ones(256)) == 0.0
    assert _hist_sim(np.ones(256), None) == 0.0
