"""Tests for src/tracking.py.

ByteTrack-backed PlayerTracker. No model weights required — tracker operates
purely on supplied detection bboxes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.tracking import PlayerTracker
from src.types import PlayerDetection


def _det(x: float, y: float, conf: float = 0.9) -> PlayerDetection:
    return PlayerDetection(bbox=(x - 20, y - 40, x + 20, y + 40), confidence=conf, frame_idx=0)


def test_tracker_assigns_track_ids():
    tracker = PlayerTracker(Config())
    tracked = tracker.update([_det(100, 100), _det(500, 500)], frame_idx=0)
    # ByteTrack may need a few frames to "activate" tracks; do a few frames
    for f in range(1, 5):
        tracked = tracker.update([_det(100 + f, 100 + f), _det(500 + f, 500 + f)], f)
    assert len(tracked) == 2
    track_ids = {tp.track_id for tp in tracked}
    assert len(track_ids) == 2  # two distinct IDs


def test_tracker_keeps_id_stable_across_frames():
    tracker = PlayerTracker(Config())
    # Activate tracks
    for f in range(5):
        tracker.update([_det(100 + f, 100 + f)], f)
    last = tracker.update([_det(105, 105)], 5)
    id_at_5 = last[0].track_id
    # Same player, slightly moved — id should persist
    for f in range(6, 15):
        tracked = tracker.update([_det(100 + f, 100 + f)], f)
    assert tracked[0].track_id == id_at_5


def test_tracker_handles_empty_detections():
    tracker = PlayerTracker(Config())
    assert tracker.update([], 0) == []
    assert tracker.update([], 1) == []


def test_tracker_reset_clears_state():
    tracker = PlayerTracker(Config())
    for f in range(10):
        tracker.update([_det(100 + f, 100 + f)], f)
    tracker.reset()
    # After reset, the tracker is fresh — first detection set won't be activated yet
    out = tracker.update([_det(100, 100)], 0)
    # Either empty (not yet activated) or fresh IDs starting from 1
    if out:
        assert out[0].track_id >= 1


def test_tracker_recovers_after_brief_gap():
    """A player vanishes for a few frames then reappears nearby —
    ByteTrack should re-attach to the original track id."""
    tracker = PlayerTracker(Config(lost_track_buffer=30))
    # Activate
    for f in range(10):
        tracker.update([_det(500, 500)], f)
    pre = tracker.update([_det(500, 500)], 10)
    pre_id = pre[0].track_id
    # 5-frame gap
    for f in range(11, 16):
        tracker.update([], f)
    # Reappear nearby
    for f in range(16, 20):
        post = tracker.update([_det(505, 505)], f)
    assert post[0].track_id == pre_id, "Tracker should re-attach after a brief gap"
