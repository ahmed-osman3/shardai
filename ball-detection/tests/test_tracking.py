"""Tests for src/tracking.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.tracking import BallTracker
from src.types import Detection


def _tracker(max_lost: int = 5) -> BallTracker:
    return BallTracker(Config(kalman_max_lost_frames=max_lost))


def _det(x: float = 100.0, y: float = 100.0) -> Detection:
    r = 10.0
    return Detection(bbox=(x - r, y - r, x + r, y + r), confidence=0.8, frame_idx=0)


def test_tracker_detected_on_detection():
    tracker = _tracker()
    result = tracker.update([_det()], frame_idx=0)
    assert result is not None
    assert result.source == "detected"
    assert result.confidence > 0


def test_tracker_none_before_first_detection():
    tracker = _tracker()
    result = tracker.update([], frame_idx=0)
    assert result is None


def test_tracker_interpolated_within_gap_window():
    tracker = _tracker(max_lost=5)
    tracker.update([_det()], frame_idx=0)
    for i in range(1, 6):
        result = tracker.update([], frame_idx=i)
        assert result is not None
        assert result.source == "interpolated", f"Expected interpolated at frame {i}"


def test_tracker_lost_after_gap_window():
    tracker = _tracker(max_lost=3)
    tracker.update([_det()], frame_idx=0)
    for i in range(1, 4):
        tracker.update([], frame_idx=i)
    result = tracker.update([], frame_idx=4)
    assert result is not None
    assert result.source == "lost"


def test_tracker_reset_clears_state():
    tracker = _tracker()
    tracker.update([_det()], frame_idx=0)
    tracker.reset()
    result = tracker.update([], frame_idx=1)
    assert result is None, "After reset, tracker should return None with no detections"


def test_tracker_recovers_after_gap():
    tracker = _tracker(max_lost=5)
    tracker.update([_det(100, 100)], frame_idx=0)
    for i in range(1, 4):
        tracker.update([], frame_idx=i)
    result = tracker.update([_det(110, 110)], frame_idx=4)
    assert result is not None
    assert result.source == "detected"
