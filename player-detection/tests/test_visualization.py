"""Smoke tests for visualization.draw_frame.

We check that draw_frame returns a frame of the right shape and doesn't mutate
the input. We don't pixel-compare — that's what notebooks are for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.types import TrackedPlayer
from src.visualization import draw_frame


def _tp(track_id: int, x: int, y: int) -> TrackedPlayer:
    return TrackedPlayer(
        frame_idx=0,
        track_id=track_id,
        bbox=(x - 20, y - 40, x + 20, y + 40),
        confidence=0.9,
    )


def test_draw_frame_returns_same_shape():
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    tracked = [_tp(0, 100, 100), _tp(1, 300, 200)]
    labels = {0: ("R7", "red"), 1: ("B3", "blue")}
    out = draw_frame(frame, tracked, labels, Config())
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_draw_frame_does_not_mutate_input():
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    original = frame.copy()
    tracked = [_tp(0, 100, 100)]
    labels = {0: ("R7", "red")}
    draw_frame(frame, tracked, labels, Config())
    assert np.array_equal(frame, original)


def test_draw_frame_handles_unresolved_label():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracked = [_tp(99, 100, 100)]
    out = draw_frame(frame, tracked, {}, Config())
    assert out.shape == frame.shape


def test_draw_frame_skips_unresolved_when_disabled():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracked = [_tp(99, 100, 100)]
    cfg = Config(draw_unresolved_boxes=False)
    out = draw_frame(frame, tracked, {}, cfg)
    # Unresolved track skipped entirely → output identical to input.
    assert np.array_equal(out, frame)
