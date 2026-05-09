"""Tests for src/detection.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.types import Detection


def _make_config(**kwargs) -> Config:
    return Config(**kwargs)


def _make_detector(**kwargs):
    """Create a BallDetector with a real model if available, else skip."""
    config = _make_config(**kwargs)
    model_path = config.resolve_model_path()
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}. Drop weights into models/ to run this test.")
    from src.detection import BallDetector
    return BallDetector(config)


# --- tile geometry tests (no model needed) ---

def _tiles(tile_size: int, tile_overlap: int, h: int, w: int):
    """Re-implement _compute_tiles to test logic independently."""
    stride = tile_size - tile_overlap
    tiles = []
    y = 0
    while True:
        x = 0
        while True:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tiles.append((x, y, x2, y2))
            if x2 == w:
                break
            x += stride
        if y2 == h:
            break
        y += stride
    return tiles


def test_compute_tiles_covers_full_frame():
    h, w = 1080, 1920
    tiles = _tiles(640, 128, h, w)
    coverage = np.zeros((h, w), dtype=bool)
    for (x1, y1, x2, y2) in tiles:
        coverage[y1:y2, x1:x2] = True
    assert coverage.all(), "Not all pixels covered"


def test_compute_tiles_overlap():
    tiles = _tiles(640, 128, 1080, 1920)
    # Check horizontal overlap between consecutive tiles in same row
    row = [t for t in tiles if t[1] == 0]
    for i in range(1, len(row)):
        prev_x2 = row[i - 1][2]
        curr_x1 = row[i][0]
        overlap = prev_x2 - curr_x1
        assert overlap == 128, f"Expected overlap 128, got {overlap}"


def test_compute_tiles_single_tile_small_frame():
    # Frame smaller than tile_size → single tile covering whole frame
    tiles = _tiles(640, 128, 300, 400)
    assert len(tiles) == 1
    assert tiles[0] == (0, 0, 400, 300)


def test_nms_removes_duplicate():
    from src.detection import BallDetector

    detector = object.__new__(BallDetector)
    detector._iou = 0.45

    high_conf = Detection(bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.9, frame_idx=0)
    low_conf = Detection(bbox=(105.0, 105.0, 205.0, 205.0), confidence=0.5, frame_idx=0)

    result = detector._nms([high_conf, low_conf])
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_detect_returns_list_on_blank_frame():
    detector = _make_detector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank, frame_idx=0)
    assert isinstance(result, list)
    # Blank frame may return empty — just check type is correct
    for d in result:
        assert isinstance(d, Detection)
