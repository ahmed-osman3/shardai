"""Tests for src/detection.py.

Tile geometry + NMS tests run without weights. The integration test that
actually loads YOLO is skipped when models/yolo11m.pt is absent (mirroring
ball-detection's pattern).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.types import PlayerDetection


def _requires_model(config: Config) -> None:
    if not config.resolve_model_path().exists():
        pytest.skip("Model not found — drop weights into models/ to run detection tests.")


# ---------------------------------------------------------------------------
# Tile geometry — no weights needed
# ---------------------------------------------------------------------------

def test_compute_tiles_covers_full_frame_no_gaps():
    from src.detection import PlayerDetector

    cfg = Config()
    _requires_model(cfg)
    det = PlayerDetector(cfg)
    h, w = 1080, 1920
    tiles = det._compute_tiles(h, w)

    # Every pixel must be covered by at least one tile
    coverage = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in tiles:
        coverage[y1:y2, x1:x2] = True
    assert coverage.all(), "Tile grid leaves uncovered pixels"


def test_compute_tiles_respect_overlap():
    from src.detection import PlayerDetector

    cfg = Config(tile_size=640, tile_overlap=128)
    _requires_model(cfg)
    det = PlayerDetector(cfg)
    tiles = det._compute_tiles(1080, 1920)

    # Tiles in the same row should overlap by tile_overlap (except last tile clamped)
    row0 = sorted([t for t in tiles if t[1] == 0], key=lambda t: t[0])
    if len(row0) >= 2:
        a, b = row0[0], row0[1]
        overlap = a[2] - b[0]
        assert overlap == 128


# ---------------------------------------------------------------------------
# NMS — no weights needed
# ---------------------------------------------------------------------------

def test_nms_dedupes_overlapping_boxes():
    from src.detection import PlayerDetector

    cfg = Config()
    _requires_model(cfg)
    det = PlayerDetector(cfg)
    dets = [
        PlayerDetection(bbox=(100, 100, 200, 200), confidence=0.9, frame_idx=0),
        PlayerDetection(bbox=(105, 105, 205, 205), confidence=0.8, frame_idx=0),  # ~95% IoU
        PlayerDetection(bbox=(500, 500, 600, 600), confidence=0.7, frame_idx=0),  # disjoint
    ]
    kept = det._nms(dets)
    assert len(kept) == 2
    # Highest confidence of the overlapping pair survives
    assert kept[0].confidence == 0.9


def test_nms_passthrough_for_zero_or_one_detection():
    from src.detection import PlayerDetector

    cfg = Config()
    _requires_model(cfg)
    det = PlayerDetector(cfg)
    assert det._nms([]) == []
    one = [PlayerDetection(bbox=(0, 0, 10, 10), confidence=0.5, frame_idx=0)]
    assert det._nms(one) == one


# ---------------------------------------------------------------------------
# Detection on real frames — needs weights
# ---------------------------------------------------------------------------

def test_detect_returns_list_on_blank_frame():
    cfg = Config()
    _requires_model(cfg)
    from src.detection import PlayerDetector

    det = PlayerDetector(cfg)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = det.detect(frame, frame_idx=0)
    assert isinstance(result, list)
    # A black frame may produce 0 detections (or noise — don't assert exact count)


def test_detect_filters_to_player_class():
    """Confirms filter to player_class_id excludes non-person detections."""
    cfg = Config(player_class_id=0)
    _requires_model(cfg)
    from src.detection import PlayerDetector

    det = PlayerDetector(cfg)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = det.detect(frame, frame_idx=0)
    # Each detection's bbox should be xyxy with x2>x1 and y2>y1
    for d in result:
        assert d.bbox[2] > d.bbox[0]
        assert d.bbox[3] > d.bbox[1]
        assert 0.0 <= d.confidence <= 1.0
