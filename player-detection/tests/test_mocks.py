"""Tests for mock modules. Run offline, no model weights required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mocks.ball_detector import MockBallDetector
from src.mocks.calibration import MockCalibration, _PITCH_LEN_M, _PITCH_WID_M
from src.mocks.ocr import NoisyMockOCR, TruthTableMockOCR
from src.mocks.player_detector import MockPlayerDetector
from src.mocks.player_tracker import MockPlayerTracker
from src.mocks.storage import MockStorage


# ---------------------------------------------------------------------------
# MockCalibration
# ---------------------------------------------------------------------------

def test_calibration_pixel_to_pitch_near_left_corner():
    cal = MockCalibration(1920, 1080)
    bl = cal.pitch_corners_px[3]
    mx, my = cal.pixel_to_pitch(*bl)
    assert abs(mx) < 1.0
    assert abs(my) < 1.0


def test_calibration_pixel_to_pitch_far_right_corner():
    cal = MockCalibration(1920, 1080)
    tr = cal.pitch_corners_px[1]
    mx, my = cal.pixel_to_pitch(*tr)
    assert abs(mx - _PITCH_LEN_M) < 1.0
    assert abs(my - _PITCH_WID_M) < 1.0


def test_calibration_is_in_goal_at_post():
    cal = MockCalibration(1920, 1080)
    posts = cal.goal_posts_px["north"]
    mid_x = (posts[0][0] + posts[1][0]) / 2
    goal_y = posts[0][1]
    assert cal.is_in_goal(mid_x, goal_y, "north")
    assert not cal.is_in_goal(mid_x, goal_y, "south")


# ---------------------------------------------------------------------------
# MockBallDetector
# ---------------------------------------------------------------------------

def test_ball_detector_always_none():
    d = MockBallDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert d.detect(frame, 0) is None
    assert d.detect(frame, 999) is None


# ---------------------------------------------------------------------------
# MockPlayerDetector
# ---------------------------------------------------------------------------

def test_player_detector_returns_14_detections():
    detector = MockPlayerDetector(n_per_team=7, seed=42)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    players = detector.detect(frame, frame_idx=0)
    assert len(players) == 14


def test_player_detector_truth_table_covers_all_synthetic_idx():
    detector = MockPlayerDetector(n_per_team=7, seed=42)
    truth = detector.truth_table()
    assert len(truth) == 14
    for i in range(14):
        assert i in truth
    # Both teams covered, numbers 1..7 each
    reds = [(c, n) for c, n in truth.values() if c == "red"]
    blues = [(c, n) for c, n in truth.values() if c == "blue"]
    assert sorted(n for _, n in reds) == list(range(1, 8))
    assert sorted(n for _, n in blues) == list(range(1, 8))


def test_player_detector_deterministic():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    d1 = MockPlayerDetector(seed=42)
    d2 = MockPlayerDetector(seed=42)
    p1 = d1.detect(frame, frame_idx=100)
    p2 = d2.detect(frame, frame_idx=100)
    for a, b in zip(p1, p2):
        assert a.bbox == pytest.approx(b.bbox)


def test_player_detector_positions_in_bounds():
    detector = MockPlayerDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    for fi in [0, 50, 200, 1000]:
        for p in detector.detect(frame, frame_idx=fi):
            cx = (p.bbox[0] + p.bbox[2]) / 2
            cy = (p.bbox[1] + p.bbox[3]) / 2
            assert 0 <= cx <= w
            assert 0 <= cy <= h


# ---------------------------------------------------------------------------
# MockPlayerTracker
# ---------------------------------------------------------------------------

def test_player_tracker_assigns_index_track_ids():
    detector = MockPlayerDetector()
    tracker = MockPlayerTracker()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detections = detector.detect(frame, 0)
    tracked = tracker.update(detections, frame_idx=0)
    assert [tp.track_id for tp in tracked] == list(range(len(detections)))


def test_player_tracker_track_id_stable_across_frames():
    detector = MockPlayerDetector()
    tracker = MockPlayerTracker()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    t0 = tracker.update(detector.detect(frame, 0), 0)
    t100 = tracker.update(detector.detect(frame, 100), 100)
    assert [tp.track_id for tp in t0] == [tp.track_id for tp in t100]


# ---------------------------------------------------------------------------
# MockStorage
# ---------------------------------------------------------------------------

def test_storage_upload_creates_file(tmp_path: Path):
    src = tmp_path / "test.json"
    src.write_text('{"ok": true}')
    storage = MockStorage(tmp_path / "store")
    url = storage.upload(src, "match/test.json")
    assert url.startswith("file://")
    assert (tmp_path / "store" / "match" / "test.json").exists()


def test_storage_round_trip(tmp_path: Path):
    src = tmp_path / "data.bin"
    src.write_bytes(b"\x00\x01\x02\x03")
    storage = MockStorage(tmp_path / "store")
    storage.upload(src, "data.bin")
    dest = tmp_path / "restored.bin"
    storage.download("data.bin", dest)
    assert dest.read_bytes() == b"\x00\x01\x02\x03"


# ---------------------------------------------------------------------------
# MockOCR
# ---------------------------------------------------------------------------

def test_truth_table_ocr_returns_configured_identity():
    ocr = TruthTableMockOCR(truth={5: ("red", 7)})
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    r = ocr.read(crop, track_id=5, frame_idx=10)
    assert r.colour == "red"
    assert r.number == 7
    assert r.frame_idx == 10
    assert r.track_id == 5


def test_truth_table_ocr_unknown_track_returns_unresolved():
    ocr = TruthTableMockOCR(truth={5: ("red", 7)})
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    r = ocr.read(crop, track_id=99, frame_idx=10)
    assert r.colour == "unknown"
    assert r.number is None


def test_noisy_mock_ocr_mostly_correct():
    truth = {0: ("red", 7)}
    ocr = NoisyMockOCR(truth=truth, accuracy=0.9, seed=42)
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    correct = sum(
        1 for _ in range(200)
        if (r := ocr.read(crop, track_id=0, frame_idx=0)).colour == "red" and r.number == 7
    )
    assert correct >= 150  # ~90% of 200, with slack
