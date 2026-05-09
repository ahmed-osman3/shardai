"""Tests for mock modules.

These tests require no model weights and run offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.mocks.calibration import MockCalibration, _PITCH_LEN_M, _PITCH_WID_M
from src.mocks.event_detector import MockEventDetector
from src.mocks.player_detector import MockPlayerDetector
from src.mocks.storage import MockStorage
from src.types import PlayerDetection, TrackedBall


# ---------------------------------------------------------------------------
# MockCalibration
# ---------------------------------------------------------------------------

def test_calibration_pixel_to_pitch_near_left_corner():
    cal = MockCalibration(1920, 1080)
    bl = cal.pitch_corners_px[3]  # BL = near-left = origin (0, 0) in pitch coords
    mx, my = cal.pixel_to_pitch(*bl)
    assert abs(mx) < 1.0, f"Expected ~0m x at BL, got {mx}"
    assert abs(my) < 1.0, f"Expected ~0m y at BL, got {my}"


def test_calibration_pixel_to_pitch_far_right_corner():
    cal = MockCalibration(1920, 1080)
    tr = cal.pitch_corners_px[1]  # TR = far-right = (50, 30) in pitch coords
    mx, my = cal.pixel_to_pitch(*tr)
    assert abs(mx - _PITCH_LEN_M) < 1.0, f"Expected ~{_PITCH_LEN_M}m x at TR, got {mx}"
    assert abs(my - _PITCH_WID_M) < 1.0, f"Expected ~{_PITCH_WID_M}m y at TR, got {my}"


def test_calibration_is_in_goal_at_post():
    cal = MockCalibration(1920, 1080)
    posts = cal.goal_posts_px["north"]
    mid_x = (posts[0][0] + posts[1][0]) / 2
    goal_y = posts[0][1]
    assert cal.is_in_goal(mid_x, goal_y, "north")
    assert not cal.is_in_goal(mid_x, goal_y, "south")


def test_calibration_not_in_goal_outside_posts():
    cal = MockCalibration(1920, 1080)
    posts = cal.goal_posts_px["north"]
    goal_y = posts[0][1]
    outside_x = posts[0][0] - 50  # well outside left post
    assert not cal.is_in_goal(outside_x, goal_y, "north")


def test_calibration_not_in_goal_wrong_y():
    cal = MockCalibration(1920, 1080)
    posts = cal.goal_posts_px["north"]
    mid_x = (posts[0][0] + posts[1][0]) / 2
    far_y = posts[0][1] + 100  # far from goal line
    assert not cal.is_in_goal(mid_x, far_y, "north")


# ---------------------------------------------------------------------------
# MockPlayerDetector
# ---------------------------------------------------------------------------

def test_player_detector_returns_14_detections():
    detector = MockPlayerDetector(n_per_team=7, seed=42)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    players = detector.detect(frame, frame_idx=0)
    assert len(players) == 14


def test_player_detector_team_ids():
    detector = MockPlayerDetector(n_per_team=7, seed=42)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    players = detector.detect(frame, frame_idx=0)
    red = [p for p in players if p.team_id == "red"]
    blue = [p for p in players if p.team_id == "blue"]
    assert len(red) == 7
    assert len(blue) == 7


def test_player_detector_player_ids():
    detector = MockPlayerDetector(n_per_team=7, seed=42)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    players = detector.detect(frame, frame_idx=0)
    ids = {p.player_id for p in players}
    for i in range(1, 8):
        assert f"R{i}" in ids
        assert f"B{i}" in ids


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
            assert 0 <= cx <= w, f"cx {cx} out of bounds"
            assert 0 <= cy <= h, f"cy {cy} out of bounds"


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


def test_storage_signed_url(tmp_path: Path):
    storage = MockStorage(tmp_path / "store")
    url = storage.signed_url("some/key.mp4")
    assert url.startswith("file://")


# ---------------------------------------------------------------------------
# MockEventDetector
# ---------------------------------------------------------------------------

def _make_ball_track(frames: int, x_start: float = 500.0, x_end: float = 500.0) -> list[TrackedBall | None]:
    track = []
    for i in range(frames):
        x = x_start + (x_end - x_start) * (i / max(frames - 1, 1))
        track.append(TrackedBall(frame_idx=i, x=x, y=540.0, source="detected", confidence=0.9))
    return track


def _make_player_track(
    frames: int,
    player_a_xy: tuple[float, float],
    player_b_xy: tuple[float, float],
) -> list[list[PlayerDetection]]:
    def _player(pid: str, tid: str, x: float, y: float) -> PlayerDetection:
        return PlayerDetection(bbox=(x - 20, y - 40, x + 20, y + 40), player_id=pid, team_id=tid, confidence=0.85)

    result = []
    for i in range(frames):
        if i < frames // 2:
            near = player_a_xy
            far = player_b_xy
        else:
            near = player_b_xy
            far = player_a_xy
        result.append([
            _player("R1", "red", near[0], near[1]),
            _player("R2", "red", far[0], far[1]),
        ])
    return result


def test_event_detector_emits_pass_on_team_handover():
    cal = MockCalibration(1920, 1080)
    config = Config()
    detector = MockEventDetector(config, cal)

    n = 100
    # Ball moves from near R1 to near R2 halfway through
    ball_track = _make_ball_track(n, x_start=500.0, x_end=500.0)
    player_track = _make_player_track(n, player_a_xy=(500.0, 540.0), player_b_xy=(700.0, 540.0))

    events = detector.process(ball_track, player_track, fps=30.0)
    pass_events = [e for e in events if e.type == "pass"]
    assert len(pass_events) >= 1
    assert pass_events[0].primary_player in ("R1", "R2")
    assert pass_events[0].secondary_player in ("R1", "R2")
    assert pass_events[0].primary_player != pass_events[0].secondary_player


def test_event_detector_no_cross_team_pass():
    cal = MockCalibration(1920, 1080)
    config = Config()
    detector = MockEventDetector(config, cal)

    n = 100
    ball_track = _make_ball_track(n)

    def _player(pid: str, tid: str, x: float, y: float) -> PlayerDetection:
        return PlayerDetection(bbox=(x - 20, y - 40, x + 20, y + 40), player_id=pid, team_id=tid, confidence=0.85)

    player_track = []
    for i in range(n):
        if i < n // 2:
            near = (500.0, 540.0)
        else:
            near = (700.0, 540.0)
        player_track.append([
            _player("R1", "red", near[0], near[1]),
            _player("B1", "blue", 700.0 if i < n // 2 else 500.0, 540.0),
        ])

    events = detector.process(ball_track, player_track, fps=30.0)
    pass_events = [e for e in events if e.type == "pass"]
    # Any pass should be within the same team
    for p in pass_events:
        assert p.primary_player[0] == p.secondary_player[0], \
            f"Cross-team pass detected: {p.primary_player} → {p.secondary_player}"
