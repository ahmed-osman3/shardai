"""Rule-based event detector.

Consumes ball track + player tracks + calibration to detect possession,
passes, shots, and goals. The rules here are close to what the real v1
will use — this is not purely a mock, just deliberately naive.
"""

from __future__ import annotations

import logging
from collections import Counter

from config import Config
from src.mocks.calibration import MockCalibration
from src.types import Event, PlayerDetection, TrackedBall

logger = logging.getLogger(__name__)

_POSSESSION_WINDOW = 30        # frames for mode smoothing
_POSSESSION_MAX_DIST_PX = 80   # ignore players further than this from ball
_PASS_MAX_GAP_FRAMES = 30      # max None-possession gap in a pass sequence
_PASS_MAX_DURATION_FRAMES = 120
_SHOT_SPEED_MPS = 12.0
_SHOT_LOOKAHEAD_FRAMES = 30
_GOAL_DEBOUNCE_FRAMES = 180
_GOAL_Y_TOLERANCE_PX = 8


class MockEventDetector:
    """Rule-based event detection over full-match ball and player tracks.

    Args:
        config: Pipeline configuration.
        calibration: Pitch geometry for coordinate conversion.
    """

    def __init__(self, config: Config, calibration: MockCalibration) -> None:
        self._config = config
        self._cal = calibration

    def process(
        self,
        ball_track: list[TrackedBall | None],
        player_track: list[list[PlayerDetection]],
        fps: float,
    ) -> list[Event]:
        """Detect events over the full match track.

        Args:
            ball_track: Per-frame ball positions (None = not yet initialised).
            player_track: Per-frame list of player detections.
            fps: Source video frame rate (for velocity and time conversion).

        Returns:
            Chronologically ordered list of detected events.
        """
        n = len(ball_track)
        events: list[Event] = []

        # --- Step 1: raw per-frame possession ---
        raw_possession: list[str | None] = []
        for i in range(n):
            ball = ball_track[i]
            players = player_track[i] if i < len(player_track) else []
            raw_possession.append(_closest_player(ball, players))

        # --- Step 2: smooth possession over 30-frame window ---
        smoothed: list[str | None] = []
        for i in range(n):
            window_start = max(0, i - _POSSESSION_WINDOW + 1)
            window = [p for p in raw_possession[window_start : i + 1] if p is not None]
            if window:
                smoothed.append(Counter(window).most_common(1)[0][0])
            else:
                smoothed.append(None)

        # --- Step 3: passes ---
        events.extend(_detect_passes(smoothed, ball_track, fps))

        # --- Step 4 & 5: shots ---
        events.extend(_detect_shots(ball_track, fps, self._cal))

        # --- Step 6: goals ---
        events.extend(_detect_goals(ball_track, fps, self._cal))

        events.sort(key=lambda e: e.frame_idx)
        return events


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _closest_player(
    ball: TrackedBall | None,
    players: list[PlayerDetection],
) -> str | None:
    if ball is None or not players:
        return None
    best_dist = float("inf")
    best_id = None
    for p in players:
        px = (p.bbox[0] + p.bbox[2]) / 2
        py = (p.bbox[1] + p.bbox[3]) / 2
        dist = ((ball.x - px) ** 2 + (ball.y - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = p.player_id
    if best_dist > _POSSESSION_MAX_DIST_PX:
        return None
    return best_id


def _detect_passes(
    smoothed: list[str | None],
    ball_track: list[TrackedBall | None],
    fps: float,
) -> list[Event]:
    events: list[Event] = []
    n = len(smoothed)
    i = 0
    while i < n:
        if smoothed[i] is None:
            i += 1
            continue
        player_a = smoothed[i]
        team_a = player_a[0]  # "R" or "B"
        # scan forward for handover to same team
        j = i + 1
        none_gap = 0
        while j < n:
            if smoothed[j] is None:
                none_gap += 1
                if none_gap > _PASS_MAX_GAP_FRAMES:
                    break
            elif smoothed[j] == player_a:
                none_gap = 0
            else:
                # possession changed
                player_b = smoothed[j]
                if player_b[0] == team_a and (j - i) <= _PASS_MAX_DURATION_FRAMES:
                    ball = ball_track[j] if ball_track[j] is not None else ball_track[i]
                    ts = j / fps if fps else 0.0
                    events.append(Event(
                        type="pass",
                        frame_idx=j,
                        ts_seconds=ts,
                        primary_player=player_a,
                        secondary_player=player_b,
                        metadata={},
                    ))
                break
            j += 1
        i = j if j > i else i + 1
    return events


def _detect_shots(
    ball_track: list[TrackedBall | None],
    fps: float,
    cal: MockCalibration,
) -> list[Event]:
    events: list[Event] = []
    n = len(ball_track)
    for i in range(5, n):
        ball = ball_track[i]
        prev = ball_track[i - 5]
        if ball is None or prev is None:
            continue
        if ball.source == "lost" or prev.source == "lost":
            continue

        dx_px = ball.x - prev.x
        dy_px = ball.y - prev.y
        bx0, by0 = cal.pixel_to_pitch(prev.x, prev.y)
        bx1, by1 = cal.pixel_to_pitch(ball.x, ball.y)
        dist_m = ((bx1 - bx0) ** 2 + (by1 - by0) ** 2) ** 0.5
        speed_mps = dist_m / (5 / fps) if fps else 0.0

        if speed_mps < _SHOT_SPEED_MPS:
            continue

        # Check if trajectory intersects a goal mouth
        for goal in ("north", "south"):
            posts = cal.goal_posts_px[goal]
            gx1, gy = posts[0]
            gx2, _ = posts[1]
            # Extrapolate ball position 30 frames forward
            t = _SHOT_LOOKAHEAD_FRAMES / 5  # in units of 5-frame steps
            ex = ball.x + dx_px * t
            ey = ball.y + dy_px * t
            if (min(gx1, gx2) <= ex <= max(gx1, gx2) and abs(ey - gy) <= 20):
                events.append(Event(
                    type="shot",
                    frame_idx=i,
                    ts_seconds=i / fps if fps else 0.0,
                    primary_player=None,
                    secondary_player=None,
                    metadata={"velocity_mps": round(speed_mps, 2), "goal_end": goal},
                ))
                break

    return events


def _detect_goals(
    ball_track: list[TrackedBall | None],
    fps: float,
    cal: MockCalibration,
) -> list[Event]:
    events: list[Event] = []
    last_goal_frame = -_GOAL_DEBOUNCE_FRAMES
    for i, ball in enumerate(ball_track):
        if ball is None or ball.source == "lost":
            continue
        if i - last_goal_frame < _GOAL_DEBOUNCE_FRAMES:
            continue
        for goal in ("north", "south"):
            if cal.is_in_goal(ball.x, ball.y, goal):
                events.append(Event(
                    type="goal",
                    frame_idx=i,
                    ts_seconds=i / fps if fps else 0.0,
                    primary_player=None,
                    secondary_player=None,
                    metadata={"goal_end": goal},
                ))
                last_goal_frame = i
                break
    return events
