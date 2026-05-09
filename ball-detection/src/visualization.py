"""Frame annotation utilities.

Draws ball trajectory tails, detection boxes, and player bounding boxes
onto video frames. Event banners are deferred to v2 — see doc/IMPLEMENTATION_PLAN.md "Deferred".
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import Config
from src.types import PlayerDetection, TrackedBall

logger = logging.getLogger(__name__)

_TAIL_LENGTH = 30
_TEAM_COLOURS = {"red": (0, 0, 220), "blue": (220, 80, 0)}  # BGR


def draw_frame(
    frame: np.ndarray,
    ball: TrackedBall | None,
    ball_history: list[TrackedBall],
    players: list[PlayerDetection],
    config: Config,
) -> np.ndarray:
    """Annotate frame with ball trajectory + player boxes.

    Args:
        frame: BGR source frame. Not modified in place.
        ball: Current ball position (may be None).
        ball_history: Recent tracked positions for drawing the trajectory tail.
        players: PlayerDetection list for this frame.
        config: Pipeline configuration.

    Returns:
        Annotated BGR frame as a new array.
    """
    out = frame.copy()

    # --- trajectory tail ---
    tail = ball_history[-_TAIL_LENGTH:]
    for i in range(1, len(tail)):
        prev, curr = tail[i - 1], tail[i]
        alpha = i / len(tail)  # 0=dim, 1=bright
        colour = _tail_colour(curr.source, alpha)
        if curr.source == "interpolated":
            _draw_dashed_line(out, (int(prev.x), int(prev.y)), (int(curr.x), int(curr.y)), colour)
        else:
            cv2.line(out, (int(prev.x), int(prev.y)), (int(curr.x), int(curr.y)), colour, 2)

    # --- ball marker ---
    if ball is not None:
        if ball.source == "detected":
            # Green bounding box — approximate from tracker centre
            r = 12
            cv2.rectangle(out, (int(ball.x - r), int(ball.y - r)),
                          (int(ball.x + r), int(ball.y + r)), (0, 220, 0), 2)
            cv2.putText(out, f"{ball.confidence:.2f}", (int(ball.x - r), int(ball.y - r - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
        elif ball.source == "interpolated":
            cv2.circle(out, (int(ball.x), int(ball.y)), 6, (0, 220, 220), -1)

    # --- player boxes (disabled by default until real detection is wired in) ---
    if config.draw_player_boxes:
        for p in players:
            colour = _TEAM_COLOURS.get(p.team_id, (180, 180, 180))
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
            cv2.putText(out, p.player_id, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

    # TODO: event banners — see doc/IMPLEMENTATION_PLAN.md "Deferred"

    return out


def setup_video_writer(
    output_path: Path,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> cv2.VideoWriter:
    """Create a cv2.VideoWriter for an annotated output clip.

    Args:
        output_path: Destination .mp4 path. Parent directory must exist.
        fps: Output frame rate.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Configured VideoWriter (caller must release when done).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")
    return writer


def _tail_colour(source: str, alpha: float) -> tuple[int, int, int]:
    if source == "interpolated":
        v = int(80 + 140 * alpha)
        return (0, v, v)
    v = int(60 + 160 * alpha)
    return (0, v, 0)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    colour: tuple[int, int, int],
    dash_len: int = 8,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    length = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
    steps = max(1, length // dash_len)
    for s in range(steps):
        if s % 2 == 0:
            sx = int(x1 + (x2 - x1) * s / steps)
            sy = int(y1 + (y2 - y1) * s / steps)
            ex = int(x1 + (x2 - x1) * (s + 1) / steps)
            ey = int(y1 + (y2 - y1) * (s + 1) / steps)
            cv2.line(img, (sx, sy), (ex, ey), colour, 2)
