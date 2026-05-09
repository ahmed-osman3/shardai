"""Mock pitch calibration.

Returns hardcoded but geometrically plausible pitch corners and goal post
positions for a 1920×1080 frame. Provides pixel↔pitch-metre homography
and goal-line crossing queries.

Coordinate convention:
  - Origin: near-left corner of pitch (bottom-left in the camera view)
  - +x: along touchline, near→far (pitch length, 50 m)
  - +y: across width, left→right (pitch width, 30 m)
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Real pitch dimensions for 7-a-side (upper bound)
_PITCH_LEN_M = 50.0
_PITCH_WID_M = 30.0

# Goal dimensions: width 7.3 m, centred on pitch width
_GOAL_WIDTH_M = 7.3
_GOAL_LEFT_M = (_PITCH_WID_M - _GOAL_WIDTH_M) / 2   # 11.35 m
_GOAL_RIGHT_M = _GOAL_LEFT_M + _GOAL_WIDTH_M          # 18.65 m


class MockCalibration:
    """Hardcoded pitch geometry for a 1920×1080 frame.

    Pitch dimensions assumed: 50 m × 30 m (upper bound for 7-a-side).
    Perspective: camera elevated at near-touchline corner.

    Args:
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
    """

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080) -> None:
        self._frame_w = frame_w
        self._frame_h = frame_h

        # Scale corner pixel positions to arbitrary frame size
        sx = frame_w / 1920
        sy = frame_h / 1080

        # Pitch corners in pixel space [TL, TR, BR, BL] for a 1920×1080 frame
        # Perspective offset: pitch doesn't fill frame edge-to-edge
        self._corners_px = [
            (int(160 * sx), int(120 * sy)),   # TL (far-left)
            (int(1760 * sx), int(120 * sy)),  # TR (far-right)
            (int(1820 * sx), int(960 * sy)),  # BR (near-right)
            (int(100 * sx), int(960 * sy)),   # BL (near-left)
        ]

        # Pitch corners in metre space [TL, TR, BR, BL]
        # TL = far-left = (x=50, y=0), TR = far-right = (x=50, y=30)
        # BR = near-right = (x=0, y=30), BL = near-left = (x=0, y=0)
        corners_m = np.array([
            [_PITCH_LEN_M, 0.0],
            [_PITCH_LEN_M, _PITCH_WID_M],
            [0.0, _PITCH_WID_M],
            [0.0, 0.0],
        ], dtype=np.float32)

        src = np.array(self._corners_px, dtype=np.float32)
        self._H = cv2.getPerspectiveTransform(src, corners_m)

        # Goal post pixel positions (hardcoded for 1920×1080, scaled to frame)
        # "north" = far end (top of frame), "south" = near end (bottom of frame)
        # Posts are at goal-line y; left/right centred on pitch width
        gx1_north = int((160 + (1760 - 160) * (_GOAL_LEFT_M / _PITCH_WID_M)) * sx)
        gx2_north = int((160 + (1760 - 160) * (_GOAL_RIGHT_M / _PITCH_WID_M)) * sx)
        gy_north = int(120 * sy)

        gx1_south = int((100 + (1820 - 100) * (_GOAL_LEFT_M / _PITCH_WID_M)) * sx)
        gx2_south = int((100 + (1820 - 100) * (_GOAL_RIGHT_M / _PITCH_WID_M)) * sx)
        gy_south = int(960 * sy)

        self._goal_posts = {
            "north": [(gx1_north, gy_north), (gx2_north, gy_north)],
            "south": [(gx1_south, gy_south), (gx2_south, gy_south)],
        }

    def pixel_to_pitch(self, x: float, y: float) -> tuple[float, float]:
        """Map pixel coordinates to pitch-metre coordinates.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            (pitch_x_metres, pitch_y_metres) — origin near-left corner.
        """
        pt = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    def is_in_goal(
        self,
        x: float,
        y: float,
        which_goal: Literal["north", "south"],
    ) -> bool:
        """Check whether a pixel position is within a goal region.

        Uses ±8 px tolerance around the goal-line y to handle Kalman jitter.

        Args:
            x: Pixel x coordinate of ball centre.
            y: Pixel y coordinate of ball centre.
            which_goal: Which end of the pitch to check.

        Returns:
            True if the position is inside the goal-line region.
        """
        posts = self._goal_posts[which_goal]
        post_x1, post_y1 = posts[0]
        post_x2, _ = posts[1]
        goal_y = post_y1  # both posts share the same y

        between_posts = min(post_x1, post_x2) <= x <= max(post_x1, post_x2)
        near_line = abs(y - goal_y) <= 8
        return between_posts and near_line

    @property
    def pitch_corners_px(self) -> list[tuple[int, int]]:
        """Four pitch corners in pixel space: [TL, TR, BR, BL]."""
        return self._corners_px

    @property
    def goal_posts_px(self) -> dict[str, list[tuple[int, int]]]:
        """Goal post pixel positions. Keys: 'north', 'south'."""
        return self._goal_posts
