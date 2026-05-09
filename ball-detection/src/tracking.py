"""Ball tracking with Kalman filter gap-fill.

Maintains a single Kalman filter (state: [x, y, vx, vy]) for the ball.
Short gaps in detection (≤ max_lost_frames) are filled by Kalman prediction
and marked as "interpolated". Longer gaps mark the track as "lost".
"""

from __future__ import annotations

import logging

import numpy as np

from config import Config
from src.types import Detection, TrackedBall

logger = logging.getLogger(__name__)


class BallTracker:
    """Single-object Kalman tracker for the football.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Config) -> None:
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State: [x, y, vx, vy]; measurement: [x, y]
        self.kf.F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.kf.R *= 10.0   # measurement noise ~10px²
        self.kf.Q *= 1.0    # process noise ~1px²
        self.kf.P *= 500.0  # initial uncertainty

        self._max_lost = config.kalman_max_lost_frames
        self._lost_frames = 0
        self._initialized = False

    def update(self, detections: list[Detection], frame_idx: int) -> TrackedBall | None:
        """Update tracker with new detections for a frame.

        Args:
            detections: Ball detections for this frame (may be empty).
            frame_idx: Current frame index.

        Returns:
            TrackedBall if track is active, None if not yet initialised.
        """
        if detections:
            if self._initialized:
                # Pick detection closest to predicted position
                pred_x, pred_y = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
                best = min(detections, key=lambda d: _dist(d, pred_x, pred_y))
            else:
                best = max(detections, key=lambda d: d.confidence)

            cx = (best.bbox[0] + best.bbox[2]) / 2
            cy = (best.bbox[1] + best.bbox[3]) / 2

            if not self._initialized:
                self.kf.x[0, 0] = cx
                self.kf.x[1, 0] = cy
                self._initialized = True
            else:
                self.kf.update([[cx], [cy]])

            self.kf.predict()
            self._lost_frames = 0
            return TrackedBall(frame_idx, cx, cy, "detected", best.confidence)

        # No detections
        if not self._initialized:
            return None

        self._lost_frames += 1
        self.kf.predict()
        px, py = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])

        if self._lost_frames > self._max_lost:
            return TrackedBall(frame_idx, px, py, "lost", 0.0)
        return TrackedBall(frame_idx, px, py, "interpolated", 0.0)

    def reset(self) -> None:
        """Reset tracker state. Call between independent video clips."""
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.kf.R *= 10.0
        self.kf.Q *= 1.0
        self.kf.P *= 500.0
        self._lost_frames = 0
        self._initialized = False


def _dist(d: Detection, px: float, py: float) -> float:
    cx = (d.bbox[0] + d.bbox[2]) / 2
    cy = (d.bbox[1] + d.bbox[3]) / 2
    return (cx - px) ** 2 + (cy - py) ** 2
