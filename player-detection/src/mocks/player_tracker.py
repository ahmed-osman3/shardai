"""Mock player tracker.

Drop-in stand-in for the real ByteTrack-backed PlayerTracker during Phase A.
Assigns stable sequential track_ids by detection index — perfect for the
synthetic MockPlayerDetector where the i-th detection is always the same
synthetic player. Real tracker swaps this out once supervision.ByteTrack
is wired into src/tracking.py.
"""

from __future__ import annotations

import logging

from src.types import PlayerDetection, TrackedPlayer

logger = logging.getLogger(__name__)


class MockPlayerTracker:
    """Index-based tracker: track_id = position of the detection in the input list.

    Works only when the upstream detector returns players in a stable order
    (e.g. MockPlayerDetector). Has no occlusion handling — tests should not
    rely on tracker robustness.
    """

    def update(
        self,
        detections: list[PlayerDetection],
        frame_idx: int,
    ) -> list[TrackedPlayer]:
        return [
            TrackedPlayer(
                frame_idx=frame_idx,
                track_id=i,
                bbox=d.bbox,
                confidence=d.confidence,
            )
            for i, d in enumerate(detections)
        ]

    def reset(self) -> None:
        pass
