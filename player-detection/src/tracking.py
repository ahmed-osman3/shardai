"""Multi-object player tracking via ByteTrack (supervision wrapper).

Per frame: convert PlayerDetection list to a `supervision.Detections` object,
call ByteTrack's `update_with_detections`, return list[TrackedPlayer] with
the tracker's stable integer track_id.

Tracker params come from config (track_activation_threshold, lost_track_buffer,
minimum_matching_threshold).

Note: supervision deprecated `ByteTrack` in v0.28 with removal in v0.30; switch
to a different tracker (or pin supervision) before upgrading past v0.30.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from config import Config
from src.types import PlayerDetection, TrackedPlayer

logger = logging.getLogger(__name__)


class PlayerTracker:
    """ByteTrack wrapper for multi-object player tracking.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._tracker = self._build_tracker()

    def update(
        self,
        detections: list[PlayerDetection],
        frame_idx: int,
    ) -> list[TrackedPlayer]:
        """Update tracker with new detections for a frame.

        Returns:
            One TrackedPlayer per surviving track this frame, with the raw
            integer track_id from ByteTrack.
        """
        from supervision import Detections

        if not detections:
            sv_det = Detections.empty()
        else:
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            confs = np.array([d.confidence for d in detections], dtype=np.float32)
            cls_id = np.zeros(len(detections), dtype=int)
            sv_det = Detections(xyxy=xyxy, confidence=confs, class_id=cls_id)

        sv_tracked = self._tracker.update_with_detections(sv_det)

        if len(sv_tracked) == 0 or sv_tracked.tracker_id is None:
            return []

        out: list[TrackedPlayer] = []
        for bbox, conf, tid in zip(
            sv_tracked.xyxy,
            sv_tracked.confidence,
            sv_tracked.tracker_id,
        ):
            out.append(TrackedPlayer(
                frame_idx=frame_idx,
                track_id=int(tid),
                bbox=tuple(float(v) for v in bbox),
                confidence=float(conf),
            ))
        return out

    def reset(self) -> None:
        """Reset tracker state (call between clips)."""
        self._tracker = self._build_tracker()

    def _build_tracker(self):
        from supervision import ByteTrack

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return ByteTrack(
                track_activation_threshold=self._config.track_activation_threshold,
                lost_track_buffer=self._config.lost_track_buffer,
                minimum_matching_threshold=self._config.minimum_matching_threshold,
                frame_rate=30,
            )
