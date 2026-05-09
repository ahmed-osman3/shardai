"""Mock ball detector.

This module's focus is players + bib OCR; ball detection is the other
module's responsibility. We mock it as a no-op so the per-frame loop and
JSON schema stay parallel to ball-detection's, with the `ball` field
always serializing to null.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MockBallDetector:
    """Always returns None — the JSON `ball` field will be null in this module's output."""

    def detect(self, frame: np.ndarray, frame_idx: int) -> None:
        return None
