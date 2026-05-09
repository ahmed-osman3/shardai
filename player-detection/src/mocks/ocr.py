"""Mock OCR.

Returns deterministic, controlled (colour, number) readings keyed by track_id.
Used to exercise the identity-resolution and track-merging logic without a real
OCR engine, and to keep the synthetic-video pipeline test runnable on CI.

Two flavours:
- TruthTableMockOCR: lookup table {track_id: (colour, number)} so a test author
  can scripts exact identity flows (including ambiguous reads, swaps, merges).
- NoisyMockOCR: returns the truth (colour, number) with probability `accuracy`
  and a random plausible misread otherwise — for stress-testing the voter.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from src.types import OCRReading

logger = logging.getLogger(__name__)


class TruthTableMockOCR:
    """Always returns the configured (colour, number) for a given track_id.

    Args:
        truth: Mapping of track_id -> (colour, number). track_ids absent from
            the table return ("unknown", None).
        confidence: Fixed OCR confidence to report on every reading.
    """

    def __init__(
        self,
        truth: dict[int, tuple[Literal["red", "blue"], int]],
        confidence: float = 0.95,
    ) -> None:
        self._truth = truth
        self._confidence = confidence

    def read(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_idx: int,
    ) -> OCRReading:
        if track_id in self._truth:
            colour, number = self._truth[track_id]
            return OCRReading(
                track_id=track_id,
                frame_idx=frame_idx,
                colour=colour,
                number=number,
                ocr_confidence=self._confidence,
            )
        return OCRReading(
            track_id=track_id,
            frame_idx=frame_idx,
            colour="unknown",
            number=None,
            ocr_confidence=0.0,
        )


class NoisyMockOCR:
    """Returns the truth (colour, number) with probability `accuracy`,
    otherwise a random plausible misread or a None number.

    Useful for verifying that the per-track voter recovers from sporadic OCR
    failure — which is the whole reason the voter exists.
    """

    def __init__(
        self,
        truth: dict[int, tuple[Literal["red", "blue"], int]],
        accuracy: float = 0.7,
        seed: int = 0,
    ) -> None:
        self._truth = truth
        self._accuracy = accuracy
        self._rng = np.random.default_rng(seed)

    def read(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_idx: int,
    ) -> OCRReading:
        if track_id not in self._truth:
            return OCRReading(track_id, frame_idx, "unknown", None, 0.0)

        colour, number = self._truth[track_id]
        if self._rng.random() < self._accuracy:
            return OCRReading(track_id, frame_idx, colour, number, 0.9)
        # Misread: random number 1..12 or None
        if self._rng.random() < 0.5:
            wrong = int(self._rng.integers(1, 13))
            return OCRReading(track_id, frame_idx, colour, wrong, 0.5)
        return OCRReading(track_id, frame_idx, "unknown", None, 0.0)
