"""Bib colour classification — dominant-hue approach.

Detects whatever colour the bib/shirt actually is rather than assuming any
specific team colours. Works by:
  1. Slicing the upper-torso ROI (bib_roi_y).
  2. Separating pixels into "colourful" (high S + V) vs "white" (low S, high V).
  3. If colourful pixels dominate, bin their hues and return the dominant name.
  4. If white pixels dominate with no strong colourful signal, return "white".
  5. Otherwise return "unknown".

Recognised colour labels: red, orange, yellow, green, cyan, blue, purple, pink,
white, unknown.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

# (hue_lo, hue_hi, name) — OpenCV HSV hue range 0–179.
# Red wraps around 0/179 so hues outside these bins default to "red".
_HUE_BINS: list[tuple[int, int, str]] = [
    (10,  25,  "orange"),
    (25,  35,  "yellow"),
    (35,  75,  "green"),
    (75,  100, "cyan"),
    (100, 130, "blue"),
    (130, 155, "purple"),
    (155, 170, "pink"),
]


def _hue_to_colour(hues: np.ndarray) -> np.ndarray:
    """Map an array of OpenCV hue values (0–179) to colour-name strings."""
    out = np.full(hues.shape, "red", dtype=object)
    for lo, hi, name in _HUE_BINS:
        out[(hues >= lo) & (hues < hi)] = name
    return out


class BibColourClassifier:
    """Dominant-hue bib/shirt colour classifier.

    Args:
        config: Pipeline configuration (hsv thresholds + bib_roi_y).
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """Detect the dominant shirt colour from a player crop.

        Args:
            crop: BGR player crop (full bbox).

        Returns:
            (colour_label, confidence). colour_label is one of: red, orange,
            yellow, green, cyan, blue, purple, pink, white, unknown.
            Confidence is the fraction of ROI pixels that voted for the winner.
        """
        if crop is None or crop.size == 0:
            return ("unknown", 0.0)

        h = crop.shape[0]
        y1 = int(self._cfg.bib_roi_y[0] * h)
        y2 = int(self._cfg.bib_roi_y[1] * h)
        roi = crop[y1:y2]
        if roi.size == 0:
            return ("unknown", 0.0)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        total = H.size

        white_mask  = (S < self._cfg.hsv_white_max_s) & (V > self._cfg.hsv_white_min_v)
        colour_mask = (S >= self._cfg.hsv_min_s) & (V >= self._cfg.hsv_min_v)

        white_frac  = int(white_mask.sum()) / total
        colour_frac = int(colour_mask.sum()) / total

        if colour_frac >= 0.10:
            hues   = H[colour_mask]
            names  = _hue_to_colour(hues)
            labels, counts = np.unique(names, return_counts=True)
            best   = counts.argmax()
            return (str(labels[best]), float(counts[best]) / total)

        if white_frac >= 0.25:
            return ("white", white_frac)

        return ("unknown", 0.0)
