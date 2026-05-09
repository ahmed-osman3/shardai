"""Tests for src/bib_colour.py — HSV bib colour classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.bib_colour import BibColourClassifier


def _swatch(bgr: tuple[int, int, int], h: int = 100, w: int = 60) -> np.ndarray:
    """Build a uniform BGR swatch."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = bgr[0]
    img[..., 1] = bgr[1]
    img[..., 2] = bgr[2]
    return img


def test_classify_solid_red_swatch():
    cls = BibColourClassifier(Config())
    label, conf = cls.classify(_swatch((0, 0, 220)))  # bright red BGR
    assert label == "red"
    assert conf > 0.5


def test_classify_solid_blue_swatch():
    cls = BibColourClassifier(Config())
    label, conf = cls.classify(_swatch((220, 50, 0)))  # bright blue BGR
    assert label == "blue"
    assert conf > 0.5


def test_classify_grey_swatch_returns_unknown():
    cls = BibColourClassifier(Config())
    label, conf = cls.classify(_swatch((128, 128, 128)))
    assert label == "unknown"
    assert conf == 0.0


def test_classify_white_swatch_returns_white():
    cls = BibColourClassifier(Config())
    label, conf = cls.classify(_swatch((255, 255, 255)))
    assert label == "white"


def test_classify_black_swatch_returns_unknown():
    cls = BibColourClassifier(Config())
    label, conf = cls.classify(_swatch((0, 0, 0)))
    assert label == "unknown"


def test_classify_empty_crop_returns_unknown():
    cls = BibColourClassifier(Config())
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    label, conf = cls.classify(empty)
    assert label == "unknown"
    assert conf == 0.0


def test_classify_handles_none_input():
    cls = BibColourClassifier(Config())
    label, conf = cls.classify(None)  # type: ignore[arg-type]
    assert label == "unknown"


def test_classify_uses_upper_torso_roi():
    """A crop where only the upper torso is red should classify as red,
    even when the lower body is blue."""
    cls = BibColourClassifier(Config())
    crop = np.zeros((200, 60, 3), dtype=np.uint8)
    # Bib region (15-55% of height = rows 30..110): red
    crop[30:110] = (0, 0, 220)
    # Lower body: blue (should NOT influence the result)
    crop[110:] = (220, 50, 0)
    label, _ = cls.classify(crop)
    assert label == "red"


def test_classify_red_high_hue_wedge():
    """Red wraps around 0 in HSV — pixels at hue ~175 are also red."""
    cls = BibColourClassifier(Config())
    # Build a swatch with hue ~175 (high red wedge)
    import cv2
    hsv = np.zeros((100, 60, 3), dtype=np.uint8)
    hsv[..., 0] = 175
    hsv[..., 1] = 200
    hsv[..., 2] = 200
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    label, _ = cls.classify(bgr)
    assert label == "red"
