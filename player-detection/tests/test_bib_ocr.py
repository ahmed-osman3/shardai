"""Tests for src/bib_ocr.py — EasyOCR-backed bib number reader.

The "real OCR" tests are skipped if easyocr isn't importable. The blank/tiny
crop tests don't need easyocr because they short-circuit before _ensure_reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.bib_ocr import BibNumberOCR


def _has_easyocr() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


def test_read_returns_none_on_blank_crop():
    ocr = BibNumberOCR(Config())
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    n, conf = ocr.read(empty)
    assert n is None
    assert conf == 0.0


def test_read_returns_none_on_tiny_crop():
    ocr = BibNumberOCR(Config())
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    n, conf = ocr.read(tiny)
    assert n is None
    assert conf == 0.0


def test_read_returns_none_on_none_input():
    ocr = BibNumberOCR(Config())
    n, conf = ocr.read(None)  # type: ignore[arg-type]
    assert n is None
    assert conf == 0.0


@pytest.mark.skipif(not _has_easyocr(), reason="easyocr not installed")
@pytest.mark.xfail(reason="OpenCV-rendered digits don't resemble bib fonts; real-clip eval is the test")
def test_read_recognises_synthetic_digit():
    """Render '7' onto a high-contrast crop. Kept as xfail because EasyOCR was
    trained on natural-scene text and doesn't reliably resolve OpenCV-rendered
    Hershey-font digits — real bib OCR works fine on the actual clip footage,
    which is what eval_identity.py covers."""
    import cv2

    crop = np.full((200, 100, 3), 255, dtype=np.uint8)
    cv2.putText(crop, "7", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
    ocr = BibNumberOCR(Config())
    n, _ = ocr.read(crop)
    assert n == 7
