"""Bib number OCR (Tesseract via pytesseract).

Per crop:
1. Slice out the upper-torso ROI (where the bib sits).
2. Convert to grayscale + CLAHE contrast normalisation.
3. Otsu binarise — Tesseract is most reliable on clean black-on-white.
4. Run pytesseract in single-word mode with digit-only allowlist.
5. Parse to int; require 1..99.

Returns (None, 0.0) rather than guessing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class BibNumberOCR:
    """Tesseract-backed bib number reader."""

    # oem 3 = LSTM only (more robust on real-world images than combined mode)
    _PSM_WORD = "--oem 3 --psm 8"   # single word
    _PSM_LINE = "--oem 3 --psm 7"   # single text line (fallback for two-digit numbers)
    _WHITELIST = "-c tessedit_char_whitelist=0123456789"

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._tess_available: bool | None = None  # checked lazily

    def _ensure_tesseract(self) -> bool:
        if self._tess_available is not None:
            return self._tess_available
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tess_available = True
            logger.info("Tesseract OCR ready")
        except Exception as e:
            logger.warning("Tesseract not available (%s). Install: brew install tesseract && pip install pytesseract", e)
            self._tess_available = False
        return self._tess_available

    def read(
        self,
        crop: np.ndarray,
        debug_dir: "Path | None" = None,
        debug_tag: str = "",
    ) -> tuple[int | None, float]:
        """Read the bib number from a single player crop.

        Args:
            crop: BGR player crop (full bbox).
            debug_dir: If set, save the preprocessed binary ROI here.
            debug_tag: Filename prefix for debug images.

        Returns:
            (number, confidence). number is an int 1..99 or None on failure.
        """
        if crop is None or crop.size == 0:
            return (None, 0.0)

        h = crop.shape[0]
        y1 = int(self._cfg.bib_roi_y[0] * h)
        y2 = int(self._cfg.bib_roi_y[1] * h)
        roi = crop[y1:y2]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            return (None, 0.0)

        binary = self._preprocess(roi)

        if debug_dir is not None:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(debug_dir) / f"{debug_tag}.png"), binary)

        if not self._ensure_tesseract():
            return (None, 0.0)

        import pytesseract

        config_str = f"{self._PSM_WORD} {self._WHITELIST}"
        result = self._run_tess(pytesseract, binary, config_str)
        if result is None:
            # retry with line mode — helps when two-digit numbers confuse word mode
            config_str = f"{self._PSM_LINE} {self._WHITELIST}"
            result = self._run_tess(pytesseract, binary, config_str)

        if result is not None:
            logger.debug("OCR %s → %d", debug_tag, result)
        return (result, 1.0) if result is not None else (None, 0.0)

    # ------------------------------------------------------------------
    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Prepare ROI for Tesseract — grayscale only, no manual binarization.

        Our earlier Otsu/adaptive passes destroyed the image by picking up
        shirt texture and background as noise. Tesseract's internal LSTM
        binarizes better than we can for these crops, so we just clean up
        contrast and hand it a grayscale image.
        """
        # Narrow to centre 70% horizontally to reduce background on the sides
        w = roi.shape[1]
        x1, x2 = int(0.15 * w), int(0.85 * w)
        if x2 - x1 >= 8:
            roi = roi[:, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Scale to standard height — both upscale and downscale
        target_h = 128
        if gray.shape[0] != target_h:
            scale = target_h / gray.shape[0]
            new_w = max(16, int(gray.shape[1] * scale))
            gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Light contrast normalisation only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # Small border so Tesseract doesn't clip edge digits
        gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

        return gray

    def _run_tess(self, pytesseract, img: np.ndarray, config_str: str) -> int | None:
        """Run Tesseract and return a valid bib number or None."""
        # Try both polarities — some bibs are white-on-dark
        for image in (img, cv2.bitwise_not(img)):
            try:
                text = pytesseract.image_to_string(image, config=config_str).strip()
            except Exception as e:
                logger.debug("Tesseract error: %s", e)
                continue
            for tok in text.split():
                try:
                    n = int(tok)
                except ValueError:
                    continue
                if 1 <= n <= 99:
                    return n
        return None
