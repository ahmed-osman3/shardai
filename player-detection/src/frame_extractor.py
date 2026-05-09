"""Frame sampling from video clips for labelling.

Supports three sampling strategies:
- uniform: N frames evenly spaced across the clip.
- motion: N frames with highest inter-frame motion (best when the bib region needs to be sharp).
- manual: every Nth frame regardless of count.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

_MOTION_W, _MOTION_H = 320, 180


class FrameExtractor:
    """Samples frames from a video and saves them as JPEG for labelling.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
        strategy: Literal["uniform", "motion", "manual"],
        count: int = 200,
        every_n: int = 10,
    ) -> list[Path]:
        """Extract frames from a video clip and save as JPEG."""
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Extracting frames from %s (%d total, strategy=%s)",
            video_path.name, total, strategy,
        )

        if strategy == "uniform":
            saved = self._uniform(cap, total, count, output_dir)
        elif strategy == "motion":
            saved = self._motion(cap, total, count, output_dir)
        elif strategy == "manual":
            saved = self._manual(cap, total, every_n, output_dir)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        cap.release()
        logger.info("Saved %d frames to %s", len(saved), output_dir)
        return saved

    def _uniform(self, cap: cv2.VideoCapture, total: int, count: int, out: Path) -> list[Path]:
        indices = np.linspace(0, total - 1, min(count, total)).astype(int)
        return self._save_at_indices(cap, indices, out)

    def _motion(self, cap: cv2.VideoCapture, total: int, count: int, out: Path) -> list[Path]:
        scores: list[float] = []
        prev_gray: np.ndarray | None = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (_MOTION_W, _MOTION_H))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                scores.append(0.0)
            else:
                scores.append(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))))
            prev_gray = gray

        top_indices = np.argsort(scores)[-min(count, len(scores)):]
        top_indices = np.sort(top_indices)
        return self._save_at_indices(cap, top_indices, out)

    def _manual(self, cap: cv2.VideoCapture, total: int, every_n: int, out: Path) -> list[Path]:
        indices = np.arange(0, total, every_n)
        return self._save_at_indices(cap, indices, out)

    def _save_at_indices(
        self,
        cap: cv2.VideoCapture,
        indices: np.ndarray,
        out: Path,
    ) -> list[Path]:
        saved: list[Path] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            dest = out / f"{int(idx):06d}.jpg"
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(dest)
        return saved
