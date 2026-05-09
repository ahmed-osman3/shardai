"""Mock player detector.

Returns synthetic player bounding boxes that drift smoothly across the pitch
using per-player sinusoidal motion. Stand-in for the real player detection +
bib OCR + tracking pipeline that will be built separately.
"""

from __future__ import annotations

import logging

import numpy as np

from src.types import PlayerDetection

logger = logging.getLogger(__name__)


class MockPlayerDetector:
    """Generates plausible synthetic player positions for pipeline integration tests.

    Players drift smoothly across the pitch using sinusoidal motion so that
    the event detector has realistic-looking inputs without any real CV.

    Args:
        n_per_team: Number of players per team (default 7 for 7-a-side).
        seed: RNG seed for reproducible player motion.
    """

    def __init__(self, n_per_team: int = 7, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._n_per_team = n_per_team
        self._players: list[dict] = []

        # Lay out players on a n_per_team × 2 grid across the frame
        cols = n_per_team
        for team_idx, (team_id, prefix) in enumerate([("red", "R"), ("blue", "B")]):
            for i in range(n_per_team):
                base_x = 160 + (i / (cols - 1)) * 1600 if cols > 1 else 960
                base_y = 300 + team_idx * 480
                self._players.append({
                    "player_id": f"{prefix}{i + 1}",
                    "team_id": team_id,
                    "base_x": base_x,
                    "base_y": base_y,
                    "freq_x": rng.uniform(0.01, 0.05),
                    "freq_y": rng.uniform(0.01, 0.05),
                    "phase_x": rng.uniform(0, 2 * np.pi),
                    "phase_y": rng.uniform(0, 2 * np.pi),
                    "amplitude": rng.uniform(20, 60),
                })

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[PlayerDetection]:
        """Return synthetic player detections for a given frame.

        Args:
            frame: Source frame (used only for shape; not processed).
            frame_idx: Current frame index, drives the motion model.

        Returns:
            List of PlayerDetection with smoothly drifting positions.
        """
        h, w = frame.shape[:2]
        result = []
        for p in self._players:
            x = p["base_x"] + p["amplitude"] * np.sin(frame_idx * p["freq_x"] + p["phase_x"])
            y = p["base_y"] + p["amplitude"] * np.sin(frame_idx * p["freq_y"] + p["phase_y"])
            x = float(np.clip(x, 50, w - 50))
            y = float(np.clip(y, 50, h - 50))
            # 40×80 pixel bbox centred on (x, y) — rough person silhouette
            result.append(PlayerDetection(
                bbox=(x - 20, y - 40, x + 20, y + 40),
                player_id=p["player_id"],
                team_id=p["team_id"],
                confidence=0.85,
            ))
        return result
