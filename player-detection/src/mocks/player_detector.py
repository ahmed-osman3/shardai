"""Mock player detector.

Mirrors the API of the real PlayerDetector so the pipeline orchestrator
can run end-to-end on synthetic frames during Phase A scaffolding, before
the real YOLO-backed detector is wired in.

Players drift smoothly across the frame using per-player sinusoidal motion;
ground-truth (colour, number) is exposed via `truth_for(synthetic_idx)` so
TruthTableMockOCR can be configured against the same source of truth.
"""

from __future__ import annotations

import logging

import numpy as np

from src.types import PlayerDetection

logger = logging.getLogger(__name__)


class MockPlayerDetector:
    """Generates plausible synthetic player bounding boxes per frame.

    Args:
        n_per_team: Number of players per team (default 7 for 7-a-side).
        seed: RNG seed for reproducible motion.
    """

    def __init__(self, n_per_team: int = 7, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._n_per_team = n_per_team
        self._players: list[dict] = []

        cols = n_per_team
        for team_idx, (team_id, prefix) in enumerate([("red", "R"), ("blue", "B")]):
            for i in range(n_per_team):
                base_x = 160 + (i / (cols - 1)) * 1600 if cols > 1 else 960
                base_y = 300 + team_idx * 480
                self._players.append({
                    "synth_idx": team_idx * n_per_team + i,
                    "team_id": team_id,
                    "number": i + 1,
                    "base_x": base_x,
                    "base_y": base_y,
                    "freq_x": rng.uniform(0.01, 0.05),
                    "freq_y": rng.uniform(0.01, 0.05),
                    "phase_x": rng.uniform(0, 2 * np.pi),
                    "phase_y": rng.uniform(0, 2 * np.pi),
                    "amplitude": rng.uniform(20, 60),
                })

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[PlayerDetection]:
        """Return synthetic player detections for a given frame."""
        h, w = frame.shape[:2]
        result: list[PlayerDetection] = []
        for p in self._players:
            x = p["base_x"] + p["amplitude"] * np.sin(frame_idx * p["freq_x"] + p["phase_x"])
            y = p["base_y"] + p["amplitude"] * np.sin(frame_idx * p["freq_y"] + p["phase_y"])
            x = float(np.clip(x, 50, w - 50))
            y = float(np.clip(y, 50, h - 50))
            result.append(PlayerDetection(
                bbox=(x - 20, y - 40, x + 20, y + 40),
                confidence=0.85,
                frame_idx=frame_idx,
            ))
        return result

    def truth_table(self) -> dict[int, tuple[str, int]]:
        """Ground-truth (colour, number) keyed by synthetic player index.

        With MockPlayerTracker(stable=True), the tracker assigns track_id =
        synthetic_idx, so this table can be passed directly to TruthTableMockOCR.
        """
        return {p["synth_idx"]: (p["team_id"], p["number"]) for p in self._players}
