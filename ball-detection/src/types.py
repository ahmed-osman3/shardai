"""Shared dataclasses for the ball detection pipeline.

Centralized so real modules don't import from mock modules.
Replace mocks with real implementations without changing any imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # xyxy, pixel coords
    confidence: float
    frame_idx: int


@dataclass
class TrackedBall:
    frame_idx: int
    x: float
    y: float
    source: Literal["detected", "interpolated", "lost"]
    confidence: float


@dataclass
class PlayerDetection:
    bbox: tuple[float, float, float, float]  # xyxy
    player_id: str
    team_id: str
    confidence: float


@dataclass
class Event:
    type: str  # "goal" | "shot" | "pass" | "possession_change"
    frame_idx: int
    ts_seconds: float
    primary_player: str | None
    secondary_player: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    json_path: Path
    video_path: Path | None  # None when --no-video
    stats: dict
