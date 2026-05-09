"""Shared dataclasses for the player + bib detection pipeline.

Centralized so real modules don't import from mock modules.
Replace mocks with real implementations without changing any imports.

Schema-compatible with the ball-detection module so downstream code can union
per-frame records (ball field stays None in this module's output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PlayerDetection:
    bbox: tuple[float, float, float, float]  # xyxy, pixel coords
    confidence: float
    frame_idx: int


@dataclass
class TrackedPlayer:
    """A detection that survived the tracker for a given frame.

    track_id is the raw ID from ByteTrack (pre-merge). After IdentityResolver
    runs, multiple track_ids may collapse to a single bib identity.
    """

    frame_idx: int
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class OCRReading:
    """A single (colour, number) sample taken on a player crop at one frame."""

    track_id: int
    frame_idx: int
    colour: str  # dominant shirt colour or "unknown"
    number: int | None  # 1..99, or None when OCR fails or returns out-of-range
    ocr_confidence: float


@dataclass
class BibIdentity:
    """The voted-on bib identity for a tracker ID after the full video is processed."""

    track_id: int
    bib_id: str | None  # e.g. "P18" / "W10" / None when unresolved
    colour: str  # dominant shirt colour or "unknown"
    number: int | None
    vote_count: int  # supporting OCR readings
    sample_count: int  # total OCR samples taken on this track
    merged_into: int | None = None  # canonical track_id when multiple raw tracks share a bib


@dataclass
class FrameRecord:
    """One frame's worth of fully resolved players, ready for JSON serialization."""

    idx: int
    ts: float
    ball: None  # always None in this module — kept for ball-detection schema parity
    players: list[dict] = field(default_factory=list)


@dataclass
class PipelineResult:
    json_path: Path
    video_path: Path | None  # None when --no-video
    stats: dict
