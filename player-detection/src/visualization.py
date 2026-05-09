"""Frame annotation utilities.

Draws player bounding boxes labelled with resolved bib IDs onto video frames.
Box colour matches the detected shirt colour; tracks with no resolved identity
render in grey with their raw track ID prefixed by '?'.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import Config
from src.types import TrackedPlayer

logger = logging.getLogger(__name__)

# BGR — one entry per colour label returned by BibColourClassifier
_TEAM_COLOURS: dict[str, tuple[int, int, int]] = {
    "red":     (0,   0,   220),
    "orange":  (0,   140, 255),
    "yellow":  (0,   220, 220),
    "green":   (0,   200, 0),
    "cyan":    (220, 220, 0),
    "blue":    (220, 80,  0),
    "purple":  (180, 0,   180),
    "pink":    (200, 100, 255),
    "white":   (220, 220, 220),
    "black":   (60,  60,  60),
    "unknown": (140, 140, 140),
}
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.55
_LABEL_THICK = 1


def draw_frame(
    frame: np.ndarray,
    tracked: list[TrackedPlayer],
    labels: dict[int, tuple[str, str]],
    config: Config,
) -> np.ndarray:
    """Annotate frame with labelled player boxes.

    Args:
        frame: BGR source frame. Not modified in place.
        tracked: TrackedPlayer entries for this frame.
        labels: Mapping of raw track_id -> (display_label, team). Unresolved
            tracks should map to ("?-{tid}", "unknown").
        config: Pipeline configuration.

    Returns:
        Annotated BGR frame as a new array.
    """
    out = frame.copy()

    for tp in tracked:
        label, team = labels.get(tp.track_id, (f"?-{tp.track_id}", "unknown"))
        is_unresolved = team == "unknown"
        if is_unresolved and not config.draw_unresolved_boxes:
            continue

        colour = _TEAM_COLOURS.get(team, _TEAM_COLOURS["unknown"])
        x1, y1, x2, y2 = (int(v) for v in tp.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        _draw_label(out, label, (x1, y1), colour)

    return out


def setup_video_writer(
    output_path: Path,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> cv2.VideoWriter:
    """Create a cv2.VideoWriter for an annotated output clip."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")
    return writer


def _draw_label(
    img: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    """Draw a filled background + text label above an anchor point."""
    (tw, th), baseline = cv2.getTextSize(text, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICK)
    x, y = anchor
    pad = 3
    bg_top_left = (x, max(0, y - th - 2 * pad))
    bg_bottom_right = (x + tw + 2 * pad, y)
    cv2.rectangle(img, bg_top_left, bg_bottom_right, colour, -1)
    cv2.putText(
        img,
        text,
        (x + pad, y - pad),
        _LABEL_FONT,
        _LABEL_SCALE,
        (255, 255, 255),
        _LABEL_THICK,
        cv2.LINE_AA,
    )
