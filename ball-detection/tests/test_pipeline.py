"""Integration tests for src/pipeline.py.

Creates a synthetic video and runs the full pipeline end-to-end.
Requires model weights in models/ — tests skip automatically if absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.pipeline import run_pipeline


def _requires_model(config: Config) -> None:
    if not config.resolve_model_path().exists():
        pytest.skip("Model not found — drop weights into models/ to run pipeline tests.")


def _make_synthetic_video(path: Path, n_frames: int = 30, w: int = 640, h: int = 480) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h),
    )
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # draw a white circle to vaguely resemble a ball
        cv2.circle(frame, (w // 2 + i * 5, h // 2), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    p = tmp_path / "test_clip.mp4"
    _make_synthetic_video(p)
    return p


def test_pipeline_creates_json_output(synthetic_video: Path, tmp_path: Path):
    config = Config()
    _requires_model(config)
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
    )
    assert result.json_path.exists()
    data = json.loads(result.json_path.read_text())
    assert data["schema_version"] == "1.0"
    assert "frames" in data
    assert "events" in data


def test_pipeline_creates_annotated_video(synthetic_video: Path, tmp_path: Path):
    config = Config()
    _requires_model(config)
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=True,
    )
    assert result.video_path is not None
    assert result.video_path.exists()
    cap = cv2.VideoCapture(str(result.video_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count > 0


def test_pipeline_no_video_flag(synthetic_video: Path, tmp_path: Path):
    config = Config()
    _requires_model(config)
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
    )
    assert result.video_path is None


def test_pipeline_stats_keys(synthetic_video: Path, tmp_path: Path):
    config = Config()
    _requires_model(config)
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
    )
    required = {"total_frames", "detection_rate", "interpolation_rate", "lost_rate", "event_count"}
    assert required.issubset(result.stats.keys())


def test_pipeline_max_frames(synthetic_video: Path, tmp_path: Path):
    config = Config()
    _requires_model(config)
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        max_frames=10,
    )
    assert result.stats["total_frames"] <= 10
