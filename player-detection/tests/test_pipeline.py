"""Integration tests for src/pipeline.py.

Phase A: runs entirely in mock_mode on a synthetic video — no model weights
required. Verifies JSON schema, video output, stats keys, and that resolved
bib identities flow through to the per-frame `players` records.
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


def _make_synthetic_video(path: Path, n_frames: int = 60, w: int = 1920, h: int = 1080) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (w, h),
    )
    for _ in range(n_frames):
        frame = np.full((h, w, 3), 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    p = tmp_path / "test_clip.mp4"
    _make_synthetic_video(p)
    return p


def test_pipeline_creates_json_output(synthetic_video: Path, tmp_path: Path):
    config = Config()
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        mock_mode=True,
    )
    assert result.json_path.exists()
    data = json.loads(result.json_path.read_text())
    assert data["schema_version"] == "1.0"
    assert "frames" in data
    assert "events" in data
    assert "stats" in data
    assert data["meta"]["ocr_engine"] == "mock"


def test_pipeline_creates_annotated_video(synthetic_video: Path, tmp_path: Path):
    config = Config()
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=True,
        mock_mode=True,
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
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        mock_mode=True,
    )
    assert result.video_path is None


def test_pipeline_stats_keys(synthetic_video: Path, tmp_path: Path):
    config = Config()
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        mock_mode=True,
    )
    required = {
        "total_frames",
        "tracks_created",
        "tracks_after_merge",
        "identities_resolved",
        "identities_unresolved",
        "mean_players_per_frame",
        "ocr_samples_total",
        "ocr_samples_with_number",
    }
    assert required.issubset(result.stats.keys())


def test_pipeline_max_frames(synthetic_video: Path, tmp_path: Path):
    config = Config()
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        max_frames=10,
        mock_mode=True,
    )
    assert result.stats["total_frames"] <= 10


def test_pipeline_resolves_all_mock_identities(synthetic_video: Path, tmp_path: Path):
    """In mock_mode, TruthTableMockOCR returns the correct (colour, number) every
    sample, so every visible synthetic player should resolve to a bib id."""
    config = Config()
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        mock_mode=True,
    )
    assert result.stats["identities_resolved"] == 14
    assert result.stats["identities_unresolved"] == 0


def test_pipeline_frames_have_resolved_bib_ids(synthetic_video: Path, tmp_path: Path):
    config = Config()
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        mock_mode=True,
    )
    data = json.loads(result.json_path.read_text())
    last_frame = data["frames"][-1]
    assert last_frame["ball"] is None  # always null in this module
    ids = {p["id"] for p in last_frame["players"]}
    expected = {f"R{i}" for i in range(1, 8)} | {f"B{i}" for i in range(1, 8)}
    assert ids == expected


def test_pipeline_ball_field_is_null_in_all_frames(synthetic_video: Path, tmp_path: Path):
    config = Config()
    result = run_pipeline(
        synthetic_video, config,
        outputs_dir=tmp_path / "out",
        write_video=False,
        mock_mode=True,
    )
    data = json.loads(result.json_path.read_text())
    for f in data["frames"]:
        assert f["ball"] is None


def test_pipeline_real_components_instantiate():
    """Phase B: the real CV classes can be instantiated and called without raising.
    Skips if the YOLO weights aren't available."""
    cfg = Config()
    if not cfg.resolve_model_path().exists():
        pytest.skip("Model weights not present — drop yolo11m.pt into models/.")
    from src.bib_colour import BibColourClassifier
    from src.detection import PlayerDetector
    from src.tracking import PlayerTracker

    pd = PlayerDetector(cfg)
    pt = PlayerTracker(cfg)
    bc = BibColourClassifier(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # detect returns a list (possibly empty on a blank frame)
    assert isinstance(pd.detect(frame, 0), list)
    # tracker handles empty detections
    assert pt.update([], 0) == []
    # colour classifier returns ("unknown", 0.0) on a blank crop
    assert bc.classify(frame)[0] == "unknown"
