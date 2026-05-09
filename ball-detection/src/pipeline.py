"""Pipeline orchestrator.

Runs ball detection + tracking + mock player/event modules on a video clip
and writes JSON + annotated video outputs.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import cv2
from tqdm import tqdm

from config import Config
from src.detection import BallDetector
from src.mocks.calibration import MockCalibration
from src.mocks.event_detector import MockEventDetector
from src.mocks.player_detector import MockPlayerDetector
from src.mocks.storage import MockStorage
from src.tracking import BallTracker
from src.types import PipelineResult, TrackedBall
from src.visualization import draw_frame, setup_video_writer

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a simple timestamp format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    video_path: Path,
    config: Config,
    *,
    outputs_dir: Path | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
    persist_tracks: bool = False,
) -> PipelineResult:
    """Run the full ball detection pipeline on a single video file.

    Args:
        video_path: Path to the input .mp4 clip.
        config: Pipeline configuration.
        outputs_dir: Override output directory (default: config.outputs_dir).
        max_frames: Stop after this many frames (None = process all).
        write_video: Write annotated MP4 output (False = JSON only).
        persist_tracks: Pickle ball+player tracks before event detection
                        so event rules can be iterated without re-running inference.

    Returns:
        PipelineResult with paths to output files and summary stats.
    """
    out_dir = outputs_dir if outputs_dir is not None else config.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # --- open video ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    frame_step = max(1, round(fps / config.target_fps)) if config.target_fps else 1
    logger.info(
        "Video: %s  %.1f fps  %dx%d  %d frames  step=%d",
        video_path.name, fps, frame_w, frame_h, total_frames, frame_step,
    )

    # --- instantiate modules ---
    detector = BallDetector(config)
    tracker = BallTracker(config)
    player_detector = MockPlayerDetector()
    calibration = MockCalibration(frame_w, frame_h)
    event_detector = MockEventDetector(config, calibration)

    # --- video writer ---
    video_path_out = out_dir / f"{stem}_annotated.mp4" if write_video else None
    writer: cv2.VideoWriter | None = None
    if write_video and video_path_out is not None:
        writer = setup_video_writer(video_path_out, fps / frame_step, frame_w, frame_h)

    # --- per-frame loop ---
    ball_track: list[TrackedBall | None] = []
    player_track: list[list] = []
    ball_history: list[TrackedBall] = []

    n_detected = n_interpolated = n_lost = 0
    frame_idx = 0
    processed = 0

    with tqdm(total=total_frames, unit="fr", desc=stem) as pbar:
        while processed < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            detections = detector.detect(frame, frame_idx)
            ball = tracker.update(detections, frame_idx)
            players = player_detector.detect(frame, frame_idx)

            ball_track.append(ball)
            player_track.append(players)

            if ball is not None:
                ball_history.append(ball)
                if ball.source == "detected":
                    n_detected += 1
                elif ball.source == "interpolated":
                    n_interpolated += 1
                else:
                    n_lost += 1

            if writer is not None:
                annotated = draw_frame(frame, ball, ball_history, players, config)
                writer.write(annotated)

            frame_idx += 1
            processed += 1
            pbar.update(1)

    cap.release()
    if writer is not None:
        writer.release()

    # --- persist raw tracks ---
    if persist_tracks:
        tracks_path = out_dir / f"{stem}_tracks.pkl"
        with open(tracks_path, "wb") as f:
            pickle.dump({"ball_track": ball_track, "player_track": player_track, "fps": fps}, f)
        logger.info("Tracks saved to %s", tracks_path)

    # --- event detection ---
    events = event_detector.process(ball_track, player_track, fps)
    logger.info("Detected %d events", len(events))

    # --- stats ---
    n_frames = len(ball_track)
    stats = {
        "total_frames": n_frames,
        "detection_rate": n_detected / n_frames if n_frames else 0.0,
        "interpolation_rate": n_interpolated / n_frames if n_frames else 0.0,
        "lost_rate": n_lost / n_frames if n_frames else 0.0,
        "event_count": len(events),
    }

    # --- serialize JSON ---
    json_path = out_dir / f"{stem}.json"
    payload = {
        "schema_version": "1.0",
        "meta": {
            "source_video": video_path.name,
            "fps": fps,
            "frame_count": n_frames,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": config.resolve_model_path().name,
        },
        "frames": [
            {
                "idx": i,
                "ts": round(i / fps, 4) if fps else 0.0,
                "ball": (
                    {
                        "x": round(b.x, 2),
                        "y": round(b.y, 2),
                        "source": b.source,
                        "conf": round(b.confidence, 4),
                    }
                    if b is not None
                    else None
                ),
                "players": [
                    {
                        "id": p.player_id,
                        "team": p.team_id,
                        "bbox": [int(v) for v in p.bbox],
                        "conf": p.confidence,
                    }
                    for p in player_track[i]
                ],
            }
            for i, b in enumerate(ball_track)
        ],
        "events": [
            {
                "type": e.type,
                "frame_idx": e.frame_idx,
                "ts_seconds": round(e.ts_seconds, 4),
                "primary_player": e.primary_player,
                "secondary_player": e.secondary_player,
                "metadata": e.metadata,
            }
            for e in events
        ],
        "stats": stats,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("JSON saved to %s", json_path)

    # --- mock storage upload ---
    storage = MockStorage(out_dir / "mock_storage")
    storage.upload(json_path, f"{stem}.json")
    if video_path_out and video_path_out.exists():
        storage.upload(video_path_out, f"{stem}_annotated.mp4")

    return PipelineResult(json_path=json_path, video_path=video_path_out, stats=stats)
