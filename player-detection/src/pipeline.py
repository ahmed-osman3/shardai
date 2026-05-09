"""Pipeline orchestrator.

Runs player detection + tracking + bib OCR + identity resolution on a video
clip and writes JSON + annotated video outputs. Uses mock components for the
ball detector, calibration, storage; the real CV components are stubs in
Phase A and are wired in during Phase B (see doc/IMPLEMENTATION_PLAN.md).

Pass `mock_mode=True` to substitute MockPlayerDetector / MockPlayerTracker /
TruthTableMockOCR for the real (stubbed) components — used by integration
tests so the pipeline runs end-to-end without model weights or a real video.

Two passes over the video:
  1. detect → track → sparse OCR sample → buffer per-frame tracks.
  2. After resolving identities at end-of-video, replay the source video and
     write annotated frames with the resolved R7 / B11 labels.
The two-pass design is forced by the voting model: a track's bib ID is only
known after we've seen all its OCR samples, so per-frame labels can't be
written during the first pass.
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
from src.identity import IdentityResolver
from src.mocks.ball_detector import MockBallDetector
from src.mocks.calibration import MockCalibration
from src.mocks.ocr import TruthTableMockOCR
from src.mocks.player_detector import MockPlayerDetector
from src.mocks.player_tracker import MockPlayerTracker
from src.mocks.storage import MockStorage
from src.types import OCRReading, PipelineResult, TrackedPlayer
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
    start_frame: int = 0,
    write_video: bool = True,
    persist_tracks: bool = False,
    ocr_every_n: int | None = None,
    mock_mode: bool = False,
    debug_ocr: bool = False,
) -> PipelineResult:
    """Run the full player + bib detection pipeline on a single video file.

    Args:
        video_path: Path to the input .mp4 clip.
        config: Pipeline configuration.
        outputs_dir: Override output directory (default: config.outputs_dir).
        max_frames: Stop after this many frames (None = process all).
        start_frame: Skip the first N frames of the source video (e.g. to skip
            intro footage). Frame indices in the output JSON remain 0-indexed
            relative to the processed segment.
        write_video: Write annotated MP4 output (False = JSON only).
        persist_tracks: Pickle raw per-frame tracks before identity resolution
            so identity logic can be iterated without re-running inference.
        ocr_every_n: Override config.ocr_every_n.
        mock_mode: Use MockPlayerDetector/MockPlayerTracker/TruthTableMockOCR
            instead of the real (Phase B) components. Used by tests so the
            scaffold runs without weights or real video.

    Returns:
        PipelineResult with paths to output files and summary stats.
    """
    out_dir = outputs_dir if outputs_dir is not None else config.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    every_n = ocr_every_n if ocr_every_n is not None else config.ocr_every_n
    debug_crops_dir = (out_dir / "debug_crops") if debug_ocr else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    remaining = max(0, source_total - start_frame)
    total_frames = remaining if max_frames is None else min(remaining, max_frames)

    frame_step = max(1, round(fps / config.target_fps)) if config.target_fps else 1
    logger.info(
        "Video: %s  %.1f fps  %dx%d  start=%d  %d frames  step=%d  ocr_every_n=%d  mock_mode=%s",
        video_path.name, fps, frame_w, frame_h, start_frame, total_frames, frame_step, every_n, mock_mode,
    )

    if mock_mode:
        mock_pd = MockPlayerDetector()
        detector = mock_pd
        tracker = MockPlayerTracker()
        # MockPlayerTracker assigns track_id = detection index, which equals
        # MockPlayerDetector's synthetic_idx — so the truth table keys line up.
        ocr_sampler = TruthTableMockOCR(truth=mock_pd.truth_table())
        colour_classifier = None
        bib_ocr = None
    else:
        from src.bib_colour import BibColourClassifier
        from src.bib_ocr import BibNumberOCR
        from src.detection import PlayerDetector
        from src.tracking import PlayerTracker

        detector = PlayerDetector(config)
        tracker = PlayerTracker(config)
        colour_classifier = BibColourClassifier(config)
        bib_ocr = BibNumberOCR(config)
        ocr_sampler = None

    ball_detector = MockBallDetector()
    _ = MockCalibration(frame_w, frame_h)  # parity with ball-detection; not used here yet
    identity_resolver = IdentityResolver(config)

    # --- Pass 1: detect, track, sparsely OCR-sample ---
    tracked_per_frame: list[list[TrackedPlayer]] = []
    ocr_samples_total = 0
    ocr_samples_with_number = 0
    frame_idx = 0
    processed = 0

    with tqdm(total=total_frames, unit="fr", desc=f"{stem} detect") as pbar:
        while processed < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            detections = detector.detect(frame, frame_idx)
            tracked = tracker.update(detections, frame_idx)
            tracked_per_frame.append(tracked)
            _ = ball_detector.detect(frame, frame_idx)  # always None — kept for parity

            if frame_idx % every_n == 0:
                for tp in tracked:
                    if (tp.bbox[3] - tp.bbox[1]) < config.min_bib_crop_h:
                        continue
                    crop = _crop_bbox(frame, tp.bbox)
                    if crop.size == 0:
                        continue
                    reading = _sample_bib(
                        crop=crop,
                        track_id=tp.track_id,
                        frame_idx=frame_idx,
                        mock_sampler=ocr_sampler,
                        colour_classifier=colour_classifier,
                        ocr=bib_ocr,
                        debug_crops_dir=debug_crops_dir,
                    )
                    identity_resolver.add_reading(reading)
                    ocr_samples_total += 1
                    if reading.number is not None:
                        ocr_samples_with_number += 1

            frame_idx += 1
            processed += 1
            pbar.update(1)

    cap.release()

    # --- resolve identities (per-track voting + cross-track merging) ---
    identity_resolver.resolve()

    # --- persist raw tracks ---
    if persist_tracks:
        tracks_path = out_dir / f"{stem}_tracks.pkl"
        with open(tracks_path, "wb") as f:
            pickle.dump({"tracked_per_frame": tracked_per_frame, "fps": fps}, f)
        logger.info("Tracks saved to %s", tracks_path)

    # --- Pass 2: replay video, write annotated frames with resolved labels ---
    video_path_out: Path | None = None
    if write_video:
        video_path_out = out_dir / f"{stem}_annotated.mp4"
        _write_annotated_video(
            video_path=video_path,
            output_path=video_path_out,
            config=config,
            fps=fps / frame_step,
            frame_w=frame_w,
            frame_h=frame_h,
            tracked_per_frame=tracked_per_frame,
            identity_resolver=identity_resolver,
            frame_step=frame_step,
            start_frame=start_frame,
        )

    # --- stats ---
    n_frames = len(tracked_per_frame)
    identities = identity_resolver.resolve()
    canonical = [i for i in identities.values() if i.merged_into is None]
    resolved = [i for i in canonical if i.bib_id is not None]
    unresolved = [i for i in canonical if i.bib_id is None]
    mean_players = sum(len(t) for t in tracked_per_frame) / n_frames if n_frames else 0.0
    stats = {
        "total_frames": n_frames,
        "tracks_created": len(identities),
        "tracks_after_merge": len(canonical),
        "identities_resolved": len(resolved),
        "identities_unresolved": len(unresolved),
        "mean_players_per_frame": round(mean_players, 2),
        "ocr_samples_total": ocr_samples_total,
        "ocr_samples_with_number": ocr_samples_with_number,
    }

    # --- serialize JSON (schema-compatible with ball-detection) ---
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
            "ocr_engine": "mock" if mock_mode else config.ocr_engine,
        },
        "frames": [
            {
                "idx": i,
                "ts": round(i / fps, 4) if fps else 0.0,
                "ball": None,
                "players": [_serialize_player(tp, identity_resolver) for tp in tracked],
            }
            for i, tracked in enumerate(tracked_per_frame)
        ],
        "events": [],
        "stats": stats,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("JSON saved to %s", json_path)

    storage = MockStorage(out_dir / "mock_storage")
    storage.upload(json_path, f"{stem}.json")
    if video_path_out and video_path_out.exists():
        storage.upload(video_path_out, f"{stem}_annotated.mp4")

    return PipelineResult(json_path=json_path, video_path=video_path_out, stats=stats)


def _sample_bib(
    *,
    crop,
    track_id: int,
    frame_idx: int,
    mock_sampler: TruthTableMockOCR | None,
    colour_classifier,
    ocr,
    debug_crops_dir=None,
) -> OCRReading:
    """Build an OCRReading from either the mock sampler or the real
    colour classifier + OCR engine combo."""
    if mock_sampler is not None:
        return mock_sampler.read(crop, track_id=track_id, frame_idx=frame_idx)
    colour, _ = colour_classifier.classify(crop)
    debug_tag = f"fr{frame_idx:05d}_t{track_id}" if debug_crops_dir else ""
    number, ocr_conf = ocr.read(crop, debug_dir=debug_crops_dir, debug_tag=debug_tag)
    logger.debug("OCR track=%d frame=%d colour=%s number=%s conf=%.2f",
                 track_id, frame_idx, colour, number, ocr_conf)
    return OCRReading(
        track_id=track_id,
        frame_idx=frame_idx,
        colour=colour,
        number=number,
        ocr_confidence=ocr_conf,
    )


def _crop_bbox(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return frame[y1:y2, x1:x2]


def _label_for(track_id: int, identity_resolver: IdentityResolver) -> tuple[str, str]:
    ident = identity_resolver.lookup(track_id)
    if ident is None or ident.bib_id is None:
        return (f"?-{track_id}", "unknown")
    return (ident.bib_id, ident.colour)


def _serialize_player(tp: TrackedPlayer, identity_resolver: IdentityResolver) -> dict:
    label, team = _label_for(tp.track_id, identity_resolver)
    return {
        "id": label,
        "team": team,
        "track_id": tp.track_id,
        "bbox": [int(v) for v in tp.bbox],
        "conf": round(float(tp.confidence), 4),
    }


def _write_annotated_video(
    *,
    video_path: Path,
    output_path: Path,
    config: Config,
    fps: float,
    frame_w: int,
    frame_h: int,
    tracked_per_frame: list[list[TrackedPlayer]],
    identity_resolver: IdentityResolver,
    frame_step: int,
    start_frame: int = 0,
) -> None:
    """Replay the source video and write annotated frames with resolved labels."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot reopen video: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = setup_video_writer(output_path, fps, frame_w, frame_h)
    try:
        frame_idx = 0
        record_idx = 0
        with tqdm(total=len(tracked_per_frame), unit="fr", desc=f"{video_path.stem} annotate") as pbar:
            while record_idx < len(tracked_per_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue
                tracked = tracked_per_frame[record_idx]
                labels = {
                    tp.track_id: _label_for(tp.track_id, identity_resolver)
                    for tp in tracked
                }
                annotated = draw_frame(frame, tracked, labels, config)
                writer.write(annotated)
                record_idx += 1
                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        writer.release()
