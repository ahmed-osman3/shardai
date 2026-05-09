"""CLI: run the player + bib detection pipeline on a video clip.

Usage:
    python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4
    python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4 --device mps
    python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4 --max-frames 300 --no-video
    python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4 --mock  # Phase A scaffold demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.pipeline import run_pipeline, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run player + bib detection pipeline on a video clip.")
    p.add_argument("--input", type=Path, required=True, help="Input .mp4 clip path.")
    p.add_argument("--output", type=Path, default=None, help="Output directory (default: data/outputs/).")
    p.add_argument("--model", type=Path, default=None, help="Model weights path (default: models/yolo11m.pt).")
    p.add_argument("--device", type=str, default="auto", help="Device: auto | cpu | mps | cuda.")
    p.add_argument("--player-class-id", type=int, default=None,
                   help="Player class ID in the model (default: 0 for COCO person).")
    p.add_argument("--tiling", action="store_true",
                   help="Tiled inference (slower but tighter at the far end of the pitch).")
    p.add_argument("--fps", type=int, default=None, help="Subsample to this FPS (default: source fps).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Process only the first N frames (for fast iteration).")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Skip the first N frames of the source video (e.g. to skip intro footage).")
    p.add_argument("--no-video", action="store_true",
                   help="Skip annotated video output (JSON only).")
    p.add_argument("--persist-tracks", action="store_true",
                   help="Save raw per-frame tracks before identity resolution.")
    p.add_argument("--ocr-every-n", type=int, default=None,
                   help="Run OCR on 1 in N frames per track (default: config.ocr_every_n = 5).")
    p.add_argument("--mock", action="store_true",
                   help="Use mock CV components — runs the pipeline end-to-end without weights/OCR.")
    p.add_argument("--debug-ocr", action="store_true",
                   help="Save preprocessed OCR crops to data/outputs/debug_crops/ for inspection.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=10 if args.verbose else 20)

    config = Config(
        model_path=args.model,
        device=args.device,
        target_fps=args.fps,
        tiling=args.tiling,
        **({"player_class_id": args.player_class_id} if args.player_class_id is not None else {}),
    )

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    result = run_pipeline(
        args.input,
        config,
        outputs_dir=args.output,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        write_video=not args.no_video,
        persist_tracks=args.persist_tracks,
        ocr_every_n=args.ocr_every_n,
        mock_mode=args.mock,
        debug_ocr=args.debug_ocr,
    )

    print(f"\nDone.")
    print(f"  JSON:  {result.json_path}")
    if result.video_path:
        print(f"  Video: {result.video_path}")
    print(f"  Stats:")
    for k, v in result.stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
