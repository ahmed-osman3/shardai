"""CLI: run the ball detection pipeline on a video clip.

Usage:
    python scripts/run_pipeline.py --input data/raw_clips/match1.mp4
    python scripts/run_pipeline.py --input data/raw_clips/match1.mp4 --device mps
    python scripts/run_pipeline.py --input data/raw_clips/match1.mp4 --max-frames 300 --no-video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.pipeline import run_pipeline, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ball detection pipeline on a video clip.")
    p.add_argument("--input", type=Path, required=True, help="Input .mp4 clip path.")
    p.add_argument("--output", type=Path, default=None, help="Output directory (default: data/outputs/).")
    p.add_argument("--model", type=Path, default=None, help="Model weights path (default: models/yolo11m.pt).")
    p.add_argument("--device", type=str, default="auto", help="Device: auto | cpu | mps | cuda.")
    p.add_argument("--ball-class-id", type=int, default=None,
                   help="Ball class ID in the model (default: 32 for COCO, 0 for fine-tuned).")
    p.add_argument("--no-tiling", action="store_true",
                   help="Whole-frame inference instead of tiled (faster on 1080p, ~10x speedup).")
    p.add_argument("--fps", type=int, default=None, help="Subsample to this FPS (default: source fps).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Process only the first N frames (for fast iteration).")
    p.add_argument("--no-video", action="store_true",
                   help="Skip annotated video output (JSON only).")
    p.add_argument("--persist-tracks", action="store_true",
                   help="Save raw ball/player tracks before event detection.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=10 if args.verbose else 20)

    config = Config(
        model_path=args.model,
        device=args.device,
        target_fps=args.fps,
        tiling=not args.no_tiling,
        **({"ball_class_id": args.ball_class_id} if args.ball_class_id is not None else {}),
    )

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    result = run_pipeline(
        args.input,
        config,
        outputs_dir=args.output,
        max_frames=args.max_frames,
        write_video=not args.no_video,
        persist_tracks=args.persist_tracks,
    )

    print(f"\nDone.")
    print(f"  JSON:  {result.json_path}")
    if result.video_path:
        print(f"  Video: {result.video_path}")
    print(f"  Stats:")
    for k, v in result.stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.1%}")
        else:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
