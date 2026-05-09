"""CLI: extract frames from a video clip for labelling.

Usage:
    python scripts/extract_frames.py \\
        --input data/raw_clips/match1.mp4 \\
        --output data/frames/match1/ \\
        --strategy motion \\
        --count 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.frame_extractor import FrameExtractor
from src.pipeline import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample frames from a video for labelling.")
    p.add_argument("--input", type=Path, required=True, help="Source video file.")
    p.add_argument("--output", type=Path, required=True, help="Output directory for JPEG frames.")
    p.add_argument(
        "--strategy",
        choices=["uniform", "motion", "manual"],
        default="motion",
        help="Sampling strategy (default: motion).",
    )
    p.add_argument("--count", type=int, default=200, help="Number of frames to extract.")
    p.add_argument("--every-n", type=int, default=10, help="Extract every Nth frame (manual strategy only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    extractor = FrameExtractor(Config())
    saved = extractor.extract(
        video_path=args.input,
        output_dir=args.output,
        strategy=args.strategy,
        count=args.count,
        every_n=args.every_n,
    )
    print(f"Saved {len(saved)} frames to {args.output}")


if __name__ == "__main__":
    main()
