"""CLI: extract frames from a video clip for manual labelling.

Usage:
    python scripts/extract_frames.py --input data/raw_clips/match_1080p.mp4 \
                                     --output data/frames/match_1080p \
                                     --strategy motion --count 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.frame_extractor import FrameExtractor
from src.pipeline import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from a video clip.")
    p.add_argument("--input", type=Path, required=True, help="Input video file.")
    p.add_argument("--output", type=Path, required=True, help="Output directory for JPEG frames.")
    p.add_argument("--strategy", choices=["uniform", "motion", "manual"], default="uniform",
                   help="Sampling strategy: uniform | motion | manual.")
    p.add_argument("--count", type=int, default=200, help="Number of frames (uniform/motion).")
    p.add_argument("--every-n", type=int, default=10, help="Save every Nth frame (manual).")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=10 if args.verbose else 20)

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    extractor = FrameExtractor(Config())
    saved = extractor.extract(
        args.input,
        args.output,
        strategy=args.strategy,
        count=args.count,
        every_n=args.every_n,
    )
    print(f"Saved {len(saved)} frames to {args.output}")


if __name__ == "__main__":
    main()
