"""Download test clips for the ball detection pipeline.

Requires yt-dlp and ffmpeg:
    brew install yt-dlp ffmpeg      # macOS
    apt install yt-dlp ffmpeg       # Ubuntu/Debian

Usage:
    python scripts/setup_clips.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CLIPS = [
    {
        "name": "match_1080p.mp4",
        "url": "https://www.youtube.com/watch?v=nGcysU7xH60",
        "section": "8:00-11:00",   # 3-min clip of active 7-aside play
        "format": "299+140",       # 1080p50 video + m4a audio
        "description": "7-aside match, 1080p50, minutes 8-11",
    },
]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw_clips"


def check_dep(cmd: str) -> bool:
    return subprocess.run(["which", cmd], capture_output=True).returncode == 0


def download(clip: dict) -> None:
    dest = OUTPUT_DIR / clip["name"]
    if dest.exists():
        print(f"  Already exists: {dest.name} — skipping")
        return

    print(f"  Downloading {clip['name']} ({clip['description']}) ...")
    cmd = [
        "yt-dlp",
        "-f", clip["format"],
        "--download-sections", f"*{clip['section']}",
        "--force-keyframes-at-cuts",
        "-o", str(dest),
        clip["url"],
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: download failed for {clip['name']}")
    else:
        size_mb = dest.stat().st_size / 1_000_000
        print(f"  Saved: {dest}  ({size_mb:.0f} MB)")


def main() -> None:
    for dep in ("yt-dlp", "ffmpeg"):
        if not check_dep(dep):
            print(f"ERROR: {dep} not installed.")
            print("  macOS:  brew install yt-dlp ffmpeg")
            print("  Linux:  apt install yt-dlp ffmpeg")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(CLIPS)} clip(s) to {OUTPUT_DIR}\n")
    for clip in CLIPS:
        download(clip)

    print("\nDone. Run the pipeline with:")
    for clip in CLIPS:
        print(f"  python scripts/run_pipeline.py --input data/raw_clips/{clip['name']} --ball-class-id 32")


if __name__ == "__main__":
    main()
