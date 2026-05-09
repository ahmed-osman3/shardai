"""Download football ball detection datasets from Roboflow and train YOLO models.

Downloads 3 curated datasets and trains a YOLO11m model from each.
Run this on a machine with a CUDA GPU — training takes ~15-30 min per model.
On CPU/MPS it will work but is much slower (~2-4 hrs per model).

Usage:
    pip install -e ".[train]"
    python scripts/download_and_train.py --api-key YOUR_ROBOFLOW_API_KEY

Get your free API key at: https://app.roboflow.com/ → Settings → API Keys

Outputs (one per dataset):
    models/{slug}/weights/best.pt   ← drop this into models/ to use in the pipeline
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Datasets to download and train. Each entry:
#   slug        : used as output folder name
#   workspace   : Roboflow workspace
#   project     : Roboflow project name
#   version     : dataset version number
#   ball_class  : class index for ball in this dataset (0 for single-class; varies for multi-class)
#   notes       : why this dataset is interesting
DATASETS = [
    {
        "slug": "rf-football-ball",
        "workspace": "roboflow-jvuqo",
        "project": "football-ball-detection-rejhg",
        "version": 2,
        "ball_class_id": 0,
        "notes": "Roboflow official — 1237 images, 92.5% mAP, Bundesliga broadcast, ball only",
    },
    {
        "slug": "soccer-ball-finding",
        "workspace": "school-5lmma",
        "project": "soccer-ball-finding",
        "version": 1,
        "ball_class_id": 0,
        "notes": "Largest single-class ball dataset — 1914 images, broadcast footage",
    },
    {
        "slug": "rf-football-players",
        "workspace": "roboflow-jvuqo",
        "project": "football-players-detection-3zvbc",
        "version": 19,
        "ball_class_id": 0,  # ball is class 0 in this dataset
        "notes": "Roboflow official — ball + players, 84% mAP, YOLO11 trained",
    },
]

YOLO_MODEL = "yolo11m.pt"   # base weights to fine-tune from
EPOCHS = 50
IMG_SIZE = 640


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Roboflow datasets and train YOLO models.")
    p.add_argument("--api-key", required=True, help="Roboflow API key (free at app.roboflow.com).")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Slugs to train (default: all). E.g. --datasets rf-football-ball")
    p.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default {EPOCHS}).")
    p.add_argument("--model", default=YOLO_MODEL, help="Base YOLO weights to fine-tune from.")
    p.add_argument("--device", default="auto", help="Training device: auto | cpu | mps | cuda:0")
    p.add_argument("--output-dir", type=Path, default=Path("models"),
                   help="Where to save trained weights (default: models/).")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "0"   # CUDA device index for ultralytics
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def download_dataset(ds: dict, api_key: str, dest_dir: Path) -> Path:
    """Download dataset in YOLOv8 format, return path to data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install -e '.[train]'")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ds["workspace"]).project(ds["project"])
    version = project.version(ds["version"])
    dataset = version.download("yolov8", location=str(dest_dir))
    return Path(dataset.location) / "data.yaml"


def train_model(data_yaml: Path, slug: str, args: argparse.Namespace, device: str) -> Path:
    """Run yolo train and return path to best.pt."""
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=IMG_SIZE,
        device=device,
        project=str(args.output_dir / slug),
        name="train",
        exist_ok=True,
        verbose=False,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    return best_pt


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = [d for d in DATASETS if args.datasets is None or d["slug"] in args.datasets]
    if not targets:
        print(f"No matching datasets. Available slugs: {[d['slug'] for d in DATASETS]}")
        sys.exit(1)

    print(f"\nTraining {len(targets)} model(s) on device={device}\n")

    results_summary = []
    for ds in targets:
        print(f"{'=' * 60}")
        print(f"  Dataset : {ds['slug']}")
        print(f"  Notes   : {ds['notes']}")
        print(f"{'=' * 60}")

        data_dir = args.output_dir / ds["slug"] / "dataset"
        print(f"  Downloading dataset → {data_dir} ...")
        try:
            data_yaml = download_dataset(ds, args.api_key, data_dir)
        except Exception as e:
            print(f"  ERROR downloading {ds['slug']}: {e}")
            continue

        print(f"  Training for {args.epochs} epochs ...")
        try:
            best_pt = train_model(data_yaml, ds["slug"], args, device)
        except Exception as e:
            print(f"  ERROR training {ds['slug']}: {e}")
            continue

        # Copy best.pt to a flat, easy-to-reference path
        final_pt = args.output_dir / f"{ds['slug']}.pt"
        shutil.copy2(best_pt, final_pt)
        print(f"  ✓ Saved: {final_pt}  (ball_class_id={ds['ball_class_id']})")
        results_summary.append((ds["slug"], final_pt, ds["ball_class_id"]))

    print(f"\n{'=' * 60}")
    print("Done. To compare models run:\n")
    model_args = " ".join(f"--models {p} --ball-class-ids {c}" for _, p, c in results_summary)
    print(f"  python scripts/compare_models.py "
          f"--frames data/frames/match1/ {model_args}")
    print(f"\nOr to run the full pipeline with a specific model:")
    for slug, pt, cls_id in results_summary:
        print(f"  python scripts/run_pipeline.py --input data/raw_clips/match1.mp4 "
              f"--model {pt} --ball-class-id {cls_id}")


if __name__ == "__main__":
    main()
