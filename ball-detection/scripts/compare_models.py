"""Compare multiple YOLO models on detection rate, mAP, and inference speed.

Runs every model against the same set of frames and prints a side-by-side table.
If YOLO-format labels are provided, also computes mAP@50 and zone/blur recall.

Usage — detection rate only (no labels needed):
    python scripts/compare_models.py \\
        --input data/raw_clips/match1.mp4 --max-frames 300 \\
        --models models/rf-football-ball.pt models/soccer-ball-finding.pt \\
        --ball-class-ids 0 0

Usage — full eval against your labelled frames:
    python scripts/compare_models.py \\
        --frames data/frames/match1/ \\
        --labels data/annotations/ \\
        --models models/rf-football-ball.pt models/soccer-ball-finding.pt \\
        --ball-class-ids 0 0

Labels must be YOLO-format .txt files (class cx cy w h, normalised) with the
same stem as the corresponding .jpg in --frames.
Export from Roboflow in "YOLOv8" format and drop the .txt files into data/annotations/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.detection import BallDetector
from src.pipeline import setup_logging

_IOU_THRESH = 0.50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side model comparison.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path,
                     help="Video file. Frames extracted on the fly (use with --max-frames).")
    src.add_argument("--frames", type=Path,
                     help="Directory of .jpg frames already on disk.")

    p.add_argument("--labels", type=Path, default=None,
                   help="Directory of YOLO-format .txt label files (same stem as frames).")
    p.add_argument("--max-frames", type=int, default=300,
                   help="Max frames to evaluate from video (ignored when --frames is used).")
    p.add_argument("--models", nargs="+", required=True, type=Path,
                   help="Model .pt paths to compare.")
    p.add_argument("--ball-class-ids", nargs="+", type=int, default=None,
                   help="Ball class ID per model (default: 0 for all). Must match --models length.")
    p.add_argument("--report", type=Path, default=Path("data/outputs/comparison_report.json"),
                   help="Path to save JSON report.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# frame loading
# ---------------------------------------------------------------------------

def load_frames_from_video(video_path: Path, max_frames: int) -> list[tuple[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    indices = np.linspace(0, total - 1, min(max_frames, total)).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append((f"frame_{idx:06d}", frame))
    cap.release()
    return frames


def load_frames_from_dir(frames_dir: Path) -> list[tuple[str, np.ndarray]]:
    paths = sorted(frames_dir.glob("*.jpg"))
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append((p.stem, img))
    return frames


# ---------------------------------------------------------------------------
# label loading
# ---------------------------------------------------------------------------

def load_labels(labels_dir: Path, stem: str, frame_w: int, frame_h: int) -> list[np.ndarray]:
    label_path = labels_dir / f"{stem}.txt"
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(x) for x in parts[:5])
            x1 = (cx - w / 2) * frame_w
            y1 = (cy - h / 2) * frame_h
            x2 = (cx + w / 2) * frame_w
            y2 = (cy + h / 2) * frame_h
            boxes.append(np.array([x1, y1, x2, y2]))
    return boxes


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _zone(cx: float, frame_w: int) -> str:
    if cx < frame_w * 0.167 or cx > frame_w * 0.833:
        return "corners"
    return "near_half" if cx < frame_w / 2 else "far_half"


def _blur_bucket(img: np.ndarray, box: np.ndarray) -> str:
    crop = img[max(0,int(box[1])):int(box[3]), max(0,int(box[0])):int(box[2])]
    if crop.size < 75:
        return "blurry"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return "sharp" if var > 100 else ("medium" if var > 20 else "blurry")


def evaluate_model(
    detector: BallDetector,
    frames: list[tuple[str, np.ndarray]],
    labels_dir: Path | None,
) -> dict:
    tp = fp = fn = detected_frames = 0
    total_ms = 0.0
    zone_tp = {"near_half": 0, "far_half": 0, "corners": 0}
    zone_fn = {"near_half": 0, "far_half": 0, "corners": 0}
    blur_tp = {"sharp": 0, "medium": 0, "blurry": 0}
    blur_fn = {"sharp": 0, "medium": 0, "blurry": 0}

    for stem, frame in frames:
        fh, fw = frame.shape[:2]

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        total_ms += (time.perf_counter() - t0) * 1000

        if dets:
            detected_frames += 1

        if labels_dir is None:
            continue

        gt_boxes = load_labels(labels_dir, stem, fw, fh)
        pred_boxes = [np.array(d.bbox) for d in dets]
        matched_pred: set[int] = set()

        for gt in gt_boxes:
            best_iou, best_pi = 0.0, -1
            for pi, pred in enumerate(pred_boxes):
                iou = _iou(gt, pred)
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            cx = (gt[0] + gt[2]) / 2
            zone = _zone(cx, fw)
            bucket = _blur_bucket(frame, gt)
            if best_iou >= _IOU_THRESH and best_pi not in matched_pred:
                tp += 1
                matched_pred.add(best_pi)
                zone_tp[zone] += 1
                blur_tp[bucket] += 1
            else:
                fn += 1
                zone_fn[zone] += 1
                blur_fn[bucket] += 1

        fp += sum(1 for pi in range(len(pred_boxes)) if pi not in matched_pred)

    n = len(frames)
    ms_per_frame = total_ms / n if n else 0.0
    detection_rate = detected_frames / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    def _zr(z: str) -> float | None:
        t = zone_tp[z] + zone_fn[z]
        return zone_tp[z] / t if t else None

    def _br(b: str) -> float | None:
        t = blur_tp[b] + blur_fn[b]
        return blur_tp[b] / t if t else None

    return {
        "frames_evaluated": n,
        "detection_rate": detection_rate,
        "ms_per_frame": round(ms_per_frame, 1),
        "mAP50_recall": recall,
        "precision": precision,
        "tp": tp, "fp": fp, "fn": fn,
        "zone_recall": {z: _zr(z) for z in zone_tp},
        "blur_recall": {b: _br(b) for b in blur_tp},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging()

    # Resolve ball class IDs — default 0 for all models
    class_ids = args.ball_class_ids or [0] * len(args.models)
    if len(class_ids) != len(args.models):
        print("ERROR: --ball-class-ids must have the same length as --models")
        sys.exit(1)

    # Load frames
    if args.input:
        print(f"Extracting up to {args.max_frames} frames from {args.input.name} ...")
        frames = load_frames_from_video(args.input, args.max_frames)
    else:
        print(f"Loading frames from {args.frames} ...")
        frames = load_frames_from_dir(args.frames)

    if not frames:
        print("ERROR: no frames loaded")
        sys.exit(1)

    has_labels = args.labels is not None and args.labels.exists()
    print(f"Loaded {len(frames)} frames  |  labels: {'yes' if has_labels else 'no (detection rate only)'}\n")

    all_results: dict[str, dict] = {}

    for model_path, class_id in zip(args.models, class_ids):
        name = model_path.stem
        print(f"Running {name}  (ball_class_id={class_id}) ...")
        config = Config(model_path=model_path, ball_class_id=class_id)
        try:
            detector = BallDetector(config)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        metrics = evaluate_model(detector, frames, args.labels if has_labels else None)
        all_results[name] = metrics

        det_rate = metrics["detection_rate"]
        ms = metrics["ms_per_frame"]
        recall = metrics["mAP50_recall"]
        print(f"  detection_rate={det_rate:.1%}  ms/frame={ms:.1f}  "
              f"recall@50={f'{recall:.1%}' if recall is not None else 'n/a (no labels)'}")

    # --- pretty table ---
    print(f"\n{'─' * 72}")
    print(f"{'Model':<28} {'DetRate':>8} {'ms/fr':>7} {'Recall@50':>10} {'Prec':>7} {'Near':>7} {'Far':>7} {'Corners':>8}")
    print(f"{'─' * 72}")
    for name, m in all_results.items():
        def _fmt(v: float | None) -> str:
            return f"{v:.1%}" if v is not None else "   n/a"
        print(
            f"{name:<28} "
            f"{_fmt(m['detection_rate']):>8} "
            f"{m['ms_per_frame']:>6.1f}ms "
            f"{_fmt(m['mAP50_recall']):>10} "
            f"{_fmt(m['precision']):>7} "
            f"{_fmt(m['zone_recall']['near_half']):>7} "
            f"{_fmt(m['zone_recall']['far_half']):>7} "
            f"{_fmt(m['zone_recall']['corners']):>8}"
        )
    print(f"{'─' * 72}")

    # Save report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull report saved to {args.report}")


if __name__ == "__main__":
    main()
