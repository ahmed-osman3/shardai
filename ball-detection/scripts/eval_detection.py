"""CLI: evaluate ball detection on a labelled frame set.

Usage:
    python scripts/eval_detection.py \\
        --model models/ball_v1.pt \\
        --labels data/annotations/ \\
        --frames data/frames/ \\
        --report data/outputs/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.detection import BallDetector
from src.pipeline import setup_logging

logger = logging.getLogger(__name__)

_IOU_50 = 0.50
_BLUR_SHARP = 100
_BLUR_MEDIUM = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ball detection against a labelled set.")
    p.add_argument("--model", type=Path, required=True, help="Model weights (.pt).")
    p.add_argument("--labels", type=Path, required=True, help="Directory of YOLO-format .txt label files.")
    p.add_argument("--frames", type=Path, required=True, help="Directory of JPEG frame images.")
    p.add_argument("--report", type=Path, default=Path("data/outputs/eval_report.json"),
                   help="Output JSON report path.")
    p.add_argument("--fp-crops", type=Path, default=Path("data/outputs/fp_crops/"),
                   help="Directory to save FP crops.")
    return p.parse_args()


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IOU between two xyxy boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _laplacian_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _blur_bucket(var: float) -> str:
    if var > _BLUR_SHARP:
        return "sharp"
    if var > _BLUR_MEDIUM:
        return "medium"
    return "blurry"


def _zone(cx_px: float, cy_px: float, frame_w: int) -> str:
    if cx_px < 320 or cx_px > frame_w - 320:
        return "corners"
    if cx_px < frame_w / 2:
        return "near_half"
    return "far_half"


def main() -> None:
    args = parse_args()
    setup_logging()

    args.fp_crops.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    config = Config(model_path=args.model)
    detector = BallDetector(config)

    label_files = sorted(args.labels.glob("*.txt"))
    if not label_files:
        logger.error("No .txt label files found in %s", args.labels)
        sys.exit(1)

    tp = fp = fn = 0
    zone_tp: dict[str, int] = {"near_half": 0, "far_half": 0, "corners": 0}
    zone_fn: dict[str, int] = {"near_half": 0, "far_half": 0, "corners": 0}
    blur_tp: dict[str, int] = {"sharp": 0, "medium": 0, "blurry": 0}
    blur_fn: dict[str, int] = {"sharp": 0, "medium": 0, "blurry": 0}
    fp_crops: list[tuple[float, np.ndarray]] = []  # (confidence, crop)

    for label_path in label_files:
        frame_path = args.frames / label_path.with_suffix(".jpg").name
        if not frame_path.exists():
            logger.warning("Frame not found for label %s, skipping", label_path.name)
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        fh, fw = frame.shape[:2]

        # Parse YOLO label: class cx cy w h (normalised)
        gt_boxes: list[np.ndarray] = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = (float(x) for x in parts[:5])
                x1 = (cx - w / 2) * fw
                y1 = (cy - h / 2) * fh
                x2 = (cx + w / 2) * fw
                y2 = (cy + h / 2) * fh
                gt_boxes.append(np.array([x1, y1, x2, y2]))

        detections = detector.detect(frame)
        pred_boxes = [np.array(d.bbox) for d in detections]

        matched_gt = set()
        matched_pred = set()

        for gi, gt in enumerate(gt_boxes):
            best_iou = 0.0
            best_pi = -1
            for pi, pred in enumerate(pred_boxes):
                iou = _box_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            cx_px = (gt[0] + gt[2]) / 2
            cy_px = (gt[1] + gt[3]) / 2
            zone = _zone(cx_px, cy_px, fw)

            crop = frame[max(0, int(gt[1])):int(gt[3]), max(0, int(gt[0])):int(gt[2])]
            if crop.size >= 5 * 5 * 3:
                bvar = _laplacian_var(crop)
                bucket = _blur_bucket(bvar)
            else:
                bucket = "blurry"

            if best_iou >= _IOU_50 and best_pi not in matched_pred:
                tp += 1
                matched_gt.add(gi)
                matched_pred.add(best_pi)
                zone_tp[zone] += 1
                blur_tp[bucket] += 1
            else:
                fn += 1
                zone_fn[zone] += 1
                blur_fn[bucket] += 1

        for pi, (pred, det) in enumerate(zip(pred_boxes, detections)):
            if pi not in matched_pred:
                fp += 1
                cx, cy = int((pred[0] + pred[2]) / 2), int((pred[1] + pred[3]) / 2)
                r = 64
                crop = frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
                if crop.size > 0:
                    fp_crops.append((det.confidence, crop))

    # Save top-20 FP crops
    fp_crops.sort(key=lambda x: -x[0])
    for i, (_, crop) in enumerate(fp_crops[:20]):
        crop_resized = cv2.resize(crop, (128, 128)) if crop.size > 0 else crop
        cv2.imwrite(str(args.fp_crops / f"fp_{i:02d}.jpg"), crop_resized)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def _zone_recall(z: str) -> float:
        total = zone_tp[z] + zone_fn[z]
        return zone_tp[z] / total if total else 0.0

    def _blur_recall(b: str) -> float:
        total = blur_tp[b] + blur_fn[b]
        return blur_tp[b] / total if total else 0.0

    report = {
        "total_gt": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall_mAP50": round(recall, 4),
        "f1": round(f1, 4),
        "zone_recall": {z: round(_zone_recall(z), 4) for z in zone_tp},
        "blur_recall": {b: round(_blur_recall(b), 4) for b in blur_tp},
        "fp_crops_dir": str(args.fp_crops),
    }

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEval report: {args.report}")
    print(f"  Precision: {precision:.1%}  Recall: {recall:.1%}  F1: {f1:.1%}")
    print(f"  Zone recall: near={_zone_recall('near_half'):.1%}  "
          f"far={_zone_recall('far_half'):.1%}  corners={_zone_recall('corners'):.1%}")
    print(f"  Blur recall: sharp={_blur_recall('sharp'):.1%}  "
          f"medium={_blur_recall('medium'):.1%}  blurry={_blur_recall('blurry'):.1%}")


if __name__ == "__main__":
    main()
