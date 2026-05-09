"""CLI: evaluate bib identity resolution accuracy against a hand-labelled file.

Inputs:
- --predictions: pipeline JSON output (from scripts/run_pipeline.py).
- --ground-truth: per-frame list of {bbox, expected} labels in the format:
    {
      "frames": [
        {"idx": 100, "players": [{"bbox": [x1,y1,x2,y2], "expected": "R7"}, ...]}
      ]
    }
  Frame indices not present in ground-truth are skipped.

Reports:
- Per-bib precision/recall (R1..R12, B1..B12 — or whatever bib IDs appear in GT).
- Track-stability: % of consecutive ground-truth frames where the matched track
  keeps the same track_id.
- Aggregate identity-resolution rate: identities_resolved / total resolvable
  (pulled directly from prediction stats).

Match criterion: predicted bbox with IoU > IOU_MATCH against the GT bbox; the
predicted player's `id` is then compared with `expected`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

IOU_MATCH = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate bib identity resolution accuracy.")
    p.add_argument("--predictions", type=Path, required=True, help="Pipeline JSON output.")
    p.add_argument("--ground-truth", type=Path, required=True, help="Hand-labelled identity.json.")
    p.add_argument("--report", type=Path, default=None, help="Where to write the eval report JSON.")
    return p.parse_args()


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _match_predictions_to_gt(
    pred_players: list[dict],
    gt_players: list[dict],
) -> list[tuple[dict, dict]]:
    """Greedy IoU-based matching of predictions to ground truth.

    Returns pairs of (gt_player, matched_pred) for each GT — pred is None if no
    prediction reached the IoU threshold.
    """
    used = set()
    pairs = []
    for gt in gt_players:
        best_iou = 0.0
        best_idx = -1
        for i, p in enumerate(pred_players):
            if i in used:
                continue
            iou = _iou(tuple(gt["bbox"]), tuple(p["bbox"]))
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= IOU_MATCH:
            used.add(best_idx)
            pairs.append((gt, pred_players[best_idx]))
        else:
            pairs.append((gt, None))
    return pairs


def _per_bib_metrics(matches: list[tuple[dict, dict | None]]) -> dict[str, dict]:
    """Compute per-bib TP/FP/FN/precision/recall over a flat list of (gt, pred) pairs.

    A bib's:
    - TP: gt[expected] == pred[id]
    - FN: pred is None or pred[id] != expected
    - FP: pred[id] is the bib but no gt with that expected matched it
          (we approximate via gt mismatches; full FP requires unmatched preds)
    """
    tp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    bibs: set[str] = set()

    for gt, pred in matches:
        expected = gt["expected"]
        bibs.add(expected)
        if pred is None:
            fn[expected] += 1
            continue
        pred_id = pred.get("id", "")
        if pred_id == expected:
            tp[expected] += 1
        else:
            fn[expected] += 1
            if pred_id and not pred_id.startswith("?"):
                fp[pred_id] += 1
                bibs.add(pred_id)

    out: dict[str, dict] = {}
    for bib in sorted(bibs):
        t, f, n = tp[bib], fp[bib], fn[bib]
        precision = t / (t + f) if (t + f) else 0.0
        recall = t / (t + n) if (t + n) else 0.0
        out[bib] = {
            "tp": t, "fp": f, "fn": n,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }
    return out


def _track_stability(per_frame_pairs: dict[int, list[tuple[dict, dict | None]]]) -> dict:
    """For each ground-truth bib id, % of consecutive GT frames where the matched
    prediction keeps the same `track_id`."""
    history: dict[str, list[tuple[int, int]]] = defaultdict(list)  # bib -> [(frame_idx, track_id)]
    for frame_idx in sorted(per_frame_pairs.keys()):
        for gt, pred in per_frame_pairs[frame_idx]:
            if pred is None:
                continue
            history[gt["expected"]].append((frame_idx, pred.get("track_id", -1)))

    out: dict[str, float] = {}
    for bib, seq in history.items():
        if len(seq) < 2:
            out[bib] = 1.0
            continue
        same = sum(1 for i in range(1, len(seq)) if seq[i][1] == seq[i - 1][1])
        out[bib] = round(same / (len(seq) - 1), 3)
    return out


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        print(f"Predictions not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)
    if not args.ground_truth.exists():
        print(f"Ground truth not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    preds = json.loads(args.predictions.read_text())
    gt = json.loads(args.ground_truth.read_text())

    pred_by_idx = {f["idx"]: f["players"] for f in preds.get("frames", [])}
    per_frame_pairs: dict[int, list[tuple[dict, dict | None]]] = {}
    all_pairs: list[tuple[dict, dict | None]] = []

    for frame in gt.get("frames", []):
        idx = frame["idx"]
        if idx not in pred_by_idx:
            continue
        pairs = _match_predictions_to_gt(pred_by_idx[idx], frame["players"])
        per_frame_pairs[idx] = pairs
        all_pairs.extend(pairs)

    per_bib = _per_bib_metrics(all_pairs)
    stability = _track_stability(per_frame_pairs)
    pred_stats = preds.get("stats", {})
    resolved = pred_stats.get("identities_resolved", 0)
    unresolved = pred_stats.get("identities_unresolved", 0)
    aggregate_rate = resolved / (resolved + unresolved) if (resolved + unresolved) else 0.0

    report = {
        "matched_frames": len(per_frame_pairs),
        "matched_pairs": len(all_pairs),
        "iou_threshold": IOU_MATCH,
        "per_bib": per_bib,
        "track_stability": stability,
        "aggregate_identity_resolution_rate": round(aggregate_rate, 3),
        "from_pipeline_stats": {
            "tracks_created": pred_stats.get("tracks_created"),
            "tracks_after_merge": pred_stats.get("tracks_after_merge"),
            "identities_resolved": resolved,
            "identities_unresolved": unresolved,
            "mean_players_per_frame": pred_stats.get("mean_players_per_frame"),
        },
    }

    print(json.dumps(report, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
