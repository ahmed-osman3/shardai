# Player + Bib Detection — Football Match Analysis MVP

Prototype player detection + bib identity pipeline for UK 5/7-a-side football. Sibling module to `ball-detection/` — the second isolated CV layer of a larger cloud analysis system. Given a clip of a match where every player wears a numbered bib, this pipeline returns frame-by-frame player boxes labelled with stable identities like `R7` / `B11`, even on frames where bib OCR didn't fire.

The trick is to **track first, OCR sparsely, vote across the track, then merge tracks that share a bib** — see [Identity resolution](#how-identity-resolution-works) below.

---

## Setup

**Requirements:** Python 3.11+, macOS (Apple Silicon) or Linux with CUDA GPU.

```bash
git clone https://github.com/ahmed-osman3/shardai
cd shardai/player-detection
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Tesseract is a system dependency for bib OCR:
```bash
brew install tesseract        # macOS
# sudo apt install tesseract-ocr   # Ubuntu/Debian
```

Pretrained `yolo11m.pt` auto-downloads on first use. Drop your own footage into `data/raw_clips/`. Video files are gitignored.

---

## Quickstart

```bash
# Smoke-test the orchestration without weights / OCR / video
pytest tests/

# Run on a real clip
python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4

# Outputs:
#   data/outputs/match_1080p.json            ← per-frame player boxes + resolved bib IDs
#   data/outputs/match_1080p_annotated.mp4   ← video with R7/B11 labels drawn on each player
```

Useful flags:
- `--start-frame N` — skip the first N frames (e.g. intro footage)
- `--max-frames N` — stop after N frames (fast iteration)
- `--no-video` — JSON only, skip video render
- `--ocr-every-n N` — sample OCR on 1 in N frames per track (default: `config.ocr_every_n = 5`)
- `--device {auto|cpu|mps|cuda}` — force device
- `--mock` — use mock CV components (orchestration sanity check; no weights/OCR needed)
- `--persist-tracks` — pickle raw track state so you can iterate identity logic without re-running inference
- `--debug-ocr` — save preprocessed OCR crops to `data/outputs/debug_crops/` for inspection
- `--tiling` — tiled inference (slower but tighter at the far end of the pitch; off by default)
- `--verbose` — enable debug logging

---

## Key config defaults (`config.py`)

| Setting | Default | Notes |
|---|---|---|
| `confidence_threshold` | `0.25` | Lower to catch far-end players |
| `player_class_id` | `0` | COCO person class |
| `max_bbox_area_pct` | `0.35` | Drop detections covering >35% of frame (spectators, talking heads) |
| `tiling` | `False` | Players are large enough for whole-frame inference |
| `lost_track_buffer` | `90` | Frames a track survives without a detection (≈3 s @ 30 fps) |
| `ocr_every_n` | `5` | OCR 1 in 5 frames per track |
| `min_bib_crop_h` | `60` | Skip OCR on crops shorter than this (far-end players) |
| `min_votes_for_identity` | `2` | Minimum matching OCR reads to resolve a bib |
| `bib_roi_y` | `(0.15, 0.55)` | Upper-torso ROI as fraction of bbox height |
| `hsv_min_s` | `80` | Saturation floor for colourful pixel counting |
| `device` | `"auto"` | Auto-selects MPS → CUDA → CPU |

To adjust bib colour ranges (e.g. pink/green instead of red/blue), set `hsv_min_s`, `hsv_min_v`, `hsv_white_max_s`, and `hsv_white_min_v` in `config.py`.

---

## How identity resolution works

OCR on a bib in the middle of a match fails on most frames: motion blur, players turned sideways, occlusion, far end of pitch. The fix is to track first, OCR sparsely, and vote across each track:

1. **Per frame** — `PlayerDetector.detect()` returns bounding boxes for every player.
2. **Per frame** — `PlayerTracker.update()` (ByteTrack) assigns stable `track_id`s, surviving brief occlusions and crossings.
3. **Sparse OCR** — every `--ocr-every-n` frames per track, crop the player box, run a HSV colour classifier on the upper-torso ROI, and run Tesseract on the same ROI for the digit. Skip crops shorter than `min_bib_crop_h` to avoid wasted OCR on far-end players.
4. **End-of-video** — `IdentityResolver.resolve()`:
   - Per track: take the most common `(colour, number)` across all readings; require ≥ `min_votes_for_identity` supporting reads.
   - Across tracks: when multiple `track_id`s resolve to the same `(red, 7)`, collapse them — keep the highest-vote-count track as canonical, link the others via `merged_into`.
5. **Backfill** — every frame's player list gets the resolved `bib_id` stamped via `IdentityResolver.lookup(track_id)`, even on frames where OCR never fired.

The output is a JSON file where every player on every frame is labelled `R{1..12}` or `B{1..12}` (or `?-{track_id}` when unresolved).

---

## Output schema (matches ball-detection for downstream merging)

```json
{
  "schema_version": "1.0",
  "meta": { "fps": 50.0, "frame_count": 9000, "frame_w": 1920, "frame_h": 1080,
            "model": "yolo11m.pt", "ocr_engine": "tesseract" },
  "frames": [
    { "idx": 0, "ts": 0.0, "ball": null,
      "players": [
        {"id": "R7", "team": "red", "track_id": 3, "bbox": [940,500,980,580], "conf": 0.85},
        {"id": "B3", "team": "blue", "track_id": 7, "bbox": [...], "conf": 0.81}
      ]
    }
  ],
  "events": [],
  "stats": { "total_frames": 9000, "tracks_after_merge": 14, "identities_resolved": 12, ... }
}
```

`ball` is always `null` — that field is owned by `ball-detection/`. Downstream code can union frames from both modules into a single per-frame view.

---

## Notebooks

```bash
jupyter notebook
```

| Notebook | Purpose |
|---|---|
| `01_baseline_player_detection.ipynb` | Run pretrained YOLO11 on a clip; per-frame detection counts, far-end recall, bbox area distribution. |
| `02_tracking_stability.ipynb` | Load persisted tracks; plot lifespans, switches/minute, track duration distribution. |
| `03_bib_identity_resolution.ipynb` | Sweep `ocr_every_n`; plot resolution rate vs OCR sample budget. |

---

## Run evaluation

```bash
python scripts/eval_identity.py \
    --predictions data/outputs/match_1080p.json \
    --ground-truth data/annotations/identity.json \
    --report data/outputs/eval_report.json
```

Reports per-bib precision/recall, track-stability per bib, and the aggregate identity-resolution rate (the ≥80% MVP target).

The ground-truth file is hand-labelled with shape:
```json
{"frames": [{"idx": 100, "players": [{"bbox": [x1,y1,x2,y2], "expected": "R7"}, ...]}]}
```

---

## Extract frames for labelling

```bash
python scripts/extract_frames.py \
    --input data/raw_clips/match_1080p.mp4 \
    --output data/frames/match_1080p/ \
    --strategy motion --count 200
```

Strategies: `uniform`, `motion` (best when bib region needs to be sharp), `manual` (every Nth).

---

## What is real vs mock

| Module | Status | Notes |
|---|---|---|
| `src/detection.py` | **Real** | YOLO11 person-class with whole-frame inference (tiling opt-in); max-area filter to reject full-frame noise |
| `src/tracking.py` | **Real** | `supervision.ByteTrack` wrapper |
| `src/bib_colour.py` | **Real** | HSV mask on upper-torso ROI; configurable hue ranges |
| `src/bib_ocr.py` | **Real** | Tesseract (`pytesseract`) with digit-only whitelist, CLAHE preprocessing, range check 1..99 |
| `src/identity.py` | **Real** | Per-track voting + cross-track merging |
| `src/visualization.py` | **Real** | R7/B11 labelled boxes |
| `src/frame_extractor.py` | **Real** | Uniform/motion/manual sampling |
| `src/mocks/ball_detector.py` | Mock | Always returns None — `ball` field stays `null` in output |
| `src/mocks/calibration.py` | Mock | Hardcoded pitch corners + homography (parity with ball-detection) |
| `src/mocks/storage.py` | Mock | Copies files locally, returns `file://` URLs |
| `src/mocks/ocr.py` | Mock | TruthTableMockOCR + NoisyMockOCR for tests |
| `src/mocks/player_detector.py` | Mock | Synthetic drifting bboxes (used by `--mock` and integration tests) |
| `src/mocks/player_tracker.py` | Mock | Index-based tracker for the synthetic detector |

---

## Smoke run findings (match_1080p.mp4, frames 4500-5000)

- **Detection works.** ~6.2 players/frame on the match section vs. the ~10–14 we expect for 7-a-side — model is missing some far-end players. Lowering `confidence_threshold` + enabling `--tiling` should help.
- **Tracking is unstable.** 77 tracks created over 500 frames is too many; ByteTrack is creating new IDs whenever a player is briefly missed. Voting across tracks recovers some of this, but identity merging is the safety net, not the main signal.
- **Bib colours don't match the defaults.** This clip uses **pink and green** bibs; the default HSV ranges are tuned for red/blue. Result: 0/77 identities resolved because every colour reading came back "unknown". To run against this clip, override the HSV config values to match pink (~150–175) and green (~40–80), or use a clip with red/blue bibs.
- **Tesseract resolved 8/623 samples (1.3%).** Even with correct colour matching, OCR is the weakest link at 1080p where bib digits are ~20–40 px tall. Options going forward: (a) train a digit classifier on cropped bib regions, (b) increase input resolution, (c) use a tighter bib-region detector before OCR. Use `--debug-ocr` to save preprocessed crops and inspect what Tesseract sees.

---

## Known limitations

- **Bib OCR is the failure mode.** Far-end players, motion blur, and side-on players will fail. Voting + merging are the mitigation; if a track never gets a confident read its identity stays unresolved (`bib_id: null`, `team: "unknown"`).
- **Bib colour ranges are fixed-width HSV bands.** They assume team colours are saturated. For other palettes (pink/green, yellow/black, etc.) the HSV ranges in `config.py` need tuning per clip.
- **No ID-to-person matching across clips.** Each clip is processed independently — track IDs and bib IDs reset between runs.
- **Mocked ball detection.** This pipeline doesn't detect the ball — `ball-detection/` owns that. The schemas are compatible for downstream merging.
- **No training code.** Fine-tune in Colab using Roboflow export; drop the `.pt` into `models/`.

---

## See also

- `doc/IMPLEMENTATION_PLAN.md` — Phase B handoff doc (component specs).
- `../ball-detection/README.md` — sibling module (ball detection + Kalman tracking).
