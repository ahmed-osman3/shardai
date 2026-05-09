# Ball Detection — Football Match Analysis MVP

Prototype ball detection pipeline for UK 5/7-a-side football. This is the first isolated layer of a larger cloud analysis system: a user records a match on a tripod-mounted iPhone, uploads it, and the pipeline returns goal events, per-player stats, match ratings, and highlight clips. Ball detection is prototyped first because it is the riskiest computer vision component — the ball is small, fast, and easily lost at the far end of the pitch.

---

## Setup

**Requirements:** Python 3.11+, macOS (Apple Silicon) or Linux with CUDA GPU.

```bash
git clone https://github.com/ahmed-osman3/shardai
cd shardai/ball-detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Download the standard test clips (requires `yt-dlp` + `ffmpeg`):
```bash
brew install yt-dlp ffmpeg   # macOS; use apt on Linux
python scripts/setup_clips.py
# Downloads data/raw_clips/match_1080p.mp4 — 1080p50, 7-aside match, minutes 8-11
```

Or drop your own footage into `data/raw_clips/`. Video files are gitignored.

The pretrained YOLO model auto-downloads from Ultralytics on first use — no manual step needed.

For a fine-tuned model, drop `ball_v1.pt` (or any single-class `.pt`) into `models/` and pass `--model models/ball_v1.pt --ball-class-id 0`.

---

## Quickstart

```bash
# Drop a clip in data/raw_clips/
python scripts/run_pipeline.py --input data/raw_clips/match1.mp4

# Outputs:
#   data/outputs/match1.json            ← ball positions + mock player positions + events
#   data/outputs/match1_annotated.mp4   ← video with trajectory overlay
```

Useful flags:
- `--max-frames N` — stop after N frames (fast iteration)
- `--no-tiling` — whole-frame inference instead of tiled (~10× faster on 1080p, lower far-end recall)
- `--no-video` — JSON only, skip video render
- `--fps N` — subsample source video to N FPS
- `--ball-class-id 0` — use with fine-tuned single-class model instead of COCO class 32
- `--model models/ball_v1.pt` — path to custom weights
- `--device {auto|cpu|mps|cuda}` — force device (default: auto-selects MPS on Apple Silicon)
- `--persist-tracks` — save raw tracks before event detection for offline analysis
- `--verbose` — enable debug logging

---

## Key config defaults (`config.py`)

| Setting | Default | Notes |
|---|---|---|
| `confidence_threshold` | `0.25` | Lower to catch far-end balls at the cost of more false positives |
| `iou_threshold` | `0.45` | NMS deduplication |
| `ball_class_id` | `32` | COCO sports ball; set to `0` for fine-tuned models |
| `tiling` | `True` | Tiled inference enabled by default; use `--no-tiling` to disable |
| `tile_size` | `640` | Tile edge length in pixels |
| `tile_overlap` | `128` | Overlap between adjacent tiles |
| `kalman_max_lost_frames` | `30` | Frames before track is dropped (≈0.5 s @ 60 fps) |
| `device` | `"auto"` | Auto-selects MPS → CUDA → CPU |

---

## Notebooks (run in order)

| Notebook | Purpose |
|---|---|
| `01_baseline_pretrained.ipynb` | Run pretrained YOLO11 (COCO class 32) on a clip. Sets the detection-rate floor. |
| `02_tile_inference_eval.ipynb` | Compare whole-frame vs tiled inference. Shows detection-rate uplift and time cost. |
| `03_kalman_gap_fill.ipynb` | Apply Kalman tracker to detection output. Shows % frames covered with gap-fill enabled. |

```bash
jupyter notebook
```

---

## Extract frames for labelling

```bash
python scripts/extract_frames.py \
    --input data/raw_clips/match1.mp4 \
    --output data/frames/match1/ \
    --strategy motion \
    --count 200
```

Strategies: `uniform` (evenly spaced), `motion` (highest frame-difference — best for ball labelling), `manual` (every Nth frame).

---

## Run evaluation on a labelled set

```bash
python scripts/eval_detection.py \
    --model models/ball_v1.pt \
    --labels data/annotations/ \
    --frames data/frames/ \
    --report data/outputs/eval_report.json
```

Reports mAP@50, mAP@50-95, per-zone recall (near-half / far-half / corners), detection rate by motion blur, and top-20 false positive crops.

---

## What is real vs mock

| Module | Status | Notes |
|---|---|---|
| `src/detection.py` | **Real** | Tiled YOLO inference |
| `src/tracking.py` | **Real** | Kalman filter gap-fill |
| `src/visualization.py` | **Real** | Trajectory + bbox overlay |
| `src/mocks/player_detector.py` | Mock | Returns synthetic drifting player bboxes |
| `src/mocks/calibration.py` | Mock | Hardcoded pitch corners + homography |
| `src/mocks/event_detector.py` | Partial mock | Rule-based; close to v1 real logic |
| `src/mocks/storage.py` | Mock | Copies files locally, returns `file://` URLs |

### Where real implementations will live (future modules)

- **Player detection + bib OCR:** already implemented in `../player-detection/` — YOLO11 person detector + Tesseract OCR on cropped bib regions
- **Pitch calibration:** `src/calibration/` — 4-corner homography from user-selected keypoints or automatic line detection
- **Event detection:** `src/events/` — promoted from `mocks/event_detector.py` with tuned thresholds
- **Storage:** `src/storage/` — real R2/S3 with presigned URLs
- **Backend API:** separate FastAPI service
- **Mobile app:** separate React Native project

---

## Known limitations

- **Single camera only.** No multi-angle reconstruction; far-end ball detection degrades with distance.
- **Far-end detection is hard.** Ball is ~10 px at 40 m on 1080p. Tiling helps but fine-tuning on far-end crops is essential.
- **No training code.** Fine-tune in Colab using Roboflow export; drop the `.pt` into `models/`.
- **Mock players are synthetic.** Per-player stats and possession logic will be inaccurate until real player detection is wired in from `../player-detection/`.
- **No temporal smoothing on events.** Event detector fires on individual frames; a short debounce window is needed in production.

---

## See also

- `doc/IMPLEMENTATION_PLAN.md` — design notes and handoff doc.
- `../player-detection/README.md` — sibling module (player detection + bib identity resolution).
