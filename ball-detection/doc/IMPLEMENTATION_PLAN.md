# Implementation Plan — Ball Detection Pipeline

Hand-off document for the next implementation conversation. Each module is described in enough detail to implement without revisiting the original spec.

---

## Architecture Overview

```
video file
    │
    ▼
FrameExtractor (cv2 VideoCapture)
    │  frame: np.ndarray
    ▼
BallDetector.detect(frame)          ← tiled YOLO inference
    │  list[Detection]
    ▼
BallTracker.update(detections, frame_idx)   ← Kalman gap-fill
    │  TrackedBall | None
    │
    ├──► MockPlayerDetector.detect(frame, frame_idx)   ← synthetic drift
    │       list[PlayerDetection]
    │
    ├──► MockCalibration                               ← hardcoded homography
    │       pixel_to_pitch(), is_in_goal()
    │
    └──► MockEventDetector.process(ball_track, player_track, calibration)
             list[Event]

All per-frame state → PipelineResult
    │
    ├──► {name}.json          (ball + player positions + events)
    ├──► {name}_annotated.mp4 (visualization)
    └──► MockStorage.upload() → local file:// URL
```

---

## Key Dataclasses

Define these in `src/types.py` (or inline in each module — keep consistent):

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # xyxy, pixel coords
    confidence: float
    frame_idx: int


@dataclass
class TrackedBall:
    frame_idx: int
    x: float           # centre x, pixels
    y: float           # centre y, pixels
    source: Literal["detected", "interpolated", "lost"]
    confidence: float  # 0.0 if interpolated/lost


@dataclass
class PlayerDetection:
    bbox: tuple[float, float, float, float]  # xyxy
    player_id: str     # e.g. "R1", "B7"
    team_id: str       # "red" | "blue"
    confidence: float


@dataclass
class Event:
    type: str          # "goal" | "shot" | "pass" | "possession_change"
    frame_idx: int
    ts_seconds: float
    primary_player: str | None
    secondary_player: str | None
    metadata: dict     # e.g. {"goal_end": "north", "velocity_mps": 14.2}


@dataclass
class PipelineResult:
    json_path: Path
    video_path: Path
    stats: dict        # detection_rate, interpolation_rate, lost_rate, total_frames
```

---

## Module-by-Module Implementation Guide

Build in this order — each module depends only on earlier ones.

---

### 1. `src/mocks/calibration.py` — MockCalibration

**Purpose:** Provide pixel↔pitch-metre mapping and goal-line queries so the event detector can reason about ball position in real-world coordinates.

**Class:** `MockCalibration`

```python
class MockCalibration:
    def __init__(self, frame_w: int = 1920, frame_h: int = 1080) -> None: ...

    def pixel_to_pitch(self, x: float, y: float) -> tuple[float, float]:
        """Map pixel coords to pitch metres (origin = near-left corner)."""

    def is_in_goal(self, x: float, y: float, which_goal: Literal["north", "south"]) -> bool:
        """Return True if pixel position is within the goal-line region."""

    @property
    def pitch_corners_px(self) -> list[tuple[int, int]]:
        """Four pitch corners in pixel space [TL, TR, BR, BL]."""

    @property
    def goal_posts_px(self) -> dict[str, list[tuple[int, int]]]:
        """Post positions: {"north": [(x1,y1),(x2,y2)], "south": [...]}."""
```

**Implementation steps:**
1. Hardcode 4 pitch corner pixels for a 1920×1080 frame with a realistic perspective offset (pitch does not fill frame edge-to-edge). Example: TL=(160, 120), TR=(1760, 120), BR=(1820, 960), BL=(100, 960).
2. Use `cv2.getPerspectiveTransform` to compute a homography H from those 4 corners to the real pitch dimensions (e.g. 50m × 30m).
3. `pixel_to_pitch`: apply H via `cv2.perspectiveTransform`.
4. Hardcode goal post pixel positions at both ends. Goal width ~7.3m, centred on pitch width.
5. `is_in_goal`: check if the ball's x falls between post x-coords AND y is within 2px of the goal-line y.

**Do NOT implement:** actual line detection, user keypoint selection, lens undistortion.

---

### 2. `src/mocks/player_detector.py` — MockPlayerDetector

**Purpose:** Produce synthetic but visually plausible player positions so the pipeline can run end-to-end and the event detector has players to reason about.

**Class:** `MockPlayerDetector`

```python
class MockPlayerDetector:
    def __init__(self, n_per_team: int = 7, seed: int = 42) -> None: ...

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[PlayerDetection]: ...
```

**Implementation steps:**
1. In `__init__`: seed `np.random.default_rng(seed)`. For each of the 14 players (7 red, 7 blue), generate:
   - `base_x`, `base_y`: grid positions spread across the frame (tile the pitch into a 7×2 grid)
   - `freq_x`, `freq_y`: random frequency in [0.01, 0.05] rad/frame
   - `phase_x`, `phase_y`: random phase in [0, 2π]
   - `amplitude`: random in [20, 60] pixels
2. In `detect(frame, frame_idx)`:
   - For each player: `x = base_x + amplitude * sin(frame_idx * freq_x + phase_x)`; clamp to [50, frame_w-50].
   - Bbox: 40×80 pixel box centred on (x, y) — rough person silhouette.
   - Confidence: fixed 0.85.
   - Return list[PlayerDetection].

**Do NOT implement:** real person detection, bib OCR, cross-frame re-identification.

---

### 3. `src/mocks/event_detector.py` — MockEventDetector

**Purpose:** Rule-based event detection consuming ball track + player tracks. The rules here are close to what the real v1 will use — this is not purely a mock.

**Class:** `MockEventDetector`

```python
class MockEventDetector:
    def __init__(self, config: Config, calibration: MockCalibration) -> None: ...

    def process(
        self,
        ball_track: list[TrackedBall | None],
        player_track: list[list[PlayerDetection]],
        fps: float,
    ) -> list[Event]: ...
```

**Implementation steps:**

**Possession** (per frame):
1. For each frame, find the player whose bbox centre is closest to the ball position.
2. Smooth possession over a 30-frame window (mode of last 30 possession labels).

**Pass** (across frames):
3. Scan the smoothed possession sequence. When possession switches from player A to player B on the same team within 120 frames (2s @ 60fps), emit `Event(type="pass", primary_player=A, secondary_player=B)`.

**Shot** (per frame):
4. Estimate ball velocity: `vx = ball[i].x - ball[i-5].x` over 5 frames, convert px/frame to m/s using `pixel_to_pitch`.
5. If speed > 12 m/s AND the ball's projected trajectory (extrapolate 30 frames) passes through the goal region: emit `Event(type="shot")`.

**Goal** (per frame):
6. Check `calibration.is_in_goal(ball.x, ball.y, "north")` and `"south"`. If True and ball source != "lost": emit `Event(type="goal")`. Debounce: suppress duplicate goals within 180 frames.

**Do NOT implement:** ML-based event detection, multi-camera geometry, audio cues.

---

### 4. `src/mocks/storage.py` — MockStorage

**Purpose:** Mimic an S3/R2 interface so pipeline code can "upload" results without real credentials.

**Class:** `MockStorage`

```python
class MockStorage:
    def __init__(self, base_dir: Path) -> None:
        """base_dir is e.g. data/outputs/mock_storage/"""

    def upload(self, local_path: Path, key: str) -> str:
        """Copy local_path to base_dir/key. Return file:// URL."""

    def download(self, key: str, local_path: Path) -> None:
        """Copy base_dir/key to local_path."""

    def signed_url(self, key: str, ttl_seconds: int = 3600) -> str:
        """Return file:// URL (TTL ignored in mock)."""
```

**Implementation steps:**
1. `upload`: `shutil.copy2(local_path, base_dir / key)`, create parent dirs. Return `f"file://{(base_dir / key).resolve()}"`.
2. `download`: `shutil.copy2(base_dir / key, local_path)`.
3. `signed_url`: return same `file://` URL.

---

### 5. `src/detection.py` — BallDetector

**Purpose:** Tiled YOLO inference. Splits each frame into overlapping tiles, runs the model on each, translates coordinates back to frame space, deduplicates with NMS.

**Class:** `BallDetector`

```python
class BallDetector:
    def __init__(self, config: Config) -> None: ...

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]: ...

    def _compute_tiles(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) tile rects covering the frame."""

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """Deduplicate detections across tile boundaries using IOU-based NMS."""
```

**Implementation steps:**

1. **Model loading** (`__init__`):
   - `self.model = YOLO(config.resolve_model_path())`
   - Store `config.ball_class_id`, `config.confidence_threshold`, `config.iou_threshold`, `config.tile_size`, `config.tile_overlap`.
   - Device is passed to `model.predict(device=...)`.

2. **`_compute_tiles(h, w)`**:
   - stride = tile_size - tile_overlap
   - `xs = range(0, w, stride)`, `ys = range(0, h, stride)`
   - For each (x, y): tile = (x, y, min(x+tile_size, w), min(y+tile_size, h))
   - Return all tiles.

3. **`detect(frame, frame_idx)`**:
   - Call `_compute_tiles(frame.shape[0], frame.shape[1])`.
   - For each tile `(x1, y1, x2, y2)`:
     - `crop = frame[y1:y2, x1:x2]`
     - `results = self.model.predict(crop, conf=confidence_threshold, iou=iou_threshold, verbose=False, device=device)`
     - For each box in results[0].boxes: filter `cls == ball_class_id`, translate `xyxy` by `(+x1, +y1, +x1, +y1)`.
     - Append `Detection(bbox=translated_xyxy, confidence=conf, frame_idx=frame_idx)`.
   - Call `_nms(all_detections)`.
   - Return deduplicated list.

4. **`_nms(detections)`**:
   - Convert to numpy arrays of boxes and scores.
   - Use `supervision.detection.utils.box_iou_batch` or a simple torchvision NMS call.
   - Keep detections whose IOU with any higher-confidence detection is < `iou_threshold`.

**Edge cases:**
- Frame smaller than tile_size: single tile covers whole frame.
- Model not found: raise `FileNotFoundError` with clear message pointing to `models/` dir.
- Empty frame (all black): return [].

---

### 6. `src/tracking.py` — BallTracker

**Purpose:** Maintain a single Kalman filter for ball position + velocity. Fill short gaps with Kalman predictions. Mark track as lost after too many consecutive missing frames.

**Class:** `BallTracker`

```python
class BallTracker:
    def __init__(self, config: Config) -> None: ...

    def update(self, detections: list[Detection], frame_idx: int) -> TrackedBall | None: ...

    def reset(self) -> None:
        """Reset tracker state (call between clips)."""
```

**Implementation steps:**

1. **`__init__`**:
   ```python
   from filterpy.kalman import KalmanFilter
   self.kf = KalmanFilter(dim_x=4, dim_z=2)
   # State: [x, y, vx, vy]
   self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
   # Measurement: [x, y]
   self.kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
   self.kf.R *= 10.0    # measurement noise ~10px²
   self.kf.Q *= 1.0     # process noise ~1px²
   self.kf.P *= 500.0   # initial uncertainty
   ```
   - `self._lost_frames = 0`
   - `self._initialized = False`
   - `self._max_lost = config.kalman_max_lost_frames`

2. **`update(detections, frame_idx)`**:
   - If detections is non-empty:
     - Pick the detection closest to current predicted position (if initialized) or highest confidence (if not).
     - If not initialized: set `kf.x[:2] = [[cx], [cy]]`, `_initialized = True`.
     - Else: `kf.update([[cx], [cy]])`.
     - `kf.predict()`
     - `_lost_frames = 0`
     - Return `TrackedBall(frame_idx, cx, cy, "detected", detection.confidence)`.
   - If no detections:
     - If not initialized: return None.
     - `_lost_frames += 1`
     - If `_lost_frames > _max_lost`:
       - Return `TrackedBall(frame_idx, px, py, "lost", 0.0)` — signal lost but still emit position from last prediction.
     - Else:
       - `kf.predict()`
       - px, py = `kf.x[0,0]`, `kf.x[1,0]`
       - Return `TrackedBall(frame_idx, px, py, "interpolated", 0.0)`.

---

### 7. `src/visualization.py`

**Purpose:** Draw ball trajectory, bbox overlays, player boxes, and event banners onto frames.

**Functions:**

```python
def draw_frame(
    frame: np.ndarray,
    ball: TrackedBall | None,
    ball_history: list[TrackedBall],  # last N tracked positions
    players: list[PlayerDetection],
    events: list[Event],              # events active this frame
    config: Config,
) -> np.ndarray:
    """Return annotated copy of frame. Does not modify input."""

def setup_video_writer(
    output_path: Path,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> cv2.VideoWriter: ...
```

**Drawing spec:**
- **Ball bbox** (if `source == "detected"`): green box, confidence label.
- **Ball interpolated**: yellow dot (no box, just a circle at centre).
- **Ball lost**: no marker.
- **Trajectory tail**: last 30 positions as a polyline. Colour fades from bright (recent) to dim (old). Use `cv2.addWeighted` per segment or vary alpha with index. `source == "interpolated"` segments: dashed yellow; `"detected"` segments: solid green.
- **Player boxes**: thin blue/red boxes with player_id label (team colour).
- **Event banners**: when a goal event is active (±60 frames), draw a full-width banner at top: "GOAL — R3" in large white text on semi-transparent dark background.

---

### 8. `src/frame_extractor.py` — FrameExtractor

**Purpose:** Sample frames from a video clip for Roboflow/labelling upload.

**Class:** `FrameExtractor`

```python
class FrameExtractor:
    def __init__(self, config: Config) -> None: ...

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
        strategy: Literal["uniform", "motion", "manual"],
        count: int = 200,
        every_n: int = 10,  # only used by "manual"
    ) -> list[Path]:
        """Extract frames and save as JPEG. Return list of saved paths."""
```

**Strategies:**
- **uniform**: `frame_indices = np.linspace(0, total_frames-1, count).astype(int)`. Seek and write each.
- **motion**: compute per-frame motion score = `np.mean(np.abs(frame - prev_frame))` for a fast pass over the video. Sample the top-`count` frames by score.
- **manual**: save every `every_n`th frame regardless of count.

Save frames as `{output_dir}/{frame_idx:06d}.jpg` at quality 95.

---

### 9. `src/pipeline.py` — run_pipeline()

**Purpose:** Orchestrate the full pipeline: open video, iterate frames, call all modules, write outputs.

```python
def run_pipeline(video_path: Path, config: Config) -> PipelineResult:
    """Run ball detection + tracking + mock player/event pipeline on a video."""
```

**Implementation steps:**

1. Open video with `cv2.VideoCapture(str(video_path))`. Read `fps`, `frame_w`, `frame_h`, `total_frames`.
2. Instantiate: `BallDetector(config)`, `BallTracker(config)`, `MockPlayerDetector()`, `MockCalibration(frame_w, frame_h)`, `MockEventDetector(config, calibration)`.
3. Set up `cv2.VideoWriter` via `visualization.setup_video_writer(...)`.
4. Per-frame loop (tqdm progress bar over `total_frames`):
   - If `config.target_fps` set: skip frames to match target FPS.
   - Read frame; if failed, break.
   - `detections = detector.detect(frame, frame_idx)`
   - `ball = tracker.update(detections, frame_idx)`
   - `players = player_detector.detect(frame, frame_idx)`
   - Append ball and players to running track lists.
   - Draw and write annotated frame.
5. After loop: `event_detector.process(ball_track, player_track, fps)`.
6. Serialize to JSON: frame-by-frame records + events list.
7. Return `PipelineResult(json_path, annotated_video_path, stats)`.

**Stats to compute:**
```python
stats = {
    "total_frames": int,
    "detection_rate": detected_frames / total_frames,
    "interpolation_rate": interpolated_frames / total_frames,
    "lost_rate": lost_frames / total_frames,
    "event_count": len(events),
}
```

---

### 10. `scripts/run_pipeline.py` — CLI

```
python scripts/run_pipeline.py --input data/raw_clips/match1.mp4 [--output data/outputs/] [--device auto]
```

Use `argparse`. Load default `Config()`, override `model_path` / `device` if passed. Call `run_pipeline()`. Print stats table to stdout.

---

### 11. `scripts/extract_frames.py` — CLI

```
python scripts/extract_frames.py \
    --input data/raw_clips/match1.mp4 \
    --output data/frames/match1/ \
    --strategy motion \
    --count 200
```

Use `argparse`. Instantiate `FrameExtractor(Config())` and call `extract()`.

---

### 12. `scripts/eval_detection.py` — CLI + eval loop

```
python scripts/eval_detection.py \
    --model models/ball_v1.pt \
    --labels data/annotations/ \
    --frames data/frames/ \
    --report data/outputs/eval_report.json
```

**Implementation steps:**
1. Load all label files (`*.txt` in `--labels`). Each line: `class cx cy w h` (YOLO format, normalised).
2. For each labelled frame: run `BallDetector.detect()`, compute IOU between predicted and ground-truth box.
3. Detection = TP if IOU ≥ 0.5, else FP/FN.
4. Compute overall mAP@50 and mAP@50-95.

**Per-zone recall** (split by frame x-coordinate of GT box centre):
- `near_half`: cx_px < 960
- `far_half`: cx_px ≥ 960
- `corners`: cx_px < 320 or cx_px > 1600, AND (cy_px < 200 or cy_px > 880)

**Motion blur bucketing:**
- Compute Laplacian variance of the GT box crop: `cv2.Laplacian(crop, cv2.CV_64F).var()`
- Bucket into: sharp (>100), medium (20–100), blurry (<20)
- Report recall per bucket.

**Top-20 FP crops:**
- For each FP detection: save a 128×128 crop centred on the predicted box to `data/outputs/fp_crops/`.

---

### 13–15. Notebooks

Each notebook should be self-contained top-to-bottom.

**01_baseline_pretrained.ipynb** — sections:
1. Setup & imports
2. Load Config + BallDetector (yolo11m.pt, no tiling, whole-frame inference)
3. Run on first 500 frames of a sample clip
4. Compute and display detection rate
5. Show 10 annotated sample frames with sv.BoxAnnotator
6. Conclusion: baseline detection rate = floor expectation

**02_tile_inference_eval.ipynb** — sections:
1. Setup & imports
2. Run whole-frame inference on 100 frames → detection list A
3. Run tiled inference on same 100 frames → detection list B
4. Compare detection rates (bar chart)
5. Compare inference time per frame (bar chart)
6. Side-by-side annotated frame comparison (picked from far-end of pitch)
7. Conclusion: when does tiling help?

**03_kalman_gap_fill.ipynb** — sections:
1. Setup & imports
2. Run BallDetector on a 30-second segment
3. Run BallTracker on detection output
4. Plot ball trajectory: x vs frame_idx, colour-coded by source (green=detected, yellow=interpolated, red=lost)
5. Print: % detected / % interpolated / % lost
6. Show 5 example interpolated segments overlaid on video frames
7. Conclusion: how much does Kalman extend coverage?

---

### 16. `tests/`

Focus on integration tests over unit tests. ML detection quality belongs in the eval script.

**`tests/test_detection.py`:**
- Test `_compute_tiles` returns tiles that cover the full frame with no gaps.
- Test tiles have correct overlap.
- Test `_nms` removes a duplicate detection with IOU > threshold.
- Test `BallDetector` returns `list[Detection]` on a synthetic blank frame (may return empty — just check type).

**`tests/test_tracking.py`:**
- Test `BallTracker` returns `"detected"` when a detection is provided.
- Test `BallTracker` returns `"interpolated"` for up to `max_lost_frames` consecutive empty frames.
- Test `BallTracker` returns `"lost"` after exceeding `max_lost_frames`.
- Test `reset()` clears state.

**`tests/test_pipeline.py`:**
- Create a 30-frame synthetic video (black frames, 640×480) using `cv2.VideoWriter`.
- Run `run_pipeline(synthetic_video, Config())` end-to-end.
- Assert JSON file is created and parseable.
- Assert annotated video is created and has correct frame count.
- Assert `PipelineResult.stats` has all expected keys.

---

## Verification — Done Criteria

```bash
cd ball-detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Drop a clip in data/raw_clips/
python scripts/run_pipeline.py --input data/raw_clips/match1.mp4
# Expect:
#   data/outputs/match1.json            ← valid JSON with "frames" and "events" keys
#   data/outputs/match1_annotated.mp4   ← playable video with overlays

# Notebooks
jupyter notebook notebooks/01_baseline_pretrained.ipynb   # runs top-to-bottom, shows detection rate
jupyter notebook notebooks/02_tile_inference_eval.ipynb   # runs top-to-bottom, shows uplift chart
jupyter notebook notebooks/03_kalman_gap_fill.ipynb       # runs top-to-bottom, shows trajectory plot

# Tests (no real video needed — synthetic frames)
pytest tests/ -v

# Frame extraction
python scripts/extract_frames.py \
    --input data/raw_clips/match1.mp4 \
    --output data/frames/match1/ \
    --strategy motion --count 200
ls data/frames/match1/ | wc -l  # expect ~200
```

**Performance target:** 5-minute 1080p60 clip should complete in under 10 minutes on Apple Silicon M-series.

---

## Deferred

Items intentionally skipped in v1 — revisit when value warrants.

- **Event banners on annotated video** — draw a top-of-frame "GOAL — R3" strip during ±60 frames around each event. Skipped because the primary goal is internal event/stats logging, not video presentation. Implementation path when needed: two-pass (re-read video after event detection runs on full tracks). See `src/visualization.py` for the TODO marker.

- **`Config.outputs_dir` override field** — currently overridden via `run_pipeline(outputs_dir=...)` argument. If multiple call sites need overrides, promote to a Config field.

- **Cross-frame tile batching** — current implementation batches all tiles within one frame into a single `model.predict([crops])` call (~3–5× faster than per-tile on M1 MPS). A further optimization is cross-frame batching (multiple frames × multiple tiles in one call). Revisit if the per-clip time target slips past 10 min.
