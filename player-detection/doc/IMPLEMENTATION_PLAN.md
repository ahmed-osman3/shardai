# Player + Bib Detection — Phase B Implementation Plan

This is the handoff doc for filling in the real CV components after Phase A scaffolding lands. Phase A produced: directory layout, config, types, mocks, identity resolver, visualization, frame extractor, pipeline orchestrator (with `mock_mode=True` integration test path), CLI shells, and tests.

The real components are stubbed in `src/{detection,tracking,bib_colour,bib_ocr}.py` and raise `NotImplementedError`. Fill them in **in order** — each step is independently runnable and testable.

> **Build order is load-bearing.** Detection comes first because tracking is meaningless without it; tracking comes next because OCR voting is meaningless without stable track IDs; colour + OCR come last because they only matter once tracks exist.

---

## Phase B step 1 — Player detection (`src/detection.py`)

**API:** `PlayerDetector(config).detect(frame, frame_idx) -> list[PlayerDetection]`

**Implementation outline:**
```python
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, config: Config) -> None:
        model_path = config.resolve_model_path()
        if model_path.exists():
            self.model = YOLO(str(model_path))
        elif config.model_path is None:
            self.model = YOLO("yolo11m.pt")  # auto-downloads
        else:
            raise FileNotFoundError(...)
        self._cls_id = config.player_class_id
        self._conf = config.confidence_threshold
        self._iou = config.iou_threshold
        self._device = config.resolve_device()
        self._tiling = config.tiling
        self._tile_size = config.tile_size
        self._tile_overlap = config.tile_overlap

    def detect(self, frame, frame_idx=0):
        # Whole-frame inference by default — players are large enough.
        # Tiling is opt-in via config.tiling for far-end tightness — copy the
        # tiled inference path from ball-detection/src/detection.py verbatim
        # (compute tiles, batch crops, translate boxes back, NMS).
        ...
```

**Reuse from ball-detection:** the weights resolution / auto-download fallback, the tile geometry helper `_compute_tiles`, the `_nms` helper. Copy these straight over and rename only the class-id filter.

**Test (`tests/test_detection.py`):**
- `test_compute_tiles_covers_frame` — tile rects cover full frame, no gaps.
- `test_nms_dedupes_overlapping_boxes` — synthetic overlapping detections collapse correctly.
- `test_detect_returns_list_on_blank_frame` — black frame returns `[]`.
- `test_detect_returns_persons_only` — synthetic frame with both ball and person yields person only when `player_class_id=0`.
- All tests skip if `models/yolo11m.pt` not present (mirror ball-detection's `_requires_model` helper).

**Sanity check on the dropped video:**
```bash
python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4 \
    --max-frames 100 --no-video --ocr-every-n 9999
# Should print mean_players_per_frame ≥ 8 for a 7-a-side clip.
```

---

## Phase B step 2 — Tracking (`src/tracking.py`)

**API:** `PlayerTracker(config).update(detections, frame_idx) -> list[TrackedPlayer]`

**Implementation outline:**
```python
import numpy as np
from supervision import ByteTrack, Detections

class PlayerTracker:
    def __init__(self, config: Config) -> None:
        self._tracker = ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=30,  # ByteTrack uses this only for buffer interpretation
        )

    def update(self, detections, frame_idx):
        if not detections:
            sv_det = Detections.empty()
        else:
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            confs = np.array([d.confidence for d in detections], dtype=np.float32)
            cls = np.zeros(len(detections), dtype=int)
            sv_det = Detections(xyxy=xyxy, confidence=confs, class_id=cls)
        sv_tracked = self._tracker.update_with_detections(sv_det)
        return [
            TrackedPlayer(
                frame_idx=frame_idx,
                track_id=int(tid),
                bbox=tuple(bbox.tolist()),
                confidence=float(conf),
            )
            for bbox, conf, tid in zip(sv_tracked.xyxy, sv_tracked.confidence, sv_tracked.tracker_id)
        ]

    def reset(self):
        self._tracker = ByteTrack(...)  # rebuild
```

**Test (`tests/test_tracking.py`):**
- `test_tracker_assigns_consistent_id_across_frames` — same detection at the same place across 10 frames keeps a single track_id.
- `test_tracker_handles_brief_occlusion` — detections drop for 5 frames then reappear at similar position; ByteTrack should re-attach (not always — depends on `lost_track_buffer`).
- `test_tracker_handles_empty_detections` — `update([], frame_idx)` returns `[]`.
- `test_reset_clears_state` — after `reset()`, fresh detections get fresh track_ids starting from 1.

---

## Phase B step 3 — Bib colour classifier (`src/bib_colour.py`)

**API:** `BibColourClassifier(config).classify(crop) -> tuple[str, float]`

**Implementation outline:**
```python
import cv2
import numpy as np

class BibColourClassifier:
    def __init__(self, config):
        self._cfg = config

    def classify(self, crop):
        if crop.size == 0:
            return ("unknown", 0.0)
        h = crop.shape[0]
        y1 = int(self._cfg.bib_roi_y[0] * h)
        y2 = int(self._cfg.bib_roi_y[1] * h)
        roi = crop[y1:y2]
        if roi.size == 0:
            return ("unknown", 0.0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sat = (S > self._cfg.hsv_min_s) & (V > self._cfg.hsv_min_v)
        red_lo = (H >= self._cfg.hsv_red_h_low[0]) & (H <= self._cfg.hsv_red_h_low[1])
        red_hi = (H >= self._cfg.hsv_red_h_high[0]) & (H <= self._cfg.hsv_red_h_high[1])
        red_mask = sat & (red_lo | red_hi)
        blue_mask = sat & (H >= self._cfg.hsv_blue_h[0]) & (H <= self._cfg.hsv_blue_h[1])
        red_n = int(red_mask.sum())
        blue_n = int(blue_mask.sum())
        total = roi.shape[0] * roi.shape[1]
        if max(red_n, blue_n) / total < 0.05:
            return ("unknown", 0.0)
        if red_n >= blue_n:
            return ("red", red_n / total)
        return ("blue", blue_n / total)
```

**Test (`tests/test_bib_colour.py`):**
Hand-build BGR swatch fixtures:
- `test_classify_solid_red_swatch` → `("red", > 0.5)`
- `test_classify_solid_blue_swatch` → `("blue", > 0.5)`
- `test_classify_grey_swatch` → `("unknown", 0.0)`
- `test_classify_empty_crop_returns_unknown` → `("unknown", 0.0)`

---

## Phase B step 4 — Bib OCR (`src/bib_ocr.py`)

**API:** `BibNumberOCR(config).read(crop) -> tuple[int | None, float]`

**Implementation outline:**
```python
import cv2
import numpy as np

class BibNumberOCR:
    def __init__(self, config):
        self._cfg = config
        self._reader = None  # lazy

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr
            gpu = config.resolve_device() in ("cuda",)  # easyocr doesn't fully support MPS
            self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, crop):
        if crop.size == 0:
            return (None, 0.0)
        self._ensure_reader()
        h = crop.shape[0]
        y1 = int(self._cfg.bib_roi_y[0] * h)
        y2 = int(self._cfg.bib_roi_y[1] * h)
        roi = crop[y1:y2]
        if roi.size == 0 or roi.shape[0] < 8:
            return (None, 0.0)
        # Resize so height is at least 64 px
        if roi.shape[0] < 64:
            scale = 64 / roi.shape[0]
            new_w = max(8, int(roi.shape[1] * scale))
            roi = cv2.resize(roi, (new_w, 64), interpolation=cv2.INTER_CUBIC)
        # Light preprocessing: contrast + sharpen
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        lab[..., 0] = l
        roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        results = self._reader.readtext(roi, allowlist="0123456789", detail=1)
        if not results:
            return (None, 0.0)
        # Highest-confidence reading, parse integer, validate range
        results.sort(key=lambda r: -r[2])
        for _, text, conf in results:
            try:
                n = int(text.strip())
            except ValueError:
                continue
            if 1 <= n <= 12:
                return (n, float(conf))
        return (None, 0.0)
```

**Notes:**
- EasyOCR is heavy to import — defer to first call.
- Returns `None` rather than guessing — over-eager OCR poisons the voter. The voter is robust to gaps; it is not robust to systematic misreads.
- MPS is not fully supported by EasyOCR as of writing — fall back to CPU on Apple Silicon. CUDA path uses `gpu=True`.

**Test (`tests/test_bib_ocr.py`):**
- `test_read_returns_none_on_blank_crop` → `(None, 0.0)`
- `test_read_returns_none_on_tiny_crop` → `(None, 0.0)`
- Optionally one fixture-driven test with a synthesized "7" rendered onto a swatch — but this is brittle, so prefer manual notebook-driven verification.

---

## Phase B step 5 — wire real components into pipeline.py

The pipeline orchestrator already conditionally imports real-vs-mock components based on `mock_mode`. Once steps 1–4 are filled in, `mock_mode=False` (the default) will route through the real path. **No edits required to `pipeline.py`.** Run:

```bash
python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4 --max-frames 300
```

Verify the JSON has resolved `R{n}` / `B{n}` IDs on most players in most frames, and that the annotated video shows correctly labelled boxes.

---

## Phase B step 6 — eval script (`scripts/eval_identity.py`)

Replace the stub with a real evaluator. Inputs:
- `--predictions`: pipeline JSON output (as produced above).
- `--ground-truth`: hand-labelled `data/annotations/identity.json` of shape:
  ```json
  {
    "frames": [
      {"idx": 100, "players": [{"bbox": [..], "expected": "R7"}, ...]}
    ]
  }
  ```

Compute:
1. **Per-bib precision/recall** — for each of `R1..R12, B1..B12`, count TP/FP/FN by matching predicted boxes to ground-truth boxes via IoU > 0.5; check label equality.
2. **Track-stability** — for each ground-truth player, the % of consecutive frames where their pipeline track_id stays the same (excluding occluded segments).
3. **Aggregate identity-resolution rate** — `identities_resolved / (identities_resolved + identities_unresolved)`.

Save the report as JSON; print a summary table to stdout.

---

## Phase B step 7 — notebooks

Three notebooks under `notebooks/`. Each should be runnable end-to-end on the dropped video.

- `01_baseline_player_detection.ipynb` — load YOLO, run on a 30-frame slice, visualize boxes, report mean detections/frame, far-end recall (boxes with `y < frame_h * 0.4`).
- `02_tracking_stability.ipynb` — run pipeline with `persist_tracks=True`, load the pickle, plot track-id timelines, count switches/merges per minute.
- `03_bib_identity_resolution.ipynb` — vary `ocr_every_n` (1, 5, 15) and plot identity-resolution rate vs OCR sample budget; show per-track vote distributions.

---

## Phase B step 8 — finalize README

Update the "What is real vs mock" table — flip Phase B stubs to **Real**, drop the Phase A scaffolding banner from the top, add real numbers from the smoke runs (e.g. "12/14 identities resolved on match_1080p.mp4 with default settings").

---

## Verification at end of Phase B

- `pytest tests/` — all green (the new tests for steps 1–4 plus existing scaffold tests).
- `python scripts/run_pipeline.py --input data/raw_clips/match_1080p.mp4 --max-frames 3000`:
  - Produces JSON with non-null `players` arrays per frame and `stats.identities_resolved` consistent with the ~14 visible bibs.
  - Annotated mp4 shows `R{n}` / `B{n}` labels above player boxes.
- MVP smoke targets:
  - `mean_players_per_frame ≥ 10` for a 7-a-side clip.
  - `identities_resolved / (identities_resolved + identities_unresolved) ≥ 0.8` when all 14 bibs are visible at some point.
  - Spot-check via notebook 02: a player tracked through a 5s segment keeps the same `bib_id` after backfill.

---

## Deferred (post-MVP)

- Re-identification by appearance embedding (when ByteTrack drops a track for too long, currently we create a new ID and rely on bib voting to re-merge).
- Calibration-aware filtering (drop detections outside the pitch via `MockCalibration.pixel_to_pitch`).
- Per-frame action recognition (running, walking, sprinting, kicking) — separate model.
- Real ball detection here too — currently we mock it; downstream merger combines with the `ball-detection/` module's output.
