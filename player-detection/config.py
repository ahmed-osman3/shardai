"""Central configuration for the player + bib detection pipeline.

All tunables live here. Pass a Config instance explicitly — no global state.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """All pipeline tunables in one place.

    Args:
        project_root: Repo root. Defaults to the directory containing this file.
        model_path: Path to YOLO .pt weights. Defaults to pretrained yolo11m.pt.
        confidence_threshold: Minimum detection confidence to keep.
        iou_threshold: IOU threshold for NMS deduplication.
        player_class_id: COCO class 0 = person. Set to 0 for fine-tuned single-class.
        tile_size: Tile edge length in pixels for tiled inference (opt-in).
        tile_overlap: Overlap between adjacent tiles in pixels.
        tiling: Disabled by default — players are big enough for whole-frame inference.
        target_fps: Subsample video to this FPS. None = use source FPS.
        device: Inference device. "auto" selects MPS on Apple Silicon, CUDA on Linux, CPU otherwise.

        # Tracking (ByteTrack via supervision)
        track_activation_threshold: Score threshold to start a track.
        lost_track_buffer: Frames a track survives without a detection (default 30 ≈ 1s @ 30fps).
        minimum_matching_threshold: IOU cost cap when matching detections to tracks.

        # Bib OCR
        ocr_every_n: Run OCR on 1 in N frames per track (default 5).
        min_bib_crop_h: Skip OCR on player crops shorter than this (px) — far end / tiny boxes.
        ocr_engine: Identifier recorded in JSON metadata.

        # Bib colour (HSV) — colour-agnostic dominant-hue classifier
        hsv_min_s: Saturation floor — pixels below this are not counted as colourful.
        hsv_min_v: Value floor — discards near-black / shadow pixels.
        hsv_white_max_s: Saturation ceiling for classifying pixels as white.
        hsv_white_min_v: Value floor for classifying pixels as white.

        # Identity resolution
        min_votes_for_identity: Need ≥ this many matching (colour, number) OCR readings to resolve a track.

        # Bib ROI (fraction of bbox height — slice out the chest area where the bib sits)
        bib_roi_y: (top, bottom) as fractions of bbox height.

        # Visualization
        draw_unresolved_boxes: Draw boxes for tracks with no resolved bib id (greyed out).
    """

    # --- paths ---
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def raw_clips_dir(self) -> Path:
        return self.project_root / "data" / "raw_clips"

    @property
    def frames_dir(self) -> Path:
        return self.project_root / "data" / "frames"

    @property
    def annotations_dir(self) -> Path:
        return self.project_root / "data" / "annotations"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "data" / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    # --- detection ---
    model_path: Path = field(default=None)  # None → resolved to models_dir/yolo11m.pt at runtime
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    player_class_id: int = 0  # COCO person; 0 also for fine-tuned single-class
    max_bbox_area_pct: float = 0.35  # drop person detections covering > 35% of frame
                                     # (camera-edge artifacts, foreground spectators, intro talking-heads)

    # --- tiled inference (opt-in for far-end tightness) ---
    tiling: bool = False
    tile_size: int = 640
    tile_overlap: int = 128

    # --- tracking (ByteTrack via supervision) ---
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 90
    minimum_matching_threshold: float = 0.5

    # --- bib OCR ---
    ocr_every_n: int = 5
    min_bib_crop_h: int = 60
    ocr_engine: str = "easyocr"

    # --- bib colour HSV thresholds ---
    hsv_min_s: int = 80       # saturation floor for colourful pixels
    hsv_min_v: int = 40       # value floor (discard shadows)
    hsv_white_max_s: int = 50  # saturation ceiling → white
    hsv_white_min_v: int = 180 # value floor → white

    # --- identity resolution ---
    min_votes_for_identity: int = 2

    # --- bib ROI ---
    bib_roi_y: tuple[float, float] = (0.15, 0.55)

    # --- video ---
    target_fps: int | None = None  # None = source fps

    # --- device ---
    device: str = "auto"  # auto | cpu | mps | cuda

    # --- visualization ---
    draw_unresolved_boxes: bool = True

    def resolve_model_path(self) -> Path:
        """Return the effective model path, defaulting to yolo11m.pt in models_dir."""
        if self.model_path is not None:
            return self.model_path
        return self.models_dir / "yolo11m.pt"

    def resolve_device(self) -> str:
        """Resolve 'auto' to the best available device for this machine."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
