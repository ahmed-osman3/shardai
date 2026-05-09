"""Central configuration for the ball detection pipeline.

All tunables live here. Pass a Config instance explicitly — no global state.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """All pipeline tunables in one place.

    Args:
        project_root: Repo root. Defaults to the directory containing this file.
        raw_clips_dir: Input footage directory.
        frames_dir: Extracted frames for labelling.
        annotations_dir: YOLO-format label files.
        outputs_dir: Pipeline outputs (JSON, annotated video).
        models_dir: Model weight files.
        model_path: Path to YOLO .pt weights. Defaults to pretrained yolo11m.pt.
        confidence_threshold: Minimum detection confidence to keep.
        iou_threshold: IOU threshold for NMS deduplication.
        ball_class_id: COCO class 32 = sports ball. Set to 0 for fine-tuned single-class model.
        tile_size: Tile edge length in pixels for tiled inference.
        tile_overlap: Overlap between adjacent tiles in pixels.
        kalman_max_lost_frames: Frames without a detection before track is marked lost (30 = 0.5s @ 60fps).
        target_fps: Subsample video to this FPS. None = use source FPS.
        device: Inference device. "auto" selects MPS on Apple Silicon, CUDA on Linux, CPU otherwise.
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
    ball_class_id: int = 32  # COCO sports ball; 0 for fine-tuned single-class

    # --- tiled inference ---
    tile_size: int = 640
    tile_overlap: int = 128

    # --- tracking ---
    kalman_max_lost_frames: int = 30  # 0.5s @ 60fps

    # --- video ---
    target_fps: int | None = None  # None = source fps; e.g. 30 to subsample

    # --- device ---
    device: str = "auto"  # auto | cpu | mps | cuda

    # --- inference ---
    tiling: bool = True  # disable for faster whole-frame inference on 1080p footage

    # --- visualization ---
    draw_player_boxes: bool = False  # mock player boxes interfere until real detection is wired in

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
