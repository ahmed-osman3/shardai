"""Player detection via YOLO inference.

Whole-frame inference by default — players are big enough on 1080p footage
that tiling adds latency without much recall benefit. Tiling is opt-in via
`config.tiling` for very-far-end situations (e.g. wider 7-a-side pitches
where the goalkeeper is < 30 px tall).

Filters detections to `config.player_class_id` (default 0 = COCO person).
A higher confidence threshold than the ball detector (default 0.4) suppresses
sideline / spectator false positives that the ball pipeline doesn't see.
"""

from __future__ import annotations

import logging

import numpy as np

from config import Config
from src.types import PlayerDetection

logger = logging.getLogger(__name__)


class PlayerDetector:
    """YOLO-backed player detection.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Config) -> None:
        from ultralytics import YOLO

        model_path = config.resolve_model_path()
        if model_path.exists():
            self.model = YOLO(str(model_path))
        elif config.model_path is None:
            # Default yolo11m.pt — let ultralytics auto-download
            self.model = YOLO("yolo11m.pt")
        else:
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. "
                f"Drop the .pt file into {config.models_dir}/."
            )
        self._cls_id = config.player_class_id
        self._conf = config.confidence_threshold
        self._iou = config.iou_threshold
        self._tile_size = config.tile_size
        self._tile_overlap = config.tile_overlap
        self._device = config.resolve_device()
        self._tiling = config.tiling
        self._max_area_pct = config.max_bbox_area_pct

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[PlayerDetection]:
        """Run player detection on one frame.

        Args:
            frame: BGR image as returned by cv2.
            frame_idx: Frame index for populating PlayerDetection.frame_idx.

        Returns:
            Deduplicated list of player detections in frame coordinates (xyxy pixels).
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        max_area = self._max_area_pct * h * w

        if not self._tiling:
            results = self.model.predict(
                frame,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
                device=self._device,
            )
            detections: list[PlayerDetection] = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls) != self._cls_id:
                        continue
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    if (bx2 - bx1) * (by2 - by1) > max_area:
                        continue
                    detections.append(PlayerDetection(
                        bbox=(bx1, by1, bx2, by2),
                        confidence=float(box.conf),
                        frame_idx=frame_idx,
                    ))
            return detections

        tiles = self._compute_tiles(h, w)
        crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]

        # Single batched MPS/CUDA call — ~3-5x faster than per-tile
        results = self.model.predict(
            crops,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
            device=self._device,
        )

        detections = []
        for (x1, y1, _, _), result in zip(tiles, results):
            if result.boxes is None:
                continue
            for box in result.boxes:
                if int(box.cls) != self._cls_id:
                    continue
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                if (bx2 - bx1) * (by2 - by1) > max_area:
                    continue
                detections.append(PlayerDetection(
                    bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    confidence=float(box.conf),
                    frame_idx=frame_idx,
                ))

        return self._nms(detections)

    def _compute_tiles(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        """Compute tile rects (x1, y1, x2, y2) that cover the frame."""
        stride = self._tile_size - self._tile_overlap
        tiles = []
        y = 0
        while True:
            x = 0
            while True:
                x2 = min(x + self._tile_size, w)
                y2 = min(y + self._tile_size, h)
                tiles.append((x, y, x2, y2))
                if x2 == w:
                    break
                x += stride
            if y2 == h:
                break
            y += stride
        return tiles

    def _nms(self, detections: list[PlayerDetection]) -> list[PlayerDetection]:
        """IOU-based NMS to deduplicate detections across tile boundaries."""
        if len(detections) <= 1:
            return detections

        import torch
        from torchvision.ops import nms

        boxes = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
        keep = nms(boxes, scores, self._iou)
        return [detections[i] for i in keep.tolist()]
