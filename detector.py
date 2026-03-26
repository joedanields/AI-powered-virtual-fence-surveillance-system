"""
Detector module — YOLOv8-based human detection.

Wraps Ultralytics YOLOv8 with configurable parameters.
Filters results to person-class only and returns structured Detection objects.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import DetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected person."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) — top-left / bottom-right
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[int, int]:
        """Geometric center of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Bottom-center — better proxy for foot position."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class HumanDetector:
    """
    YOLOv8-based person detector.

    Lazy-loads the model on first inference call so that
    import-time stays fast.
    """

    def __init__(self, config: DetectorConfig):
        self._config = config
        self._model = None
        self._inference_time: float = 0.0

    def _load_model(self):
        """Load the YOLOv8 model (deferred until first use)."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        logger.info("Loading YOLOv8 model: %s", self._config.model_path)
        self._model = YOLO(self._config.model_path)

        # Move to target device
        device = self._config.device
        if device:
            logger.info("Using device: %s", device)
        else:
            logger.info("Device auto-select (CUDA if available, else CPU)")

        # Warm-up inference
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            device=device if device else None,
            verbose=False,
        )
        logger.info("YOLOv8 model loaded and warmed up")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of Detection objects for class=person only.
        """
        if self._model is None:
            self._load_model()

        start = time.perf_counter()

        results = self._model.predict(
            frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            classes=self._config.target_classes,
            imgsz=self._config.img_size,
            device=self._config.device if self._config.device else None,
            half=self._config.half_precision,
            max_det=self._config.max_detections,
            verbose=False,
        )

        self._inference_time = (time.perf_counter() - start) * 1000  # ms

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self._config.target_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_id=cls_id,
                    class_name="person",
                ))

        return detections

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._inference_time
