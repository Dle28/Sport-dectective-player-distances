"""
Player detection utilities using YOLO or similar object detectors.

This module wraps an underlying detection backend (e.g. ultralytics YOLOv8)
behind a small interface that returns bounding boxes for player detections.

The main entrypoint is :func:`create_detector` which returns an object with a
`detect(frame)` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Protocol, Optional, Sequence

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore[assignment]


BoundingBox = Tuple[int, int, int, int]
Frame = np.ndarray[Any, np.dtype[np.uint8]]


class Detector(Protocol):
    """
    Protocol for detector implementations.

    Concrete implementations must provide a :meth:`detect` method.
    """

    def detect(self, frame: Frame) -> List[Tuple[BoundingBox, float]]:
        """
        Run person/player detection on a single frame.

        Args:
            frame: Input image in BGR format (OpenCV convention).

        Returns:
            A list of tuples (bbox, confidence) where:
                - bbox is (x1, y1, x2, y2) in pixel coordinates.
                - confidence is a float between 0 and 1.
        """
        ...


@dataclass
class YoloDetector:
    """
    Simple YOLO-based detector wrapper.

    Attributes:
        model_path: Path to the YOLO weights file.
        conf_threshold: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.
        classes: Optional list of class IDs to keep.

    Note:
        This implementation assumes the `ultralytics` package is installed.
        Install via `pip install ultralytics` and download appropriate weights.
    """
    model_path: str
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45
    classes: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Install it with `pip install ultralytics`."
            )
        self._model = YOLO(self.model_path)

    def detect(self, frame: Frame) -> List[Tuple[BoundingBox, float]]:
        """
        Run YOLO detection on the given frame and return player bounding boxes.

        This implementation filters for the 'person' class (COCO id 0)
        by default, unless `classes` is provided explicitly.
        """
        results = self._model(  # type: ignore[misc]
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )
        detections: List[Tuple[BoundingBox, float]] = []

        for result in results:  # type: ignore[attr-defined]
            boxes = result.boxes  # type: ignore[attr-defined]
            if boxes is None:
                continue
            cls = boxes.cls.cpu().numpy().astype(int)  # type: ignore[attr-defined]
            conf = boxes.conf.cpu().numpy()  # type: ignore[attr-defined]
            xyxy = boxes.xyxy.cpu().numpy()  # type: ignore[attr-defined]
            for c, score, box in zip(cls, conf, xyxy):  # type: ignore[arg-type]
                # If no explicit class filter is provided, keep only 'person' (COCO id 0).
                if self.classes is None and c != 0:
                    continue
                coords = box.astype(int)  # type: ignore[attr-defined]
                x1: int = int(coords[0])  # type: ignore[arg-type]
                y1: int = int(coords[1])  # type: ignore[arg-type]
                x2: int = int(coords[2])  # type: ignore[arg-type]
                y2: int = int(coords[3])  # type: ignore[arg-type]
                bbox: BoundingBox = (x1, y1, x2, y2)
                detections.append((bbox, float(score)))  # type: ignore[arg-type]
        return detections


class DummyDetector:
    """
    Fallback detector that returns no detections.

    This is mainly useful for debugging the rest of the pipeline when a
    real detector is not yet available.

    TODO: Replace this with a simple heuristic detector (e.g., color threshold)
    for synthetic demo videos.
    """

    def detect(self, frame: Frame) -> List[Tuple[BoundingBox, float]]:
        return []


def create_detector(
    model_path: str,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.45,
    classes: Optional[Sequence[int]] = None,
    use_dummy_if_unavailable: bool = True,
) -> Detector:
    """
    Factory for creating a detector instance.

    Args:
        model_path: Path to YOLO weights.
        conf_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS inside the detector.
        classes: Optional sequence of class IDs to keep from the detector.
        use_dummy_if_unavailable: If True, fall back to :class:`DummyDetector`
            when YOLO cannot be imported.

    Returns:
        An object implementing the :class:`Detector` protocol.
    """
    if YOLO is None:
        if use_dummy_if_unavailable:
            # TODO: Log a warning about using the dummy detector.
            return DummyDetector()
        raise ImportError(
            "ultralytics is required for YOLO detection. Install it or enable dummy mode."
        )
    return YoloDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
    )


def detect_players(
    detector: Detector, frame: Frame
) -> List[Tuple[BoundingBox, float, int]]:
    """
    High-level helper to detect players in a frame.

    Args:
        detector: A detector implementing :class:`Detector` protocol.
        frame: BGR image.

    Returns:
        List of (bbox, confidence, class_id). For the current YOLO
        configuration this is typically class_id == 0 (person).
    """
    base_detections = detector.detect(frame)
    return [(bbox, score, 0) for bbox, score in base_detections]
