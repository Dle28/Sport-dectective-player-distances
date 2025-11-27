"""
Object tracking utilities for associating detections across frames.

This module defines a minimal tracking interface and provides a simple
IOU-based tracker that can be replaced with more advanced methods such
as DeepSORT or ByteTrack.

TODO: Integrate a proper tracker (e.g., Norfair, DeepSORT, or ByteTrack)
for more robust long-term tracking and re-identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Protocol, Optional

import numpy as np

from .detection import BoundingBox

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    DeepSort = None  # type: ignore[assignment,misc]


TrackID = int
Frame = np.ndarray[Any, np.dtype[np.uint8]]


class Tracker(Protocol):
    """
    Protocol for tracker implementations.

    Concrete implementations must provide an :meth:`update` method.
    """

    def update(
        self,
        detections: List[Tuple[BoundingBox, float]],
        frame: Optional[Frame] = None,
    ) -> Dict[TrackID, BoundingBox]:
        """
        Update tracker state with detections from the current frame.

        Args:
            detections: List of (bbox, confidence) tuples.

        Returns:
            A mapping from persistent track IDs to their current bounding boxes.
        """
        ...


def iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter_area / float(area_a + area_b - inter_area)


@dataclass
class SimpleIouTracker:
    """
    Minimal IOU-based tracker assigning persistent IDs.

    This tracker performs greedy data association based on IoU between
    detections and existing tracks. It is not meant for production use
    but suffices as a minimal working example.

    Attributes:
        max_age: Number of frames a track is kept without matching new detections.
        iou_threshold: Minimum IoU for association.
    """

    max_age: int = 30
    iou_threshold: float = 0.3

    _next_id: int = field(default=1, init=False)
    _tracks: Dict[TrackID, BoundingBox] = field(default_factory=dict, init=False)  # type: ignore[misc]
    _age: Dict[TrackID, int] = field(default_factory=dict, init=False)  # type: ignore[misc]

    def update(
        self,
        detections: List[Tuple[BoundingBox, float]],
        frame: Optional[Frame] = None,
    ) -> Dict[TrackID, BoundingBox]:
        """
        Update tracks with detections from the current frame.
        """
        assigned_tracks: Dict[TrackID, BoundingBox] = {}

        # Keep track timestamp/age
        for track_id in list(self._tracks.keys()):
            self._age[track_id] += 1

        # Greedy matching: for each detection, find the best track.
        for bbox, _score in detections:
            best_iou = self.iou_threshold
            best_track_id: TrackID | None = None
            for track_id, track_box in self._tracks.items():
                current_iou = iou(bbox, track_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self._tracks[best_track_id] = bbox
                self._age[best_track_id] = 0
                assigned_tracks[best_track_id] = bbox
            else:
                # Create new track
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = bbox
                self._age[track_id] = 0
                assigned_tracks[track_id] = bbox

        # Remove old tracks
        for track_id in list(self._tracks.keys()):
            if self._age[track_id] > self.max_age:
                del self._tracks[track_id]
                del self._age[track_id]

        return dict(self._tracks)


@dataclass
class DeepSortTracker:
    """
    Wrapper around the `deep-sort-realtime` implementation.

    This tracker uses appearance features and motion to maintain identities
    more robustly than the simple IoU tracker, when the optional dependency
    is installed.

    Attributes:
        max_age: Maximum number of frames to keep a track without detections.
        n_init: Number of frames before a track is confirmed.
        nms_max_overlap: NMS overlap used internally by DeepSORT.
    """

    max_age: int = 30
    n_init: int = 3
    nms_max_overlap: float = 1.0

    def __post_init__(self) -> None:
        if DeepSort is None:
            raise ImportError(
                "deep-sort-realtime is not installed. Install it with "
                "`pip install deep-sort-realtime` or use the simple IOU tracker."
            )
        self._tracker = DeepSort(  # type: ignore[misc]
            max_age=self.max_age,
            n_init=self.n_init,
            nms_max_overlap=self.nms_max_overlap,
        )

    def update(
        self,
        detections: List[Tuple[BoundingBox, float]],
        frame: Optional[Frame] = None,
    ) -> Dict[TrackID, BoundingBox]:
        """
        Update DeepSORT tracker with detections from the current frame.
        """
        if frame is None:
            raise ValueError("DeepSortTracker.update requires the current frame.")

        ds_detections = []  # type: ignore[var-annotated]
        for bbox, score in detections:
            x1, y1, x2, y2 = bbox
            ds_detections.append([x1, y1, x2, y2, score, 0])  # type: ignore[attr-defined] # class 0: player/person

        tracks = self._tracker.update_tracks(ds_detections, frame=frame)  # type: ignore[attr-defined]
        result: Dict[TrackID, BoundingBox] = {}
        for track in tracks:  # type: ignore[attr-defined]
            if not track.is_confirmed():  # type: ignore[attr-defined]
                continue
            l, t, r, b = track.to_ltrb()  # type: ignore[attr-defined]
            bbox: BoundingBox = (int(l), int(t), int(r), int(b))  # type: ignore[arg-type]
            result[int(track.track_id)] = bbox  # type: ignore[arg-type,attr-defined]
        return result


def create_tracker(
    tracker_type: str = "simple_iou",
    max_age: int = 30,
    iou_threshold: float = 0.3,
) -> Tracker:
    """
    Factory for creating a tracker instance.

    Args:
        tracker_type: Which backend to use: "simple_iou" or "deepsort".
        max_age: Maximum number of frames to keep tracks without detections.
        iou_threshold: Association IoU threshold for the simple IOU tracker.
    """
    tracker_type = tracker_type.lower()
    if tracker_type == "deepsort":
        return DeepSortTracker(max_age=max_age)
    # Default: simple IOU tracker
    return SimpleIouTracker(max_age=max_age, iou_threshold=iou_threshold)
