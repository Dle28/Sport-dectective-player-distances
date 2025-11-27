"""
Configuration utilities for the football player tracking pipeline.

This module centralizes configurable parameters such as:
- Paths to input videos, model weights, and output directories.
- Detection and tracking thresholds.
- Pitch dimensions and calibration file paths.

The default `Config` dataclass is meant for experimentation and can be
extended or replaced by loading configuration from YAML/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class Config:
    """
    High-level configuration for a single experiment / run.

    Attributes:
        input_video: Path to the broadcast-style input video.
        output_dir: Directory where outputs (stats, heatmaps, videos) are stored.
        yolo_model_path: Path to YOLO model weights.
        detection_conf_threshold: Minimum confidence for detected players.
        detection_iou_threshold: IoU threshold for detector NMS.
        detection_classes: Optional list of class IDs to keep.
        tracking_max_age: Maximum number of frames to keep a lost track alive.
        tracking_iou_threshold: IOU threshold for data association.
        tracker_type: Name of tracker backend ("simple_iou", "deepsort").
        pitch_length_m: Length of the pitch in meters.
        pitch_width_m: Width of the pitch in meters.
        calibration_file: Path to a file containing image-to-pitch correspondences.
        frame_stride: Process every N-th frame for speed.
        enable_visualization: Whether to draw overlays.
        write_annotated_video: Whether to save annotated video to disk.
        draw_trajectories: Whether to overlay short trajectories.
    """

    input_video: Path = Path("data") / "sample_match.mp4"
    output_dir: Path = Path("outputs")
    yolo_model_path: Path = Path("models") / "yolov8n.pt"

    detection_conf_threshold: float = 0.4
    detection_iou_threshold: float = 0.45
    detection_classes: Optional[List[int]] = None

    # Simple tracker parameters; these are generic and not tied to a specific library.
    tracking_max_age: int = 30
    tracking_iou_threshold: float = 0.3
    tracker_type: str = "simple_iou"  # or "deepsort"

    # Standard football pitch dimensions (can be adjusted).
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    calibration_file: Optional[Path] = Path("data") / "calibration_points.json"

    frame_stride: int = 1

    enable_visualization: bool = True
    write_annotated_video: bool = True
    draw_trajectories: bool = False

    # TODO: Add options for choosing tracker device (CPU/GPU, half precision).

    def ensure_output_dirs(self) -> None:
        """
        Create output directories if they do not exist.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def heatmaps_dir(self) -> Path:
        """
        Directory for saving generated heatmaps.
        """
        path = self.output_dir / "heatmaps"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def stats_dir(self) -> Path:
        """
        Directory for saving per-player statistics in JSON/CSV format.
        """
        path = self.output_dir / "stats"
        path.mkdir(parents=True, exist_ok=True)
        return path


DEFAULT_CONFIG = Config()
