"""
Camera-to-pitch calibration and coordinate transformations.

This module deals with mapping pixel coordinates (image space) to metric
coordinates on the pitch using a homography.

The homography is estimated from manually selected correspondences between
image points and known pitch coordinates in meters.

Expected calibration file format (JSON):
    {
        "image_points": [[x, y], ...],
        "pitch_points": [[X, Y], ...]
    }

TODO: Add utilities for interactive point selection using OpenCV mouse callbacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]
HomographyMatrix = np.ndarray[Any, np.dtype[np.float32]]


@dataclass
class HomographyCalibrator:
    """
    Handles estimation and application of a planar homography.

    Attributes:
        H: 3x3 homography matrix mapping image → pitch coordinates.
    """

    H: HomographyMatrix

    @classmethod
    def from_correspondences(
        cls, image_points: List[Point2D], pitch_points: List[Point2D]
    ) -> "HomographyCalibrator":
        """
        Estimate homography from corresponding points.
        """
        img = np.array(image_points, dtype=np.float32)
        pitch = np.array(pitch_points, dtype=np.float32)
        H, _ = cv2.findHomography(img, pitch, method=cv2.RANSAC)
        if H is None:  # type: ignore[reportUnnecessaryComparison]
            raise ValueError("Failed to compute homography; check your points.")
        H_array: HomographyMatrix = np.asarray(H, dtype=np.float32)
        return cls(H=H_array)

    @classmethod
    def from_json(cls, path: Path) -> "HomographyCalibrator":
        """
        Load homography correspondences from a JSON file and compute H.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        image_points = data["image_points"]
        pitch_points = data["pitch_points"]
        return cls.from_correspondences(image_points, pitch_points)

    def image_to_pitch(self, point: Point2D) -> Point2D:
        """
        Map an image point (x, y) to pitch coordinates (X, Y) in meters.
        """
        x, y = point
        src = np.array([[x, y, 1.0]], dtype=np.float32).T
        dst = self.H @ src
        dst /= dst[2, 0] + 1e-8
        X, Y = float(dst[0, 0]), float(dst[1, 0])
        return X, Y

    def bbox_center_to_pitch(self, bbox: Tuple[int, int, int, int]) -> Point2D:
        """
        Convenience helper: map bounding-box center to pitch coordinates.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return self.image_to_pitch((cx, cy))


def load_calibrator(calibration_file: Path) -> HomographyCalibrator:
    """
    Load a :class:`HomographyCalibrator` from a JSON file.

    TODO: Support caching computed homographies and multiple camera setups.
    """
    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration file {calibration_file} not found. "
            f"Create it with manually selected correspondences."
        )
    return HomographyCalibrator.from_json(calibration_file)
