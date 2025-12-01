"""
Homography-based calibration for mapping image pixels to pitch meters.

This module assumes a fixed, high-angle tactical camera where the full 11v11
pitch is visible (goalkeepers may hug the edges and occasionally leave the
frame). Distances must be computed **after** mapping to pitch coordinates; never
use pixel distances directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .calibration_data import CalibrationSample, CalibrationSet, load_calibration
from .data_structures import PitchMeta, PitchPointUV, PitchPointXY

PointUV = Tuple[float, float]
PointXY = Tuple[float, float]
Point2D = Tuple[float, float]
Homography = np.ndarray


def _compute_homography(uv_points: Sequence[PointUV], xy_points: Sequence[PointXY]) -> Homography:  # type: ignore[misc]
    """
    Estimate a 3x3 homography mapping image (u, v) -> pitch (x, y) using RANSAC.
    """
    uv = np.asarray(uv_points, dtype=np.float32)
    xy = np.asarray(xy_points, dtype=np.float32)
    if uv.shape != xy.shape or uv.shape[0] < 4:
        raise ValueError("Homography requires >=4 point pairs with matching shapes.")
    H, _ = cv2.findHomography(uv, xy, method=cv2.RANSAC)  # type: ignore[misc]
    if H is None:  # type: ignore[unreachable]
        raise ValueError("Failed to compute homography; check calibration points.")
    return H.astype(np.float32)


@dataclass
class PitchCalibrator:
    """
    Applies a planar homography to map pixel coordinates to pitch meters.

    Attributes:
        calibration: CalibrationSet containing pitch metadata and point pairs.
        H: 3x3 homography matrix mapping (u, v, 1) -> (x, y, w).
    """

    calibration: CalibrationSet
    H: Homography  # type: ignore[misc]

    @classmethod
    def from_calibration(cls, calibration: CalibrationSet) -> "PitchCalibrator":
        """
        Build a calibrator from a CalibrationSet.
        """
        uv_points = [(s.image_point.u, s.image_point.v) for s in calibration.samples]
        xy_points = [(s.world_point.x, s.world_point.y) for s in calibration.samples]
        H: Homography = _compute_homography(uv_points, xy_points)  # type: ignore[misc]
        return cls(calibration=calibration, H=H)

    def image_to_pitch(self, u: float, v: float) -> Tuple[float, float]:
        """
        Map a single pixel coordinate (u, v) to pitch meters (x, y).
        """
        pts = np.asarray([[[u, v]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)  # type: ignore[arg-type]
        x, y = mapped[0]
        return float(x), float(y)

    def batch_image_to_pitch(self, points_uv: Iterable[PointUV]) -> List[PointXY]:
        """
        Map a batch of pixel coordinates to pitch meters.
        """
        pts_list = list(points_uv)
        if not pts_list:
            return []
        pts = np.asarray(pts_list, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)  # type: ignore[arg-type]
        return [(float(x), float(y)) for x, y in mapped]


def load_calibrator(
    calibration_file: Path,
    pitch_length_m: float,
    pitch_width_m: float,
) -> PitchCalibrator:
    """
    Compatibility helper for loading a calibrator from disk.

    Supports both the new CalibrationSet schema and the legacy
    ``{\"image_points\": [[...]], \"pitch_points\": [[...]]}`` layout.
    """
    with calibration_file.open("r", encoding="utf-8") as f:
        import json

        data = json.load(f)

    if "samples" in data or "pitch_meta" in data:
        calib_set = load_calibration(calibration_file)
        return PitchCalibrator.from_calibration(calib_set)

    # Legacy format.
    image_points = data["image_points"]
    pitch_points = data["pitch_points"]
    pitch_meta = PitchMeta(
        pitch_type=str(data.get("pitch_type", "unknown")),
        length_m=float(data.get("pitch_length_m", pitch_length_m)),
        width_m=float(data.get("pitch_width_m", pitch_width_m)),
    )
    samples = [
        CalibrationSample(
            image_point=PitchPointUV(u=float(u), v=float(v)),
            world_point=PitchPointXY(x=float(x), y=float(y)),
        )
        for (u, v), (x, y) in zip(image_points, pitch_points)
    ]
    calib_set = CalibrationSet(pitch_meta=pitch_meta, samples=samples)
    return PitchCalibrator.from_calibration(calib_set)
