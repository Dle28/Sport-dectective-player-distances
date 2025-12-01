"""
Camera-to-pitch calibration and coordinate transformations.

This module deals with mapping pixel coordinates (image space) to metric
coordinates on the pitch using a planar homography.

The key idea is that **all distances must be measured on the pitch in
meters**, not directly in pixels. A player who moves the same number of
pixels near the camera and far away does not necessarily cover the same
physical distance because of perspective effects. The homography uses
manually selected point correspondences to map from image coordinates
``(u, v)`` to pitch coordinates ``(x, y)`` in meters.

Expected calibration file format (JSON):

    {
        "image_points": [[u1, v1], [u2, v2], ...],
        "pitch_points": [[x1, y1], [x2, y2], ...]
    }

Each video (and camera viewpoint) should have its own calibration file
and its own pitch dimensions. This allows the same code to work for
5-a-side, 7-a-side, and 11-a-side pitches without mixing distances
between different field sizes.

The design also leaves room for a future semi-automatic mode where the
pitch lines (sidelines, boxes, center circle) are detected automatically
and used to infer the scale and homography.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]
HomographyMatrix = np.ndarray[Any, np.dtype[np.float32]]


def compute_homography(
    image_points: Sequence[Point2D],
    pitch_points: Sequence[Point2D],
) -> HomographyMatrix:
    """
    Estimate a 3x3 homography matrix from corresponding points.

    Args:
        image_points: Points in image/pixel coordinates ``(u, v)``.
        pitch_points: Corresponding points in pitch coordinates ``(x, y)``
            in meters, using any consistent pitch coordinate system.

    Returns:
        A ``3x3`` homography matrix ``H`` that maps ``(u, v, 1)`` to
        ``(x, y, w)`` in homogeneous coordinates.

    Raises:
        ValueError: If homography estimation fails (e.g., degenerate
        or insufficient point configuration).
    """
    img = np.asarray(image_points, dtype=np.float32)
    pitch = np.asarray(pitch_points, dtype=np.float32)
    if img.shape != pitch.shape or img.shape[0] < 4:
        raise ValueError(
            "Homography requires at least 4 point pairs with matching shapes."
        )
    H, _ = cv2.findHomography(img, pitch, method=cv2.RANSAC)  # type: ignore[misc]
    # cv2.findHomography can theoretically return None, but typing says otherwise
    if H is None:  # type: ignore[unreachable]  # pragma: no cover - defensive check
        raise ValueError("Failed to compute homography; check your points.")
    return np.asarray(H, dtype=np.float32)


def image_to_pitch(
    points_uv: Iterable[Point2D],
    H: HomographyMatrix,
) -> List[Point2D]:
    """
    Transform a batch of image points to pitch coordinates.

    Args:
        points_uv: Iterable of image coordinates ``(u, v)`` in pixels.
        H: Homography matrix mapping image to pitch coordinates.

    Returns:
        List of pitch coordinates ``(x, y)`` in meters.
    """
    pts_list = list(points_uv)
    if not pts_list:
        return []
    pts = np.asarray(pts_list, dtype=np.float32).reshape(-1, 1, 2)
    pts_pitch = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts_pitch]


@dataclass
class PitchCalibrator:
    """
    Handles estimation and application of a planar homography for a pitch.

    Attributes:
        H: 3x3 homography matrix mapping image coordinates to pitch coordinates.
        pitch_length_m: Length of the pitch in meters.
        pitch_width_m: Width of the pitch in meters.

    The world coordinate system is assumed to be:
        - Origin ``(0, 0)`` at one pitch corner.
        - ``x`` axis along the pitch length.
        - ``y`` axis along the pitch width.

    Distances for players and the ball should **always** be computed after
    mapping to this metric coordinate system. Direct distances in pixels
    are not physically meaningful because of perspective distortion.
    """

    H: HomographyMatrix
    pitch_length_m: float
    pitch_width_m: float

    @classmethod
    def from_correspondences(
        cls,
        image_points: List[Point2D],
        pitch_points: List[Point2D],
        pitch_length_m: float,
        pitch_width_m: float,
    ) -> "PitchCalibrator":
        """
        Create a calibrator from point correspondences.

        Args:
            image_points: At least four points in image coordinates.
            pitch_points: Corresponding points in pitch coordinates (meters).
            pitch_length_m: Length of the pitch in meters.
            pitch_width_m: Width of the pitch in meters.

        Returns:
            A :class:`PitchCalibrator` instance with the estimated homography.
        """
        H = compute_homography(image_points, pitch_points)
        return cls(H=H, pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m)

    @classmethod
    def from_json(
        cls,
        path: Path,
        pitch_length_m: float | None = None,
        pitch_width_m: float | None = None,
    ) -> "PitchCalibrator":
        """
        Load point correspondences from JSON and compute the homography.

        The JSON file should contain ``image_points`` and ``pitch_points``.
        Pitch dimensions can either be provided here in the JSON (optional)
        or passed explicitly via ``pitch_length_m`` and ``pitch_width_m``
        from a config file or CLI arguments.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_points = data["image_points"]
        pitch_points = data["pitch_points"]

        # Allow optional pitch dimensions inside the JSON as a fallback.
        json_length = data.get("pitch_length_m") or data.get("pitch_length")
        json_width = data.get("pitch_width_m") or data.get("pitch_width")

        length = float(pitch_length_m if pitch_length_m is not None else json_length or 0.0)
        width = float(pitch_width_m if pitch_width_m is not None else json_width or 0.0)

        # As a last resort, if dimensions are still zero, infer them from
        # the max extents of the world points. This is only a heuristic.
        if (length <= 0.0 or width <= 0.0) and pitch_points:
            xs = [p[0] for p in pitch_points]
            ys = [p[1] for p in pitch_points]
            length = float(max(xs) - min(xs))
            width = float(max(ys) - min(ys))

        if length <= 0.0 or width <= 0.0:
            raise ValueError(
                "Pitch dimensions could not be determined. Provide pitch_length_m "
                "and pitch_width_m via config/CLI or in the calibration JSON."
            )

        return cls.from_correspondences(
            image_points=image_points,
            pitch_points=pitch_points,
            pitch_length_m=length,
            pitch_width_m=width,
        )

    def image_point_to_pitch(self, point: Point2D) -> Point2D:
        """
        Map a single image point ``(u, v)`` to pitch coordinates ``(x, y)``.
        """
        x, y = image_to_pitch([point], self.H)[0]
        return x, y

    def bbox_center_to_pitch(self, bbox: Tuple[int, int, int, int]) -> Point2D:
        """
        Convenience helper: map bounding-box center to pitch coordinates.

        Args:
            bbox: Bounding box ``(x1, y1, x2, y2)`` in pixels.

        Returns:
            Center of the box in pitch coordinates ``(x, y)`` in meters.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return self.image_point_to_pitch((cx, cy))


def load_calibrator(
    calibration_file: Path,
    pitch_length_m: float,
    pitch_width_m: float,
) -> PitchCalibrator:
    """
    Load a :class:`PitchCalibrator` from a JSON file.

    The calibration file contains the point correspondences, while the
    pitch dimensions (length/width in meters) are provided by the user
    via a config file or CLI arguments. This ensures that distances are
    always expressed in the correct units for the specific pitch size.

    TODO: Support caching computed homographies and multiple camera setups.
    """
    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration file {calibration_file} not found. "
            f"Create it with manually selected correspondences."
        )
    return PitchCalibrator.from_json(
        calibration_file,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
    )
