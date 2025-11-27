"""
Heatmap generation for visualizing player movement.

This module generates simple 2D heatmaps over the pitch for each player,
based on their positions in pitch coordinates.

The output can be saved as an image or used as an overlay in video.

TODO: Support kernel density estimation for smoother heatmaps and
normalization across entire matches or multiple players.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from .calibration import Point2D

HeatmapArray = np.ndarray[Any, np.dtype[np.uint8]]


def create_heatmap(
    positions: List[Point2D],
    pitch_size: tuple[float, float],
    resolution: tuple[int, int] = (1050, 680),
) -> HeatmapArray:
    """
    Create a simple heatmap image from a list of positions.

    Args:
        positions: List of (X, Y) pitch coordinates in meters.
        pitch_size: (length_m, width_m) of the pitch.
        resolution: Output image resolution in pixels (width, height).

    Returns:
        Heatmap image as a BGR uint8 array.
    """
    length_m, width_m = pitch_size
    width_px, height_px = resolution
    heatmap = np.zeros((height_px, width_px), dtype=np.float32)

    if not positions:
        empty_heatmap: HeatmapArray = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)  # type: ignore[assignment]
        return empty_heatmap

    for X, Y in positions:
        # Normalize to [0, 1] range
        u = np.clip(X / length_m, 0, 1)
        v = np.clip(Y / width_m, 0, 1)
        x_px = int(u * (width_px - 1))
        y_px = int(v * (height_px - 1))
        heatmap[y_px, x_px] += 1.0

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=10, sigmaY=10)
    heatmap_norm = heatmap / (heatmap.max() + 1e-6)
    heatmap_img_raw = (heatmap_norm * 255).astype(np.uint8)
    heatmap_img: HeatmapArray = cv2.applyColorMap(heatmap_img_raw, cv2.COLORMAP_JET)  # type: ignore[assignment]
    return heatmap_img  # type: ignore[return-value]


def save_heatmap(path: Path, heatmap_img: HeatmapArray) -> None:
    """
    Save a heatmap image to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), heatmap_img)
