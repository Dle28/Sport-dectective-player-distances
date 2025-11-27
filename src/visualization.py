"""
Visualization helpers for drawing detections, tracks, and team labels.

This module provides simple utilities to overlay bounding boxes, track IDs,
team labels, and trajectories on video frames.

TODO: Add richer overlays such as speed, distance covered, and tactical
zones, and support compositing heatmaps with transparency.
"""

from __future__ import annotations

from typing import Any, Dict, cast

import cv2
import numpy as np

from .tracking import TrackID
from .detection import BoundingBox
from .team_classifier import TeamLabel

Frame = np.ndarray[Any, np.dtype[np.uint8]]


def draw_tracks(
    frame: Frame,
    tracks: Dict[TrackID, BoundingBox],
    team_labels: Dict[TrackID, TeamLabel] | None = None,
) -> Frame:
    """
    Draw bounding boxes and track IDs (and optional team labels) on a frame.
    """
    vis = frame.copy()
    for track_id, bbox in tracks.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)
        if team_labels and track_id in team_labels:
            label = team_labels[track_id]
            color = (0, 0, 255) if label == "A" else (255, 0, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = str(track_id)
        if team_labels and track_id in team_labels:
            text += f" ({team_labels[track_id]})"
        cv2.putText(
            vis,
            text,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def overlay_heatmap_on_pitch(
    heatmap_img: Frame,
    alpha: float = 0.6,
    pitch_color: tuple[int, int, int] = (0, 100, 0),
) -> Frame:
    """
    Overlay a heatmap on a synthetic pitch background.

    This is a simple way to visualize the heatmap without having the
    original camera view.
    """
    h, w = heatmap_img.shape[:2]
    pitch = np.full((h, w, 3), pitch_color, dtype=np.uint8)
    overlay = cv2.addWeighted(pitch, 1 - alpha, heatmap_img, alpha, 0)
    return cast(Frame, overlay)
