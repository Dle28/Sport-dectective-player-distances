"""
Visualization helpers for drawing detections, tracks, and debug views.

Includes simple helpers to draw track centers in pixel space and to render a
synthetic top-down pitch view in meters. The dataset uses a static, high-angle
camera; goalkeepers may hug the edges or leave the frame, so invisible points
should be skipped.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple, cast

import cv2
import numpy as np

from .tracking import TrackID
from .detection import BoundingBox
from .team_classifier import TeamLabel
from .data_structures import PitchMeta, PlayerPitchPoint, PlayerTrackPoint

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
            if label == "A":
                color = (0, 0, 255)
            elif label == "B":
                color = (255, 0, 0)
            elif label.lower().startswith("ref"):
                color = (0, 215, 255)  # gold/yellow for referee
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


def draw_track_points_px(
    frame: Frame,
    track_points: Iterable[PlayerTrackPoint],
    radius: int = 4,
) -> Frame:
    """
    Draw track points (u, v) on a frame; skips invisible points.
    """
    vis = frame.copy()
    for pt in track_points:
        if not pt.visible:
            continue
        color = _id_color(pt.player_id)
        cv2.circle(vis, (int(pt.u), int(pt.v)), radius, color, -1)
        cv2.putText(
            vis,
            str(pt.player_id),
            (int(pt.u) + radius + 2, int(pt.v)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def render_topdown_tracks(
    tracks_xy: Dict[int, Iterable[PlayerPitchPoint]],
    pitch_meta: PitchMeta,
    output_size: Tuple[int, int] = (1050, 680),
    radius: int = 4,
) -> Frame:
    """
    Render a synthetic top-down view of track points in pitch coordinates.

    Args:
        tracks_xy: Mapping player_id -> iterable of PlayerPitchPoint.
        pitch_meta: Pitch dimensions used to scale coordinates.
        output_size: (width_px, height_px) of the synthetic pitch image.
    """
    width_px, height_px = output_size
    canvas = np.full((height_px, width_px, 3), (0, 100, 0), dtype=np.uint8)
    scale_x = width_px / float(pitch_meta.length_m)
    scale_y = height_px / float(pitch_meta.width_m)

    for player_id, points in tracks_xy.items():
        color = _id_color(player_id)
        for pt in points:
            if not pt.visible or np.isnan(pt.x) or np.isnan(pt.y):
                continue
            x_px = int(pt.x * scale_x)
            y_px = int(pt.y * scale_y)
            cv2.circle(canvas, (x_px, y_px), radius, color, -1)
    return canvas


def _id_color(idx: int) -> tuple[int, int, int]:
    """
    Deterministic pseudo-random color for a given ID.
    """
    np.random.seed(idx)
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


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
