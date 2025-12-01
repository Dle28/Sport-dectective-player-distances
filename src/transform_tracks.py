"""
Conversion utilities for mapping pixel-space tracks to pitch-space tracks.

For each :class:`PlayerTrackPoint`:
- If ``visible`` is False, emit a :class:`PlayerPitchPoint` with ``x=y=nan`` and
  ``visible=False`` so downstream metrics can skip gaps (common for goalkeepers
  near frame edges).
- Otherwise map (u, v) pixels to (x, y) meters using the provided calibrator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from .calibration import PitchCalibrator
from .data_structures import PlayerPitchPoint, PlayerTrackPoint
from .tracking_data import PitchTrackStore


def transform_tracks_to_pitch(
    tracks_px: Dict[int, List[PlayerTrackPoint]],
    calibrator: PitchCalibrator,
) -> Dict[int, List[PlayerPitchPoint]]:
    """
    Convert pixel-space tracks into pitch-space tracks using a homography.
    """
    tracks_xy: Dict[int, List[PlayerPitchPoint]] = {}
    for player_id, points in tracks_px.items():
        tracks_xy[player_id] = []
        for p in points:
            if not p.visible:
                tracks_xy[player_id].append(
                    PlayerPitchPoint(
                        frame_index=p.frame_index,
                        player_id=player_id,
                        x=math.nan,
                        y=math.nan,
                        visible=False,
                    )
                )
                continue
            x, y = calibrator.image_to_pitch(p.u, p.v)
            tracks_xy[player_id].append(
                PlayerPitchPoint(
                    frame_index=p.frame_index,
                    player_id=player_id,
                    x=x,
                    y=y,
                    visible=True,
                )
            )
    return tracks_xy


def save_pitch_tracks_json(
    tracks_xy: Dict[int, List[PlayerPitchPoint]],
    path: Path,
    fps: float,
    video_path: str | Path | None = None,
) -> None:
    """
    Persist pitch-space tracks to JSON using the standard schema.
    """
    store = PitchTrackStore(fps=fps, video_path=video_path, tracks=tracks_xy)
    store.to_json(path)


def save_pitch_tracks_csv(
    tracks_xy: Dict[int, List[PlayerPitchPoint]],
    path: Path,
    fps: float,
    video_path: str | Path | None = None,
) -> None:
    """
    Persist pitch-space tracks to CSV using the standard schema.
    """
    store = PitchTrackStore(fps=fps, video_path=video_path, tracks=tracks_xy)
    store.to_csv(path)

