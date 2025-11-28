"""
Computation of player movement metrics such as distance and speed.

This module takes tracked positions over time (in pitch coordinates) and
computes aggregate statistics per player.

Distances are computed in meters and can optionally be converted to km.

TODO: Extend metrics to include instantaneous and average speed,
accelerations, and high-intensity running statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .tracking import TrackID
from .calibration import Point2D


@dataclass
class PlayerMetrics:
    """
    Stores per-player movement metrics.

    Attributes:
        total_distance_m: Total distance covered in meters.
        positions: List of (X, Y) positions in meters over time.
    """

    total_distance_m: float
    positions: List[Point2D]

    @property
    def total_distance_km(self) -> float:
        """
        Total distance covered in kilometers.
        """
        return self.total_distance_m / 1000.0


@dataclass
class BallMetrics:
    """
    Stores ball movement metrics.

    Attributes:
        total_distance_m: Total distance the ball traveled in meters.
        positions: List of (X, Y) positions in meters over time.
    """

    total_distance_m: float
    positions: List[Point2D]


def compute_distance(positions: List[Point2D]) -> float:
    """
    Compute total path length from a sequence of positions.
    """
    if len(positions) < 2:
        return 0.0
    pts = np.array(positions, dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(segment_lengths.sum())


def compute_player_metrics(
    track_positions: Dict[TrackID, List[Point2D]]
) -> Dict[TrackID, PlayerMetrics]:
    """
    Compute basic metrics for each player.

    Args:
        track_positions: Mapping from track ID to list of pitch positions over time.

    Returns:
        Mapping from track ID to :class:`PlayerMetrics`.
    """
    metrics: Dict[TrackID, PlayerMetrics] = {}
    for track_id, positions in track_positions.items():
        dist = compute_distance(positions)
        metrics[track_id] = PlayerMetrics(total_distance_m=dist, positions=positions)
    return metrics


def compute_ball_metrics(
    ball_positions: Dict[TrackID, List[Point2D]]
) -> Dict[TrackID, BallMetrics]:
    """
    Compute basic metrics for each ball track.

    Args:
        ball_positions: Mapping from ball track ID to list of pitch positions.

    Returns:
        Mapping from ball track ID to :class:`BallMetrics`.
    """
    metrics: Dict[TrackID, BallMetrics] = {}
    for ball_id, positions in ball_positions.items():
        dist = compute_distance(positions)
        metrics[ball_id] = BallMetrics(total_distance_m=dist, positions=positions)
    return metrics
