"""
Distance and basic movement metrics computed in pitch coordinates (meters).

Distances are only valid after calibration; never use pixel distances. The video
dataset uses a static, high-angle camera where nearly the full pitch is visible.
Goalkeepers may leave the frame; mark those segments ``visible=False`` to avoid
adding distance during gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isnan
from typing import Dict, List, Mapping, Tuple

from .data_structures import PlayerPitchPoint

Point2D = Tuple[float, float]


@dataclass
class PlayerDistance:
    """
    Aggregate distance metrics for a player.
    """

    player_id: int
    distance_m: float

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000.0


@dataclass
class PlayerMetrics:
    """
    Compatibility struct for pre-existing pipeline code.
    """

    total_distance_m: float
    positions: List[Point2D]

    @property
    def total_distance_km(self) -> float:
        return self.total_distance_m / 1000.0


@dataclass
class BallMetrics:
    total_distance_m: float
    positions: List[Point2D]


def compute_distance(positions: List[Point2D]) -> float:
    """
    Compute total path length from a sequence of (x, y) points.
    """
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(positions[:-1], positions[1:]):
        total += hypot(x1 - x0, y1 - y0)
    return total


def _valid_segment(p0: PlayerPitchPoint, p1: PlayerPitchPoint) -> bool:
    """
    Check visibility and finite coordinates before computing displacement.
    """
    if not (p0.visible and p1.visible):
        return False
    if any(isnan(v) for v in (p0.x, p0.y, p1.x, p1.y)):
        return False
    return True


def compute_distance_per_player(
    tracks_xy: Mapping[int, List[PlayerPitchPoint]],
    fps: float = 25.0,
    max_speed_m_per_s: float = 10.0,
) -> Dict[int, float]:
    """
    Compute total distance covered per player in meters.

    Rules:
    - Segments with ``visible=False`` are skipped.
    - Speed is derived from frame delta / fps; segments exceeding
      ``max_speed_m_per_s`` are treated as outliers and ignored.
    """
    distances: Dict[int, float] = {}
    for player_id, pts in tracks_xy.items():
        if len(pts) < 2:
            distances[player_id] = 0.0
            continue
        pts_sorted = sorted(pts, key=lambda p: p.frame_index)
        total = 0.0
        for p0, p1 in zip(pts_sorted[:-1], pts_sorted[1:]):
            if p1.frame_index <= p0.frame_index:
                continue
            if not _valid_segment(p0, p1):
                continue
            dt = (p1.frame_index - p0.frame_index) / fps
            if dt <= 0:
                continue
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            dist = hypot(dx, dy)
            speed = dist / dt
            if speed > max_speed_m_per_s:
                continue
            total += dist
        distances[player_id] = total
    return distances


def summarize_distances_km(distances_m: Mapping[int, float]) -> Dict[int, float]:
    """
    Convert per-player distances (meters) to kilometers.
    """
    return {pid: dist_m / 1000.0 for pid, dist_m in distances_m.items()}


def compute_player_metrics(
    track_positions: Mapping[int, List[Point2D]],
) -> Dict[int, PlayerMetrics]:
    """
    Compatibility helper: compute per-player distance for plain (x, y) tuples.
    """
    metrics: Dict[int, PlayerMetrics] = {}
    for track_id, positions in track_positions.items():
        dist = compute_distance(positions)
        metrics[track_id] = PlayerMetrics(total_distance_m=dist, positions=positions)
    return metrics


def compute_ball_metrics(
    ball_positions: Mapping[int, List[Point2D]],
) -> Dict[int, BallMetrics]:
    """
    Compatibility helper mirroring compute_player_metrics.
    """
    metrics: Dict[int, BallMetrics] = {}
    for ball_id, positions in ball_positions.items():
        dist = compute_distance(positions)
        metrics[ball_id] = BallMetrics(total_distance_m=dist, positions=positions)
    return metrics
