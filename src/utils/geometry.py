"""
Geometry helper functions for coordinate transforms and distances.

This module contains reusable geometric operations that are shared
between calibration, metrics, and visualization components.

TODO: Add functions for line/polygon intersections, offside line
projections, and zone definitions.
"""

from __future__ import annotations

# No typing imports needed

import numpy as np

from ..calibration import PointXY


def euclidean_distance(a: PointXY, b: PointXY) -> float:
    """
    Compute Euclidean distance between two 2D points.
    """
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))

