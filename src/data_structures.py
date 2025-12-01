"""
Core data structures for player tracking, calibration, and downstream metrics.

The dataset currently assumes a single static, high-angle tactical camera that
shows nearly the entire pitch for an 11v11 match. Goalkeepers can sit near the
top/bottom edges and occasionally move out of frame; the ``visible`` flag on
track points should be set to ``False`` (or omitted for that frame) when a
player is outside the view or heavily occluded so distance is not accumulated
for those gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Pixel bounding box in absolute image coordinates (top-left origin).
BBoxXYXY = Tuple[float, float, float, float]


@dataclass
class FrameInfo:
    """
    Index and timestamp for a video frame.

    Attributes:
        frame_index: Zero-based frame index.
        timestamp_s: Timestamp in seconds (frame_index / fps).
    """

    frame_index: int
    timestamp_s: float

    @classmethod
    def from_index(cls, frame_index: int, fps: float) -> "FrameInfo":
        """
        Convenience constructor computing the timestamp from frame index and fps.
        """
        return cls(frame_index=frame_index, timestamp_s=frame_index / fps)


@dataclass
class PlayerDetection:
    """
    Single player detection in pixel coordinates.

    Attributes:
        frame_index: Zero-based frame index.
        bbox_xyxy: Bounding box (x1, y1, x2, y2) in pixels.
        confidence: Detector confidence score.
        team_id: Optional team label from a classifier (e.g., 0/1).
        is_goalkeeper: True when the detection is a goalkeeper.
    """

    frame_index: int
    bbox_xyxy: BBoxXYXY
    confidence: float
    team_id: int | None
    is_goalkeeper: bool = False


@dataclass
class PlayerTrackPoint:
    """
    Tracked player position in pixel coordinates.

    Attributes:
        frame_index: Zero-based frame index.
        player_id: Stable track identifier.
        u: Horizontal pixel position (e.g., bbox center x).
        v: Vertical pixel position (e.g., bbox center y).
        visible: False when the player is outside the frame or heavily occluded.
    """

    frame_index: int
    player_id: int
    u: float
    v: float
    visible: bool = True


@dataclass
class PlayerPitchPoint:
    """
    Tracked player position in metric pitch coordinates (meters).

    Attributes:
        frame_index: Zero-based frame index.
        player_id: Stable track identifier.
        x: Horizontal position on the pitch in meters.
        y: Vertical position on the pitch in meters.
        visible: False when the player is outside the frame or heavily occluded.
    """

    frame_index: int
    player_id: int
    x: float
    y: float
    visible: bool = True


@dataclass
class PitchPointUV:
    """
    Image coordinate in pixels (origin at top-left of the video frame).
    """

    u: float
    v: float


@dataclass
class PitchPointXY:
    """
    World/pitch coordinate in meters (origin and axes depend on calibration).
    """

    x: float
    y: float


@dataclass
class PitchMeta:
    """
    Metadata describing the pitch layout.

    Attributes:
        pitch_type: Format label, e.g. "11v11".
        length_m: Pitch length in meters (goal-to-goal).
        width_m: Pitch width in meters (touchline to touchline).
        description: Optional free text (e.g., stadium, camera notes).
    """

    pitch_type: str
    length_m: float
    width_m: float
    description: str | None = None

