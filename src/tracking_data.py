"""
Utilities for storing and serializing player tracks in pixel or pitch coordinates.

The structures are designed for a static, high-angle 11v11 camera where most of
the pitch is visible at all times. Goalkeepers (or any player) can move out of
frame; mark those frames with ``visible=False`` or omit them so distance
aggregation can skip gaps instead of adding spurious travel.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, TypeVar, cast

from .data_structures import PlayerPitchPoint, PlayerTrackPoint, PitchMeta

PixelTracks = Dict[int, List[PlayerTrackPoint]]
PitchTracks = Dict[int, List[PlayerPitchPoint]]
T = TypeVar("T")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sort_points(points: Sequence[T], key: str = "frame_index") -> List[T]:
    return sorted(points, key=lambda p: getattr(p, key))  # type: ignore[arg-type]


@dataclass
class PixelTrackStore:
    """
    Container for player tracks in pixel coordinates.

    Attributes:
        fps: Video frame rate used to derive timestamps elsewhere.
        video_path: Optional absolute or relative video path reference.
        tracks: Mapping of player_id -> list of PlayerTrackPoint.
    """

    fps: float
    video_path: str | Path | None = None
    tracks: PixelTracks = field(default_factory=dict)  # type: ignore[misc]

    def add_point(self, point: PlayerTrackPoint) -> None:
        """
        Append a track point; frames with ``visible=False`` are kept so gaps are explicit.
        """
        self.tracks.setdefault(point.player_id, []).append(point)

    def to_json(self, path: Path) -> None:
        """
        Serialize tracks to JSON with string keys for player IDs.
        """
        _ensure_parent(path)
        payload: Dict[str, Any] = {
            "fps": self.fps,
            "video_path": str(self.video_path) if self.video_path is not None else None,
            "tracks": {
                str(player_id): [
                    {
                        "frame_index": p.frame_index,
                        "u": p.u,
                        "v": p.v,
                        "visible": bool(p.visible),
                    }
                    for p in _sort_points(points)
                ]
                for player_id, points in self.tracks.items()
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "PixelTrackStore":
        """
        Load tracks from a JSON file produced by :meth:`to_json`.
        """
        with path.open("r", encoding="utf-8") as f:
            data: Mapping[str, object] = json.load(f)

        fps = float(cast(float, data.get("fps", 0.0)))
        video_path = cast(str | Path | None, data.get("video_path"))

        tracks_raw: Dict[str, Any] = cast(Dict[str, Any], data.get("tracks", {}))
        tracks: PixelTracks = {}
        for player_id_str, entries in tracks_raw.items():
            player_id = int(player_id_str)
            tracks[player_id] = []
            for entry in entries:
                entry_map: Dict[str, Any] = cast(Dict[str, Any], entry)
                tracks[player_id].append(
                    PlayerTrackPoint(
                        frame_index=int(cast(int, entry_map["frame_index"])),
                        player_id=player_id,
                        u=float(cast(float, entry_map["u"])),
                        v=float(cast(float, entry_map["v"])),
                        visible=_to_bool(cast(bool, entry_map.get("visible", True))),
                    )
                )

        return cls(fps=fps, video_path=video_path, tracks=tracks)

    def to_csv(self, path: Path) -> None:
        """
        Serialize tracks to CSV with columns: frame_index,player_id,u,v,visible.
        """
        _ensure_parent(path)
        fieldnames = ["frame_index", "player_id", "u", "v", "visible"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for player_id, points in self.tracks.items():
                for p in _sort_points(points):
                    writer.writerow(
                        {
                            "frame_index": p.frame_index,
                            "player_id": player_id,
                            "u": p.u,
                            "v": p.v,
                            "visible": int(bool(p.visible)),
                        }
                    )

    @classmethod
    def from_csv(cls, path: Path, fps: float, video_path: str | Path | None = None) -> "PixelTrackStore":
        """
        Load tracks from CSV. FPS/video path are provided because CSV stores only rows.
        """
        tracks: PixelTracks = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_id = int(row["player_id"])
                tracks.setdefault(player_id, []).append(
                    PlayerTrackPoint(
                        frame_index=int(row["frame_index"]),
                        player_id=player_id,
                        u=float(row["u"]),
                        v=float(row["v"]),
                        visible=_to_bool(row.get("visible", "1")),
                    )
                )
        return cls(fps=fps, video_path=video_path, tracks=tracks)


@dataclass
class PitchTrackStore:
    """
    Container for player tracks mapped into pitch coordinates (meters).

    Attributes:
        fps: Video frame rate used to derive timestamps elsewhere.
        video_path: Optional absolute or relative video path reference.
        tracks: Mapping of player_id -> list of PlayerPitchPoint.
    """

    fps: float
    video_path: str | Path | None = None
    pitch_meta: PitchMeta | None = None
    tracks: PitchTracks = field(default_factory=dict)  # type: ignore[misc]

    def add_point(self, point: PlayerPitchPoint) -> None:
        """
        Append a metric-space track point.
        """
        self.tracks.setdefault(point.player_id, []).append(point)

    def to_json(self, path: Path) -> None:
        """
        Serialize pitch-space tracks to JSON.
        """
        _ensure_parent(path)
        payload: Dict[str, Any] = {
            "fps": self.fps,
            "video_path": str(self.video_path) if self.video_path is not None else None,
            "pitch_meta": asdict(self.pitch_meta) if self.pitch_meta is not None else None,
            "tracks": {
                str(player_id): [
                    {
                        "frame_index": p.frame_index,
                        "x": p.x,
                        "y": p.y,
                        "visible": bool(p.visible),
                    }
                    for p in _sort_points(points)
                ]
                for player_id, points in self.tracks.items()
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "PitchTrackStore":
        """
        Load pitch-space tracks from JSON.
        """
        with path.open("r", encoding="utf-8") as f:
            data: Mapping[str, object] = json.load(f)

        fps = float(cast(float, data.get("fps", 0.0)))
        video_path = cast(str | Path | None, data.get("video_path"))
        pitch_meta_data = data.get("pitch_meta")
        pitch_meta = PitchMeta(**pitch_meta_data) if isinstance(pitch_meta_data, dict) else None

        tracks_raw: Dict[str, Any] = cast(Dict[str, Any], data.get("tracks", {}))
        tracks: PitchTracks = {}
        for player_id_str, entries in tracks_raw.items():
            player_id = int(player_id_str)
            tracks[player_id] = []
            for entry in entries:
                entry_map: Dict[str, Any] = cast(Dict[str, Any], entry)
                tracks[player_id].append(
                    PlayerPitchPoint(
                        frame_index=int(cast(int, entry_map["frame_index"])),
                        player_id=player_id,
                        x=float(cast(float, entry_map["x"])),
                        y=float(cast(float, entry_map["y"])),
                        visible=_to_bool(cast(bool, entry_map.get("visible", True))),
                    )
                )

        return cls(fps=fps, video_path=video_path, pitch_meta=pitch_meta, tracks=tracks)

    def to_csv(self, path: Path) -> None:
        """
        Serialize pitch-space tracks to CSV with columns: frame_index,player_id,x,y,visible.
        """
        _ensure_parent(path)
        fieldnames = ["frame_index", "player_id", "x", "y", "visible"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for player_id, points in self.tracks.items():
                for p in _sort_points(points):
                    writer.writerow(
                        {
                            "frame_index": p.frame_index,
                            "player_id": player_id,
                            "x": p.x,
                            "y": p.y,
                            "visible": int(bool(p.visible)),
                        }
                    )

    @classmethod
    def from_csv(cls, path: Path, fps: float, video_path: str | Path | None = None) -> "PitchTrackStore":
        """
        Load pitch-space tracks from CSV. FPS/video path are provided externally.
        """
        tracks: PitchTracks = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_id = int(row["player_id"])
                tracks.setdefault(player_id, []).append(
                    PlayerPitchPoint(
                        frame_index=int(row["frame_index"]),
                        player_id=player_id,
                        x=float(row["x"]),
                        y=float(row["y"]),
                        visible=_to_bool(row.get("visible", "1")),
                    )
                )
        return cls(fps=fps, video_path=video_path, tracks=tracks)
