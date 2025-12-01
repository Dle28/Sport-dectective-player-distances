"""
Data containers and helpers for image-to-pitch calibration.

Designed for a static, high-angle camera where the full 11v11 pitch is visible.
Collect a handful of pixel/pitch correspondences, store them here, and reuse
them when mapping tracks from pixels to meters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Mapping

from .data_structures import PitchMeta, PitchPointUV, PitchPointXY


@dataclass
class CalibrationSample:
    """
    Single correspondence between an image point (pixels) and pitch point (meters).
    """

    image_point: PitchPointUV
    world_point: PitchPointXY


@dataclass
class CalibrationSet:
    """
    Collection of calibration samples tied to a specific pitch description.

    Attributes:
        pitch_meta: Pitch dimensions/type the calibration targets.
        samples: List of pixel-to-pitch correspondences.
    """

    pitch_meta: PitchMeta
    samples: List[CalibrationSample]

    def to_dict(self) -> Mapping[str, object]:
        """
        Convert to a JSON-serializable dictionary.
        """
        return {
            "pitch_meta": asdict(self.pitch_meta),
            "samples": [
                {
                    "image_point": asdict(sample.image_point),
                    "world_point": asdict(sample.world_point),
                }
                for sample in self.samples
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CalibrationSet":
        """
        Construct a CalibrationSet from a mapping (e.g., loaded JSON).
        """
        pitch_meta = PitchMeta(**data["pitch_meta"])  # type: ignore[arg-type]
        samples: List[CalibrationSample] = []
        for sample_data in data.get("samples", []):  # type: ignore[assignment]
            sample_map: Mapping[str, object] = sample_data  # type: ignore[assignment]
            image_point = PitchPointUV(**sample_map["image_point"])  # type: ignore[arg-type]
            world_point = PitchPointXY(**sample_map["world_point"])  # type: ignore[arg-type]
            samples.append(CalibrationSample(image_point=image_point, world_point=world_point))
        return cls(pitch_meta=pitch_meta, samples=samples)


def load_calibration(path: Path | str) -> CalibrationSet:
    """
    Load a calibration set from ``path`` (e.g., data/calibration/calibration.json).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibrationSet.from_dict(data)


def save_calibration(calibration: CalibrationSet, path: Path | str) -> None:
    """
    Save a calibration set to JSON at ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(calibration.to_dict(), f, indent=2)
