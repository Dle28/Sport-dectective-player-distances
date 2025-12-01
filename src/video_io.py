"""
Thin OpenCV wrapper for reading football match videos.

The primary dataset uses a fixed, high-angle tactical camera that shows almost
the full 11v11 pitch. This class exposes only the essentials (fps, resolution,
frame count) and an iterator over frames so the rest of the codebase stays free
of OpenCV specifics. Use ``stride`` to subsample frames when experimenting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Tuple
from types import TracebackType

import cv2
import numpy as np

Frame = np.ndarray[Any, np.dtype[np.uint8]]


@dataclass
class VideoReader:
    """
    Read video frames with optional subsampling.

    Attributes:
        path: Path to the input video.
        stride: Keep every N-th frame (1 = keep all).
    """

    path: Path
    stride: int = 1

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

    @property
    def fps(self) -> float:
        """
        Frames per second reported by the video container (defaults to 25 if missing).
        """
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)

    @property
    def frame_count(self) -> int:
        """
        Total number of frames (0 if unknown).
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def width(self) -> int:
        """
        Frame width in pixels.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    @property
    def height(self) -> int:
        """
        Frame height in pixels.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def __iter__(self) -> Generator[Tuple[int, Frame], None, None]:
        """
        Iterate over frames as (frame_index, frame).
        """
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx % self.stride == 0:
                yield idx, frame  # type: ignore[misc]
            idx += 1

    def release(self) -> None:
        """
        Release the underlying OpenCV handle.
        """
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

