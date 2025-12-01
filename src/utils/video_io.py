"""
Video reading and writing utilities.

This module wraps basic OpenCV video I/O to make the rest of the codebase
cleaner and to centralize any future improvements (e.g., handling variable
frame rates, logging, or progress bars).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Tuple

import cv2
import numpy as np

Frame = np.ndarray[Any, np.dtype[np.uint8]]


@dataclass
class VideoReader:
    """
    Simple OpenCV-based video reader.

    Attributes:
        path: Path to the video file.
    """

    path: Path

    def __post_init__(self) -> None:
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

    @property
    def fps(self) -> float:
        """
        Frames per second of the video.
        """
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)

    @property
    def frame_count(self) -> int:
        """
        Total number of frames in the video (if known).
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def frames(self, stride: int = 1) -> Generator[Tuple[int, Frame], None, None]:
        """
        Iterate over frames in the video.

        Args:
            stride: Process every N-th frame.

        Yields:
            (frame_index, frame) where frame_index is zero-based.
        """
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx % stride == 0:
                yield idx, frame  # type: ignore[misc]
            idx += 1

    def release(self) -> None:
        """
        Release the underlying video capture object.
        """
        self._cap.release()


def open_video(path: Path) -> VideoReader:
    """
    Convenience function to create a :class:`VideoReader`.
    """
    return VideoReader(path=path)


@dataclass
class VideoWriter:
    """
    Simple OpenCV-based video writer for annotated outputs.

    Attributes:
        path: Path to the output video file.
        fps: Target frames per second.
        frame_size: (width, height) in pixels.
        codec: FourCC codec string, e.g., "mp4v", "XVID".
    """

    path: Path
    fps: float
    frame_size: Tuple[int, int]
    codec: str = "mp4v"

    def __post_init__(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore[attr-defined]
        w, h = self.frame_size
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))  # type: ignore[arg-type]
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer: {self.path}")

    def write(self, frame: Frame) -> None:
        """
        Write a single BGR frame to the video.
        """
        self._writer.write(frame)

    def release(self) -> None:
        """
        Release the underlying video writer.
        """
        self._writer.release()


def open_video_writer(path: Path, fps: float, frame_size: Tuple[int, int], codec: str = "mp4v") -> VideoWriter:
    """
    Convenience function to create a :class:`VideoWriter`.
    """
    return VideoWriter(path=path, fps=fps, frame_size=frame_size, codec=codec)
