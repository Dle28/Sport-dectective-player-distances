"""
Demonstration of the data flow: video -> pixel tracks -> calibration -> pitch tracks -> distance.

Assumptions:
- Static, high-angle 11v11 camera showing the full pitch (data/raw/5ph.mp4).
- A calibration file exists at data/calibration/5ph_calibration.json with >=4 point pairs.
- Goalkeepers may leave the frame; mark visibility accordingly.
"""

from __future__ import annotations

from pathlib import Path

from src.video_io import VideoReader
from src.data_structures import PlayerTrackPoint
from src.tracking_data import PixelTrackStore
from src.calibration_data import load_calibration
from src.calibration import PitchCalibrator
from src.transform_tracks import (
    save_pitch_tracks_csv,
    save_pitch_tracks_json,
    transform_tracks_to_pitch,
)
from src.metrics import compute_distance_per_player, summarize_distances_km


def main() -> None:
    video_path = Path("data/raw/5ph.mp4")
    calibration_path = Path("data/calibration/5ph_calibration.json")
    if not video_path.exists():
        raise FileNotFoundError(f"Expected video at {video_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Expected calibration file at {calibration_path}. "
            "Create it with at least four (u,v) <-> (x,y) pairs for this camera."
        )

    # 1) Read video and collect a toy set of pixel tracks (replace with detector + tracker).
    reader = VideoReader(video_path, stride=10)
    px_store = PixelTrackStore(fps=reader.fps, video_path=video_path)
    max_frames = 200
    for frame_idx, _frame in reader:
        if frame_idx > max_frames:
            break
        # Synthetic example: two players drifting horizontally; visible throughout.
        px_store.add_point(
            PlayerTrackPoint(
                frame_index=frame_idx,
                player_id=3,
                u=reader.width / 2 + 0.5 * frame_idx,
                v=reader.height / 2,
                visible=True,
            )
        )
        px_store.add_point(
            PlayerTrackPoint(
                frame_index=frame_idx,
                player_id=5,
                u=reader.width / 2 - 0.4 * frame_idx,
                v=reader.height / 2 + 20,
                visible=True,
            )
        )
    tracks_px = px_store.tracks
    px_store.to_json(Path("data/tracks_px/5ph_demo_tracks.json"))
    px_store.to_csv(Path("data/tracks_px/5ph_demo_tracks.csv"))

    # 2) Load calibration and build homography.
    calib_set = load_calibration(calibration_path)
    calibrator = PitchCalibrator.from_calibration(calib_set)

    # 3) Transform tracks to pitch coordinates (meters) and persist.
    tracks_xy = transform_tracks_to_pitch(tracks_px, calibrator)
    save_pitch_tracks_json(tracks_xy, Path("data/tracks_xy/5ph_demo_tracks.json"), fps=reader.fps, video_path=video_path)
    save_pitch_tracks_csv(tracks_xy, Path("data/tracks_xy/5ph_demo_tracks.csv"), fps=reader.fps, video_path=video_path)

    # 4) Compute per-player distances in meters/kilometers.
    distances_m = compute_distance_per_player(tracks_xy, fps=reader.fps)
    distances_km = summarize_distances_km(distances_m)

    print("Distance covered (m):", distances_m)
    print("Distance covered (km):", distances_km)


if __name__ == "__main__":
    main()

