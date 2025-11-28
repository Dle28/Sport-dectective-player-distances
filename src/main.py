"""
Entry point for the football player tracking pipeline.

This script demonstrates a minimal working example that:
- Loads a broadcast-style video.
- Runs player detection and simple IOU-based tracking.
- Assigns team labels based on jersey color.
- Maps positions to pitch coordinates via a homography.
- Computes per-player distance covered.
- Saves a heatmap image for one selected player.

Requirements (install via pip):
    pip install opencv-python numpy ultralytics

Before running:
    1. Place your input video at the path configured in `Config`.
    2. Provide a calibration JSON file with image/pitch correspondences.
    3. Optionally, download YOLO weights (e.g., yolov8n.pt) to `models/`.

Run:
    python -m src.main
"""

from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Dict, List, Tuple

import cv2

from .config import DEFAULT_CONFIG, Config
from .detection import BoundingBox, YoloDetector, create_detector
from .tracking import create_tracker, TrackID
from .team_classifier import create_team_classifier
from .calibration import load_calibrator, Point2D
from .metrics import compute_ball_metrics, compute_player_metrics
from .heatmap import create_heatmap, save_heatmap
from .visualization import draw_tracks, overlay_heatmap_on_pitch
from .utils.video_io import open_video, open_video_writer


def run_pipeline(config: Config = DEFAULT_CONFIG) -> None:
    """
    Run the full detection → tracking → metrics → heatmap pipeline.
    """
    config.ensure_output_dirs()

    # Initialize components
    detector = create_detector(
        model_path=str(config.yolo_model_path),
        conf_threshold=config.detection_conf_threshold,
        iou_threshold=config.detection_iou_threshold,
        classes=config.detection_classes,
        use_dummy_if_unavailable=True,
    )
    player_tracker = create_tracker(
        tracker_type=config.tracker_type,
        max_age=config.tracking_max_age,
        iou_threshold=config.tracking_iou_threshold,
    )
    ball_tracker = create_tracker(
        tracker_type=config.tracker_type,
        max_age=config.tracking_max_age,
        iou_threshold=config.tracking_iou_threshold,
    )
    team_classifier = create_team_classifier()
    calibrator = load_calibrator(config.calibration_file) if config.calibration_file else None

    video = open_video(config.input_video)
    fps = video.fps

    # Track positions in pitch coordinates for each player and ball over time.
    track_positions: Dict[TrackID, List[Point2D]] = {}
    ball_positions: Dict[TrackID, List[Point2D]] = {}

    # Approximate ball possession: number of frames each player is closest
    # to the ball within a small radius in pitch space.
    possession_frames: Dict[TrackID, int] = defaultdict(int)

    # Optional: write an annotated video for debugging.
    output_video_path = config.output_dir / "annotated.mp4"
    writer = None

    try:
        for _, frame in video.frames(stride=config.frame_stride):
            # Run detection; if using YOLO, get both players and ball.
            if isinstance(detector, YoloDetector):
                player_detections, ball_detections = detector.detect_players_and_ball(frame)
            else:
                player_detections = detector.detect(frame)
                ball_detections: List[Tuple[BoundingBox, float]] = []

            player_tracks = player_tracker.update(player_detections, frame=frame)
            ball_tracks = ball_tracker.update(ball_detections, frame=frame) if ball_detections else {}

            # Classify each track into a team.
            team_labels = team_classifier.classify_tracks(frame, player_tracks)

            # Convert track positions from image to pitch coordinates.
            if calibrator is not None:
                frame_player_positions: Dict[TrackID, Point2D] = {}
                for track_id, bbox in player_tracks.items():
                    pt = calibrator.bbox_center_to_pitch(bbox)
                    track_positions.setdefault(track_id, []).append(pt)
                    frame_player_positions[track_id] = pt

                frame_ball_positions: Dict[TrackID, Point2D] = {}
                for ball_id, bbox in ball_tracks.items():
                    pt = calibrator.bbox_center_to_pitch(bbox)
                    ball_positions.setdefault(ball_id, []).append(pt)
                    frame_ball_positions[ball_id] = pt

                # Update possession statistics: which player is closest to
                # the ball this frame (within 2 meters).
                if frame_ball_positions and frame_player_positions:
                    _, ball_pt = next(iter(frame_ball_positions.items()))
                    closest_id: TrackID | None = None
                    closest_dist = float("inf")
                    for track_id, player_pt in frame_player_positions.items():
                        dist = hypot(player_pt[0] - ball_pt[0], player_pt[1] - ball_pt[1])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_id = track_id
                    if closest_id is not None and closest_dist < 2.0:
                        possession_frames[closest_id] += 1

            # Draw overlays for visualization/debugging.
            if config.enable_visualization or config.write_annotated_video:
                vis = draw_tracks(frame, player_tracks, team_labels)
            else:
                vis = frame

            # Lazy-init video writer once we know frame size.
            if config.write_annotated_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = open_video_writer(
                        output_video_path,
                        fps=fps,
                        frame_size=(w, h),
                        codec="mp4v",
                    )
                writer.write(vis)

            # OPTIONAL: show preview window
            # cv2.imshow("Tracking", vis)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    finally:
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    # Compute metrics once all frames have been processed.
    metrics = compute_player_metrics(track_positions)
    _ball_metrics = compute_ball_metrics(ball_positions) if ball_positions else {}

    # Convert possession frame counts into seconds.
    possession_seconds: Dict[TrackID, float] = {}
    for track_id, frame_count in possession_frames.items():
        possession_seconds[track_id] = frame_count / fps

    # Save a simple heatmap for one player (e.g., the first track).
    if metrics:
        first_track_id = next(iter(metrics.keys()))
        player_metrics = metrics[first_track_id]
        heatmap_img = create_heatmap(
            positions=player_metrics.positions,
            pitch_size=(config.pitch_length_m, config.pitch_width_m),
        )
        heatmap_overlay = overlay_heatmap_on_pitch(heatmap_img)

        heatmap_path = config.heatmaps_dir / f"player_{first_track_id}_heatmap.png"
        overlay_path = config.heatmaps_dir / f"player_{first_track_id}_heatmap_overlay.png"

        save_heatmap(heatmap_path, heatmap_img)
        save_heatmap(overlay_path, heatmap_overlay)

    # Optionally, save metrics to disk as JSON/CSV.
    # TODO: Implement JSON/CSV serialization for per-player stats.


def main() -> None:
    """
    CLI entrypoint for running the pipeline with default configuration.

    TODO: Add argparse to override config parameters from the command line,
    such as input video path or model weights.
    """
    run_pipeline()


if __name__ == "__main__":
    main()
