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

from typing import Dict, List

import cv2

from .config import DEFAULT_CONFIG, Config
from .detection import create_detector
from .tracking import create_tracker, TrackID
from .team_classifier import create_team_classifier
from .calibration import load_calibrator, Point2D
from .metrics import compute_player_metrics
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
    tracker = create_tracker(
        tracker_type=config.tracker_type,
        max_age=config.tracking_max_age,
        iou_threshold=config.tracking_iou_threshold,
    )
    team_classifier = create_team_classifier()
    calibrator = load_calibrator(config.calibration_file) if config.calibration_file else None

    video = open_video(config.input_video)

    # Track positions in pitch coordinates for each player over time.
    track_positions: Dict[TrackID, List[Point2D]] = {}

    # Optional: write an annotated video for debugging.
    output_video_path = config.output_dir / "annotated.mp4"
    writer = None

    try:
        for _, frame in video.frames(stride=config.frame_stride):
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame=frame)

            # Classify each track into a team.
            team_labels = team_classifier.classify_tracks(frame, tracks)

            # Convert track positions from image to pitch coordinates.
            if calibrator is not None:
                for track_id, bbox in tracks.items():
                    pt = calibrator.bbox_center_to_pitch(bbox)
                    track_positions.setdefault(track_id, []).append(pt)

            # Draw overlays for visualization/debugging.
            if config.enable_visualization or config.write_annotated_video:
                vis = draw_tracks(frame, tracks, team_labels)
            else:
                vis = frame

            # Lazy-init video writer once we know frame size.
            if config.write_annotated_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = open_video_writer(
                        output_video_path,
                        fps=video.fps,
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
