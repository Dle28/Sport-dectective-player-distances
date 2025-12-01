"""
Entry point for the football player tracking pipeline.

This script wires together:
- Player detection (YOLO or dummy)
- Multi-object tracking
- Pitch calibration and coordinate conversion
- Distance and simple possession metrics
- Optional heatmap generation

It can be run from the command line, for example:

    python -m src.main \\
        --video_path data/sample_match.mp4 \\
        --pitch_type 11 \\
        --pitch_length 105 \\
        --pitch_width 68 \\
        --calib_points data/calibration_points.json \\
        --output_dir outputs/

Pitch dimensions can also be configured via ``config.yaml``; see the README.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, cast

import cv2

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

from .config import DEFAULT_CONFIG, Config
from .detection import Detection, YoloDetector, create_detector
from .tracking import TrackID, create_player_tracker
from .team_classifier import TeamLabel, create_team_classifier
from .calibration import PointXY, load_calibrator
from .data_structures import PlayerPitchPoint, PitchMeta
from .metrics import (
    compute_distance_per_player,
    compute_player_metrics,
)
from .tracking_data import PitchTrackStore
from .heatmap import create_heatmap, save_heatmap
from .visualization import draw_tracks, overlay_heatmap_on_pitch
from .utils.video_io import open_video, open_video_writer


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file if it exists.

    The file is optional; when missing, an empty dict is returned.
    """
    if not config_path.exists():
        return {}
    if yaml is None:
        raise ImportError(
            "pyyaml is required to load config.yaml. "
            "Install it with `pip install pyyaml` or via requirements.txt."
        )
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}  # type: ignore[no-untyped-call]
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping.")
    return cast(Dict[str, Any], data)


def _get_pitch_size_from_config(
    cfg: Dict[str, Any],
    pitch_type: int,
) -> Tuple[float, float]:
    """
    Look up default pitch dimensions for a given ``pitch_type``.

    Values are read from ``config.yaml`` under ``pitch_types``, with keys
    like ``\"5\"``, ``\"7\"``, and ``\"11\"``. If they are not found,
    reasonable hard-coded defaults are used.
    """
    pitch_cfg: Dict[str, Any] = cast(Dict[str, Any], cfg.get("pitch_types", {}) or {})
    key = str(pitch_type)
    if key in pitch_cfg:
        dims: Dict[str, Any] = cast(Dict[str, Any], pitch_cfg.get(key, {}) or {})
        length = float(cast(Any, dims.get("length_m")) or cast(Any, dims.get("length")) or 0.0)
        width = float(cast(Any, dims.get("width_m")) or cast(Any, dims.get("width")) or 0.0)
        if length > 0.0 and width > 0.0:
            return length, width

    # Simple hard-coded defaults as a fallback.
    if pitch_type == 5:
        return 40.0, 20.0
    if pitch_type == 7:
        return 50.0, 30.0
    # Default to a standard 11-a-side pitch.
    return 105.0, 68.0


def _get_model_path_from_config(cfg: Dict[str, Any], default_path: Path) -> Path:
    """
    Resolve YOLO model path, optionally overridden by ``config.yaml``.
    """
    model_cfg: Dict[str, Any] = cast(Dict[str, Any], cfg.get("model", {}) or {})
    path_str: Optional[str] = cast(Optional[str], model_cfg.get("yolo_model_path"))
    if path_str:
        return Path(path_str)
    return default_path


def build_config_from_args(args: argparse.Namespace) -> Config:
    """
    Construct a :class:`Config` instance from CLI arguments and optional YAML.
    """
    # Start from code defaults.
    config = Config()

    # Load optional YAML config near the project root.
    default_config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config_path = Path(args.config) if args.config is not None else default_config_path
    yaml_cfg = _load_yaml_config(config_path) if config_path.exists() else {}

    # Paths
    if args.video_path is not None:
        config.input_video = Path(args.video_path)
    if args.output_dir is not None:
        config.output_dir = Path(args.output_dir)

    # Model path: CLI > YAML > default.
    if args.model_path is not None:
        config.yolo_model_path = Path(args.model_path)
    else:
        config.yolo_model_path = _get_model_path_from_config(
            yaml_cfg, config.yolo_model_path
        )

    # Calibration file.
    if args.calib_points is not None:
        config.calibration_file = Path(args.calib_points)

    # Pitch dimensions: pitch_type from YAML, then CLI overrides.
    if args.pitch_type is not None:
        length, width = _get_pitch_size_from_config(yaml_cfg, args.pitch_type)
        config.pitch_length_m = length
        config.pitch_width_m = width

    if args.pitch_length is not None:
        config.pitch_length_m = float(args.pitch_length)
    if args.pitch_width is not None:
        config.pitch_width_m = float(args.pitch_width)
    if getattr(args, "frame_stride", None) is not None:
        config.frame_stride = max(1, int(args.frame_stride))
    if getattr(args, "max_frames", None) is not None:
        config.max_frames = int(args.max_frames)
    if getattr(args, "force_yolo", False):
        config.use_dummy_detector = False
    if getattr(args, "sample_interval_s", None) is not None:
        config.sample_interval_s = float(args.sample_interval_s)

    return config


def run_pipeline(config: Config = DEFAULT_CONFIG) -> None:
    """
    Run the full detection + tracking + metrics + visualization pipeline.
    """
    config.ensure_output_dirs()

    # Initialize components
    detector = create_detector(
        model_path=str(config.yolo_model_path),
        conf_threshold=config.detection_conf_threshold,
        iou_threshold=config.detection_iou_threshold,
        classes=config.detection_classes,
        use_dummy_if_unavailable=config.use_dummy_detector,
    )
    player_tracker = create_player_tracker(
        tracker_type=config.tracker_type,
        max_age=config.tracking_max_age,
        iou_threshold=config.tracking_iou_threshold,
    )
    ball_tracker = create_player_tracker(
        tracker_type=config.tracker_type,
        max_age=config.tracking_max_age,
        iou_threshold=config.tracking_iou_threshold,
    )
    team_classifier = create_team_classifier()
    calibrator = (
        load_calibrator(
            config.calibration_file,
            pitch_length_m=config.pitch_length_m,
            pitch_width_m=config.pitch_width_m,
        )
        if config.calibration_file is not None
        else None
    )

    video = open_video(config.input_video)
    fps = video.fps
    # If requested, derive frame_stride from sample interval.
    if config.sample_interval_s is not None:
        config.frame_stride = max(1, int(round(fps * config.sample_interval_s)))

    # Track positions in pitch coordinates for each player and ball over time.
    track_positions: Dict[TrackID, List[PointXY]] = {}
    ball_positions: Dict[TrackID, List[PointXY]] = {}
    # Track positions with frame indices for robust distance computation.
    track_points_xy: Dict[TrackID, List[PlayerPitchPoint]] = {}
    # Optional: raw pixel-space positions (frame_idx, u, v) for debugging/export.
    pixel_track_points_uv: Dict[TrackID, List[Tuple[int, float, float]]] = {}

    # Approximate ball possession: number of frames each player is closest
    # to the ball within a small radius in pitch space.
    possession_frames: Dict[TrackID, int] = defaultdict(int)

    # Keep a stable team label per track (last seen label wins).
    track_team_labels: Dict[TrackID, TeamLabel] = {}

    # Optional: write an annotated video for debugging.
    output_video_path = config.output_dir / "annotated.mp4"
    writer = None

    try:
        for frame_idx, frame in video.frames(stride=config.frame_stride):
            if config.max_frames is not None and frame_idx >= config.max_frames:
                break
            # Run detection; if using YOLO, get both players and ball.
            if isinstance(detector, YoloDetector):
                player_detections, ball_detections = detector.detect_players_and_ball(
                    frame,
                    player_class_ids=config.player_class_ids,
                    ball_class_ids=config.ball_class_ids,
                )
            else:
                player_detections = detector.detect(frame)
                ball_detections: List[Detection] = []

            player_tracks = player_tracker.update(player_detections, frame=frame)
            ball_tracks = (
                ball_tracker.update(ball_detections, frame=frame)
                if ball_detections
                else {}
            )

            # Filter tracks to only those inside the pitch bounds (with small margin) if calibration is available.
            filtered_player_tracks: Dict[TrackID, Tuple[int, int, int, int]] = {}
            frame_player_centers_uv: Dict[TrackID, Tuple[float, float]] = {}
            margin_m = 3.0
            for track_id, bbox in player_tracks.items():
                x1, y1, x2, y2 = bbox
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                center_uv = (float(u), float(v))

                if calibrator is not None:
                    px, py = calibrator.image_to_pitch(center_uv[0], center_uv[1])
                    if (
                        px < -margin_m
                        or px > config.pitch_length_m + margin_m
                        or py < -margin_m
                        or py > config.pitch_width_m + margin_m
                    ):
                        # Skip people clearly outside the pitch (spectators/benches).
                        continue

                frame_player_centers_uv[track_id] = center_uv
                filtered_player_tracks[track_id] = bbox
                pixel_track_points_uv.setdefault(track_id, []).append(
                    (frame_idx, center_uv[0], center_uv[1])
                )

            # Classify each track into a team/ref.
            team_labels = team_classifier.classify_tracks(frame, filtered_player_tracks)
            track_team_labels.update(team_labels)

            # Convert track positions from image to pitch coordinates.
            if calibrator is not None:
                frame_player_positions: Dict[TrackID, PointXY] = {}
                for track_id, center_uv in frame_player_centers_uv.items():
                    u, v = center_uv
                    pt = calibrator.image_to_pitch(u, v)
                    track_positions.setdefault(track_id, []).append(pt)
                    frame_player_positions[track_id] = pt
                    # Store time-stamped pitch coordinates for distance metrics.
                    track_points_xy.setdefault(track_id, []).append(
                        PlayerPitchPoint(
                            frame_index=frame_idx,
                            player_id=track_id,
                            x=pt[0],
                            y=pt[1],
                            visible=True,
                        )
                    )

                frame_ball_positions: Dict[TrackID, PointXY] = {}
                for ball_id, bbox in ball_tracks.items():
                    x1, y1, x2, y2 = bbox
                    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    pt = calibrator.image_to_pitch(u, v)
                    ball_positions.setdefault(ball_id, []).append(pt)
                    frame_ball_positions[ball_id] = pt

                # Update possession statistics: which player is closest to
                # the ball this frame (within configured radius).
                if frame_ball_positions and frame_player_positions:
                    # For now, assume a single detected ball per frame.
                    _, ball_pt = next(iter(frame_ball_positions.items()))
                    closest_id: TrackID | None = None
                    closest_dist = float("inf")
                    for track_id, player_pt in frame_player_positions.items():
                        dist = hypot(player_pt[0] - ball_pt[0], player_pt[1] - ball_pt[1])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_id = track_id
                    if closest_id is not None and closest_dist < config.possession_radius_m:
                        possession_frames[closest_id] += 1
            else:
                frame_ball_positions = {}

            # Draw overlays for visualization/debugging.
            if config.enable_visualization or config.write_annotated_video:
                vis = draw_tracks(frame, filtered_player_tracks, team_labels)
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

    finally:
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    # Select active player IDs: team A/B only, prioritize tracks with most observations.
    def _select_active_players(
        min_players: int = 20, max_players: int = 22
    ) -> List[TrackID]:
        candidates: List[Tuple[TrackID, int]] = []
        for track_id, label in track_team_labels.items():
            if label not in ("A", "B"):
                continue
            candidates.append((track_id, len(track_points_xy.get(track_id, []))))
        candidates.sort(key=lambda p: p[1], reverse=True)
        if len(candidates) >= min_players:
            chosen = candidates[:max_players]
        else:
            chosen = candidates
        return [tid for tid, _ in chosen]

    active_player_ids = set(_select_active_players())
    track_positions = {tid: pts for tid, pts in track_positions.items() if tid in active_player_ids}
    track_points_xy = {tid: pts for tid, pts in track_points_xy.items() if tid in active_player_ids}
    track_team_labels = {tid: lbl for tid, lbl in track_team_labels.items() if tid in active_player_ids}
    possession_frames = {tid: cnt for tid, cnt in possession_frames.items() if tid in active_player_ids}

    if calibrator is None:
        # Without calibration, we cannot compute metric distances.
        print("No calibration provided; skipping distance metrics.")
        return

    # Compute metrics once all frames have been processed.
    metrics = compute_player_metrics(track_positions)

    # Distance per player with simple outlier rejection.
    distance_per_player = compute_distance_per_player(track_points_xy, fps=fps)

    # Convert possession frame counts into seconds.
    possession_seconds: Dict[TrackID, float] = {
        track_id: frame_count / fps for track_id, frame_count in possession_frames.items()
    }

    # Export pitch-space tracks for the frontend dashboard.
    tracks_store = PitchTrackStore(
        fps=fps,
        video_path=str(config.input_video),
        pitch_meta=PitchMeta(
            pitch_type=str(config.pitch_length_m),
            length_m=config.pitch_length_m,
            width_m=config.pitch_width_m,
            description=None,
        ),
        tracks=track_points_xy,
    )
    tracks_store.to_json(config.stats_dir / "player_tracks_xy.json")

    # Save a simple heatmap for one player (e.g., the first active track).
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

    # Export per-player stats to JSON and CSV.
    stats_rows: List[Dict[str, Any]] = []
    for track_id, distance_m in distance_per_player.items():
        team_label = track_team_labels.get(track_id, "")
        row: Dict[str, Any] = {
            "player_id": int(track_id),
            "team": team_label,
            "total_distance_m": float(distance_m),
            "total_distance_km": float(distance_m) / 1000.0,
            "possession_seconds": float(possession_seconds.get(track_id, 0.0)),
        }
        stats_rows.append(row)

    if stats_rows:
        stats_dir = config.stats_dir
        json_path = stats_dir / "player_distances.json"
        csv_path = stats_dir / "player_distances.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(stats_rows, f, indent=2)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(
                f,
                fieldnames=[
                    "player_id",
                    "team",
                    "total_distance_m",
                    "total_distance_km",
                    "possession_seconds",
                ],
            )
            writer_csv.writeheader()
            writer_csv.writerows(stats_rows)

        # Also print a compact summary to the console.
        print("Per-player summary:")
        for row in stats_rows:
            team_display = row["team"] or "?"
            print(
                f"  Player {row['player_id']} (team {team_display}): "
                f"{row['total_distance_m']:.1f} m "
                f"({row['total_distance_km']:.2f} km), "
                f"possession {row['possession_seconds']:.1f} s"
            )
    else:
        print("No player tracks with valid calibration; no stats to export.")


def main() -> None:
    """
    CLI entrypoint for running the full pipeline.

    Use ``python -m src.main --help`` for available options.
    """
    parser = argparse.ArgumentParser(
        description="Football player tracking and distance analysis.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--pitch_type",
        type=int,
        choices=[5, 7, 11],
        help=(
            "Pitch type: 5, 7, or 11-a-side. Used together with "
            "config.yaml to choose default pitch dimensions."
        ),
    )
    parser.add_argument(
        "--pitch_length",
        type=float,
        help="Pitch length in meters (overrides pitch_type/default).",
    )
    parser.add_argument(
        "--pitch_width",
        type=float,
        help="Pitch width in meters (overrides pitch_type/default).",
    )
    parser.add_argument(
        "--calib_points",
        type=str,
        help="Path to calibration points JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for output videos, stats, and heatmaps.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to YOLO model weights file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Optional path to YAML config file "
            "(defaults to config.yaml in the project root)."
        ),
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        help="Process every N-th frame for speed/coverage (default from config.py).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        help="Process only the first N frames (useful for quick tests).",
    )
    parser.add_argument(
        "--force_yolo",
        action="store_true",
        help="Disable dummy detector fallback; require YOLO to be available.",
    )
    parser.add_argument(
        "--sample_interval_s",
        type=float,
        help="Sample every N seconds (overrides frame_stride; stride is derived from video fps).",
    )

    args = parser.parse_args()
    config = build_config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
