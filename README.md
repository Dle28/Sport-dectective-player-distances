# Sport-dectective-player-distances

Minimal end-to-end prototype for football (soccer) player tracking,
distance estimation, and heatmap generation from a broadcast-style video.

## Features

- Player detection using YOLO (via `ultralytics`) with a dummy fallback.
- Simple IOU-based multi-object tracker with persistent IDs.
- Team assignment based on jersey color in HSV space.
- Homography-based pitch calibration (image → metric coordinates).
- Per-player distance covered (meters / kilometers).
- Heatmap generation for player movement over the pitch.

## Project structure

- `src/`
  - `main.py` – orchestrates the whole pipeline.
  - `config.py` – paths, thresholds, pitch size, etc.
  - `detection.py` – player detection using YOLO or dummy detector.
  - `tracking.py` – simple IOU-based tracker (stub for DeepSORT/ByteTrack).
  - `team_classifier.py` – jersey color-based team classification.
  - `calibration.py` – homography estimation and image→pitch mapping.
  - `metrics.py` – distance per player and basic stats.
  - `heatmap.py` – per-player heatmap generation.
  - `visualization.py` – drawing boxes, IDs, and heatmap overlays.
  - `utils/video_io.py` – video reading utilities.
  - `utils/geometry.py` – simple geometric helpers.
- `models/` – YOLO weights (e.g., `yolov8n.pt`).
- `data/` – input videos and calibration JSON files.
- `outputs/` – annotated videos, stats, and heatmap images.

## Installation

Create a virtual environment and install dependencies:

```bash
pip install numpy opencv-python ultralytics
```

> Note: `ultralytics` is optional – if it is missing, the code falls
> back to a dummy detector that outputs no players.

## Calibration file

Create a JSON file at `data/calibration_points.json` with at least four
corresponding points:

```json
{
  "image_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "pitch_points": [[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]]
}
```

- `image_points` are pixel coordinates in the video frame.
- `pitch_points` are coordinates in meters on the pitch (origin and axes
  are up to you, e.g., (0, 0) in one corner).

## Running the demo

1. Place a match video at `data/sample_match.mp4` or adjust `Config.input_video`.
2. Download YOLO weights (e.g., `yolov8n.pt`) into `models/` or adjust `Config.yolo_model_path`.
3. Create the calibration JSON as described above.

Then run:

```bash
python -m src.main
```

Outputs:

- `outputs/annotated.mp4` – video with boxes, IDs, and team labels.
- `outputs/heatmaps/player_<id>_heatmap.png` – raw heatmap.
- `outputs/heatmaps/player_<id>_heatmap_overlay.png` – heatmap on a synthetic pitch.

## TODO / next steps

- Replace the simple IOU tracker with DeepSORT or ByteTrack.
- Improve jersey color clustering and robustness to lighting changes.
- Add interactive calibration and sanity checks for homography quality.
- Export per-player metrics (distance, speed) as JSON/CSV.
- Add CLI arguments and experiment configuration files (YAML/JSON).
