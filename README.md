# Sport-dectective-player-distances

Minimal end-to-end toolkit for football (soccer) player tracking,
distance estimation, and heatmap generation from a single static-camera
video.

The code is organized into small, testable modules for detection,
tracking, calibration, metrics, and visualization, and exposes a simple
CLI for running the full pipeline on a new match video.

## Features

- Player detection using YOLO (via `ultralytics`) with a dummy fallback.
- Multi-object tracking with persistent IDs (simple IoU tracker and
  optional DeepSORT backend).
- Team assignment based on jersey color in HSV space.
- Homography-based pitch calibration (image → metric pitch coordinates).
- Per-player distance covered (meters and kilometers) with basic
  outlier rejection.
- Simple ball tracking and approximate possession time.
- Per-player heatmap generation in pitch coordinates.
- CLI with configurable pitch type/size and calibration file.
- `config.yaml` for default model path and pitch sizes (5/7/11-a-side).

## Project structure

- `src/`
  - `main.py` – CLI + main pipeline (detection → tracking → calibration
    → metrics → visualization).
  - `config.py` – paths, thresholds, default pitch size, etc.
  - `detection.py` – YOLO-based `YoloDetector` and `Detection` dataclass.
  - `tracking.py` – simple IoU tracker and `PlayerTracker` wrapper
    (stub for DeepSORT/ByteTrack).
  - `team_classifier.py` – jersey color-based team classification.
  - `calibration.py` – `PitchCalibrator`, homography helpers, and
    image → pitch mapping.
  - `metrics.py` – distance per player, smoothing, and basic stats.
  - `heatmap.py` – per-player heatmap generation in pitch coordinates.
  - `visualization.py` – drawing boxes, IDs, team labels, and heatmap
    overlays.
  - `utils/video_io.py` – video reading/writing utilities.
  - `utils/geometry.py` – simple geometric helpers.
- `models/` – YOLO weights (e.g., `yolov8n.pt`).
- `data/` – input videos and calibration JSON files.
- `outputs/`
  - `annotated.mp4` – annotated video with boxes, IDs, and team labels.
  - `heatmaps/player_<id>_*.png` – raw and overlay heatmaps.
  - `stats/player_distances.(json|csv)` – per-player distance metrics.
- `config.yaml` – default model path and pitch sizes for 5/7/11-a-side.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notes:

- `ultralytics` (YOLO) is optional – if it is missing, the code falls
  back to a dummy detector that outputs no players.
- `deep-sort-realtime` is optional – only needed if you switch
  `tracker_type` to `deepsort`.

## Configuration (`config.yaml`)

`config.yaml` provides defaults for the detector model path and pitch
sizes for different formats (5/7/11-a-side). Example:

```yaml
model:
  yolo_model_path: models/yolov8n.pt

pitch_types:
  "5":
    length_m: 40.0
    width_m: 20.0
  "7":
    length_m: 50.0
    width_m: 30.0
  "11":
    length_m: 105.0
    width_m: 68.0
```

From the CLI you can override:

- Model path: `--model_path path/to/weights.pt`
- Pitch type: `--pitch_type 5|7|11` (uses `config.yaml` values).
- Pitch dimensions: `--pitch_length`, `--pitch_width` (override type).
- Calibration file: `--calib_points path/to/calibration_points.json`.

## Calibration: mapping pixels to meters

Distance must **never** be computed directly in pixels. Because of
perspective, the same pixel displacement near the camera and far away
corresponds to different real-world distances. Instead, we:

1. Collect at least 4–8 point correspondences between the image and the
   pitch in meters.
2. Compute a 3×3 homography `H` using OpenCV (`cv2.findHomography`).
3. Map all player positions `(u, v)` in pixels to pitch coordinates
   `(x, y)` in meters using `H`.
4. Compute distances in the metric pitch coordinate system.

### Calibration file format

Create a JSON file, e.g. `data/calibration_points.json`:

```json
{
  "image_points": [
    [812, 1040],   // bottom-right corner flag in pixels
    [150, 1020],   // bottom-left corner
    [860, 120],    // top-right corner
    [140, 130]     // top-left corner
  ],
  "pitch_points": [
    [105.0, 0.0],  // (x, y) in meters on a 105x68m pitch
    [0.0, 0.0],
    [105.0, 68.0],
    [0.0, 68.0]
  ]
}
```

For non-standard pitches (e.g. 7-a-side ~50×30m or 5-a-side
~40×20m), adjust `pitch_points` accordingly and ensure
`--pitch_length` / `--pitch_width` (or `config.yaml`) match the actual
field dimensions.

Internally, `PitchCalibrator`:

- Uses the correspondences to compute the homography.
- Maps bounding-box centers `(u, v)` to `(x, y)` in meters.
- Works with any pitch size, as long as you provide consistent
  calibration and dimensions per video.

## How distance is computed

For each processed frame and each tracked player:

1. The tracker outputs a stable `player_id` and bounding box in pixels.
2. The center `(u, v)` of the box is mapped to `(x, y)` in meters.
3. For each player we store `[(frame_index, x, y), ...]`.
4. Distances are accumulated using:

   ```text
   d_t = sqrt((x_{t+1} - x_t)^2 + (y_{t+1} - y_t)^2)
   total_distance_m = sum_t d_t
   total_distance_km = total_distance_m / 1000.0
   ```

5. A simple outlier rejection is applied:
   if the displacement between two consecutive frames is greater than
   15 m, that segment is ignored as tracking noise.

This ensures that distances are expressed correctly in meters and
robust to occasional ID switches or jittery detections.

## Running the pipeline

Prepare:

1. Place a match video at `data/sample_match.mp4` (or choose your own).
2. Download YOLO weights (e.g. `yolov8n.pt`) into `models/` and update
   `config.yaml` if needed.
3. Create `data/calibration_points.json` as described above.

Example for a standard 11-a-side 105×68m pitch:

```bash
python -m src.main \
  --video_path data/sample_match.mp4 \
  --pitch_type 11 \
  --pitch_length 105 \
  --pitch_width 68 \
  --calib_points data/calibration_points.json \
  --output_dir outputs/
```

For a 7-a-side game on ~50×30m pitch:

```bash
python -m src.main \
  --video_path data/sample_7aside.mp4 \
  --pitch_type 7 \
  --pitch_length 50 \
  --pitch_width 30 \
  --calib_points data/calibration_7aside.json \
  --output_dir outputs_7aside/
```

Outputs:

- `outputs/annotated.mp4` – video with boxes, IDs, and team labels.
- `outputs/heatmaps/player_<id>_heatmap.png` – raw heatmap.
- `outputs/heatmaps/player_<id>_heatmap_overlay.png` – heatmap on a
  synthetic pitch.
- `outputs/stats/player_distances.json` – per-player distances (meters,
  km, possession time).
- `outputs/stats/player_distances.csv` – same metrics in CSV form.

Example `player_distances.json` snippet:

```json
[
  {
    "player_id": 3,
    "team": "A",
    "total_distance_m": 10123.4,
    "total_distance_km": 10.1234,
    "possession_seconds": 95.2
  },
  {
    "player_id": 7,
    "team": "B",
    "total_distance_m": 8240.7,
    "total_distance_km": 8.2407,
    "possession_seconds": 12.8
  }
]
```

## Why meters, not pixels?

- Pixels depend on camera zoom, resolution, and viewpoint.
- A player running along the far touchline may move only a few pixels
  per frame but still cover several meters.
- Homography-based calibration ensures `(x, y)` are in meters on the
  chosen pitch coordinate system, so distances are comparable across
  different cameras and pitch sizes (5/7/11-a-side).

Always calibrate **per video** with the correct pitch dimensions; do not
reuse `H` or pitch sizes between different camera setups or fields.

## Data layer usage

Suggested folders for multiple matches:

- `data/raw/` – source mp4 files (e.g., `data/raw/5ph.mp4`).
- `data/calibration/` – calibration JSON per video.
- `data/tracks_px/` – pixel-space tracks (`frame_index,u,v,visible`).
- `data/tracks_xy/` – pitch-space tracks (`frame_index,x,y,visible`).
- `data/metrics/` – per-player distance and aggregated stats.

Minimal example with the new data structures:

```python
from pathlib import Path
from src.video_io import VideoReader
from src.data_structures import FrameInfo, PlayerTrackPoint, PlayerPitchPoint, PitchMeta, PitchPointUV, PitchPointXY
from src.tracking_data import PixelTrackStore, PitchTrackStore
from src.calibration_data import CalibrationSample, CalibrationSet, save_calibration

# 1) Read frames (subsample every 5th frame for a quick prototype run).
reader = VideoReader(Path("data/raw/5ph.mp4"), stride=5)
pixel_tracks = PixelTrackStore(fps=reader.fps, video_path=reader.path)
for frame_idx, frame in reader:
    frame_info = FrameInfo.from_index(frame_idx, reader.fps)
    # Example: add a detected player center; mark visible=False if they leave the frame (goalkeepers often hug edges).
    pixel_tracks.add_point(
        PlayerTrackPoint(frame_index=frame_idx, player_id=3, u=123.4, v=567.8, visible=True)
    )
pixel_tracks.to_json(Path("data/tracks_px/5ph_tracks.json"))
pixel_tracks.to_csv(Path("data/tracks_px/5ph_tracks.csv"))

# 2) Save a calibration for this static tactical cam.
pitch_meta = PitchMeta(pitch_type="11v11", length_m=105.0, width_m=68.0, description="static high-angle, 25 fps")
calibration = CalibrationSet(
    pitch_meta=pitch_meta,
    samples=[
        CalibrationSample(PitchPointUV(u=150.0, v=1020.0), PitchPointXY(x=0.0, y=0.0)),
        CalibrationSample(PitchPointUV(u=860.0, v=120.0), PitchPointXY(x=105.0, y=68.0)),
    ],
)
save_calibration(calibration, Path("data/calibration/5ph_calibration.json"))

# 3) After applying homography elsewhere, store pitch-space tracks.
pitch_tracks = PitchTrackStore(fps=reader.fps, video_path=reader.path)
pitch_tracks.add_point(PlayerPitchPoint(frame_index=10, player_id=3, x=52.3, y=33.8, visible=True))
pitch_tracks.to_json(Path("data/tracks_xy/5ph_tracks.json"))
```

## TODO / next steps

- Replace the simple IoU tracker with a stronger backend (DeepSORT,
  ByteTrack, Norfair) by default.
- Improve jersey color clustering and robustness to lighting changes.
- Add interactive calibration and sanity checks for homography quality.
- Extend metrics with per-interval speed, accelerations, and
  high-intensity running stats.
- Add more advanced heatmaps (e.g., kernel density estimation,
  matplotlib rendering, tactical zones).
