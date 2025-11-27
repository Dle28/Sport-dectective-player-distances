# Project Issues / Checklist

Use this file to track progress on the main features. You can check items off directly in this markdown file.

---

## Issue 1: Initialize project structure and basic config

- [x] Create `src/`, `models/`, `data/`, `outputs/` directories
- [x] Add `src/config.py` to store paths and basic parameters
- [x] Add `requirements.txt` with OpenCV, numpy, matplotlib, ultralytics (tracker library still TBD)

---

## Issue 2: Implement video I/O utilities

- [x] Create `src/utils/video_io.py`
- [x] Implement function/class to iterate through frames of a video (`VideoReader.frames(...)`)
- [x] Implement helper to write out a video with annotations (wrap `cv2.VideoWriter` instead of handling it only in `main.py`)

---

## Issue 3: Integrate player detection model

- [x] Create `src/detection.py`
- [x] Load pre-trained model (e.g., YOLOv8 via `ultralytics`)
- [x] Implement a high-level `detect_players(frame)` that returns bounding boxes, scores, and classes (wrapper around the current `Detector.detect`)
- [x] Add explicit config options for NMS / detection filters in `Config` (confidence, IoU, class filters, etc.)

---

## Issue 4: Implement multi-object tracking

- [x] Create `src/tracking.py`
- [x] Integrate a stronger tracker backend (e.g., DeepSORT, ByteTrack, or Norfair) behind the `Tracker` interface
- [x] Implement `update_tracks(detections, frame)`-equivalent API (current `Tracker.update(...)` returning active tracks with IDs)
- [x] Ensure consistent IDs across frames (Simple IoU tracker already provides basic persistence)

---

## Issue 5: Team classification based on jersey colors

- [x] Create `src/team_classifier.py`
- [x] Implement method to extract dominant color from each track’s bounding box (mean HSV)
- [x] Cluster players into two teams based on color (e.g., KMeans with `k=2`), rather than only running color prototypes
- [x] Assign `team_id` to each track and keep it stable over time

---

## Issue 6: Camera calibration and pitch coordinates

- [x] Create `src/calibration.py`
- [ ] Implement manual/interactive selection of 4–8 reference points in the video (OpenCV mouse callbacks)
- [x] Map selected points to real pitch coordinates (e.g., 105m × 68m) via JSON calibration file
- [x] Compute homography and expose utility functions like `image_to_pitch(x, y)` and `bbox_center_to_pitch(...)`

---

## Issue 7: Distance and metrics per player

- [x] Create `src/metrics.py`
- [x] Implement accumulation of track positions in pitch coordinates (currently done via `track_positions` in `main.py`)
- [x] Compute per-player total distance (meters and km via `PlayerMetrics`)
- [ ] Export metrics to JSON/CSV under `outputs/stats/` (one file per match or per player)

---

## Issue 8: Heatmap generation per player

- [x] Create `src/heatmap.py`
- [x] Implement function to build a 2D histogram of player positions on the pitch and blur it (current OpenCV-based heatmap)
- [ ] Add optional matplotlib-based rendering for more customizable heatmaps
- [x] Save heatmap images under `outputs/heatmaps/player_<id>_*.png` (raw and overlay)

---

## Issue 9: Visualization and debugging tools

- [x] Create `src/visualization.py`
- [x] Draw bounding boxes, IDs, and team colors on frames (`draw_tracks`)
- [ ] Optionally overlay short trajectories per player (using stored recent positions)
- [x] Add CLI flag or config option to enable/disable visualization and video writing

---

## Issue 10: Main pipeline script

- [x] Create `src/main.py`
- [ ] Parse CLI arguments (input video, config path, output directory, flags)
- [x] Run full pipeline: load video → detect → track → classify team → compute metrics → generate heatmaps
- [ ] Print a simple summary for each player (track ID, team, total distance, etc.) at the end of the run

---

## Issue 11: Documentation and examples

- [x] Add example command line usage to `README.md`
- [ ] Provide a small sample video and calibration file in `data/` (or link + clear instructions for creating them)
- [x] Document limitations and future improvements (better calibration, re-identification, stronger tracker, etc.)
