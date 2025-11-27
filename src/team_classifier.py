"""
Team classification utilities using jersey color.

This module assigns each track to a team label (e.g., 'A' or 'B') based
on the dominant color within the player's bounding box.

The current implementation uses simple color clustering in HSV space
and maintains a running estimate of team color centroids.

TODO: Make the color model more robust (e.g., per-frame updates, outlier
rejection, using k-means with temporal smoothing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np

from .tracking import TrackID
from .detection import BoundingBox


TeamLabel = str  # e.g., "A", "B", "Ref"
Frame = np.ndarray[Any, np.dtype[np.uint8]]


@dataclass
class SimpleColorTeamClassifier:
    """
    Minimal jersey color-based team classifier.

    The classifier keeps running mean HSV colors for each team and assigns
    a team label to each track by comparing the player's jersey color to
    these prototypes.

    Attributes:
        team_colors_hsv: Mapping from team label to running mean HSV color.
        learning_rate: Update rate for the running mean (0..1).
        min_samples_for_kmeans: Minimum number of players needed to run
            a simple k-means-like clustering into two teams.
        kmeans_max_iters: Maximum iterations for the k-means refinement.
    """

    learning_rate: float = 0.1
    min_samples_for_kmeans: int = 4
    kmeans_max_iters: int = 10
    team_colors_hsv: Dict[TeamLabel, np.ndarray[Any, np.dtype[np.float32]]] = field(default_factory=dict)  # type: ignore[misc]

    def _extract_dominant_color(self, frame: Frame, bbox: BoundingBox) -> np.ndarray[Any, np.dtype[np.float32]]:  # type: ignore[misc]
        """
        Extract a simple dominant color (mean HSV) from the player's bounding box.
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_hsv = hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
        return mean_hsv

    def _fit_kmeans_two_clusters(self, colors: np.ndarray[Any, np.dtype[np.float32]]) -> None:  # type: ignore[misc]
        """
        Run a simple 2-means clustering on the given HSV colors.

        This avoids adding a heavy dependency on scikit-learn while still
        behaving similarly to KMeans with k=2.
        """
        if colors.shape[0] < 2:  # type: ignore[attr-defined]
            return

        # Initialize centroids with two distinct samples.
        c1 = colors[0].copy()
        c2 = colors[-1].copy()

        for _ in range(self.kmeans_max_iters):
            d1 = np.linalg.norm(colors - c1, axis=1)  # type: ignore[arg-type]
            d2 = np.linalg.norm(colors - c2, axis=1)  # type: ignore[arg-type]

            mask = d1 <= d2
            cluster_a = colors[mask]  # type: ignore[call-overload]
            cluster_b = colors[~mask]  # type: ignore[call-overload]

            if cluster_a.size == 0 or cluster_b.size == 0:
                break

            new_c1 = cluster_a.mean(axis=0)
            new_c2 = cluster_b.mean(axis=0)

            if np.allclose(new_c1, c1) and np.allclose(new_c2, c2):
                c1, c2 = new_c1, new_c2
                break

            c1, c2 = new_c1, new_c2

        self.team_colors_hsv["A"] = c1.astype(np.float32)
        self.team_colors_hsv["B"] = c2.astype(np.float32)

    def _assign_team_for_color(self, color_hsv: np.ndarray[Any, np.dtype[np.float32]]) -> TeamLabel:  # type: ignore[misc]
        """
        Assign the closest existing team label to the given color.

        If no team colors exist yet, initialize 'A', then 'B'.
        """
        if not self.team_colors_hsv:
            # Should have been initialized via k-means when enough samples exist.
            self.team_colors_hsv["A"] = color_hsv  # type: ignore[assignment]
            return "A"
        if len(self.team_colors_hsv) == 1:  # type: ignore[arg-type]
            # Initialize second team with sufficiently different colors
            existing_color = next(iter(self.team_colors_hsv.values()))  # type: ignore[arg-type]
            if np.linalg.norm(color_hsv - existing_color) > 20:  # type: ignore[arg-type]
                self.team_colors_hsv["B"] = color_hsv  # type: ignore[assignment]
                return "B"

        # Find nearest team color
        best_label: TeamLabel = "A"
        best_dist = float("inf")
        for label, prototype in self.team_colors_hsv.items():  # type: ignore[attr-defined]
            dist = float(np.linalg.norm(color_hsv - prototype))  # type: ignore[arg-type]
            if dist < best_dist:
                best_dist = dist
                best_label = label

        # Update prototype with running mean
        proto = self.team_colors_hsv[best_label]  # type: ignore[index]
        self.team_colors_hsv[best_label] = (1 - self.learning_rate) * proto + self.learning_rate * color_hsv  # type: ignore[assignment]
        return best_label

    def classify_tracks(
        self, frame: Frame, tracks: Dict[TrackID, BoundingBox]
    ) -> Dict[TrackID, TeamLabel]:
        """
        Classify each track into a team based on jersey color.

        Args:
            frame: Current BGR frame.
            tracks: Mapping of track IDs to bounding boxes.

        Returns:
            Mapping from track ID to team label.
        """
        labels: Dict[TrackID, TeamLabel] = {}
        if not tracks:
            return labels

        # Extract colors for all tracks first.
        track_ids: List[TrackID] = []
        colors: List[np.ndarray[Any, np.dtype[np.float32]]] = []  # type: ignore[misc]
        for track_id, bbox in tracks.items():
            color_hsv = self._extract_dominant_color(frame, bbox)  # type: ignore[misc]
            track_ids.append(track_id)
            colors.append(color_hsv)  # type: ignore[arg-type]

        if not self.team_colors_hsv and len(colors) >= self.min_samples_for_kmeans:  # type: ignore[arg-type]
            # Initialize team prototypes using a simple k-means two-cluster fit.
            color_array = np.stack(colors, axis=0)  # type: ignore[arg-type]
            self._fit_kmeans_two_clusters(color_array)  # type: ignore[arg-type]

        for track_id, color_hsv in zip(track_ids, colors):  # type: ignore[arg-type]
            label = self._assign_team_for_color(color_hsv)  # type: ignore[arg-type]
            labels[track_id] = label
        return labels


def create_team_classifier() -> SimpleColorTeamClassifier:
    """
    Factory for creating a team classifier.

    TODO: Allow passing predefined team color ranges (e.g., known jersey colors).
    """
    return SimpleColorTeamClassifier()
