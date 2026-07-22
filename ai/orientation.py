"""
ai/orientation.py

Phase 5 - Head Orientation Estimator

Responsibilities
----------------
- Estimate head orientation using YOLO pose keypoints
- Classify:
    FORWARD
    LEFT
    RIGHT
    DOWN
    UNKNOWN
- Maintain temporal history for stable predictions

Phase:
    Phase 5
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

import cv2
import numpy as np
from loguru import logger


class OrientationEstimator:
    """
    Estimate head orientation using YOLO pose keypoints.
    """

    # -------------------------------------------------
    # Orientation Labels
    # -------------------------------------------------

    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"

    # -------------------------------------------------

    def __init__(
        self,
        history_size: int = 7,
        nose_offset_ratio: float = 0.12,
        down_ratio: float = 0.80,
    ) -> None:
        """
        Parameters
        ----------
        history_size
            Number of previous predictions to retain.

        nose_offset_ratio
            Horizontal nose deviation threshold.

        down_ratio
            Nose vertical threshold relative to shoulders.
        """

        self.history_size = history_size

        self.nose_offset_ratio = nose_offset_ratio

        self.down_ratio = down_ratio

        # Track history for each student
        self.history: dict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

        logger.success("Orientation Estimator initialized.")

    # -------------------------------------------------

    @staticmethod
    def _valid(point: tuple[int, int]) -> bool:
        """
        Returns True if a keypoint appears valid.
        """

        if point is None:
            return False

        x, y = point

        return not (x == 0 and y == 0)

    # -------------------------------------------------

    @staticmethod
    def _distance(
        p1: tuple[int, int],
        p2: tuple[int, int],
    ) -> float:

        return float(
            np.linalg.norm(
                np.array(p1) - np.array(p2)
            )
        )

    # -------------------------------------------------

    @staticmethod
    def _midpoint(
        p1: tuple[int, int],
        p2: tuple[int, int],
    ) -> tuple[int, int]:

        return (
            int((p1[0] + p2[0]) / 2),
            int((p1[1] + p2[1]) / 2),
        )

    # -------------------------------------------------

    @staticmethod
    def _majority_vote(
        history: deque[str],
    ) -> str:
        """
        Return the most frequent orientation.
        """

        if not history:
            return OrientationEstimator.UNKNOWN

        counts = Counter(history)

        return counts.most_common(1)[0][0]

    # -------------------------------------------------

    def _update_history(
        self,
        track_id: int,
        orientation: str,
    ) -> str:
        """
        Update orientation history and return
        the smoothed prediction.
        """

        self.history[track_id].append(
            orientation
        )

        return self._majority_vote(
            self.history[track_id]
        )

    # -------------------------------------------------

    def reset(
        self,
        track_id: int,
    ) -> None:
        """
        Remove history for a student.
        """

        if track_id in self.history:
            del self.history[track_id]

        # -------------------------------------------------

    def estimate(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Estimate head orientation for a single tracked student.

        The detection dictionary is updated in-place.

        Returns
        -------
        Updated detection dictionary.
        """

        track_id = detection["track_id"]

        keypoints = detection["keypoints"]

        nose = keypoints.get("nose")

        left_ear = keypoints.get("left_ear")
        right_ear = keypoints.get("right_ear")

        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")

        # ------------------------------------------
        # Validate required keypoints
        # ------------------------------------------

        if (
            not self._valid(nose)
            or not self._valid(left_shoulder)
            or not self._valid(right_shoulder)
        ):

            orientation = self.UNKNOWN

            detection["orientation"] = orientation
            detection["orientation_confidence"] = 0.0

            return detection

        # ------------------------------------------
        # Shoulder Geometry
        # ------------------------------------------

        shoulder_center = self._midpoint(
            left_shoulder,
            right_shoulder,
        )

        shoulder_width = self._distance(
            left_shoulder,
            right_shoulder,
        )

        if shoulder_width < 5:

            orientation = self.UNKNOWN

            detection["orientation"] = orientation
            detection["orientation_confidence"] = 0.0

            return detection

        # ------------------------------------------
        # Nose Position
        # ------------------------------------------

        nose_dx = (
            nose[0] - shoulder_center[0]
        ) / shoulder_width

        nose_dy = (
            nose[1] - shoulder_center[1]
        ) / shoulder_width

        # ------------------------------------------
        # Ear Visibility
        # ------------------------------------------

        left_visible = self._valid(left_ear)
        right_visible = self._valid(right_ear)

        # ------------------------------------------
        # Decision Logic
        # ------------------------------------------

        orientation = self.FORWARD
        confidence = 0.70

        # Looking Down

        if nose_dy > self.down_ratio:

            orientation = self.DOWN
            confidence = 0.95

        else:

            score = 0

            # ----------------------------------
            # Nose relative to shoulders
            # ----------------------------------

            if nose_dx < -self.nose_offset_ratio:
                score -= 2

            elif nose_dx > self.nose_offset_ratio:
                score += 2

            # ----------------------------------
            # Ear geometry
            # ----------------------------------

            if left_visible and right_visible:

                left_distance = self._distance(
                    nose,
                    left_ear,
                )

                right_distance = self._distance(
                    nose,
                    right_ear,
                )

                diff = (
                    left_distance - right_distance
                ) / shoulder_width

                if diff > 0.08:
                    score += 1

                elif diff < -0.08:
                    score -= 1

            elif left_visible and not right_visible:

                score += 2

            elif right_visible and not left_visible:

                score -= 2

            # ----------------------------------
            # Final Decision
            # ----------------------------------

            if score <= -2:

                orientation = self.LEFT
                confidence = 0.92

            elif score >= 2:

                orientation = self.RIGHT
                confidence = 0.92

            else:

                orientation = self.FORWARD
                confidence = 0.88
                # ------------------------------------------
                # Temporal Smoothing
                # ------------------------------------------

                smoothed = self._update_history(
                    track_id,
                    orientation,
                )

                detection["orientation"] = smoothed
                detection["orientation_confidence"] = confidence

                return detection

    # -------------------------------------------------

    def estimate_all(
        self,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Estimate orientation for every tracked student.
        """

        for detection in detections:

            self.estimate(
                detection,
            )

        return detections
    # -------------------------------------------------

    def visualize(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw estimated orientation.
        """

        colors = {
            self.FORWARD: (0, 255, 0),
            self.LEFT: (255, 0, 0),
            self.RIGHT: (0, 165, 255),
            self.DOWN: (0, 0, 255),
            self.UNKNOWN: (120, 120, 120),
        }

        for detection in detections:

            x1, y1, _, _ = detection["bbox"]

            orientation = detection.get(
                "orientation",
                self.UNKNOWN,
            )

            confidence = detection.get(
                "orientation_confidence",
                0.0,
            )

            color = colors.get(
                orientation,
                (255, 255, 255),
            )

            cv2.putText(
                frame,
                f"{orientation} ({confidence:.2f})",
                (x1, y1 - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return frame