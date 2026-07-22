"""
ai/orientation.py

Phase 5 - Head Orientation Estimation

Uses YOLO11 Pose keypoints to estimate:

- FORWARD
- LEFT
- RIGHT
- DOWN
- UNKNOWN

Input:
    Detection dictionaries from PoseTracker

Output:
    Adds:
        orientation
        orientation_confidence
        orientation_score
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from collections import deque
from typing import Any

import numpy as np
from loguru import logger


class OrientationEstimator:
    """
    Estimate head orientation using YOLO pose keypoints.

    Required keypoints:
        nose
        left_eye
        right_eye
        left_ear
        right_ear
        left_shoulder
        right_shoulder
    """

    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        history_size: int = 7,
    ) -> None:
        """
        Parameters
        ----------
        history_size
            Number of previous predictions retained for
            temporal smoothing.
        """

        self.history = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        logger.success(
            "Production Orientation Estimator initialized."
        )

    # ----------------------------------------------------------
    # Helper Functions
    # ----------------------------------------------------------

    @staticmethod
    def valid(
        point: tuple[int, int] | None,
    ) -> bool:
        """
        Check whether a keypoint is valid.
        """

        if point is None:
            return False

        x, y = point

        return not (x == 0 and y == 0)

    # ----------------------------------------------------------

    @staticmethod
    def distance(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> float:
        """
        Euclidean distance.
        """

        return float(
            np.linalg.norm(
                np.array(a) - np.array(b)
            )
        )

    # ----------------------------------------------------------

    @staticmethod
    def midpoint(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Midpoint between two points.
        """

        return (
            int((a[0] + b[0]) / 2),
            int((a[1] + b[1]) / 2),
        )

    # ----------------------------------------------------------

    @staticmethod
    def clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Clamp value to range.
        """

        return max(
            minimum,
            min(value, maximum),
        )

    # ----------------------------------------------------------
    # Temporal Smoothing
    # ----------------------------------------------------------

    def smooth(
        self,
        track_id: int,
        orientation: str,
    ) -> str:
        """
        Majority vote over recent predictions.
        """

        history = self.history[track_id]

        history.append(
            orientation
        )

        counts = Counter(history)

        return counts.most_common(1)[0][0]

    # ----------------------------------------------------------
    # Keypoint Extraction
    # ----------------------------------------------------------

    def _get_points(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:

        kp = detection["keypoints"]

        return {

            "nose":
                kp.get("nose"),

            "left_eye":
                kp.get("left_eye"),

            "right_eye":
                kp.get("right_eye"),

            "left_ear":
                kp.get("left_ear"),

            "right_ear":
                kp.get("right_ear"),

            "left_shoulder":
                kp.get("left_shoulder"),

            "right_shoulder":
                kp.get("right_shoulder"),
        }

    # ----------------------------------------------------------
    # Single Detection
    # ----------------------------------------------------------

    def estimate(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Estimate orientation for one person.

        Implemented in Part 2.
        """

        raise NotImplementedError(
            "Implemented in Part 2."
        )

    # ----------------------------------------------------------
    # Batch Processing
    # ----------------------------------------------------------

    def estimate_all(
        self,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Estimate orientation for every detection.
        """

        output: list[dict[str, Any]] = []

        for detection in detections:

            output.append(
                self.estimate(
                    detection
                )
            )

        return output

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------

    def visualize(
        self,
        frame,
        detections,
    ):
        """
        Implemented in Part 3.
        """

        raise NotImplementedError(
            "Implemented in Part 3."
        )
        # ----------------------------------------------------------
    # Single Detection
    # ----------------------------------------------------------

    def estimate(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Estimate head orientation for a single detection.
        """

        points = self._get_points(detection)

        nose = points["nose"]

        left_eye = points["left_eye"]
        right_eye = points["right_eye"]

        left_ear = points["left_ear"]
        right_ear = points["right_ear"]

        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]

        orientation = self.UNKNOWN
        confidence = 0.0
        final_score = 0.0

        # ------------------------------------------
        # Shoulders are mandatory
        # ------------------------------------------

        if not (
            self.valid(left_shoulder)
            and self.valid(right_shoulder)
        ):

            detection["orientation"] = orientation
            detection["orientation_confidence"] = confidence
            detection["orientation_score"] = final_score

            return detection

        shoulder_center = self.midpoint(
            left_shoulder,
            right_shoulder,
        )

        shoulder_width = self.distance(
            left_shoulder,
            right_shoulder,
        )

        if shoulder_width < 5:

            detection["orientation"] = orientation
            detection["orientation_confidence"] = confidence
            detection["orientation_score"] = final_score

            return detection

        score = 0.0
        weight = 0.0

        body_score = 0.0
        eye_score = 0.0
        ear_score = 0.0

        # ======================================================
        # BODY SCORE
        # ======================================================

        if self.valid(nose):

            nose_dx = (
                nose[0] - shoulder_center[0]
            ) / shoulder_width

            body_score = self.clamp(
                nose_dx * 2.5,
                -1.0,
                1.0,
            )

            score += body_score * 0.35
            weight += 0.35

        # ======================================================
        # EYE SCORE
        # ======================================================

        if (
            self.valid(left_eye)
            and self.valid(right_eye)
            and self.valid(nose)
        ):

            left_distance = self.distance(
                nose,
                left_eye,
            )

            right_distance = self.distance(
                nose,
                right_eye,
            )

            total = (
                left_distance
                + right_distance
            )

            if total > 0:

                eye_score = (
                    right_distance
                    - left_distance
                ) / total

                eye_score = self.clamp(
                    eye_score * 2.0,
                    -1.0,
                    1.0,
                )

                score += eye_score * 0.40
                weight += 0.40

        # ======================================================
        # EAR SCORE
        # ======================================================

        if (
            self.valid(left_ear)
            and self.valid(right_ear)
            and self.valid(nose)
        ):

            left_distance = self.distance(
                nose,
                left_ear,
            )

            right_distance = self.distance(
                nose,
                right_ear,
            )

            total = (
                left_distance
                + right_distance
            )

            if total > 0:

                ear_score = (
                    right_distance
                    - left_distance
                ) / total

                ear_score = self.clamp(
                    ear_score * 2.0,
                    -1.0,
                    1.0,
                )

                score += ear_score * 0.25
                weight += 0.25

        elif self.valid(left_ear):

            ear_score = -0.50

            score += ear_score * 0.25
            weight += 0.25

        elif self.valid(right_ear):

            ear_score = 0.50

            score += ear_score * 0.25
            weight += 0.25

        # ======================================================
        # DOWN DETECTION
        # ======================================================

        if (
            self.valid(nose)
            and self.valid(left_eye)
            and self.valid(right_eye)
        ):

            eye_center = self.midpoint(
                left_eye,
                right_eye,
            )

            dy = (
                nose[1]
                - eye_center[1]
            ) / shoulder_width

            if dy > 0.22:

                confidence = min(
                    dy * 2.0,
                    1.0,
                )

                orientation = self.smooth(
                    detection["track_id"],
                    self.DOWN,
                )

                detection["orientation"] = orientation
                detection["orientation_confidence"] = confidence
                detection["orientation_score"] = 0.0

                detection["debug"] = {
                    "body": round(body_score, 3),
                    "eye": round(eye_score, 3),
                    "ear": round(ear_score, 3),
                    "score": 0.0,
                }

                return detection

        # ======================================================
        # FINAL SCORE
        # ======================================================

        if weight > 0:

            final_score = score / weight

        if final_score <= -0.28:

            orientation = self.LEFT

        elif final_score >= 0.28:

            orientation = self.RIGHT

        else:

            orientation = self.FORWARD

        confidence = min(
            abs(final_score),
            1.0,
        )

        orientation = self.smooth(
            detection["track_id"],
            orientation,
        )

        detection["orientation"] = orientation
        detection["orientation_confidence"] = confidence
        detection["orientation_score"] = final_score

        detection["debug"] = {
            "body": round(body_score, 3),
            "eye": round(eye_score, 3),
            "ear": round(ear_score, 3),
            "score": round(final_score, 3),
        }

        return detection
        # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------

    def visualize(
        self,
        frame,
        detections,
        show_debug: bool = True,
    ):
        """
        Draw orientation labels and optional debug scores.
        """

        import cv2

        colors = {
            self.FORWARD: (0, 255, 0),
            self.LEFT: (255, 0, 0),
            self.RIGHT: (0, 165, 255),
            self.DOWN: (0, 0, 255),
            self.UNKNOWN: (150, 150, 150),
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

            score = detection.get(
                "orientation_score",
                0.0,
            )

            color = colors.get(
                orientation,
                (255, 255, 255),
            )

            # ----------------------------------
            # Orientation Label
            # ----------------------------------

            cv2.putText(
                frame,
                f"{orientation} ({confidence:.2f})",
                (x1, max(30, y1 - 90)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

            # ----------------------------------
            # Final Score
            # ----------------------------------

            cv2.putText(
                frame,
                f"Score: {score:.2f}",
                (x1, max(50, y1 - 65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # ----------------------------------
            # Debug Overlay
            # ----------------------------------

            if show_debug:

                debug = detection.get(
                    "debug",
                    {},
                )

                body = debug.get(
                    "body",
                    0.0,
                )

                eye = debug.get(
                    "eye",
                    0.0,
                )

                ear = debug.get(
                    "ear",
                    0.0,
                )

                cv2.putText(
                    frame,
                    f"B:{body:.2f}",
                    (x1, max(70, y1 - 45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    frame,
                    f"E:{eye:.2f}",
                    (x1, max(90, y1 - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    frame,
                    f"A:{ear:.2f}",
                    (x1, max(110, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

        return frame