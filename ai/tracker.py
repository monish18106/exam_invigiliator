"""
ai/tracker.py

Phase 4 - YOLOv8 Pose + ByteTrack Tracker

Responsibilities
----------------
- Load YOLOv8 Pose model once
- Enable ByteTrack
- Maintain persistent track IDs
- Extract required keypoints
- Return structured tracked detections
- Provide visualization utilities

Phase:
    Phase 4
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from config import settings


class PoseTracker:
    """
    YOLOv8 Pose + ByteTrack wrapper.

    This class performs pose estimation and multi-person
    tracking in a single inference pass.
    """

    # ----------------------------
    # COCO Keypoint Indices
    # ----------------------------

    NOSE = 0

    LEFT_EYE = 1
    RIGHT_EYE = 2

    LEFT_EAR = 3
    RIGHT_EAR = 4

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8

    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    # ----------------------------

    def __init__(
        self,
        confidence: float = 0.40,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        """
        Parameters
        ----------
        confidence
            Detection confidence threshold.

        tracker
            Tracker configuration file.
        """

        self.confidence = confidence
        self.tracker = tracker

        logger.info(
            f"Loading YOLO Pose Tracker: {settings.model_path}"
        )

        self.model = YOLO(settings.model_path)

        logger.success(
            "YOLO Pose Tracker loaded successfully."
        )

    # ---------------------------------------------------------

    @staticmethod
    def _point(
        keypoints: np.ndarray,
        index: int,
    ) -> tuple[int, int]:
        """
        Safely extract a keypoint.

        Returns
        -------
        tuple[int, int]
        """

        x = int(keypoints[index][0])
        y = int(keypoints[index][1])

        return x, y

    # ---------------------------------------------------------

    def _create_detection(
        self,
        track_id: int,
        bbox: np.ndarray,
        confidence: float,
        keypoints: np.ndarray,
    ) -> dict[str, Any]:
        """
        Build a standardized tracked detection dictionary.
        """

        x1, y1, x2, y2 = bbox.astype(int)

        return {
            "track_id": track_id,
            "bbox": (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ),
            "confidence": confidence,
            "keypoints": {

                # Face
                "nose": self._point(
                    keypoints,
                    self.NOSE,
                ),

                "left_ear": self._point(
                    keypoints,
                    self.LEFT_EAR,
                ),

                "right_ear": self._point(
                    keypoints,
                    self.RIGHT_EAR,
                ),

                # Upper Body
                "left_shoulder": self._point(
                    keypoints,
                    self.LEFT_SHOULDER,
                ),

                "right_shoulder": self._point(
                    keypoints,
                    self.RIGHT_SHOULDER,
                ),

                # Hands
                "left_wrist": self._point(
                    keypoints,
                    self.LEFT_WRIST,
                ),

                "right_wrist": self._point(
                    keypoints,
                    self.RIGHT_WRIST,
                ),
            },
        }
        # ---------------------------------------------------------

    def track(
        self,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Perform YOLOv8 Pose + ByteTrack inference.

        Parameters
        ----------
        frame : np.ndarray

        Returns
        -------
        list
            List of tracked detections.
        """

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]

        if (
            result.boxes is None
            or result.keypoints is None
            or result.boxes.id is None
        ):
            return detections

        boxes = result.boxes
        keypoints = result.keypoints

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        kps = keypoints.xy.cpu().numpy()

        for i in range(len(ids)):

            # Person class only
            if classes[i] != 0:
                continue

            confidence = float(scores[i])

            if confidence < self.confidence:
                continue

            detection = self._create_detection(
                track_id=int(ids[i]),
                bbox=xyxy[i],
                confidence=confidence,
                keypoints=kps[i],
            )

            detections.append(detection)

        return detections
        # ---------------------------------------------------------

    def visualize(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw tracked detections on the frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image.

        detections : list
            Output from self.track()

        Returns
        -------
        np.ndarray
            Annotated frame.
        """

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            track_id = det["track_id"]
            confidence = det["confidence"]

            # -------------------------
            # Bounding Box
            # -------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # -------------------------
            # ID + Confidence
            # -------------------------

            label = f"ID:{track_id}  {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # -------------------------
            # Keypoints
            # -------------------------

            keypoint_colors = {
                "nose": (0, 0, 255),            # Red
                "left_ear": (255, 0, 255),      # Magenta
                "right_ear": (255, 255, 0),     # Cyan
                "left_shoulder": (0, 255, 255), # Yellow
                "right_shoulder": (0, 255, 255),
                "left_wrist": (255, 0, 0),      # Blue
                "right_wrist": (255, 0, 0),
            }

            for name, point in det["keypoints"].items():

                if point is None:
                    continue

                px, py = point

                cv2.circle(
                    frame,
                    (px, py),
                    5,
                    keypoint_colors.get(name, (0, 0, 255)),
                    -1,
                )
        return frame