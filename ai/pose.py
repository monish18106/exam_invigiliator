"""
ai/pose.py

YOLOv8 Pose Detector

Responsibilities
----------------
- Load YOLOv8 Pose model once
- Detect persons
- Extract required keypoints
- Return structured detections
- Provide visualization utilities

Phase:
    Phase 3
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from config import settings


class PoseDetector:
    """
    Wrapper around Ultralytics YOLOv8 Pose.
    """

    # COCO Keypoint Indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    def __init__(
        self,
        confidence: float = 0.4,
    ) -> None:
        """
        Parameters
        ----------
        confidence : float
            Minimum confidence threshold.
        """

        self.confidence = confidence

        logger.info(
            f"Loading YOLO Pose model: {settings.model_path}"
        )

        self.model = YOLO(settings.model_path)

        logger.success("YOLO Pose model loaded.")

    # ---------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Detect persons.

        Parameters
        ----------
        frame : np.ndarray

        Returns
        -------
        list
            Structured detections.
        """

        results = self.model(
            frame,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        if len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes
        keypoints = result.keypoints

        for i in range(len(boxes)):

            cls = int(boxes.cls[i])

            # Person class only
            if cls != 0:
                continue

            confidence = float(boxes.conf[i])

            if confidence < self.confidence:
                continue

            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .astype(int)
            )

            kp = (
                keypoints.xy[i]
                .cpu()
                .numpy()
            )

            detections.append(
                {
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ),
                    "confidence": confidence,
                    "keypoints": {
                        "nose": tuple(
                            kp[self.NOSE]
                        ),
                        "left_shoulder": tuple(
                            kp[self.LEFT_SHOULDER]
                        ),
                        "right_shoulder": tuple(
                            kp[self.RIGHT_SHOULDER]
                        ),
                        "left_wrist": tuple(
                            kp[self.LEFT_WRIST]
                        ),
                        "right_wrist": tuple(
                            kp[self.RIGHT_WRIST]
                        ),
                    },
                }
            )

        return detections

    # ---------------------------------------------------------

    def visualize(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw detections.

        Parameters
        ----------
        frame : np.ndarray

        detections : list

        Returns
        -------
        np.ndarray
        """

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{det['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            for point in det["keypoints"].values():

                px = int(point[0])
                py = int(point[1])

                cv2.circle(
                    frame,
                    (px, py),
                    5,
                    (0, 0, 255),
                    -1,
                )

        return frame