"""
ai/head_pose.py

Phase 5 - Head Pose Estimation

Responsibilities
----------------
- Detect face landmarks using MediaPipe Face Mesh
- Estimate head pose using solvePnP
- Compute yaw, pitch and roll
- Provide visualization utilities

Phase:
    Phase 5
"""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger


class HeadPoseEstimator:
    """
    MediaPipe Face Mesh based Head Pose Estimator.
    """

    # --------------------------------------------------
    # MediaPipe Landmark Indices
    # --------------------------------------------------
    #
    # Nose Tip      : 1
    # Chin          : 199
    # Left Eye      : 33
    # Right Eye     : 263
    # Left Mouth    : 61
    # Right Mouth   : 291
    #
    # These six landmarks are sufficient for solvePnP.
    #

    LANDMARK_IDS = {
        "nose": 1,
        "chin": 199,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }

    # --------------------------------------------------

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initialize MediaPipe Face Mesh.
        """

        logger.info("Initializing MediaPipe Face Mesh...")

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        logger.success("Head Pose Estimator initialized.")

    # --------------------------------------------------

    @staticmethod
    def _camera_matrix(
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Create an approximate intrinsic camera matrix.
        """

        focal_length = width

        center = (
            width / 2,
            height / 2,
        )

        return np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    # --------------------------------------------------

    @staticmethod
    def _dist_coeffs() -> np.ndarray:
        """
        Assume zero lens distortion.
        """

        return np.zeros(
            (4, 1),
            dtype=np.float64,
        )

    # --------------------------------------------------

    @staticmethod
    def _model_points() -> np.ndarray:
        """
        Generic 3D facial model points.

        Coordinates are expressed in millimeters.
        """

        return np.array(
            [
                (0.0, 0.0, 0.0),          # Nose
                (0.0, -63.6, -12.5),      # Chin
                (-43.3, 32.7, -26.0),     # Left Eye
                (43.3, 32.7, -26.0),      # Right Eye
                (-28.9, -28.9, -24.1),    # Left Mouth
                (28.9, -28.9, -24.1),     # Right Mouth
            ],
            dtype=np.float64,
        )

    # --------------------------------------------------

    @staticmethod
    def _crop_face(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: int = 20,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Crop face ROI from the tracked person's bounding box.

        Returns
        -------
        roi
            Cropped image.

        offset
            Top-left corner of ROI in original frame.
        """

        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)

        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        roi = frame[y1:y2, x1:x2]

        return roi, (x1, y1)

    # --------------------------------------------------

    @staticmethod
    def _extract_image_points(
        landmarks,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks into image coordinates.
        """

        ids = HeadPoseEstimator.LANDMARK_IDS

        image_points = []

        for idx in ids.values():

            lm = landmarks.landmark[idx]

            image_points.append(
                (
                    lm.x * width,
                    lm.y * height,
                )
            )

        return np.asarray(
            image_points,
            dtype=np.float64,
        )
        # --------------------------------------------------

    @staticmethod
    def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles.

        Returns
        -------
        yaw, pitch, roll (degrees)
        """

        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 +
            rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:

            pitch = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy,
            )

            roll = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0],
            )

        else:

            pitch = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1],
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy,
            )

            roll = 0.0

        return (
            float(np.degrees(yaw)),
            float(np.degrees(pitch)),
            float(np.degrees(roll)),
        )

    # --------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray,
        detection: dict,
    ) -> Optional[dict]:
        """
        Estimate head pose for a tracked person.

        Parameters
        ----------
        frame
            Original frame.

        detection
            Detection dictionary from PoseTracker.

        Returns
        -------
        {
            "yaw": float,
            "pitch": float,
            "roll": float,
            "nose": (x, y),
            "rotation_vector": np.ndarray,
            "translation_vector": np.ndarray
        }

        Returns None if estimation fails.
        """

        bbox = detection.get("bbox")

        if bbox is None:
            return None

        roi, offset = self._crop_face(frame, bbox)

        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB,
        )

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        roi_h, roi_w = roi.shape[:2]

        image_points = self._extract_image_points(
            face_landmarks,
            roi_w,
            roi_h,
        )

        model_points = self._model_points()

        camera_matrix = self._camera_matrix(
            roi_w,
            roi_h,
        )

        dist_coeffs = self._dist_coeffs()

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        yaw, pitch, roll = self._rotation_matrix_to_euler(
            rotation_matrix
        )

        nose = (
            int(image_points[0][0] + offset[0]),
            int(image_points[0][1] + offset[1]),
        )

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "nose": nose,
            "rotation_vector": rotation_vector,
            "translation_vector": translation_vector,
        }

    # --------------------------------------------------

    def estimate_all(
        self,
        frame: np.ndarray,
        detections: list[dict],
    ) -> list[dict]:
        """
        Estimate head pose for every tracked person.

        The input detections are updated in-place by adding a
        `head_pose` field.

        Returns
        -------
        Updated detections list.
        """

        for detection in detections:

            pose = self.estimate(
                frame,
                detection,
            )

            detection["head_pose"] = pose

        return detections
        # --------------------------------------------------

    def visualize(
        self,
        frame: np.ndarray,
        detections: list[dict],
    ) -> np.ndarray:
        """
        Draw head pose information on the frame.
        """

        for detection in detections:

            head_pose = detection.get("head_pose")

            if head_pose is None:
                continue

            bbox = detection["bbox"]
            track_id = detection.get("track_id", -1)

            x1, y1, x2, y2 = bbox

            yaw = head_pose["yaw"]
            pitch = head_pose["pitch"]
            roll = head_pose["roll"]

            nose = head_pose["nose"]

            rotation_vector = head_pose["rotation_vector"]
            translation_vector = head_pose["translation_vector"]

            roi, offset = self._crop_face(frame, bbox)

            roi_h, roi_w = roi.shape[:2]

            camera_matrix = self._camera_matrix(
                roi_w,
                roi_h,
            )

            dist_coeffs = self._dist_coeffs()

            axis = np.float32(
                [
                    [50, 0, 0],
                    [0, 50, 0],
                    [0, 0, 50],
                ]
            )

            axis_points, _ = cv2.projectPoints(
                axis,
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs,
            )

            axis_points = axis_points.reshape(-1, 2)

            origin = np.array(nose)

            x_axis = (
                int(axis_points[0][0] + offset[0]),
                int(axis_points[0][1] + offset[1]),
            )

            y_axis = (
                int(axis_points[1][0] + offset[0]),
                int(axis_points[1][1] + offset[1]),
            )

            z_axis = (
                int(axis_points[2][0] + offset[0]),
                int(axis_points[2][1] + offset[1]),
            )

            cv2.circle(
                frame,
                tuple(origin),
                4,
                (0, 255, 255),
                -1,
            )

            cv2.line(
                frame,
                tuple(origin),
                x_axis,
                (0, 0, 255),
                2,
            )

            cv2.line(
                frame,
                tuple(origin),
                y_axis,
                (0, 255, 0),
                2,
            )

            cv2.line(
                frame,
                tuple(origin),
                z_axis,
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Yaw : {yaw:.1f}",
                (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Pitch : {pitch:.1f}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Roll : {roll:.1f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return frame

    # --------------------------------------------------

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        if self.face_mesh is not None:
            self.face_mesh.close()

            logger.info("Head Pose Estimator released.")

    # --------------------------------------------------

    def __del__(self):
        """
        Destructor.
        """

        try:
            self.close()
        except Exception:
            pass