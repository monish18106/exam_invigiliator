"""
detector.py

Phase 5 - Head Pose Estimation Test

Purpose
-------
Validate the complete camera + tracking + head pose pipeline.

Pipeline
--------
Camera
    ↓
Frame Queue
    ↓
YOLOv8 Pose + ByteTrack
    ↓
Head Pose Estimation
    ↓
Visualization
    ↓
Display

Press 'q' to quit.
"""

from __future__ import annotations

import time

import cv2
from loguru import logger

from ai.orientation import OrientationEstimator
from ai.tracker import PoseTracker
from camera.cctv import Camera


def main() -> None:
    """
    Phase 5 entry point.
    """

    camera = Camera()
    tracker = PoseTracker()
    orientation = OrientationEstimator()

    try:
        camera.start()

        logger.info("Starting head pose estimation...")

        previous_time = time.perf_counter()

        while camera.is_running():

            frame = camera.read()

            if frame is None:
                continue

            # -------------------------------------------------
            # Tracking
            # -------------------------------------------------

            detections = tracker.track(frame)

            # -------------------------------------------------
            # Head Pose Estimation
            # -------------------------------------------------

            detections = orientation.estimate_all(detections)

            # -------------------------------------------------
            # Visualization
            # -------------------------------------------------

            frame = tracker.visualize(
                frame,
                detections,
            )

            frame = orientation.visualize(frame, detections)

            # -------------------------------------------------
            # FPS
            # -------------------------------------------------

            current_time = time.perf_counter()

            fps = 1.0 / max(
                current_time - previous_time,
                1e-6,
            )

            previous_time = current_time

            # -------------------------------------------------
            # Camera Information
            # -------------------------------------------------

            width, height = camera.get_resolution()

            # -------------------------------------------------
            # Overlay
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"FPS : {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Resolution : {width} x {height}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Persons : {len(detections)}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Phase 5 : Head Pose Estimation",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            # -------------------------------------------------
            # Display
            # -------------------------------------------------

            cv2.imshow(
                "AI Assisted Exam Proctoring",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Exit requested by user.")
                break

    except Exception:
        logger.exception("Head pose estimation failed.")

    finally:


        camera.stop()

        cv2.destroyAllWindows()

        logger.info("Head pose estimation finished.")


if __name__ == "__main__":
    main()