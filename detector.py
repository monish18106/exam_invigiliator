"""
detector.py

Phase 2 - Camera Validation

Purpose
-------
Validate the threaded camera pipeline before
introducing AI modules.

Press 'q' to quit.
"""

from __future__ import annotations

import time

import cv2
from loguru import logger

from camera.cctv import Camera


def main() -> None:
    """
    Camera validation entry point.
    """

    camera = Camera()

    try:
        camera.start()

        logger.info("Starting live camera preview...")

        previous_time = time.perf_counter()

        while camera.is_running():

            frame = camera.read()

            if frame is None:
                continue

            current_time = time.perf_counter()

            fps = 1.0 / max(
                current_time - previous_time,
                1e-6,
            )

            previous_time = current_time

            width, height = camera.get_resolution()

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
                "Phase 2 : Camera Test",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.imshow(
                "AI Assisted Exam Proctoring",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except Exception as exc:
        logger.exception(exc)

    finally:

        camera.stop()

        cv2.destroyAllWindows()

        logger.info("Camera validation finished.")


if __name__ == "__main__":
    main()