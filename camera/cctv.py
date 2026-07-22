"""
camera/cctv.py

Threaded CCTV camera reader.

Supported Sources
-----------------
- Webcam
- USB Camera
- RTSP streams
- DVR/NVR streams
- Video files

Frames are continuously captured in a background thread
and pushed to the FrameQueue.

Only the latest frame is kept.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Union

import cv2
import numpy as np
from loguru import logger

from config import settings
from camera.frame_queue import FrameQueue


class Camera:
    """
    Threaded camera reader.
    """

    def __init__(
        self,
        source: Optional[Union[int, str]] = None,
    ) -> None:

        if source is None:
            source = settings.camera_source

        # Convert webcam indices from string to int
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.source = source

        self.capture: Optional[cv2.VideoCapture] = None

        self.frame_queue = FrameQueue()

        self.thread: Optional[threading.Thread] = None

        self.running = False

        self.fps = settings.fps_target

    # -------------------------------------------------------------

    def start(self) -> None:
        """
        Start camera thread.
        """

        if self.running:
            return

        logger.info(f"Opening camera source: {self.source}")

        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera source: {self.source}"
            )

        # Reduce OpenCV internal buffering (backend dependent)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraThread",
        )

        self.thread.start()

        logger.success("Camera started.")

    # -------------------------------------------------------------

    def stop(self) -> None:
        """
        Stop camera thread.
        """

        self.running = False

        if self.thread is not None:
            self.thread.join()

        if self.capture is not None:
            self.capture.release()

        self.frame_queue.clear()

        logger.info("Camera stopped.")

    # -------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Background capture loop.
        """

        frame_delay = 1.0 / self.fps

        while self.running:

            if self.capture is None:
                break

            success, frame = self.capture.read()

            if not success:

                logger.warning("Frame capture failed.")

                time.sleep(0.1)

                continue

            self.frame_queue.put(frame)

            time.sleep(frame_delay)

    # -------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """
        Get latest frame.

        Returns
        -------
        np.ndarray | None
        """

        return self.frame_queue.get()

    # -------------------------------------------------------------

    def has_frame(self) -> bool:
        """
        Check whether a frame exists.
        """

        return self.frame_queue.has_frame()

    # -------------------------------------------------------------

    def is_running(self) -> bool:
        """
        Camera running state.
        """

        return self.running

    # -------------------------------------------------------------

    def get_resolution(self) -> tuple[int, int]:
        """
        Current camera resolution.
        """

        if self.capture is None:
            return (0, 0)

        width = int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        return width, height

    # -------------------------------------------------------------

    def get_fps(self) -> float:
        """
        Camera FPS.

        Falls back to configured FPS if camera
        does not report one.
        """

        if self.capture is None:
            return 0.0

        fps = self.capture.get(cv2.CAP_PROP_FPS)

        if fps <= 1:
            return float(self.fps)

        return fps