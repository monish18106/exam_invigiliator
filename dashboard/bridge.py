"""
dashboard/bridge.py

Bridge between the AI pipeline and the Streamlit dashboard.

Responsibilities:
- Save the latest annotated frame.
- Update runtime dashboard status.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dashboard.live_status import LiveStatus


FRAME_PATH = Path("dashboard/assets/latest_frame.jpg")


class DashboardBridge:
    """
    Handles communication between the AI pipeline
    and the Streamlit dashboard.
    """

    def __init__(self) -> None:
        FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)

        self._status = LiveStatus()

    def update(
        self,
        *,
        frame: np.ndarray,
        fps: float,
        camera_connected: bool = True,
        pipeline_running: bool = True,
    ) -> None:
        """
        Update dashboard assets.

        Parameters
        ----------
        frame:
            Latest annotated frame.

        fps:
            Current processing FPS.
        """

        cv2.imwrite(str(FRAME_PATH), frame)

        self._status.write(
            camera_connected=camera_connected,
            pipeline_running=pipeline_running,
            fps=fps,
        )