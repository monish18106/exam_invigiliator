"""
dashboard/live_status.py

Shared runtime status using a JSON file.

This module enables communication between app.py
and the Streamlit dashboard running in separate
processes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


STATUS_FILE = Path("dashboard/assets/live_status.json")


class LiveStatus:
    """Read and write runtime dashboard status."""

    def __init__(self, status_file: Path = STATUS_FILE) -> None:
        self._status_file = status_file
        self._status_file.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        camera_connected: bool,
        pipeline_running: bool,
        fps: float,
    ) -> None:
        """Write current runtime status."""

        payload = {
            "camera_connected": camera_connected,
            "pipeline_running": pipeline_running,
            "fps": round(fps, 2),
            "last_update": time.time(),
        }

        self._status_file.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def read(self) -> dict[str, Any]:
        """Read current runtime status."""

        if not self._status_file.exists():
            return {
                "camera_connected": False,
                "pipeline_running": False,
                "fps": 0.0,
                "last_update": 0.0,
            }

        try:
            return json.loads(
                self._status_file.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return {
                "camera_connected": False,
                "pipeline_running": False,
                "fps": 0.0,
                "last_update": 0.0,
            }