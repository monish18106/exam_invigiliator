"""
camera/frame_queue.py

Thread-safe latest-frame queue.

This queue always stores ONLY the latest frame.

Purpose
-------
Prevent latency caused by accumulating frames while
the AI pipeline is processing previous ones.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

import numpy as np


class FrameQueue:
    """
    Thread-safe queue that stores only the latest frame.
    """

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = Lock()

    def put(self, frame: np.ndarray) -> None:
        """
        Store the latest frame.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV frame.
        """

        with self._lock:
            self._frame = frame.copy()

    def get(self) -> Optional[np.ndarray]:
        """
        Retrieve the latest frame.

        Returns
        -------
        Optional[np.ndarray]
            Latest frame or None.
        """

        with self._lock:
            if self._frame is None:
                return None

            return self._frame.copy()

    def empty(self) -> bool:
        """
        Check whether the queue is empty.

        Returns
        -------
        bool
        """

        with self._lock:
            return self._frame is None

    def clear(self) -> None:
        """
        Remove stored frame.
        """

        with self._lock:
            self._frame = None

    def has_frame(self) -> bool:
        """
        Returns
        -------
        bool
            True if a frame is available.
        """

        with self._lock:
            return self._frame is not None

    @property
    def size(self) -> int:
        """
        Queue size.

        Returns
        -------
        int
            0 or 1
        """

        with self._lock:
            return 0 if self._frame is None else 1