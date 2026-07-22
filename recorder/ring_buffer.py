"""
recorder/ring_buffer.py

Phase 10 - Frame Ring Buffer

Stores the most recent frames in memory
for evidence recording.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class BufferedFrame:
    """
    Frame stored in memory.
    """

    frame: Any

    timestamp: float


class RingBuffer:
    """
    Circular frame buffer.

    Keeps the latest N frames in memory.
    """

    def __init__(
        self,
        max_frames: int,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero.")

        self._buffer = deque(maxlen=max_frames)


    def add_frame(
        self,
        frame: Any,
        timestamp: float,
    ) -> None:
        """
        Add a frame to the buffer.
        """

        self._buffer.append(
            BufferedFrame(
                frame=frame,
                timestamp=timestamp,
            )
        )

    def get_frames(
        self,
    ) -> list[BufferedFrame]:
        """
        Return buffered frames in chronological order.
        """

        return list(self._buffer)

    def clear(
        self,
    ) -> None:
        """
        Remove all buffered frames.
        """

        self._buffer.clear()

    def size(
        self,
    ) -> int:
        """
        Current number of buffered frames.
        """

        return len(self._buffer)

    def capacity(
        self,
    ) -> int:
        """
        Maximum buffer capacity.
        """

        return self._buffer.maxlen or 0
    
