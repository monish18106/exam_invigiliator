"""
recorder/recorder.py

Phase 10 - Evidence Recorder

Records evidence clips for suspicious events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from recorder.ring_buffer import BufferedFrame
from recorder.ring_buffer import RingBuffer


@dataclass(slots=True)
class EvidenceResult:
    """
    Result returned after an evidence clip is saved.
    """

    student_id: int
    event_type: str
    clip_path: str
    created_at: float


class EvidenceRecorder:
    """
    Records evidence clips using a frame ring buffer.
    """

    def __init__(
        self,
        buffer: RingBuffer,
        output_dir: str,
        fps: int,
    ) -> None:

        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        self._buffer = buffer

        self._output_dir = Path(output_dir)

        self._fps = fps

        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._recording = False

        self._student_id: int | None = None

        self._event_type: str | None = None

        self._pre_frames: list[BufferedFrame] = []

        self._post_frames: list[BufferedFrame] = []

        self._post_frame_target = self._fps * 5

    def add_frame(
        self,
        frame: Any,
        timestamp: float,
    ) -> None:
        """
        Add a frame to the ring buffer.
        """

        self._buffer.add_frame(
            frame=frame,
            timestamp=timestamp,
        )

    def is_recording(
        self,
    ) -> bool:
        """
        Return whether recording is active.
        """

        return self._recording

    def start_recording(
        self,
        student_id: int,
        event_type: str,
    ) -> bool:
        """
        Start evidence recording.

        Returns
        -------
        bool
            True if recording started,
            False if already recording.
        """

        if self._recording:
            return False

        self._recording = True

        self._student_id = student_id

        self._event_type = event_type

        self._pre_frames = self._buffer.get_frames()

        self._post_frames = []

        return True

    def update(
        self,
        frame: Any,
        timestamp: float,
    ) -> EvidenceResult | None:
        """
        Update active recording.

        Returns
        -------
        EvidenceResult | None
            Saved evidence information when recording
            finishes, otherwise None.
        """

        self.add_frame(
            frame=frame,
            timestamp=timestamp,
        )

        if not self._recording:
            return None

        self._post_frames.append(
            BufferedFrame(
                frame=frame,
                timestamp=timestamp,
            )
        )

        if len(self._post_frames) < self._post_frame_target:
            return None

        try:
            clip_path = self.save_clip()

            evidence = EvidenceResult(
                student_id=self._student_id,
                event_type=self._event_type,
                clip_path=clip_path,
                created_at=timestamp,
            )

            return evidence

        finally:
            self._recording = False
            self._student_id = None
            self._event_type = None
            self._pre_frames.clear()
            self._post_frames.clear()

    def save_clip(
        self,
    ) -> str:
        """
        Save the recorded evidence clip.
        """

        frames = self._pre_frames + self._post_frames

        if not frames:
            raise RuntimeError(
                "No frames available to save."
            )

        first_frame = frames[0].frame

        height, width = first_frame.shape[:2]

        filename = (
            f"student_{self._student_id}_"
            f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )

        output_path = self._output_dir / filename

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open video writer: {output_path}"
            )

        try:
            for buffered_frame in frames:
                writer.write(buffered_frame.frame)
        finally:
            writer.release()

        return str(output_path)