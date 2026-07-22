"""
utils/timers.py

Reusable timer utilities.
"""

from __future__ import annotations

import time


class SimpleTimer:
    """
    Basic elapsed timer.
    """

    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def reset(self) -> None:
        """
        Restart timer.
        """

        self.start_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """
        Elapsed seconds.
        """

        return time.perf_counter() - self.start_time


class PerformanceTimer:
    """
    Context manager for measuring execution time.
    """

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


class ElapsedTimer:
    """
    Countdown-style timer.

    Useful for rule triggering and state transitions.
    """

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.reset()

    def reset(self) -> None:
        self.start_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def expired(self) -> bool:
        return self.elapsed >= self.duration

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration - self.elapsed)