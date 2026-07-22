"""
utils/helpers.py

Reusable helper functions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """
    Create directory if it does not exist.

    Returns
    -------
    Path
        Directory path.
    """

    directory = Path(path)

    directory.mkdir(parents=True, exist_ok=True)

    return directory


def verify_path(path: str | Path) -> bool:
    """
    Verify a filesystem path exists.

    Returns
    -------
    bool
    """

    return Path(path).exists()


def get_timestamp() -> str:
    """
    Current timestamp.

    Example
    -------
    20260722_143501
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_filename(prefix: str, extension: str) -> str:
    """
    Create timestamp-based filename.

    Example
    -------
    alert_20260722_143501.mp4
    """

    extension = extension.lstrip(".")

    return f"{prefix}_{get_timestamp()}.{extension}"


def format_duration(seconds: float) -> str:
    """
    Format seconds.

    Example
    -------
    3.42 sec
    """

    return f"{seconds:.2f} sec"