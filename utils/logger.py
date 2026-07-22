"""
utils/logger.py

Centralized Loguru logger configuration.
"""

from __future__ import annotations

import sys

from loguru import logger

from utils.constants import (
    LOG_FILE,
    LOG_FORMAT,
    LOG_RETENTION,
    LOG_ROTATION,
)


def setup_logger(level: str = "INFO") -> None:
    """
    Configure application logger.

    Parameters
    ----------
    level : str
        Logging level.
    """

    logger.remove()

    # Console
    logger.add(
        sys.stdout,
        level=level,
        format=LOG_FORMAT,
        colorize=True,
    )

    # File
    logger.add(
        LOG_FILE,
        level=level,
        format=LOG_FORMAT,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.info("Logger initialized.")