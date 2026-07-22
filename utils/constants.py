"""
utils/constants.py

Application-wide constants.
"""

from __future__ import annotations

APP_NAME = "AI Assisted Exam Proctoring"

APP_VERSION = "1.0.0"

DEFAULT_FPS = 10

QUEUE_SIZE = 1

BUFFER_SECONDS = 10

CLIP_DURATION_SECONDS = 10

PRE_EVENT_SECONDS = 5

POST_EVENT_SECONDS = 5

LOG_FILE = "logs/app.log"

LOG_ROTATION = "10 MB"

LOG_RETENTION = "10 days"

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)