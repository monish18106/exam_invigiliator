"""
utils package

Shared utility modules for the AI Assisted Exam Proctoring System.
"""

from .helpers import ensure_directory, verify_path
from .logger import setup_logger

__all__ = [
    "ensure_directory",
    "verify_path",
    "setup_logger",
]