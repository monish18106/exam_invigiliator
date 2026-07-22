"""
config.py

Centralized configuration management for the
AI Assisted Exam Proctoring System.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Application configuration loaded from .env
    """

    # Application
    app_name: str
    app_version: str

    # Camera
    camera_source: str
    fps_target: int

    # Logging
    log_level: str

    # Paths
    database_path: str
    model_path: str

    # API
    groq_api_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create Settings object from environment variables.
        """

        settings = cls(
            app_name=os.getenv(
                "APP_NAME",
                "AI Assisted Exam Proctoring",
            ),
            app_version=os.getenv(
                "APP_VERSION",
                "1.0.0",
            ),
            camera_source=os.getenv(
                "CAMERA_SOURCE",
                "0",
            ),
            fps_target=int(
                os.getenv(
                    "FPS_TARGET",
                    "10",
                )
            ),
            log_level=os.getenv(
                "LOG_LEVEL",
                "INFO",
            ).upper(),
            database_path=os.getenv(
                "DATABASE_PATH",
                "database/proctoring.db",
            ),
            model_path=os.getenv(
                "MODEL_PATH",
                "models/yolov8m-pose.pt",
            ),
            groq_api_key=os.getenv(
                "GROQ_API_KEY",
                "",
            ),
        )

        settings.validate()

        return settings

    def validate(self) -> None:
        """
        Validate configuration values.
        """

        valid_levels = {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if self.fps_target <= 0:
            raise ValueError(
                "FPS_TARGET must be greater than zero."
            )

        if self.log_level not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {self.log_level}"
            )

        if not self.database_path:
            raise ValueError(
                "DATABASE_PATH cannot be empty."
            )

        if not self.model_path:
            raise ValueError(
                "MODEL_PATH cannot be empty."
            )


# Singleton configuration object
settings = Settings.from_env()