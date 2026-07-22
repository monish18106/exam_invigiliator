"""
app.py

AI Assisted Exam Proctoring System
Phase 1 - Project Setup

Responsibilities
----------------
- Load application configuration
- Initialize logging
- Create required directories
- Validate startup environment
- Display startup banner

No AI models or camera modules are loaded in this phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config import settings
from utils.helpers import ensure_directory, verify_path
from utils.logger import setup_logger


REQUIRED_DIRECTORIES = [
    "logs",
    "clips",
    "models",
    "database",
]


def print_banner() -> None:
    """
    Display application banner.
    """

    banner = f"""
========================================================
{settings.app_name}
Version : {settings.app_version}
========================================================
"""

    print(banner)


def create_required_directories() -> None:
    """
    Create required project directories.
    """

    logger.info("Verifying project directories...")

    for directory in REQUIRED_DIRECTORIES:
        ensure_directory(directory)

    logger.success("Directories verified.")


def validate_environment() -> None:
    """
    Validate required paths and configuration.
    """

    logger.info("Validating application configuration...")

    verify_path(
        Path(settings.database_path).parent
    )

    verify_path(
        Path(settings.model_path).parent
    )

    logger.success("Environment validation completed.")


def startup() -> None:
    """
    Perform application startup sequence.
    """

    print_banner()

    setup_logger(settings.log_level)

    logger.info("Application starting...")

    logger.info("Configuration loaded successfully.")

    create_required_directories()

    validate_environment()

    logger.success("System Ready")

    logger.info("Application initialized successfully.")
    logger.info("Waiting for camera module...")


def shutdown() -> None:
    """
    Graceful shutdown.
    """

    logger.info("Application shutdown completed.")


def main() -> None:
    """
    Main application entry point.
    """

    try:
        startup()

    except KeyboardInterrupt:
        logger.warning("Application interrupted by user.")

    except Exception as exc:
        logger.exception(f"Startup failed: {exc}")
        sys.exit(1)

    finally:
        shutdown()


if __name__ == "__main__":
    main()