"""
app.py

AI Assisted Exam Proctoring System

Main application entry point.

Responsibilities
----------------
- Initialize configuration
- Initialize logging
- Create AI components
- Start CCTV processing
- Run AI pipeline
- Update dashboard
- Save alerts/evidence
- Gracefully shutdown
"""

from __future__ import annotations

import sys
import time

import cv2
from loguru import logger

from config import settings

from camera.cctv import Camera

from ai.tracker import PoseTracker
from ai.desk_mapper import DeskMapper
from ai.orientation import OrientationEstimator
from ai.head_pose import HeadPoseEstimator
from ai.rules import RuleEngine
from ai.alerts import AlertManager
from ai.pipeline import Pipeline

from ai.desk_calibration import DeskCalibration

from recorder.ring_buffer import RingBuffer
from recorder.recorder import EvidenceRecorder

from dashboard.bridge import DashboardBridge

from database.session import SessionLocal
from database.crud import DatabaseCRUD

from utils.logger import setup_logger
from utils.helpers import ensure_directory


def main() -> None:
    """
    Application entry point.
    """

    setup_logger()

    logger.info(
        "{} v{}",
        settings.app_name,
        settings.app_version,
    )

    ensure_directory("logs")
    ensure_directory("evidence")

    db = SessionLocal()

    try:
        # ---------------------------------------------------------
        # Initialize Database
        # ---------------------------------------------------------

        database = DatabaseCRUD(db)

        # ---------------------------------------------------------
        # Dashboard
        # ---------------------------------------------------------

        dashboard = DashboardBridge()

        logger.success(
            "Application initialized."
        )

        # ---------------------------------------------------------
        # Load Desk Calibration
        # ---------------------------------------------------------

        calibration = DeskCalibration(
            settings.desk_file,
        )

        polygons = calibration.get_polygons()

        desk_map = {
            desk["desk_id"]: desk["points"]
            for desk in polygons
        }

        # ---------------------------------------------------------
        # Initialize AI Modules
        # ---------------------------------------------------------

        tracker = PoseTracker()

        desk_mapper = DeskMapper(
            polygons=polygons,
        )

        orientation_estimator = OrientationEstimator()

        head_pose_estimator = HeadPoseEstimator()

        rule_engine = RuleEngine(
            desk_map=desk_map,
        )

        alert_manager = AlertManager()

        # ---------------------------------------------------------
        # Initialize Evidence Recorder
        # ---------------------------------------------------------

        ring_buffer = RingBuffer(
            max_frames=settings.fps_target * 5,
        )

        evidence_recorder = EvidenceRecorder(
            buffer=ring_buffer,
            output_dir="evidence",
            fps=settings.fps_target,
        )

        # ---------------------------------------------------------
        # Create Processing Pipeline
        # ---------------------------------------------------------

        pipeline = Pipeline(
            tracker=tracker,
            desk_mapper=desk_mapper,
            orientation_estimator=orientation_estimator,
            head_pose_estimator=head_pose_estimator,
            rule_engine=rule_engine,
            alert_manager=alert_manager,
            evidence_recorder=evidence_recorder,
            database=database,
        )

        # ---------------------------------------------------------
        # Initialize Camera
        # ---------------------------------------------------------

        camera = Camera(
            source=settings.camera_source,
        )

        camera.start()

        logger.success(
            "Camera started successfully."
        )

        # ---------------------------------------------------------
        # Main Processing Loop
        # ---------------------------------------------------------

        logger.info(
            "Starting processing loop..."
        )

        previous_time = time.perf_counter()

        while True:

            if not camera.has_frame():
                time.sleep(0.01)
                continue

            frame = camera.read()

            if frame is None:
                continue

            result = pipeline.process(frame)

            current_time = time.perf_counter()

            fps = 1.0 / max(
                current_time - previous_time,
                1e-6,
            )

            previous_time = current_time

            dashboard.update(
                frame=result.frame,
                fps=fps,
                camera_connected=True,
                pipeline_running=True,
            )

            cv2.imshow(
                settings.app_name,
                result.frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                logger.info(
                    "Exit requested."
                )
                break

    finally:

        logger.info(
            "Shutting down application..."
        )

        if "camera" in locals():
            camera.stop()

        if "head_pose_estimator" in locals():
            head_pose_estimator.close()

        cv2.destroyAllWindows()

        db.close()

        logger.success(
            "Application shutdown complete."
        )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        logger.warning(
            "Interrupted by user."
        )

        sys.exit(0)

    except Exception:

        logger.exception(
            "Fatal application error."
        )

        sys.exit(1)