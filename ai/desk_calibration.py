"""
ai/desk_calibration.py

Desk calibration module for the AI-Assisted Exam Proctoring System.

This module allows faculty to manually draw desk polygons on the CCTV
frame, save them to disk, reload them, and render them on subsequent runs.

Author: AI-Assisted Exam Proctoring System
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from loguru import logger


Point = Tuple[int, int]


class DeskCalibration:
    """
    Handles manual desk calibration using OpenCV.

    Features
    --------
    - Draw desk polygons
    - Save polygons to JSON
    - Load polygons
    - Render desk overlays
    """

    def __init__(self, calibration_file: str):
        self.calibration_file = Path(calibration_file)

        self.polygons: List[dict] = []
        self.current_polygon: List[Point] = []

        self.window_name = "Desk Calibration"

        self.load()

    # ------------------------------------------------------------------ #
    # JSON Persistence
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """
        Load desk polygons from disk.
        """

        if not self.calibration_file.exists():
            logger.info("No calibration file found.")
            return

        try:
            with self.calibration_file.open("r", encoding="utf-8") as file:
                self.polygons = json.load(file)

            logger.info(
                "Loaded {} desk polygons.",
                len(self.polygons),
            )

        except Exception as exc:
            logger.exception(
                "Failed loading calibration: {}",
                exc,
            )

    def save(self) -> None:
        """
        Save polygons to disk.
        """

        try:
            self.calibration_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.calibration_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    self.polygons,
                    file,
                    indent=4,
                )

            logger.success(
                "Saved {} desk polygons.",
                len(self.polygons),
            )

        except Exception as exc:
            logger.exception(
                "Failed saving calibration: {}",
                exc,
            )

    # ------------------------------------------------------------------ #
    # Polygon Editing
    # ------------------------------------------------------------------ #

    def reset_current(self) -> None:
        """
        Clear current polygon.
        """
        self.current_polygon.clear()

    def finish_polygon(self) -> bool:
        """
        Finish the polygon currently being drawn.

        Returns
        -------
        bool
            True if polygon saved.
        """

        if len(self.current_polygon) < 3:
            logger.warning(
                "Polygon requires at least 3 points."
            )
            return False

        desk = {
            "desk_id": len(self.polygons) + 1,
            "points": self.current_polygon.copy(),
        }

        self.polygons.append(desk)

        logger.success(
            "Desk {} created.",
            desk["desk_id"],
        )

        self.current_polygon.clear()

        return True

    # ------------------------------------------------------------------ #
    # Mouse Callback
    # ------------------------------------------------------------------ #

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ) -> None:

        if event == cv2.EVENT_LBUTTONDOWN:

            self.current_polygon.append((x, y))

            logger.debug(
                "Point added ({}, {})",
                x,
                y,
            )

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def render(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Draw desk polygons.

        Parameters
        ----------
        frame
            Current video frame.

        Returns
        -------
        np.ndarray
            Annotated frame.
        """

        output = frame.copy()

        # Finished polygons

        for desk in self.polygons:

            pts = np.array(
                desk["points"],
                dtype=np.int32,
            )

            cv2.polylines(
                output,
                [pts],
                True,
                (0, 255, 255),
                2,
            )

            center = np.mean(
                pts,
                axis=0,
            ).astype(int)

            cv2.putText(
                output,
                f"Desk {desk['desk_id']}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Current polygon

        if len(self.current_polygon) > 0:

            pts = np.array(
                self.current_polygon,
                dtype=np.int32,
            )

            for point in pts:

                cv2.circle(
                    output,
                    tuple(point),
                    4,
                    (0, 255, 0),
                    -1,
                )

            if len(pts) > 1:

                cv2.polylines(
                    output,
                    [pts],
                    False,
                    (0, 255, 0),
                    2,
                )

        return output

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def get_polygons(self) -> List[dict]:
        """
        Return all desk polygons.
        """
        return self.polygons

    def start_calibration(
        self,
        frame_provider,
    ) -> None:
        """
        Launch calibration window.

        Controls
        --------
        Left Click : Add Point
        N          : Finish Desk
        R          : Reset Current Polygon
        S          : Save
        Q / ESC    : Exit
        """

        cv2.namedWindow(self.window_name)

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback,
        )

        logger.info("Calibration started.")

        while True:

            frame = frame_provider()

            if frame is None:
                continue

            display = self.render(frame)

            cv2.imshow(
                self.window_name,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                self.finish_polygon()

            elif key == ord("r"):
                self.reset_current()

            elif key == ord("s"):
                self.save()

            elif key in (27, ord("q")):
                break

        cv2.destroyWindow(
            self.window_name,
        )

        logger.info("Calibration finished.")