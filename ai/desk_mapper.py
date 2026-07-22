"""
ai/desk_mapper.py

Maps tracked students to calibrated desk polygons.

Each tracked student is assigned to exactly one desk using the
bottom-center point of the tracked bounding box.

Author: AI-Assisted Exam Proctoring System
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class DeskMapper:
    """
    Maps tracked students to desk polygons.
    """

    def __init__(self, polygons: List[dict]):
        """
        Parameters
        ----------
        polygons
            List of desk polygons loaded from calibration.
        """
        self.polygons = polygons
        self.student_desk_map: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Polygon Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bottom_center(
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        """
        Calculate the bottom-center point of a bounding box.

        Parameters
        ----------
        bbox
            (x1, y1, x2, y2)

        Returns
        -------
        Tuple[int, int]
            Bottom-center point.
        """
        x1, y1, x2, y2 = bbox

        return (
            int((x1 + x2) / 2),
            int(y2),
        )

    @staticmethod
    def _point_inside_polygon(
        point: Tuple[int, int],
        polygon: List[List[int]],
    ) -> bool:
        """
        Check whether a point lies inside a polygon.

        Parameters
        ----------
        point
            (x, y)

        polygon
            List of polygon points.

        Returns
        -------
        bool
        """
        contour = np.array(
            polygon,
            dtype=np.int32,
        )

        return (
            cv2.pointPolygonTest(
                contour,
                point,
                False,
            )
            >= 0
        )

    # ------------------------------------------------------------------ #
    # Student Assignment
    # ------------------------------------------------------------------ #

    def assign_student(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[int]:
        """
        Assign a tracked student to a desk.

        Parameters
        ----------
        track_id
            ByteTrack ID.

        bbox
            Bounding box.

        Returns
        -------
        Optional[int]
            Desk ID if assigned, otherwise None.
        """

        reference_point = self._bottom_center(bbox)

        for desk in self.polygons:

            if self._point_inside_polygon(
                reference_point,
                desk["points"],
            ):

                desk_id = desk["desk_id"]

                previous = self.student_desk_map.get(track_id)

                if previous != desk_id:
                    logger.info(
                        "Track {} assigned to Desk {}",
                        track_id,
                        desk_id,
                    )

                self.student_desk_map[track_id] = desk_id

                return desk_id

        if track_id in self.student_desk_map:
            logger.debug(
                "Track {} left calibrated desks.",
                track_id,
            )

            del self.student_desk_map[track_id]

        return None

    # ------------------------------------------------------------------ #
    # Batch Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        tracks: List[dict],
    ) -> Dict[int, int]:
        """
        Update mappings for all tracked students.

        Expected track format
        ---------------------
        {
            "track_id": 5,
            "bbox": (x1, y1, x2, y2)
        }

        Parameters
        ----------
        tracks
            Active ByteTrack objects.

        Returns
        -------
        Dict[int, int]
            Current student → desk mapping.
        """

        active_tracks = set()

        for track in tracks:

            track_id = track["track_id"]
            bbox = track["bbox"]

            active_tracks.add(track_id)

            self.assign_student(
                track_id,
                bbox,
            )

        self.clear_missing_tracks(active_tracks)

        return self.student_desk_map.copy()

    # ------------------------------------------------------------------ #
    # Mapping Utilities
    # ------------------------------------------------------------------ #

    def get_student_desk(
        self,
        track_id: int,
    ) -> Optional[int]:
        """
        Return assigned desk for a student.

        Parameters
        ----------
        track_id
            ByteTrack ID.

        Returns
        -------
        Optional[int]
        """
        return self.student_desk_map.get(track_id)

    def clear_missing_tracks(
        self,
        active_tracks: set[int],
    ) -> None:
        """
        Remove mappings for tracks that no longer exist.

        Parameters
        ----------
        active_tracks
            Set of currently active ByteTrack IDs.
        """

        missing = set(self.student_desk_map.keys()) - active_tracks

        for track_id in missing:

            logger.debug(
                "Removing inactive Track {}",
                track_id,
            )

            del self.student_desk_map[track_id]

    # ------------------------------------------------------------------ #
    # Public Utilities
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Clear all mappings.
        """
        self.student_desk_map.clear()

    def get_all_mappings(self) -> Dict[int, int]:
        """
        Return all student-to-desk mappings.
        """
        return self.student_desk_map.copy()

    def update_polygons(
        self,
        polygons: List[dict],
    ) -> None:
        """
        Replace calibration polygons.

        Useful after recalibration.

        Parameters
        ----------
        polygons
            New desk polygons.
        """
        self.polygons = polygons

        logger.info(
            "Desk calibration updated ({} desks).",
            len(polygons),
        )