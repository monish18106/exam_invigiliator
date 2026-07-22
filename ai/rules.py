"""
ai/rules.py

Phase 7 - Rule Engine

Responsibilities
----------------
- Common rule definitions
- Rule severity levels
- Rule result data models
- Student timer management
- Helper utilities

Rule evaluation is implemented in later parts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any




# ---------------------------------------------------------------------
# Rule Thresholds (Locked MVP)
# ---------------------------------------------------------------------

HANDS_OFF_DESK_YELLOW = 2.0
HANDS_OFF_DESK_RED = 4.0

PEEPING_YELLOW = 2.0
PEEPING_RED = 4.0

ITEM_EXCHANGE_YELLOW = 1.5
ITEM_EXCHANGE_RED = 3.0


# ---------------------------------------------------------------------
# Rule Levels
# ---------------------------------------------------------------------


class RuleLevel(str, Enum):
    """
    Rule severity level.
    """

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


# ---------------------------------------------------------------------
# Rule Result
# ---------------------------------------------------------------------


@dataclass(slots=True)
class RuleResult:
    """
    Result returned by a rule evaluation.
    """

    active: bool = False
    duration: float = 0.0
    level: RuleLevel = RuleLevel.GREEN
    confidence: float = 0.0

    def reset(self) -> None:
        """
        Reset rule result.
        """

        self.active = False
        self.duration = 0.0
        self.level = RuleLevel.GREEN
        self.confidence = 0.0


# ---------------------------------------------------------------------
# Student Rule Timers
# ---------------------------------------------------------------------


@dataclass(slots=True)
class StudentRuleTimers:
    """
    Stores independent timers for every rule.

    Each tracked student owns one instance.
    """

    hands_off_start: float | None = None
    peeping_start: float | None = None
    exchange_start: float | None = None

    def reset(self) -> None:
        """
        Reset all timers.
        """

        self.hands_off_start = None
        self.peeping_start = None
        self.exchange_start = None


# ---------------------------------------------------------------------
# Student Rule Results
# ---------------------------------------------------------------------


@dataclass(slots=True)
class StudentRuleResults:
    """
    Stores evaluation results for one student.
    """

    hands_off_desk: RuleResult = field(default_factory=RuleResult)

    peeping: RuleResult = field(default_factory=RuleResult)

    item_exchange: RuleResult = field(default_factory=RuleResult)


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------


def start_timer(current: float | None) -> float:
    """
    Start a timer if not already running.

    Returns
    -------
    float
        Timer start timestamp.
    """

    if current is None:
        return time.monotonic()

    return current


def stop_timer() -> None:
    """
    Stop timer.

    Returns
    -------
    None
    """

    return None


def elapsed_seconds(start_time: float | None) -> float:
    """
    Calculate elapsed seconds.

    Parameters
    ----------
    start_time:
        Timer start.

    Returns
    -------
    float
        Seconds elapsed.
    """

    if start_time is None:
        return 0.0

    return time.monotonic() - start_time


def determine_level(
    duration: float,
    yellow_threshold: float,
    red_threshold: float,
) -> RuleLevel:
    """
    Determine rule severity based on duration.
    """

    if duration >= red_threshold:
        return RuleLevel.RED

    if duration >= yellow_threshold:
        return RuleLevel.YELLOW

    return RuleLevel.GREEN


# ---------------------------------------------------------------------
# Type Alias
# ---------------------------------------------------------------------

StudentData = dict[str, Any]
# ---------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------


class RuleEngine:
    """
    Phase 7 Rule Engine.

    Responsibilities
    ----------------
    - Maintain independent timers for every tracked student.
    - Store rule evaluation results.
    - Provide a common interface for later rule implementations.

    Actual rule logic is implemented in subsequent parts.
    """

    def __init__(
        self,
        desk_map: dict[int, list[tuple[int, int]]],
    ) -> None:
        """
        Initialize Rule Engine.

        Parameters
        ----------
        desk_map
            Dictionary mapping desk_id to calibrated desk ROI.
        """

        self._desk_map = desk_map

        self._timers: dict[int, StudentRuleTimers] = {}

        self._results: dict[int, StudentRuleResults] = {}
    # -------------------------------------------------------------

    def _ensure_student(
        self,
        student_id: int,
    ) -> None:
        """
        Allocate timer/result storage for a student.
        """

        if student_id not in self._timers:
            self._timers[student_id] = StudentRuleTimers()

        if student_id not in self._results:
            self._results[student_id] = StudentRuleResults()

    # -------------------------------------------------------------

    def update(
        self,
        student: StudentData,
        students: list[StudentData],
    ) -> StudentRuleResults:
        """
        Update rule engine for one student.

        Executes every rule sequentially.

        Parameters
        ----------
        student
            Student tracking dictionary.

        Returns
        -------
        StudentRuleResults
        """

        student_id = student["student_id"]

        self._ensure_student(student_id)

        timers = self._timers[student_id]

        results = self._results[student_id]

        # ---------------------------------------------------------
        # Rule 1
        # ---------------------------------------------------------

        self._evaluate_hands_off_desk(
            student=student,
            timers=timers,
            results=results,
        )

        # ---------------------------------------------------------
        # Rule 2
        # ---------------------------------------------------------

        self._evaluate_peeping(
            student=student,
            timers=timers,
            results=results,
        )

        # ---------------------------------------------------------
        # Rule 3
        # ---------------------------------------------------------

        self._evaluate_item_exchange(
            student=student,
            students=students,
            timers=timers,
            results=results,
        )

        return results

    # -------------------------------------------------------------

    def _evaluate_hands_off_desk(
        self,
        student: StudentData,
        timers: StudentRuleTimers,
        results: StudentRuleResults,
    ) -> None:
        """
        Evaluate Hands Off Desk rule.
        """

        inside = self._both_wrists_inside(student)

        # ------------------------------------------
        # Hands returned to desk
        # ------------------------------------------

        if inside:

            results.hands_off_desk.reset()
            timers.hands_off_start = stop_timer()

            return

        # ------------------------------------------
        # Hands outside desk
        # ------------------------------------------

        timers.hands_off_start = start_timer(
            timers.hands_off_start      
        )

        elapsed = elapsed_seconds(
            timers.hands_off_start
        )
        # ------------------------------------------
        # GREEN
        # ------------------------------------------

        if elapsed < HANDS_OFF_DESK_YELLOW:

            results.hands_off_desk.level = RuleLevel.GREEN
            results.hands_off_desk.active = False
            results.hands_off_desk.duration = elapsed
            results.hands_off_desk.confidence = 0.0

            return

        # ------------------------------------------
        # YELLOW
        # ------------------------------------------

        if elapsed < HANDS_OFF_DESK_RED:

            results.hands_off_desk.level = RuleLevel.YELLOW
            results.hands_off_desk.active = True
            results.hands_off_desk.duration = elapsed
            results.hands_off_desk.confidence = min(
                1.0,
                elapsed / HANDS_OFF_DESK_RED,
            )

            return

        # ------------------------------------------
        # RED
        # ------------------------------------------

        results.hands_off_desk.level = RuleLevel.RED
        results.hands_off_desk.active = True
        results.hands_off_desk.duration = elapsed
        results.hands_off_desk.confidence = 1.0

    # -------------------------------------------------------------

    def _evaluate_peeping(
        self,
        student: StudentData,
        timers: StudentRuleTimers,
        results: StudentRuleResults,
    ) -> None:
        """
        Evaluate Peeping rule.
        """

        head_pose = student.get("head_pose")

        if head_pose is None:
            yaw = 0.0
        else:
            yaw = abs(head_pose.get("yaw", 0.0))

        # ------------------------------------------
        # Looking forward
        # ------------------------------------------

        if yaw <= 35.0:

            results.peeping.reset()
            timers.peeping_start = stop_timer()

            return

        # ------------------------------------------
        # Looking away
        # ------------------------------------------

        timers.peeping_start = start_timer(
            timers.peeping_start
        )

        elapsed = elapsed_seconds(
            timers.peeping_start
        )

        # ------------------------------------------
        # GREEN
        # ------------------------------------------

        if elapsed < PEEPING_YELLOW:

            results.peeping.level = RuleLevel.GREEN
            results.peeping.active = False
            results.peeping.duration = elapsed
            results.peeping.confidence = 0.0

            return

        # ------------------------------------------
        # YELLOW
        # ------------------------------------------

        if elapsed < PEEPING_RED:

            results.peeping.level = RuleLevel.YELLOW
            results.peeping.active = True
            results.peeping.duration = elapsed
            results.peeping.confidence = min(
                1.0,
                elapsed / PEEPING_RED,
            )

            return

        # ------------------------------------------
        # RED
        # ------------------------------------------

        results.peeping.level = RuleLevel.RED
        results.peeping.active = True
        results.peeping.duration = elapsed
        results.peeping.confidence = 1.0


    # -------------------------------------------------------------

    def _evaluate_item_exchange(
        self,
        student: StudentData,
        students: list[StudentData],
        timers: StudentRuleTimers,
        results: StudentRuleResults,
    ) -> None:
        current_desk = student.get("desk_id")

        exchange_detected = False

        for other in students:

            if other["student_id"] == student["student_id"]:
                continue

            other_desk = other.get("desk_id")

            if other_desk == current_desk:
                continue

            polygon = self._desk_map.get(other_desk)

            if polygon is None:
                continue

            left_inside = self._is_wrist_inside_desk(
                student.get("left_wrist"),
                polygon,
            )

            right_inside = self._is_wrist_inside_desk(
                student.get("right_wrist"),
                polygon,
            )

            if left_inside or right_inside:
                exchange_detected = True
                break

        # ------------------------------------------
        # No exchange
        # ------------------------------------------

        if not exchange_detected:

            results.item_exchange.reset()
            timers.exchange_start = stop_timer()

            return

        # ------------------------------------------
        # Exchange detected
        # ------------------------------------------

        timers.exchange_start = start_timer(
            timers.exchange_start
        )

        elapsed = elapsed_seconds(
            timers.exchange_start
        )

        # ------------------------------------------
        # GREEN
        # ------------------------------------------

        if elapsed < ITEM_EXCHANGE_YELLOW:

            results.item_exchange.level = RuleLevel.GREEN
            results.item_exchange.active = False
            results.item_exchange.duration = elapsed
            results.item_exchange.confidence = 0.0

            return

        # ------------------------------------------
        # YELLOW
        # ------------------------------------------

        if elapsed < ITEM_EXCHANGE_RED:

            results.item_exchange.level = RuleLevel.YELLOW
            results.item_exchange.active = True
            results.item_exchange.duration = elapsed
            results.item_exchange.confidence = min(
                1.0,
                elapsed / ITEM_EXCHANGE_RED,
            )

            return

        # ------------------------------------------
        # RED
        # ------------------------------------------

        results.item_exchange.level = RuleLevel.RED
        results.item_exchange.active = True
        results.item_exchange.duration = elapsed
        results.item_exchange.confidence = 1.0


        # -------------------------------------------------------------
    # Geometry Helpers
    # -------------------------------------------------------------

    @staticmethod
    def _point_in_polygon(
        point: tuple[int | float, int | float] | None,
        polygon: list[tuple[int, int]] | None,
    ) -> bool:
        """
        Check whether a point lies inside a polygon using the
        ray-casting algorithm.

        Parameters
        ----------
        point
            (x, y) point.

        polygon
            List of polygon vertices.

        Returns
        -------
        bool
        """

        if point is None:
            return False

        if polygon is None:
            return False

        if len(polygon) < 3:
            return False

        x, y = point

        inside = False

        n = len(polygon)

        j = n - 1

        for i in range(n):

            xi, yi = polygon[i]

            xj, yj = polygon[j]

            intersects = (
                (yi > y) != (yj > y)
                and x
                < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
            )

            if intersects:
                inside = not inside

            j = i

        return inside

    # -------------------------------------------------------------

    def _is_wrist_inside_desk(
        self,
        wrist: tuple[int, int] | None,
        polygon: list[tuple[int, int]] | None,
    ) -> bool:
        """
        Returns True if wrist lies inside desk polygon.
        """

        return self._point_in_polygon(
            wrist,
            polygon,
        )

    # -------------------------------------------------------------

    def _both_wrists_inside(
        self,
        student: StudentData,
    ) -> bool:
        """
        Returns True if both wrists lie inside the
        assigned desk ROI.
        """

        desk_id = student.get("desk_id")

        polygon = self._desk_map.get(desk_id)

        if polygon is None:
            return True

        left_inside = self._is_wrist_inside_desk(
            student.get("left_wrist"),
            polygon,
        )

        right_inside = self._is_wrist_inside_desk(
            student.get("right_wrist"),
            polygon,
        )

        return left_inside and right_inside

    # -------------------------------------------------------------

    def reset_student(
        self,
        student_id: int,
    ) -> None:
        """
        Reset one student's timers/results.
        """

        if student_id in self._timers:
            self._timers[student_id].reset()

        if student_id in self._results:
            self._results[student_id] = StudentRuleResults()

    # -------------------------------------------------------------

    def remove_student(
        self,
        student_id: int,
    ) -> None:
        """
        Remove student from memory.

        Called when ByteTrack permanently loses a track.
        """

        self._timers.pop(student_id, None)

        self._results.pop(student_id, None)

    # -------------------------------------------------------------

    def clear(
        self,
    ) -> None:
        """
        Remove every tracked student.
        """

        self._timers.clear()

        self._results.clear()

    # -------------------------------------------------------------

    def get_results(
        self,
    ) -> dict[int, StudentRuleResults]:
        """
        Return all rule results.
        """

        return self._results

    # -------------------------------------------------------------

    def get_student_result(
        self,
        student_id: int,
    ) -> StudentRuleResults | None:
        """
        Return one student's rule results.
        """

        return self._results.get(student_id)

    # -------------------------------------------------------------

    def student_count(
        self,
    ) -> int:
        """
        Number of tracked students.
        """

        return len(self._results)


# ---------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Phase 7 Rule Engine")
    print("=" * 60)

    desk_map = {
        1: [
            (100, 100),
            (300, 100),
            (300, 250),
            (100, 250),
        ]
    }

    engine = RuleEngine(desk_map)

    student = {
        "student_id": 1,
        "bbox": (100, 100, 200, 300),
        "left_wrist": (120, 260),
        "right_wrist": (180, 260),
        "left_shoulder": (130, 150),
        "right_shoulder": (170, 150),
        "nose": (150, 120),
        "left_eye": (142, 116),
        "right_eye": (158, 116),
        "left_ear": (134, 122),
        "right_ear": (166, 122),
        "head_pose": {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        },
        "desk_id": 1,
    }

    students = [student]

    result = engine.update(
        student=student,
        students=students,
    )

    print()

    print("Tracked students :", engine.student_count())

    print()

    print("Rule Results")

    print(result)

    print()

    print("Initialization successful.")
