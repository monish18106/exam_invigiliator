"""
ai/alerts.py

Phase 8 - Alert Manager

Responsibilities
----------------
- Alert definitions
- Alert data models
- Alert state management
- Duplicate suppression
- Alert history
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from ai.rules import RuleLevel
from ai.rules import StudentRuleResults



class AlertLevel(str, Enum):
    """
    Alert severity.
    """

    YELLOW = "YELLOW"
    RED = "RED"

class AlertType(str, Enum):
    """
    Supported alert types.
    """

    HANDS_OFF_DESK = "HANDS_OFF_DESK"

    PEEPING = "PEEPING"

    ITEM_EXCHANGE = "ITEM_EXCHANGE"

@dataclass(slots=True)
class AlertEvent:
    """
    Single alert event.
    """

    student_id: int

    alert_type: AlertType

    level: AlertLevel

    timestamp: float

    duration: float

    confidence: float

def current_timestamp() -> float:
    """
    Current monotonic timestamp.
    """

    return time.monotonic()

class AlertManager:
    """
    Generates alert events from rule results.

    Prevents duplicate alerts while maintaining
    alert history.
    """

    def __init__(self) -> None:

        self._previous_levels: dict[
            int,
            dict[AlertType, RuleLevel],
        ] = {}

        self._history: list[AlertEvent] = []

    def _ensure_student(
        self,
        student_id: int,
    ) -> None:
        """
        Allocate alert state for a student.
        """

        if student_id not in self._previous_levels:

            self._previous_levels[student_id] = {

                AlertType.HANDS_OFF_DESK: RuleLevel.GREEN,

                AlertType.PEEPING: RuleLevel.GREEN,

                AlertType.ITEM_EXCHANGE: RuleLevel.GREEN,
            }

    def _check_transition(
        self,
        student_id: int,
        alert_type: AlertType,
        rule,
    ) -> AlertEvent | None:
        """
        Detect rule level transitions.

        Returns an AlertEvent only when the rule
        changes from one alert level to another.
        """

        previous = self._previous_levels[student_id][alert_type]

        current = rule.level

        if previous == current:
            return None

        self._previous_levels[student_id][alert_type] = current

        if current == RuleLevel.GREEN:
            return None

        level = AlertLevel(current.value)

        event = AlertEvent(
            student_id=student_id,
            alert_type=alert_type,
            level=level,
            timestamp=current_timestamp(),
            duration=rule.duration,
            confidence=rule.confidence,
        )

        self._history.append(event)

        return event

    def update(
        self,
        student_id: int,
        results: StudentRuleResults,
    ) -> list[AlertEvent]:
        """
        Process one student's rule results.

        Returns newly generated alerts.
        """

        self._ensure_student(student_id)

        events: list[AlertEvent] = []

        checks = [
            (
                AlertType.HANDS_OFF_DESK,
                results.hands_off_desk,
            ),
            (
                AlertType.PEEPING,
                results.peeping,
            ),
            (
                AlertType.ITEM_EXCHANGE,
                results.item_exchange,
            ),
        ]

        for alert_type, rule in checks:

            event = self._check_transition(
                student_id=student_id,
                alert_type=alert_type,
                rule=rule,
            )

            if event is not None:
                events.append(event)

        return events

    def get_history(
        self,
    ) -> list[AlertEvent]:
        """
        Return alert history.
        """

        return self._history

    def clear(
        self,
    ) -> None:
        """
        Clear all stored alerts.
        """

        self._previous_levels.clear()

        self._history.clear()