"""
database/crud.py

Database CRUD operations.
"""

from __future__ import annotations

from sqlalchemy import desc
from sqlalchemy.orm import Session

from .models import Alert
from .models import Evidence


class DatabaseCRUD:
    """
    Encapsulates all database operations.
    """

    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    # ---------------------------------------------------------
    # Alerts
    # ---------------------------------------------------------

    def create_alert(
        self,
        student_id: int,
        rule_name: str,
        alert_level: str,
        created_at: float,
    ) -> Alert:
        """
        Insert a new alert.
        """

        alert = Alert(
            student_id=student_id,
            rule_name=rule_name,
            alert_level=alert_level,
            created_at=created_at,
        )

        self._db.add(alert)
        self._db.commit()
        self._db.refresh(alert)

        return alert

    def get_recent_alerts(
        self,
        limit: int = 100,
    ) -> list[Alert]:
        """
        Return latest alerts.
        """

        return (
            self._db.query(Alert)
            .order_by(desc(Alert.created_at))
            .limit(limit)
            .all()
        )

    def get_student_alerts(
        self,
        student_id: int,
    ) -> list[Alert]:
        """
        Return alerts for a student.
        """

        return (
            self._db.query(Alert)
            .filter(Alert.student_id == student_id)
            .order_by(desc(Alert.created_at))
            .all()
        )

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------

    def create_evidence(
        self,
        student_id: int,
        event_type: str,
        clip_path: str,
        created_at: float,
    ) -> Evidence:
        """
        Insert evidence metadata.
        """

        evidence = Evidence(
            student_id=student_id,
            event_type=event_type,
            clip_path=clip_path,
            created_at=created_at,
        )

        self._db.add(evidence)
        self._db.commit()
        self._db.refresh(evidence)

        return evidence

    def get_recent_evidence(
        self,
        limit: int = 100,
    ) -> list[Evidence]:
        """
        Return latest evidence.
        """

        return (
            self._db.query(Evidence)
            .order_by(desc(Evidence.created_at))
            .limit(limit)
            .all()
        )