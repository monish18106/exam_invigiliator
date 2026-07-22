"""
dashboard/db.py

Database access layer for the Streamlit dashboard.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import Alert, Evidence
from database.session import SessionLocal


class DashboardDB:
    """
    Data access layer for the Streamlit dashboard.
    """

    def __init__(self) -> None:
        self._session: Session = SessionLocal()

    def close(self) -> None:
        """Close the database session."""
        self._session.close()

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """
        Return summary metrics for the dashboard.
        """

        total_alerts = self._session.query(Alert).count()

        total_evidence = self._session.query(Evidence).count()

        active_alerts = (
            self._session.query(Alert)
            .filter(Alert.alert_level.in_(["YELLOW", "RED"]))
            .count()
        )

        students = (
            self._session.query(func.count(func.distinct(Alert.student_id)))
            .scalar()
            or 0
        )

        return {
            "students": students,
            "active_alerts": active_alerts,
            "evidence": total_evidence,
            "fps": 0.0,  # Updated later from app.py
        }

    # --------------------------------------------------
    # Alerts
    # --------------------------------------------------

    def get_recent_alerts(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Return recent alerts.
        """

        alerts = (
            self._session.query(Alert)
            .order_by(Alert.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "student_id": alert.student_id,
                "rule_name": alert.rule_name,
                "alert_level": alert.alert_level,
                "created_at": alert.created_at,
            }
            for alert in alerts
        ]

    # --------------------------------------------------
    # Student Monitoring
    # --------------------------------------------------

    def get_student_status(self) -> list[dict[str, Any]]:
        """
        Return the latest known status for each student.

        NOTE:
        Desk assignment and live status will be integrated
        with the AI pipeline in Phase 13.
        """

        alerts = (
            self._session.query(Alert)
            .order_by(Alert.created_at.desc())
            .all()
        )

        latest_students: dict[int, dict[str, Any]] = {}

        for alert in alerts:
            if alert.student_id in latest_students:
                continue

            latest_students[alert.student_id] = {
                "student_id": alert.student_id,
                "desk_id": "-",
                "status": (
                    "SUSPICIOUS"
                    if alert.alert_level == "RED"
                    else (
                        "WARNING"
                        if alert.alert_level == "YELLOW"
                        else "NORMAL"
                    )
                ),
                "current_rule": alert.rule_name,
            }

        return list(latest_students.values())

    # --------------------------------------------------
    # Evidence
    # --------------------------------------------------

    def get_recent_evidence(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Return recent evidence clips.
        """

        evidence = (
            self._session.query(Evidence)
            .order_by(Evidence.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "student_id": item.student_id,
                "event_type": item.event_type,
                "created_at": item.created_at,
                "clip_path": item.clip_path,
            }
            for item in evidence
        ]