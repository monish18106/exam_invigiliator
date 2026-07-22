"""
dashboard/state.py

Dashboard state manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dashboard.db import DashboardDB


@dataclass
class DashboardState:
    """
    Holds the current dashboard state.

    Refreshes all dashboard data from PostgreSQL.
    """

    metrics: dict[str, Any] = field(default_factory=dict)

    alerts: list[dict[str, Any]] = field(default_factory=list)

    students: list[dict[str, Any]] = field(default_factory=list)

    evidence: list[dict[str, Any]] = field(default_factory=list)

    def refresh(self) -> None:
        """
        Refresh all dashboard data.
        """

        db = DashboardDB()

        try:
            self.metrics = db.get_dashboard_metrics()

            self.alerts = db.get_recent_alerts()

            self.students = db.get_student_status()

            self.evidence = db.get_recent_evidence()

        finally:
            db.close()