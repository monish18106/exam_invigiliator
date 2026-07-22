"""
dashboard/components/alerts.py

Live alerts panel for the AI-Assisted Exam Proctoring Dashboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


LEVEL_ICONS = {
    "GREEN": "🟢",
    "YELLOW": "🟡",
    "RED": "🔴",
}


def render_alerts(alerts: list[dict[str, Any]] | None = None) -> None:
    """
    Render the live alerts panel.

    Args:
        alerts:
            List of alert dictionaries.

            Expected format:
            [
                {
                    "student_id": 1,
                    "rule_name": "Looking Away",
                    "alert_level": "YELLOW",
                    "created_at": "10:42:15"
                }
            ]
    """

    st.subheader("🚨 Live Alerts")

    if not alerts:
        st.success("No active alerts.")
        return

    rows = []

    for alert in alerts:
        rows.append(
            {
                "Level": LEVEL_ICONS.get(
                    str(alert["alert_level"]).upper(),
                    "⚪",
                ),
                "Student": alert["student_id"],
                "Rule": alert["rule_name"],
                "Time": alert["created_at"],
            }
        )

    dataframe = pd.DataFrame(rows)

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        height=500,
    )