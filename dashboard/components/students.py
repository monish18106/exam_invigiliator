"""
dashboard/components/students.py

Student monitoring component for the
AI-Assisted Exam Proctoring Dashboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


STATUS_ICONS = {
    "NORMAL": "🟢",
    "WARNING": "🟡",
    "SUSPICIOUS": "🔴",
}


def render_students(
    students: list[dict[str, Any]] | None = None,
) -> None:
    """
    Render the student monitoring table.

    Expected format
    ---------------
    [
        {
            "student_id": 1,
            "desk_id": 3,
            "status": "NORMAL",
            "current_rule": "-"
        }
    ]
    """

    st.subheader("👨‍🎓 Student Monitoring")

    if not students:
        st.info("No students detected.")
        return

    rows = []

    for student in students:
        rows.append(
            {
                "Status": STATUS_ICONS.get(
                    str(student["status"]).upper(),
                    "⚪",
                ),
                "Student": student["student_id"],
                "Desk": student["desk_id"],
                "Current Rule": student["current_rule"],
            }
        )

    dataframe = pd.DataFrame(rows)

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        height=320,
    )