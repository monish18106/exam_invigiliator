"""
dashboard/components/header.py

Header component for the AI-Assisted Exam Proctoring Dashboard.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st


def render_header() -> None:
    """
    Render the dashboard header.

    Displays:
    - Dashboard title
    - Current timestamp
    - System status placeholders
    """

    title_col, status_col = st.columns([4, 1])

    with title_col:
        st.title("🎓 AI-Assisted Exam Proctoring Dashboard")
        st.caption("Real-time Examination Monitoring System")

    with status_col:
        st.metric(
            label="Time",
            value=datetime.now().strftime("%H:%M:%S"),
        )

    camera_col, database_col, pipeline_col = st.columns(3)

    with camera_col:
        st.success("📷 Camera: Connected")

    with database_col:
        st.success("🗄️ Database: Connected")

    with pipeline_col:
        st.success("🧠 AI Pipeline: Running")