"""
dashboard/components/metrics.py

Dashboard metric cards.
"""

from __future__ import annotations

import streamlit as st


def render_metrics(
    total_students: int = 0,
    active_alerts: int = 0,
    evidence_count: int = 0,
    fps: float = 0.0,
) -> None:
    """
    Render dashboard metric cards.

    Args:
        total_students: Number of detected students.
        active_alerts: Number of active alerts.
        evidence_count: Total evidence clips.
        fps: Current processing FPS.
    """

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="👨‍🎓 Students",
            value=total_students,
        )

    with col2:
        st.metric(
            label="🚨 Active Alerts",
            value=active_alerts,
        )

    with col3:
        st.metric(
            label="🎥 Evidence Clips",
            value=evidence_count,
        )

    with col4:
        st.metric(
            label="⚡ FPS",
            value=f"{fps:.1f}",
        )