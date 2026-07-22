"""
dashboard/dashboard.py

Main Streamlit dashboard entry point for the
AI-Assisted Exam Proctoring System.
"""

from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard.state import DashboardState
from dashboard.live_status import LiveStatus

# Dashboard Components
from dashboard.components.header import render_header
from dashboard.components.metrics import render_metrics
from dashboard.components.live_feed import render_live_feed
from dashboard.components.alerts import render_alerts
from dashboard.components.students import render_students
from dashboard.components.evidence import render_evidence


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Assisted Exam Proctoring Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Auto Refresh
# --------------------------------------------------

st_autorefresh(
    interval=1000,  # milliseconds
    key="dashboard_refresh",
)

# --------------------------------------------------
# Load Dashboard Data
# --------------------------------------------------

state = DashboardState()
state.refresh()

runtime = LiveStatus().read()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("⚙️ System")

    st.divider()

    st.markdown("### 📷 Camera")

    if runtime.get("camera_connected", False):
        st.success("Connected")
    else:
        st.error("Disconnected")

    st.markdown("### 🧠 AI Pipeline")

    if runtime.get("pipeline_running", False):
        st.success("Running")
    else:
        st.error("Stopped")

    st.markdown("### ⚡ FPS")

    st.metric(
        label="Current FPS",
        value=f"{runtime.get('fps', 0.0):.1f}",
    )

    st.markdown("### 🔄 Refresh")
    st.write("Every 1 second")

    st.markdown("### 🏷 Version")
    st.write("MVP v1.0")

# --------------------------------------------------
# Header
# --------------------------------------------------

render_header()

st.divider()

# --------------------------------------------------
# Metrics
# --------------------------------------------------

render_metrics(
    total_students=state.metrics.get("students", 0),
    active_alerts=state.metrics.get("active_alerts", 0),
    evidence_count=state.metrics.get("evidence", 0),
    fps=runtime.get("fps", 0.0),
)

st.divider()

# --------------------------------------------------
# Live Feed + Alerts
# --------------------------------------------------

left_column, right_column = st.columns([3, 1])

with left_column:
    render_live_feed()

with right_column:
    render_alerts(state.alerts)

st.divider()

# --------------------------------------------------
# Student Monitoring
# --------------------------------------------------

render_students(state.students)

st.divider()

# --------------------------------------------------
# Evidence Viewer
# --------------------------------------------------

render_evidence(state.evidence)