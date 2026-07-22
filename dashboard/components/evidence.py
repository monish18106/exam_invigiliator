"""
dashboard/components/evidence.py

Evidence viewer component for the
AI-Assisted Exam Proctoring Dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def render_evidence(
    evidence_records: list[dict[str, Any]] | None = None,
) -> None:
    """
    Render suspicious evidence records and video player.

    Expected Format
    ---------------
    [
        {
            "student_id": 3,
            "event_type": "Phone Detected",
            "created_at": "10:42:17",
            "clip_path": "clips/student_3_phone.mp4"
        }
    ]
    """

    st.subheader("🎥 Evidence Clips")

    if not evidence_records:
        st.info("No evidence clips available.")
        return

    rows = []

    for record in evidence_records:
        rows.append(
            {
                "Student": record["student_id"],
                "Event": record["event_type"],
                "Time": record["created_at"],
            }
        )

    dataframe = pd.DataFrame(rows)

    selection = st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = selection.selection.rows

    if not selected_rows:
        return

    selected_index = selected_rows[0]

    selected_record = evidence_records[selected_index]

    clip_path = Path(selected_record["clip_path"])

    st.divider()

    st.markdown("### ▶ Evidence Playback")

    st.write(f"**Student:** {selected_record['student_id']}")
    st.write(f"**Event:** {selected_record['event_type']}")
    st.write(f"**Time:** {selected_record['created_at']}")

    if clip_path.exists():
        st.video(str(clip_path))
    else:
        st.warning("Evidence clip not found.")