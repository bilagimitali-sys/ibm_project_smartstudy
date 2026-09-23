"""
Feature: Progress Dashboard
Tracks and visualises the student's activity within the session.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _session() -> Dict[str, Any]:
    """Return the Streamlit session_state dict (imported lazily to keep
    this module testable without a running Streamlit server)."""
    import streamlit as st  # noqa: PLC0415
    return st.session_state


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------

def record_event(event_type: str, detail: str = "") -> None:
    """
    Append a timestamped event to the session activity log.

    event_type : one of "summary", "exam_questions", "flashcards",
                         "mcq_quiz", "study_plan", "upload"
    """
    state = _session()
    if "activity_log" not in state:
        state["activity_log"] = []
    state["activity_log"].append({
        "type": event_type,
        "detail": detail,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    # Increment per-feature counters
    key = f"count_{event_type}"
    state[key] = state.get(key, 0) + 1


def record_quiz_result(score: int, total: int) -> None:
    """Track quiz scores for the score history chart."""
    state = _session()
    if "quiz_scores" not in state:
        state["quiz_scores"] = []
    pct = round(score / total * 100) if total else 0
    state["quiz_scores"].append({
        "attempt": len(state["quiz_scores"]) + 1,
        "score": score,
        "total": total,
        "percentage": pct,
        "timestamp": datetime.now().strftime("%H:%M"),
    })


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

def render_dashboard() -> None:
    """Render the full progress dashboard inside the current Streamlit page."""
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    state = _session()

    st.markdown("## 📊 Progress Dashboard")
    st.caption("Your activity this session at a glance.")

    # ── Metric cards ────────────────────────────────────────────────────────
    cols = st.columns(5)
    metrics = [
        ("📄 Summaries",    "count_summary",         "#4F8EF7"),
        ("❓ Exam Qs",      "count_exam_questions",  "#F76B4F"),
        ("🃏 Flashcards",   "count_flashcards",      "#4FF7A0"),
        ("📝 MCQ Quizzes",  "count_mcq_quiz",        "#F7D24F"),
        ("📅 Study Plans",  "count_study_plan",      "#C04FF7"),
    ]
    for col, (label, key, color) in zip(cols, metrics):
        val = state.get(key, 0)
        col.metric(label, val)

    st.divider()

    # ── Quiz score history ──────────────────────────────────────────────────
    quiz_scores: List[Dict] = state.get("quiz_scores", [])
    if quiz_scores:
        st.markdown("### 🎯 Quiz Score History")
        df_scores = pd.DataFrame(quiz_scores)
        fig = px.line(
            df_scores,
            x="attempt",
            y="percentage",
            markers=True,
            labels={"attempt": "Attempt #", "percentage": "Score (%)"},
            title="Quiz Performance Over Time",
        )
        fig.update_traces(line_color="#4F8EF7", marker_color="#F7D24F")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Complete a quiz to see your score history here.")

    # ── Activity breakdown pie ──────────────────────────────────────────────
    activity_data = {
        label: state.get(key, 0)
        for label, key, _ in metrics
    }
    if sum(activity_data.values()) > 0:
        st.markdown("### 🔄 Feature Usage")
        fig2 = go.Figure(go.Pie(
            labels=list(activity_data.keys()),
            values=list(activity_data.values()),
            hole=0.4,
            marker_colors=["#4F8EF7", "#F76B4F", "#4FF7A0", "#F7D24F", "#C04FF7"],
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Recent activity log ─────────────────────────────────────────────────
    log: List[Dict] = state.get("activity_log", [])
    if log:
        st.markdown("### 🕐 Recent Activity")
        for entry in reversed(log[-10:]):
            icon_map = {
                "upload": "📁", "summary": "📄", "exam_questions": "❓",
                "flashcards": "🃏", "mcq_quiz": "📝", "study_plan": "📅",
            }
            icon = icon_map.get(entry["type"], "•")
            st.markdown(
                f"`{entry['timestamp']}` {icon} **{entry['type'].replace('_', ' ').title()}** "
                + (f"— {entry['detail']}" if entry["detail"] else "")
            )
    else:
        st.info("No activity yet. Start by uploading a document!")
