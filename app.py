"""
StudyMate AI — Smart Study Generator Agent
Main Streamlit application entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal imports ─────────────────────────────────────────────────────────
from src.rag.document_processor import process_uploaded_files
from src.rag.vector_store import build_vectorstore, load_vectorstore
from src.features.summary import generate_summary
from src.features.exam_questions import generate_exam_questions
from src.features.flashcards import generate_flashcards
from src.features.mcq_quiz import generate_mcq_quiz
from src.features.study_plan import generate_study_plan
from src.features.dashboard import render_dashboard, record_event, record_quiz_result


# ── Session state initialisation ─────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "vectorstore": None,
        "uploaded_filenames": [],
        "activity_log": [],
        "quiz_scores": [],
        "current_quiz": None,       # List[MCQQuestion]
        "quiz_answers": {},         # {q_index: chosen_option}
        "quiz_submitted": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> str:
    """Render sidebar and return the selected page name."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/graduation-cap.png",
            width=72,
        )
        st.title("StudyMate AI")
        st.caption("Your AI-powered study companion 🎓")
        st.divider()

        page = st.radio(
            "Navigate",
            options=[
                "📁 Upload Documents",
                "📄 Summary",
                "❓ Exam Questions",
                "🃏 Flashcards",
                "📝 MCQ Quiz",
                "📅 Study Plan",
                "📊 Dashboard",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # Uploaded files indicator
        files = st.session_state.get("uploaded_filenames", [])
        if files:
            st.markdown("**📂 Loaded Documents**")
            for fn in files:
                st.markdown(f"- `{fn}`")
        else:
            st.info("No documents loaded yet.")

        st.divider()
        st.caption("Powered by Groq · LangChain · FAISS")
        st.caption("© 2025 StudyMate AI")

    return page


# ── Upload page ───────────────────────────────────────────────────────────────
def page_upload() -> None:
    st.markdown("# 📁 Upload Your Study Documents")
    st.markdown(
        "Upload your **PDFs** or **text notes** and StudyMate AI will index them "
        "so you can generate summaries, questions, flashcards, and more."
    )

    uploaded = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown",
    )

    if uploaded:
        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            with st.spinner("Reading and indexing your documents…"):
                try:
                    docs = process_uploaded_files(uploaded)
                    vectorstore = build_vectorstore(docs)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["uploaded_filenames"] = [f.name for f in uploaded]
                    record_event("upload", f"{len(uploaded)} file(s), {len(docs)} chunks")
                    st.success(
                        f"✅ Successfully indexed **{len(uploaded)} file(s)** "
                        f"into **{len(docs)} chunks**. "
                        "Use the sidebar to start generating study content!"
                    )
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")

    # If already loaded, show info
    if st.session_state.get("vectorstore") is not None:
        st.divider()
        st.success(
            "📚 Documents are loaded and ready. "
            "Navigate to any feature in the sidebar."
        )


# ── Helper: guard pages that need a vector store ─────────────────────────────
def _require_documents() -> bool:
    if st.session_state.get("vectorstore") is None:
        st.warning("⚠️ Please upload and process your study documents first (📁 Upload Documents).")
        return False
    return True


# ── Summary page ──────────────────────────────────────────────────────────────
def page_summary() -> None:
    st.markdown("# 📄 AI Summary")
    st.markdown("Get a clear, structured summary of your study material.")
    if not _require_documents():
        return

    topic = st.text_input(
        "Focus topic (optional)",
        placeholder="e.g. photosynthesis, World War II, derivatives…",
    )
    if st.button("✨ Generate Summary", type="primary", use_container_width=True):
        with st.spinner("Generating summary…"):
            try:
                result = generate_summary(st.session_state["vectorstore"], topic)
                record_event("summary", topic or "full document")
                st.markdown(result)
                st.divider()
                st.download_button(
                    "⬇️ Download Summary",
                    data=result,
                    file_name="summary.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"❌ {e}")


# ── Exam Questions page ───────────────────────────────────────────────────────
def page_exam_questions() -> None:
    st.markdown("# ❓ Exam Questions")
    st.markdown("Generate targeted exam-style questions with model answers.")
    if not _require_documents():
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        topic = st.text_input("Topic (optional)", placeholder="e.g. cell biology")
    with col2:
        num_q = st.slider("Number of questions", 5, 20, 10)
    with col3:
        difficulty = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])

    if st.button("🎯 Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating exam questions…"):
            try:
                result = generate_exam_questions(
                    st.session_state["vectorstore"],
                    topic=topic,
                    num_questions=num_q,
                    difficulty=difficulty,
                )
                record_event("exam_questions", f"{num_q} {difficulty} questions")
                st.markdown(result)
                st.divider()
                st.download_button(
                    "⬇️ Download Questions",
                    data=result,
                    file_name="exam_questions.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"❌ {e}")


# ── Flashcards page ───────────────────────────────────────────────────────────
def page_flashcards() -> None:
    st.markdown("# 🃏 Flashcards")
    st.markdown("Flip through AI-generated flashcards to test your recall.")
    if not _require_documents():
        return

    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic (optional)", placeholder="e.g. organic chemistry")
    with col2:
        num_cards = st.slider("Number of flashcards", 5, 30, 15)

    if st.button("🃏 Generate Flashcards", type="primary", use_container_width=True):
        with st.spinner("Generating flashcards…"):
            try:
                raw, cards = generate_flashcards(
                    st.session_state["vectorstore"],
                    topic=topic,
                    num_cards=num_cards,
                )
                record_event("flashcards", f"{len(cards)} cards generated")
                st.session_state["flashcards_data"] = cards
            except Exception as e:
                st.error(f"❌ {e}")
                return

    # Display flashcards
    cards = st.session_state.get("flashcards_data", [])
    if cards:
        st.markdown(f"### {len(cards)} Flashcards Ready")
        for card in cards:
            with st.expander(f"🃏 Card {card.index}: {card.question[:80]}…" if len(card.question) > 80 else f"🃏 Card {card.index}: {card.question}"):
                st.markdown(f"**Q:** {card.question}")
                st.markdown("---")
                st.markdown(f"**A:** {card.answer}")


# ── MCQ Quiz page ─────────────────────────────────────────────────────────────
def page_mcq_quiz() -> None:
    st.markdown("# 📝 MCQ Quiz")
    st.markdown("Test your knowledge with AI-generated multiple-choice questions.")
    if not _require_documents():
        return

    # ── Generation controls ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic (optional)", placeholder="e.g. thermodynamics")
    with col2:
        num_q = st.slider("Number of questions", 5, 20, 10)

    if st.button("🔀 Generate New Quiz", type="primary", use_container_width=True):
        with st.spinner("Generating quiz…"):
            try:
                raw, questions = generate_mcq_quiz(
                    st.session_state["vectorstore"],
                    topic=topic,
                    num_questions=num_q,
                )
                record_event("mcq_quiz", f"{len(questions)} questions")
                st.session_state["current_quiz"] = questions
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_submitted"] = False
            except Exception as e:
                st.error(f"❌ {e}")
                return

    questions = st.session_state.get("current_quiz")
    if not questions:
        return

    # ── Quiz form ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 📋 Quiz — {len(questions)} Questions")

    answers: dict = st.session_state["quiz_answers"]
    submitted: bool = st.session_state["quiz_submitted"]

    with st.form("quiz_form"):
        for q in questions:
            st.markdown(f"**Q{q.index}. {q.question}**")
            opt_labels = [o[0] for o in q.options]   # ["A", "B", "C", "D"]
            chosen = st.radio(
                f"q_{q.index}",
                options=q.options,
                key=f"radio_{q.index}",
                label_visibility="collapsed",
                disabled=submitted,
            )
            answers[q.index] = chosen[0] if chosen else None   # store just the letter
            st.markdown("")

        if not submitted:
            submit_btn = st.form_submit_button(
                "✅ Submit Quiz", type="primary", use_container_width=True
            )
            if submit_btn:
                st.session_state["quiz_submitted"] = True
                st.rerun()

    # ── Results ────────────────────────────────────────────────────────────
    if submitted:
        correct = sum(
            1 for q in questions if answers.get(q.index) == q.correct_option
        )
        total = len(questions)
        pct = round(correct / total * 100)
        record_quiz_result(correct, total)

        if pct >= 80:
            grade_msg = "🏆 Excellent work!"
            color = "green"
        elif pct >= 60:
            grade_msg = "👍 Good effort!"
            color = "orange"
        else:
            grade_msg = "📚 Keep studying!"
            color = "red"

        st.markdown(f"### Results: :{color}[{correct}/{total} ({pct}%) — {grade_msg}]")
        st.progress(pct / 100)

        st.divider()
        st.markdown("### 📖 Answer Review")
        for q in questions:
            user_ans = answers.get(q.index)
            is_correct = user_ans == q.correct_option
            icon = "✅" if is_correct else "❌"
            with st.expander(f"{icon} Q{q.index}: {q.question[:70]}…" if len(q.question) > 70 else f"{icon} Q{q.index}: {q.question}"):
                for opt in q.options:
                    letter = opt[0]
                    if letter == q.correct_option:
                        st.markdown(f"**✅ {opt}** ← Correct Answer")
                    elif letter == user_ans:
                        st.markdown(f"**❌ {opt}** ← Your Answer")
                    else:
                        st.markdown(f"{opt}")
                if q.explanation:
                    st.info(f"💡 {q.explanation}")

        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state["quiz_submitted"] = False
            st.session_state["quiz_answers"] = {}
            st.rerun()


# ── Study Plan page ───────────────────────────────────────────────────────────
def page_study_plan() -> None:
    st.markdown("# 📅 Personalized Study Plan")
    st.markdown("Get a day-by-day study schedule tailored to your material and timeline.")
    if not _require_documents():
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        exam_date = st.date_input("📅 Exam Date")
    with col2:
        hours = st.slider("⏰ Study hours per day", 1, 10, 3)
    with col3:
        level = st.selectbox("📊 Your level", ["Beginner", "Intermediate", "Advanced"])

    weak_topics = st.text_area(
        "😓 Topics you find difficult (optional)",
        placeholder="e.g. integration, organic reactions, market equilibrium…",
        height=80,
    )

    if st.button("📅 Generate Study Plan", type="primary", use_container_width=True):
        with st.spinner("Creating your personalised study plan…"):
            try:
                result = generate_study_plan(
                    st.session_state["vectorstore"],
                    exam_date=str(exam_date),
                    hours_per_day=hours,
                    weak_topics=weak_topics,
                    student_level=level,
                )
                record_event("study_plan", f"exam {exam_date}, {hours}h/day")
                st.markdown(result)
                st.divider()
                st.download_button(
                    "⬇️ Download Study Plan",
                    data=result,
                    file_name="study_plan.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"❌ {e}")


# ── Main router ───────────────────────────────────────────────────────────────
def main() -> None:
    _init_state()
    page = _render_sidebar()

    page_map = {
        "📁 Upload Documents": page_upload,
        "📄 Summary":          page_summary,
        "❓ Exam Questions":   page_exam_questions,
        "🃏 Flashcards":       page_flashcards,
        "📝 MCQ Quiz":         page_mcq_quiz,
        "📅 Study Plan":       page_study_plan,
        "📊 Dashboard":        render_dashboard,
    }

    page_fn = page_map.get(page, page_upload)
    page_fn()


if __name__ == "__main__":
    main()
