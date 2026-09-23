"""
Feature: Personalized Study Plan
Generates a day-by-day study schedule tailored to the student's material,
available time, and exam date.
"""
from __future__ import annotations

from src.rag.llm_client import generate_large
from src.rag.vector_store import retrieve_context
from langchain_community.vectorstores import FAISS


SYSTEM = (
    "You are an expert academic coach who creates personalised, "
    "realistic, and motivating study plans for students."
)


def generate_study_plan(
    vectorstore: FAISS,
    exam_date: str,
    hours_per_day: int = 3,
    weak_topics: str = "",
    student_level: str = "Intermediate",
) -> str:
    """
    Generate a personalised study plan.

    Parameters
    ----------
    vectorstore : FAISS
    exam_date : str
        Target exam date (e.g. "2025-07-20").
    hours_per_day : int
        How many hours the student can study each day.
    weak_topics : str
        Comma-separated topics the student finds difficult.
    student_level : str
        "Beginner", "Intermediate", or "Advanced".
    """
    context = retrieve_context(
        vectorstore,
        "main topics overview table of contents chapters",
        k=8,
    )

    weak_section = (
        f"\nThe student finds these topics particularly challenging: {weak_topics}."
        if weak_topics.strip()
        else ""
    )

    prompt = f"""
A {student_level}-level student needs to study for an exam on {exam_date}.
They can study {hours_per_day} hours per day.{weak_section}

Here is their study material:

--- STUDY MATERIAL ---
{context}
--- END OF MATERIAL ---

Create a personalised, day-by-day study plan from today until {exam_date}.

Structure the plan as:

## 🎯 Study Goal
(One sentence goal statement)

## 📅 Daily Study Schedule

### Week 1
| Day | Topics | Duration | Activities |
|-----|--------|----------|------------|
(Fill in each day)

### Week 2 (if applicable)
(Continue...)

## ⚡ Key Strategies
(3-5 bullet points with specific study techniques for this material)

## 📊 Milestones
(List 3-4 checkpoints to track progress)

## 💪 Final Week Revision Plan
(Focused review strategy for the last 3 days before the exam)

Be realistic about the time available and prioritise the most important topics.
"""
    return generate_large(prompt, system=SYSTEM, max_tokens=1800)
