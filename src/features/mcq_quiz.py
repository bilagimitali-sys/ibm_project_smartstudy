"""
Feature: MCQ Quiz
Generates multiple-choice questions with 4 options and correct answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.rag.llm_client import generate_large
from src.rag.vector_store import retrieve_context
from langchain_community.vectorstores import FAISS


SYSTEM = (
    "You are a quiz master creating well-crafted multiple-choice questions "
    "for university-level examinations."
)


@dataclass
class MCQQuestion:
    index: int
    question: str
    options: List[str]          # ["A. ...", "B. ...", "C. ...", "D. ..."]
    correct_option: str         # "A", "B", "C", or "D"
    explanation: str
    user_answer: Optional[str] = None   # filled in by the UI


def _parse_mcqs(raw: str) -> List[MCQQuestion]:
    """
    Parse LLM output into MCQQuestion objects.

    Expected block format:
        Q1. <question>
        A. option
        B. option
        C. option
        D. option
        ✅ Answer: A
        Explanation: ...
    """
    questions: List[MCQQuestion] = []

    # Split into question blocks
    blocks = re.split(r"\n(?=Q\d+[\.\)])", raw.strip())

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 6:
            continue

        # Question line
        q_line = re.sub(r"^Q\d+[\.\)]\s*", "", lines[0])

        # Options
        opts = []
        for line in lines[1:5]:
            if re.match(r"^[A-D][\.\)]\s", line):
                opts.append(line)

        if len(opts) != 4:
            continue

        # Correct answer
        answer_match = re.search(r"Answer\s*:\s*([A-D])", block, re.IGNORECASE)
        correct = answer_match.group(1).upper() if answer_match else "A"

        # Explanation
        exp_match = re.search(r"Explanation\s*:\s*(.+?)(?=Q\d+|$)", block, re.DOTALL | re.IGNORECASE)
        explanation = exp_match.group(1).strip() if exp_match else ""

        questions.append(MCQQuestion(
            index=len(questions) + 1,
            question=q_line,
            options=opts,
            correct_option=correct,
            explanation=explanation,
        ))

    return questions


def generate_mcq_quiz(
    vectorstore: FAISS,
    topic: str = "",
    num_questions: int = 10,
) -> tuple[str, List[MCQQuestion]]:
    """
    Generate a multiple-choice quiz from the study material.

    Returns
    -------
    raw : str
    questions : List[MCQQuestion]
    """
    query = topic if topic.strip() else "important concepts and facts"
    context = retrieve_context(vectorstore, query, k=8)

    focus = f" about: {topic}" if topic.strip() else ""
    prompt = f"""
Create {num_questions} multiple-choice questions{focus} based on the study material below.

--- STUDY MATERIAL ---
{context}
--- END OF MATERIAL ---

Format EVERY question EXACTLY like this:

Q[number]. [Question text]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
✅ Answer: [A/B/C/D]
Explanation: [1-2 sentence explanation of why the answer is correct]

Generate all {num_questions} questions now.
"""
    raw = generate_large(prompt, system=SYSTEM, max_tokens=1800)
    questions = _parse_mcqs(raw)
    return raw, questions
