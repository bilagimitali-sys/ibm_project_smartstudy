"""
Feature: Exam Questions
Generates targeted exam-style questions from the study material.
"""
from __future__ import annotations

from src.rag.llm_client import generate_large
from src.rag.vector_store import retrieve_context
from langchain_community.vectorstores import FAISS


SYSTEM = (
    "You are an experienced university professor creating exam questions. "
    "Your questions test deep understanding, not just memorisation."
)


def generate_exam_questions(
    vectorstore: FAISS,
    topic: str = "",
    num_questions: int = 10,
    difficulty: str = "Mixed",
) -> str:
    """
    Generate exam questions from the study material.

    Parameters
    ----------
    vectorstore : FAISS
    topic : str
        Optional specific topic to focus on.
    num_questions : int
        How many questions to generate (5–20).
    difficulty : str
        "Easy", "Medium", "Hard", or "Mixed".
    """
    query = topic if topic.strip() else "key concepts and important topics"
    context = retrieve_context(vectorstore, query, k=8)

    focus = f" specifically about: {topic}" if topic.strip() else ""
    prompt = f"""
Based on the following study material{focus}, generate {num_questions} exam questions.
Difficulty level: {difficulty}

--- STUDY MATERIAL ---
{context}
--- END OF MATERIAL ---

Format each question as:

**Q[number]. [Question text]**
*Difficulty: [Easy/Medium/Hard]*
💡 **Model Answer:** [Detailed answer with explanation]

---

Include a mix of:
- Short answer questions
- Analytical / "explain why" questions  
- Compare and contrast questions
- Application questions

Generate exactly {num_questions} questions now.
"""
    return generate_large(prompt, system=SYSTEM, max_tokens=4096)
