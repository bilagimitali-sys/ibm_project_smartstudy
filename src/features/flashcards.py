"""
Feature: Flashcards
Generates Q&A flashcard pairs from the study material.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.rag.llm_client import generate_large
from src.rag.vector_store import retrieve_context
from langchain_community.vectorstores import FAISS


SYSTEM = (
    "You are a study-aid creator. "
    "You make concise, accurate flashcards that help students memorise key facts."
)


@dataclass
class Flashcard:
    question: str
    answer: str
    index: int = 0


def _parse_flashcards(raw: str) -> List[Flashcard]:
    """
    Parse the LLM output into a list of Flashcard objects.
    Expected format per card:
        FRONT: <question>
        BACK: <answer>
    """
    cards: List[Flashcard] = []
    # Split on card boundaries
    blocks = re.split(r"\n\s*---+\s*\n", raw)
    for i, block in enumerate(blocks):
        front_m = re.search(r"FRONT\s*:\s*(.+?)(?=BACK\s*:|$)", block, re.DOTALL | re.IGNORECASE)
        back_m = re.search(r"BACK\s*:\s*(.+?)$", block, re.DOTALL | re.IGNORECASE)
        if front_m and back_m:
            cards.append(Flashcard(
                question=front_m.group(1).strip(),
                answer=back_m.group(1).strip(),
                index=i + 1,
            ))
    return cards


def generate_flashcards(
    vectorstore: FAISS,
    topic: str = "",
    num_cards: int = 15,
) -> tuple[str, List[Flashcard]]:
    """
    Generate flashcards from the study material.

    Returns
    -------
    raw : str
        Raw LLM output (for display in a text area if desired).
    cards : List[Flashcard]
        Parsed flashcard objects for the interactive UI.
    """
    query = topic if topic.strip() else "key terms definitions and concepts"
    context = retrieve_context(vectorstore, query, k=8)

    focus = f" on the topic: {topic}" if topic.strip() else ""
    prompt = f"""
Create {num_cards} flashcards{focus} from the study material below.

--- STUDY MATERIAL ---
{context}
--- END OF MATERIAL ---

Format EVERY flashcard exactly like this (keep the separators):

FRONT: [Question or term]
BACK: [Answer or definition — keep it under 3 sentences]

---

Generate all {num_cards} flashcards now.
"""
    raw = generate_large(prompt, system=SYSTEM, max_tokens=4096)
    cards = _parse_flashcards(raw)
    return raw, cards
