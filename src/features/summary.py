"""
Feature: AI Summary
Generates a concise, structured summary of the uploaded study material.
"""
from __future__ import annotations

from src.rag.llm_client import generate_large
from src.rag.vector_store import retrieve_context
from langchain_community.vectorstores import FAISS


SYSTEM = (
    "You are an expert academic tutor. "
    "Your summaries are clear, well-structured, and exam-focused."
)


def generate_summary(vectorstore: FAISS, topic: str = "") -> str:
    """
    Retrieve the most relevant content from the vector store and
    return a detailed markdown-formatted summary.

    Parameters
    ----------
    vectorstore : FAISS
        The populated vector store built from the student's documents.
    topic : str
        Optional focus topic. If empty, summarises the whole document.
    """
    query = topic if topic.strip() else "main topics and key concepts"
    context = retrieve_context(vectorstore, query, k=6)

    focus = f" focusing on: {topic}" if topic.strip() else ""
    prompt = f"""
You are summarising study material for a student{focus}.

--- STUDY MATERIAL ---
{context}
--- END OF MATERIAL ---

Please provide a comprehensive summary structured as follows:

## 📚 Overview
(2-3 sentence high-level overview)

## 🔑 Key Concepts
(Bullet list of the most important concepts, terms, and definitions)

## 📌 Important Points to Remember
(Numbered list of critical facts and insights)

## 🔗 How Ideas Connect
(Brief paragraph explaining relationships between the main ideas)

Keep your language clear and student-friendly.
"""
    return generate_large(prompt, system=SYSTEM)
