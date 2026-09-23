"""
FAISS-backed vector store — build, persist, and query.

We use HuggingFace sentence-transformers for embeddings so that
the RAG pipeline works fully offline without an extra API key.
"""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
TOP_K = 5


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(documents: List[Document]) -> FAISS:
    """
    Create a FAISS index from *documents*, persist it to disk,
    and return the store object.
    """
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore


def load_vectorstore() -> Optional[FAISS]:
    """
    Load a previously persisted FAISS index from disk.
    Returns None if no index exists yet.
    """
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def get_retriever(vectorstore: FAISS, k: int = TOP_K):
    """Return a LangChain retriever that fetches the top-*k* chunks."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_context(vectorstore: FAISS, query: str, k: int = TOP_K) -> str:
    """
    Run a similarity search and concatenate the retrieved chunks into
    a single context string suitable for passing to the LLM.
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
