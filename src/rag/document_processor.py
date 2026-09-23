"""
PDF / text ingestion — splits documents into overlapping chunks
and returns LangChain Document objects ready for embedding.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _extract_text_from_pdf(file_path: str | Path) -> str:
    """Return the full plain-text content of a PDF file."""
    reader = PdfReader(str(file_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_text_from_txt(file_path: str | Path) -> str:
    """Return the content of a plain-text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(file_path: str | Path) -> str:
    """
    Load a single document (PDF or TXT) and return its raw text.
    Raises ValueError for unsupported extensions.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_text_from_pdf(path)
    elif ext in {".txt", ".md"}:
        return _extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_documents(text: str, source: str = "uploaded_file") -> List[Document]:
    """
    Split raw text into overlapping chunks and wrap each in a
    LangChain Document with metadata so the retriever can cite sources.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=chunk, metadata={"source": source, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]
    return docs


def process_uploaded_files(uploaded_files: list, save_dir: str = "uploaded_docs") -> List[Document]:
    """
    Accept a list of Streamlit UploadedFile objects, persist them to
    *save_dir*, and return a flat list of LangChain Document chunks.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        dest = Path(save_dir) / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            raw_text = load_document(dest)
            chunks = split_documents(raw_text, source=uploaded_file.name)
            all_docs.extend(chunks)
        except Exception as e:
            # Surface the error to the caller; the UI will display it
            raise RuntimeError(f"Could not process '{uploaded_file.name}': {e}") from e

    return all_docs
