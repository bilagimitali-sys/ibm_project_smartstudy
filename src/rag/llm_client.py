"""
Groq LLM client — thin wrapper around the Groq Python SDK.

Reads GROQ_API_KEY from the environment (loaded via python-dotenv).
Falls back gracefully if the key is missing so the UI can display
a helpful error rather than crashing.
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Default model — fast and capable; can be overridden per call
DEFAULT_MODEL = "qwen/qwen3.8-27b"
# Larger model for complex generation tasks
LARGE_MODEL = "qwen/qwen3.8-27b"


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Create a .env file from .env.example and add your key."
        )
    return Groq(api_key=api_key)


def generate(
    prompt: str,
    system: str = "You are a helpful AI study assistant.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    """
    Send a chat-completion request to Groq and return the response text.
    Raises EnvironmentError if GROQ_API_KEY is missing.
    Raises RuntimeError for all other API errors.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}") from e


def generate_large(
    prompt: str,
    system: str = "You are a helpful AI study assistant.",
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> str:
    """Use the larger model for tasks that need deeper reasoning."""
    return generate(prompt, system=system, model=LARGE_MODEL,
                    temperature=temperature, max_tokens=max_tokens)
