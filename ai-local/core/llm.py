"""Utilities for loading chat language models."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


def _load_env() -> None:
    """Ensure environment variables from a .env file are loaded."""
    load_dotenv()


@lru_cache(maxsize=1)
def get_chat_model():
    """Return a LangChain compatible chat model.

    Preference order:
    1. OpenAI (when an API key is provided and the optional dependency is installed).
    2. Ollama running locally (default).
    """

    _load_env()

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore

            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                timeout=60,
            )
        except Exception:
            # Fall back to Ollama if the optional dependency or configuration is missing.
            pass

    from langchain_community.chat_models import ChatOllama

    model = os.getenv("LLM_MODEL", "llama3")
    base_url: Optional[str] = os.getenv("OLLAMA_HOST")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.2,
        streaming=False,
    )


__all__ = ["get_chat_model"]
