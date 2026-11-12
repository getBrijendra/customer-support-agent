"""Provider-agnostic factories for chat models and embeddings.

Adding another provider (OpenAI, Anthropic, ...) means one new branch in each factory.
"""

from __future__ import annotations

import threading

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


def _google_key(settings: Settings) -> str | None:
    return settings.GOOGLE_API_KEY.get_secret_value() if settings.GOOGLE_API_KEY else None


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            google_api_key=_google_key(settings),
        )
    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def get_embeddings(settings: Settings, model: str | None = None) -> Embeddings:
    if settings.LLM_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model or settings.EMBEDDING_MODEL,
            google_api_key=_google_key(settings),
        )
    raise ValueError(f"Unsupported embeddings provider: {settings.LLM_PROVIDER}")


class EmbeddingsRegistry:
    """Creates each embeddings client once per model name and reuses it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._clients: dict[str, Embeddings] = {}
        self._lock = threading.Lock()

    def __call__(self, model: str) -> Embeddings:
        with self._lock:
            if model not in self._clients:
                self._clients[model] = get_embeddings(self._settings, model)
            return self._clients[model]
