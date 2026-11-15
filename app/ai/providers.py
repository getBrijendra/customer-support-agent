"""Provider-agnostic factories for chat models and embeddings, plus a structured-output helper.

Adding another provider (OpenAI, Anthropic, ...) means one new branch in each factory.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.core.config import Settings

Role = Literal["generator", "compiler", "router"]


class LLMCallError(Exception):
    """An LLM call failed, timed out, or returned unparseable output."""


def _google_key(settings: Settings) -> str | None:
    return settings.GOOGLE_API_KEY.get_secret_value() if settings.GOOGLE_API_KEY else None


def model_name_for(settings: Settings, role: Role) -> str:
    return {
        "generator": settings.LLM_MODEL,
        "compiler": settings.COMPILER_MODEL,
        "router": settings.ROUTER_MODEL,
    }[role]


def get_chat_model(settings: Settings, role: Role = "generator") -> BaseChatModel:
    if settings.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name_for(settings, role),
            temperature=0.0 if role == "router" else settings.LLM_TEMPERATURE,
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
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
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


@dataclass
class StructuredResult[M: BaseModel]:
    output: M
    input_tokens: int | None = None
    output_tokens: int | None = None


async def invoke_structured[T: BaseModel](
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    schema: type[T],
    inputs: dict[str, Any],
    *,
    timeout_s: float,
) -> StructuredResult[T]:
    """prompt | llm.with_structured_output(schema, include_raw=True) with a hard timeout."""
    chain = prompt | llm.with_structured_output(schema, include_raw=True)
    try:
        result = await asyncio.wait_for(chain.ainvoke(inputs), timeout=timeout_s)
    except TimeoutError as exc:
        raise LLMCallError("LLM call timed out") from exc
    except Exception as exc:
        raise LLMCallError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc
    parsed = result.get("parsed") if isinstance(result, dict) else result
    raw = result.get("raw") if isinstance(result, dict) else None
    if parsed is None:
        err = result.get("parsing_error") if isinstance(result, dict) else None
        raise LLMCallError(f"LLM returned unparseable output: {err}")
    if isinstance(parsed, dict):
        parsed = schema.model_validate(parsed)
    in_tok = out_tok = None
    if isinstance(raw, AIMessage) and raw.usage_metadata:
        in_tok = raw.usage_metadata.get("input_tokens")
        out_tok = raw.usage_metadata.get("output_tokens")
    return StructuredResult(output=parsed, input_tokens=in_tok, output_tokens=out_tok)  # type: ignore[arg-type]
