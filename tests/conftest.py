"""Shared test setup for every tier: fake embeddings, fake LLM generators, settings factory.

No test needs a Gemini key or hosted Supabase.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from app.ai.guidance_chain import GenerationError, GenerationResult
from app.ai.schemas import GuidanceOutput
from app.core.config import Settings
from app.vectorstore.base import RetrievedChunk

AGENT_KEY = "agent-test-key"
ADMIN_KEY = "admin-test-key"
AGENT_HEADERS = {"X-API-Key": AGENT_KEY}
ADMIN_HEADERS = {"X-API-Key": ADMIN_KEY}

_TIERS = {"unit": "unit", "integration": "integration", "live": "live"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag every test with its tier marker based on its folder (tests/unit, ...)."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        for folder, marker in _TIERS.items():
            if folder in parts:
                item.add_marker(getattr(pytest.mark, marker))


# --------------------------------------------------------------------- fakes
class FakeEmbeddingsFactory:
    """Same deterministic fake for every model name; records which models were requested."""

    def __init__(self, size: int = 64) -> None:
        self.size = size
        self.requested: list[str] = []
        self._emb = DeterministicFakeEmbedding(size=size)

    def __call__(self, model: str) -> Embeddings:
        self.requested.append(model)
        return self._emb


def default_guidance(chunks: list[RetrievedChunk]) -> GuidanceOutput:
    return GuidanceOutput(
        recommended_response="Hi there, thanks for reaching out. Here is what we can do...",
        handling_instructions=["Check the order in the admin panel", "Offer the standard option"],
        requires_escalation=False,
        escalation_reason=None,
        detected_intent="refund_request",
        guideline_references=[c.chunk_id for c in chunks[:2]] + ["made-up:9999"],
        confidence="high",
        missing_information=["Order number"],
    )


class FakeGuidanceGenerator:
    """Implements the GuidanceGenerator interface with a fixed GuidanceOutput."""

    model_name = "fake-llm"

    def __init__(
        self,
        output_factory: Callable[[list[RetrievedChunk]], GuidanceOutput] = default_guidance,
        input_tokens: int = 120,
        output_tokens: int = 40,
    ) -> None:
        self._factory = output_factory
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        self.calls.append(
            {
                "company_name": company_name,
                "customer_message": customer_message,
                "agent_notes": agent_notes,
                "channel": channel,
                "chunks": chunks,
            }
        )
        return GenerationResult(
            output=self._factory(chunks),
            model=self.model_name,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


class FailingGuidanceGenerator:
    model_name = "fake-llm"

    def __init__(self, message: str = "provider exploded (503)") -> None:
        self.message = message

    async def generate(self, **_: Any) -> GenerationResult:
        raise GenerationError(self.message)


# ------------------------------------------------------------------ settings
def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "APP_ENV": "test",
        "INGESTION_SYNC": True,
        "AGENT_API_KEYS": AGENT_KEY,
        "ADMIN_API_KEYS": f"tester:{ADMIN_KEY}",
        "EMBEDDING_MODEL": "fake-embedding",
        "LLM_MODEL": "fake-llm",
        "CHUNK_SIZE": 300,
        "CHUNK_OVERLAP": 40,
        "RAG_TOP_K": 4,
        "LOG_LEVEL": "WARNING",
        "COMPANY_CACHE_TTL_SECONDS": 30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@asynccontextmanager
async def running_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app's lifespan and yield an httpx client bound to it (no network)."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingsFactory:
    return FakeEmbeddingsFactory()


@pytest.fixture
def fake_generator() -> FakeGuidanceGenerator:
    return FakeGuidanceGenerator()


@pytest.fixture
def failing_generator() -> FailingGuidanceGenerator:
    return FailingGuidanceGenerator()


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    path = tmp_path / "chroma"
    path.mkdir()
    return path
