"""U7 — GuidanceService with in-memory fake repos, a fake vector store and fake generators."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.ai.prompts import PROMPT_VERSION
from app.core.errors import CompanyNotFound, LLMError, NoActiveGuidelines
from app.db.models import Chat, Company, GuidelineVersion
from app.repositories.companies import CompanyWithActive
from app.services.common import AppMetrics, CompanyContextCache
from app.services.guidance_service import ChatRequestData, GuidanceService
from tests.conftest import FailingGuidanceGenerator, FakeGuidanceGenerator, make_settings


class StaticRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.docs


class FakeVectorStore:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.retriever_calls: list[dict[str, Any]] = []

    def collection_name(self, company_id: uuid.UUID, version_number: int) -> str:
        return f"gl_{company_id.hex[:12]}_v{version_number}"

    def as_retriever(self, collection: str, **kwargs: Any) -> BaseRetriever:
        self.retriever_calls.append({"collection": collection, **kwargs})
        return StaticRetriever(docs=self.docs)


class FakeCompanies:
    def __init__(self, rows: dict[str, CompanyWithActive]) -> None:
        self.rows = rows
        self.lookups = 0

    async def get_with_active(self, slug: str) -> CompanyWithActive | None:
        self.lookups += 1
        return self.rows.get(slug)


class FakeChats:
    def __init__(self) -> None:
        self.saved: list[Chat] = []

    async def add(self, chat: Chat) -> Chat:
        self.saved.append(chat)
        return chat


class FakeUow:
    def __init__(self, companies: FakeCompanies, chats: FakeChats) -> None:
        self.companies = companies
        self.chats = chats
        self.versions = None
        self.metrics = None
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        pass


def _company(active: bool = True, status: str = "ready") -> CompanyWithActive:
    company = Company(id=uuid.uuid4(), slug="bookshop", name="Bookshop")
    version = None
    if active:
        version = GuidelineVersion(
            id=uuid.uuid4(),
            company_id=company.id,
            version_number=3,
            status=status,
            vector_collection="gl_x_v3",
            embedding_model="emb-from-ingest-time",
        )
        company.active_version_id = version.id
    return CompanyWithActive(company=company, active_version=version)


def _docs() -> list[Document]:
    return [
        Document(
            id="vid:0000",
            page_content="Refunds in 30 days",
            metadata={"section": "REFUNDS", "score": 0.91, "company_slug": "bookshop"},
        ),
        Document(
            id="vid:0001",
            page_content="Greet by name",
            metadata={"section": "TONE", "score": 0.5, "company_slug": "bookshop"},
        ),
    ]


def _service(
    row: CompanyWithActive | None, generator: Any
) -> tuple[GuidanceService, FakeUow, FakeVectorStore]:
    uow = FakeUow(FakeCompanies({"bookshop": row} if row else {}), FakeChats())

    @asynccontextmanager
    async def factory() -> AsyncIterator[FakeUow]:
        yield uow

    vs = FakeVectorStore(_docs())
    svc = GuidanceService(
        factory,  # type: ignore[arg-type]
        vs,  # type: ignore[arg-type]
        generator,
        make_settings(),
        CompanyContextCache(30),
        AppMetrics(),
    )
    return svc, uow, vs


REQ = ChatRequestData(
    company="bookshop",
    customer_message="Can I get a refund?",
    agent_notes="order 12",
    channel="email",
)


async def test_success_builds_response_and_persists() -> None:
    gen = FakeGuidanceGenerator()
    svc, uow, vs = _service(_company(), gen)
    outcome = await svc.handle(REQ, request_id="req-1")

    chat = outcome.chat
    assert outcome.company_slug == "bookshop"
    assert outcome.guideline_version == 3
    assert chat.status == "success"
    assert chat.guidance is not None
    assert chat.recommended_response == chat.guidance["recommended_response"]
    # hallucinated chunk id "made-up:9999" is dropped; only retrieved ids remain
    assert chat.guidance["guideline_references"] == ["vid:0000", "vid:0001"]
    assert chat.prompt_version == PROMPT_VERSION
    assert chat.llm_model == "fake-llm"
    assert chat.input_tokens == 120 and chat.output_tokens == 40
    assert chat.request_id == "req-1"
    assert chat.latency_ms_total is not None and chat.latency_ms_retrieval is not None
    assert chat.retrieved_chunks and chat.retrieved_chunks[0]["chunk_id"] == "vid:0000"
    assert chat.retrieved_chunks[0]["score"] == 0.91
    assert uow.chats.saved == [chat]
    # retriever built for the active version's collection with the INGEST-time embedding model
    assert vs.retriever_calls[0]["collection"] == "gl_x_v3"
    assert vs.retriever_calls[0]["embedding_model"] == "emb-from-ingest-time"
    # generator got the agent notes and retrieved chunks
    assert gen.calls[0]["agent_notes"] == "order 12"
    assert [c.chunk_id for c in gen.calls[0]["chunks"]] == ["vid:0000", "vid:0001"]


async def test_generator_error_maps_to_llm_error_and_still_persists() -> None:
    svc, uow, _ = _service(_company(), FailingGuidanceGenerator("boom"))
    with pytest.raises(LLMError) as exc:
        await svc.handle(REQ, request_id="req-2")
    assert exc.value.code == "LLM_ERROR" and exc.value.status_code == 502
    assert len(uow.chats.saved) == 1
    saved = uow.chats.saved[0]
    assert saved.status == "error"
    assert "boom" in (saved.error_message or "")
    assert saved.guidance is None
    assert saved.latency_ms_total is not None


async def test_no_active_version() -> None:
    svc, uow, _ = _service(_company(active=False), FakeGuidanceGenerator())
    with pytest.raises(NoActiveGuidelines):
        await svc.handle(REQ, request_id=None)
    assert uow.chats.saved == []


async def test_active_version_not_ready() -> None:
    svc, _, _ = _service(_company(status="processing"), FakeGuidanceGenerator())
    with pytest.raises(NoActiveGuidelines):
        await svc.handle(REQ, request_id=None)


async def test_unknown_company() -> None:
    svc, _, _ = _service(None, FakeGuidanceGenerator())
    with pytest.raises(CompanyNotFound):
        await svc.handle(REQ, request_id=None)


async def test_company_context_is_cached() -> None:
    svc, uow, _ = _service(_company(), FakeGuidanceGenerator())
    await svc.handle(REQ, request_id=None)
    await svc.handle(REQ, request_id=None)
    assert uow.companies.lookups == 1
