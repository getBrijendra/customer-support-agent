"""Orchestrates one chat request end to end: resolve -> retrieve -> generate -> persist."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.ai.guidance_chain import GenerationResult, GuidanceGenerator
from app.ai.prompts import PROMPT_VERSION
from app.ai.retriever import build_query, build_retriever, retrieve_chunks
from app.core.config import Settings
from app.core.errors import CompanyNotFound, LLMError, NoActiveGuidelines, RetrievalError
from app.core.logging import get_logger, text_fingerprint
from app.db.models import Chat, ChatStatus, Company, GuidelineVersion, VersionStatus, utcnow
from app.services.common import AppMetrics, CompanyContext, CompanyContextCache, UowFactory
from app.vectorstore.base import RetrievedChunk, VectorStoreGateway

logger = get_logger(__name__)


@dataclass
class ChatRequestData:
    company: str
    customer_message: str
    agent_notes: str | None = None
    channel: str | None = None


@dataclass
class ChatOutcome:
    chat: Chat
    company_slug: str
    guideline_version: int | None


def _ms(start: float, end: float) -> int:
    return round((end - start) * 1000)


class GuidanceService:
    def __init__(
        self,
        uow_factory: UowFactory,
        vector_store: VectorStoreGateway,
        generator: GuidanceGenerator,
        settings: Settings,
        cache: CompanyContextCache,
        metrics: AppMetrics,
    ) -> None:
        self._uow = uow_factory
        self._vs = vector_store
        self._generator = generator
        self._settings = settings
        self._cache = cache
        self._metrics = metrics

    async def resolve_company(self, slug: str) -> CompanyContext:
        cached = self._cache.get(slug)
        if cached is not None:
            return cached
        async with self._uow() as uow:
            found = await uow.companies.get_with_active(slug)
        if found is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        company: Company = found.company
        active: GuidelineVersion | None = found.active_version
        ctx = CompanyContext(
            company_id=company.id,
            slug=company.slug,
            name=company.name,
            active_version_id=active.id if active else None,
            active_version_number=active.version_number if active else None,
            active_version_ready=bool(active and active.status == VersionStatus.READY),
            collection=active.vector_collection if active else None,
            embedding_model=active.embedding_model if active else None,
        )
        self._cache.set(ctx)
        return ctx

    async def handle(self, req: ChatRequestData, *, request_id: str | None) -> ChatOutcome:
        t_start = time.perf_counter()
        ctx = await self.resolve_company(req.company)
        if not ctx.active_version_ready or not ctx.collection or not ctx.embedding_model:
            raise NoActiveGuidelines(f"Company '{ctx.slug}' has no active, ready guidelines")

        chat = Chat(
            id=uuid.uuid4(),
            company_id=ctx.company_id,
            guideline_version_id=ctx.active_version_id,
            guideline_version_number=ctx.active_version_number,
            customer_message=req.customer_message,
            agent_notes=req.agent_notes,
            channel=req.channel,
            llm_model=self._generator.model_name,
            prompt_version=PROMPT_VERSION,
            request_id=request_id,
        )

        # --- retrieval ---------------------------------------------------------
        chunks: list[RetrievedChunk] = []
        t_retr = time.perf_counter()
        try:
            retriever = build_retriever(
                self._vs,
                collection=ctx.collection,
                embedding_model=ctx.embedding_model,
                settings=self._settings,
            )
            chunks = await retrieve_chunks(
                retriever, build_query(req.customer_message, req.agent_notes)
            )
        except Exception as exc:
            t_end = time.perf_counter()
            chat.latency_ms_retrieval = _ms(t_retr, t_end)
            await self._fail(
                chat, ctx, t_start, t_end, f"Retrieval failed: {type(exc).__name__}: {exc}"
            )
            raise RetrievalError("Could not retrieve guideline excerpts") from exc
        t_retr_end = time.perf_counter()
        chat.latency_ms_retrieval = _ms(t_retr, t_retr_end)
        chat.retrieved_chunks = [c.to_record() for c in chunks]

        # --- generation --------------------------------------------------------
        try:
            result: GenerationResult = await self._generator.generate(
                company_name=ctx.name,
                customer_message=req.customer_message,
                agent_notes=req.agent_notes,
                channel=req.channel,
                chunks=chunks,
            )
        except Exception as exc:
            t_end = time.perf_counter()
            chat.latency_ms_llm = _ms(t_retr_end, t_end)
            await self._fail(chat, ctx, t_start, t_end, str(exc) or type(exc).__name__)
            raise LLMError("The AI provider failed to generate guidance") from exc
        t_end = time.perf_counter()

        output = result.output
        # Only keep citations that refer to excerpts we actually supplied.
        known = {c.chunk_id for c in chunks}
        output = output.model_copy(
            update={"guideline_references": [r for r in output.guideline_references if r in known]}
        )
        chat.status = ChatStatus.SUCCESS
        chat.guidance = output.model_dump(mode="json")
        chat.recommended_response = output.recommended_response
        chat.llm_model = result.model
        chat.input_tokens = result.input_tokens
        chat.output_tokens = result.output_tokens
        chat.latency_ms_llm = _ms(t_retr_end, t_end)
        chat.latency_ms_total = _ms(t_start, t_end)
        await self._persist(chat)

        self._metrics.chat_requests.labels(company=ctx.slug, status="success").inc()
        self._metrics.chat_latency.observe(t_end - t_start)
        if result.input_tokens:
            self._metrics.llm_tokens.labels(direction="input").inc(result.input_tokens)
        if result.output_tokens:
            self._metrics.llm_tokens.labels(direction="output").inc(result.output_tokens)
        logger.info("chat.completed", extra=self._log_fields(chat, ctx))
        return ChatOutcome(
            chat=chat, company_slug=ctx.slug, guideline_version=ctx.active_version_number
        )

    async def _fail(
        self, chat: Chat, ctx: CompanyContext, t_start: float, t_end: float, message: str
    ) -> None:
        chat.status = ChatStatus.ERROR
        chat.error_message = message[:2000]
        chat.latency_ms_total = _ms(t_start, t_end)
        try:
            await self._persist(chat)
        except Exception:
            logger.exception("chat.persist_failed", extra={"company": ctx.slug})
        self._metrics.chat_requests.labels(company=ctx.slug, status="error").inc()
        logger.warning("chat.failed", extra={**self._log_fields(chat, ctx), "error": message[:500]})

    async def _persist(self, chat: Chat) -> None:
        if chat.created_at is None:
            chat.created_at = utcnow()
        async with self._uow() as uow:
            await uow.chats.add(chat)
            await uow.commit()

    @staticmethod
    def _log_fields(chat: Chat, ctx: CompanyContext) -> dict[str, Any]:
        return {
            "chat_id": str(chat.id),
            "company": ctx.slug,
            "version": ctx.active_version_number,
            "latency_ms_total": chat.latency_ms_total,
            "latency_ms_retrieval": chat.latency_ms_retrieval,
            "latency_ms_llm": chat.latency_ms_llm,
            "input_tokens": chat.input_tokens,
            "output_tokens": chat.output_tokens,
            "customer_message": text_fingerprint(chat.customer_message),
        }
