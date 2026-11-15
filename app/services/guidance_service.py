"""Orchestrates one chat request: resolve the company context, run the LangGraph harness,
map the final state to the response. The harness always persists the chat (success or error)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.ai.schemas import GuidanceOutput
from app.capabilities.adapters.base import BindingSnapshot
from app.core.config import Settings
from app.core.errors import CompanyNotFound, LLMError, NoActiveGuidelines, RetrievalError
from app.db.models import (
    Chat,
    Company,
    GuidelineVersion,
    ProposedAction,
    ReviewStatus,
    VersionStatus,
)
from app.harness.graph import recursion_limit
from app.harness.state import GuidanceState
from app.services.common import CompanyContext, CompanyContextCache, UowFactory


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
    guidance: GuidanceOutput
    actions: list[ProposedAction]


BindingsLoader = Callable[[uuid.UUID], Awaitable[dict[str, BindingSnapshot]]]


class GuidanceService:
    def __init__(
        self,
        uow_factory: UowFactory,
        graph: Any,
        settings: Settings,
        cache: CompanyContextCache,
        bindings_loader: BindingsLoader,
    ) -> None:
        self._uow = uow_factory
        self._graph = graph
        self._settings = settings
        self._cache = cache
        self._bindings = bindings_loader

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
            rules_row = await uow.playbooks.get_rules(active.id) if active else None
        usable = bool(
            active
            and active.status == VersionStatus.READY
            and active.review_status == ReviewStatus.APPROVED
        )
        ctx = CompanyContext(
            company_id=company.id,
            slug=company.slug,
            name=company.name,
            active_version_id=active.id if active else None,
            active_version_number=active.version_number if active else None,
            active_version_usable=usable,
            embedding_model=active.embedding_model if active else None,
            embedding_dimensions=active.embedding_dimensions if active else None,
            rules=dict(rules_row.rules) if rules_row else {},
            bindings=await self._bindings(company.id) if self._settings.TOOLS_ENABLED else {},
        )
        self._cache.set(ctx)
        return ctx

    async def handle(self, req: ChatRequestData, *, request_id: str | None) -> ChatOutcome:
        started = time.perf_counter()
        ctx = await self.resolve_company(req.company)
        if not ctx.active_version_usable or not ctx.embedding_model:
            raise NoActiveGuidelines(f"Company '{ctx.slug}' has no active, approved guidelines")
        state: GuidanceState = {
            "chat_id": uuid.uuid4(),
            "request_id": request_id,
            "company": ctx,
            "customer_message": req.customer_message,
            "agent_notes": req.agent_notes,
            "channel": req.channel,
            "started_at": started,
            "deadline": time.monotonic() + self._settings.AGENT_TOTAL_TIMEOUT_SECONDS,
            "tool_records": [],
            "timings": {},
            "errors": [],
        }
        final: GuidanceState = await self._graph.ainvoke(
            state, {"recursion_limit": recursion_limit(self._settings.AGENT_MAX_STEPS)}
        )
        errors = final.get("errors") or []
        if errors:
            kinds = {e["kind"] for e in errors}
            if kinds & {"llm", "router"}:
                raise LLMError("The AI provider failed to generate guidance")
            raise RetrievalError("Guidance could not be produced (search failed)")
        chat = final["chat"]
        return ChatOutcome(
            chat=chat,
            company_slug=ctx.slug,
            guideline_version=ctx.active_version_number,
            guidance=final["output"],
            actions=list(final.get("actions", [])),
        )
