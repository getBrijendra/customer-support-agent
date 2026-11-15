"""Routing evals: run eval cases against a version's routing index (and optionally the router
LLM), store the run, and turn agent feedback into eval cases."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.embeddings import Embeddings

from app.ai.router import IntentSelector, retrieval_no_match
from app.core.config import Settings
from app.core.errors import (
    ChatNotFound,
    CompanyNotFound,
    NoActiveGuidelines,
    NotFound,
    VersionNotFound,
    VersionNotReady,
)
from app.core.logging import get_logger
from app.db.models import (
    Company,
    GuidelineVersion,
    RoutingEvalCase,
    RoutingEvalRun,
    VersionStatus,
    utcnow,
)
from app.search.base import Query, SearchGateway
from app.services.common import UnitOfWork, UowFactory
from app.services.eval_metrics import CaseOutcome, compute_metrics

logger = get_logger(__name__)
Mode = Literal["retrieval", "full"]


class EvalService:
    def __init__(
        self,
        uow_factory: UowFactory,
        search: SearchGateway,
        embeddings_for: Callable[[str], Embeddings],
        selector: IntentSelector,
        settings: Settings,
    ) -> None:
        self._uow = uow_factory
        self._search = search
        self._embeddings_for = embeddings_for
        self._selector = selector
        self._settings = settings

    async def _company(self, uow: UnitOfWork, slug: str) -> Company:
        company: Company | None = await uow.companies.get_by_slug(slug)
        if company is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        return company

    # ------------------------------------------------------------------ cases
    async def list_cases(self, slug: str) -> list[RoutingEvalCase]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            cases: list[RoutingEvalCase] = await uow.evals.list_cases(company.id)
            return cases

    async def add_case(self, slug: str, message: str, expected: list[str]) -> RoutingEvalCase:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            case = RoutingEvalCase(
                id=uuid.uuid4(),
                company_id=company.id,
                message=message,
                expected_intent_keys=expected,
                source="manual",
                created_at=utcnow(),
            )
            await uow.evals.add_cases([case])
            await uow.commit()
            return case

    async def add_feedback(
        self, chat_id: uuid.UUID, correct_keys: list[str], by: str | None
    ) -> dict[str, Any]:
        async with self._uow() as uow:
            chat = await uow.chats.get_for_company(chat_id)
            if chat is None:
                raise ChatNotFound(f"Chat {chat_id} not found")
            selected = [i["key"] for i in (chat.routing or {}).get("selected", [])]
            feedback = {
                "correct_intent_keys": correct_keys,
                "selected_intent_keys": selected,
                "changed": set(correct_keys) != set(selected),
                "by": by,
                "at": utcnow().isoformat(),
            }
            chat.feedback = feedback
            await uow.evals.add_cases(
                [
                    RoutingEvalCase(
                        id=uuid.uuid4(),
                        company_id=chat.company_id,
                        message=chat.customer_message,
                        expected_intent_keys=correct_keys,
                        source="agent_feedback",
                        created_from_version_id=chat.guideline_version_id,
                        chat_id=chat.id,
                        created_at=utcnow(),
                    )
                ]
            )
            await uow.commit()
        logger.info(
            "chat.feedback", extra={"chat_id": str(chat_id), "changed": feedback["changed"]}
        )
        return feedback

    # ------------------------------------------------------------------- runs
    async def run_for_slug(
        self, slug: str, version_number: int | None, mode: Mode
    ) -> tuple[RoutingEvalRun, str, int]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            if version_number is None:
                if company.active_version_id is None:
                    raise NoActiveGuidelines(f"Company '{slug}' has no active version")
                version = await uow.versions.get(company.active_version_id)
            else:
                version = await uow.versions.get_by_number(company.id, version_number)
            if version is None:
                raise VersionNotFound(f"Version {version_number} not found for '{slug}'")
            if version.status != VersionStatus.READY:
                raise VersionNotReady(f"Version {version.version_number} is not ready")
        run = await self.run(company, version, mode)
        return run, company.slug, version.version_number

    async def run(self, company: Company, version: GuidelineVersion, mode: Mode) -> RoutingEvalRun:
        async with self._uow() as uow:
            cases: list[RoutingEvalCase] = await uow.evals.list_cases(company.id)
            keys = {i.key for i in await uow.playbooks.list_intents(version.id)}
        embeddings = self._embeddings_for(version.embedding_model)
        vectors = (
            await asyncio.to_thread(embeddings.embed_documents, [c.message for c in cases])
            if cases
            else []
        )
        k = self._settings.ROUTING_CANDIDATES
        outcomes: list[CaseOutcome] = []
        for case, vec in zip(cases, vectors, strict=True):
            stale = any(key not in keys for key in case.expected_intent_keys)
            candidates = await self._search.search_intents(
                Query(company.id, version.id, vec, case.message), k
            )
            if mode == "full":
                sel = await self._selector.select(
                    company_name=company.name,
                    customer_message=case.message,
                    agent_notes=None,
                    candidates=candidates,
                    max_intents=self._settings.ROUTING_MAX_INTENTS,
                )
                predicted = [i.key for i in sel.selection.intents]
            elif retrieval_no_match(candidates, self._settings.ROUTING_MIN_SCORE):
                predicted = []
            else:
                predicted = [candidates[0].key]
            outcomes.append(
                CaseOutcome(
                    message=case.message,
                    expected=list(case.expected_intent_keys),
                    predicted=predicted,
                    candidates=[c.key for c in candidates],
                    stale=stale,
                    source=case.source,
                )
            )
        metrics = compute_metrics(outcomes)
        run = RoutingEvalRun(
            id=uuid.uuid4(),
            company_id=company.id,
            version_id=version.id,
            mode=mode,
            metrics=metrics,
            results=[o.to_dict() for o in outcomes],
            created_at=utcnow(),
        )
        async with self._uow() as uow:
            await uow.evals.add_run(run)
            await uow.commit()
        logger.info(
            "eval.completed",
            extra={
                "company": company.slug,
                "version": version.version_number,
                "mode": mode,
                "cases": metrics["cases"],
                "top1": metrics["top1_accuracy"],
            },
        )
        return run

    async def get_run(self, run_id: uuid.UUID) -> tuple[RoutingEvalRun, str, int]:
        async with self._uow() as uow:
            run: tuple[RoutingEvalRun, str, int] | None = await uow.evals.get_run(run_id)
        if run is None:
            raise NotFound(f"Eval run {run_id} not found")
        return run

    async def list_runs(self, slug: str | None) -> list[tuple[RoutingEvalRun, str, int]]:
        async with self._uow() as uow:
            company_id = (await self._company(uow, slug)).id if slug else None
            rows: list[tuple[RoutingEvalRun, str, int]] = await uow.evals.list_runs(company_id)
            return rows
