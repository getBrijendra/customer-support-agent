"""Routing eval cases and runs."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Company, GuidelineVersion, RoutingEvalCase, RoutingEvalRun


class EvalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_cases(self, company_id: uuid.UUID) -> list[RoutingEvalCase]:
        stmt = (
            select(RoutingEvalCase)
            .where(RoutingEvalCase.company_id == company_id)
            .order_by(RoutingEvalCase.created_at, RoutingEvalCase.id)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def add_cases(self, cases: Sequence[RoutingEvalCase]) -> int:
        """Insert cases, skipping (message, expected keys) pairs the company already has."""
        if not cases:
            return 0
        existing = {
            (c.message, tuple(sorted(c.expected_intent_keys)))
            for c in await self.list_cases(cases[0].company_id)
        }
        new = []
        for c in cases:
            sig = (c.message, tuple(sorted(c.expected_intent_keys)))
            if sig not in existing:
                existing.add(sig)
                new.append(c)
        self.session.add_all(new)
        await self.session.flush()
        return len(new)

    async def add_run(self, run: RoutingEvalRun) -> RoutingEvalRun:
        self.session.add(run)
        await self.session.flush()
        return run

    async def get_run(self, run_id: uuid.UUID) -> tuple[RoutingEvalRun, str, int] | None:
        stmt = (
            select(RoutingEvalRun, Company.slug, GuidelineVersion.version_number)
            .join(Company, Company.id == RoutingEvalRun.company_id)
            .join(GuidelineVersion, GuidelineVersion.id == RoutingEvalRun.version_id)
            .where(RoutingEvalRun.id == run_id)
        )
        row = (await self.session.execute(stmt)).first()
        return (row[0], row[1], row[2]) if row else None

    async def list_runs(
        self, company_id: uuid.UUID | None, limit: int = 50
    ) -> list[tuple[RoutingEvalRun, str, int]]:
        stmt = (
            select(RoutingEvalRun, Company.slug, GuidelineVersion.version_number)
            .join(Company, Company.id == RoutingEvalRun.company_id)
            .join(GuidelineVersion, GuidelineVersion.id == RoutingEvalRun.version_id)
            .order_by(RoutingEvalRun.created_at.desc())
            .limit(limit)
        )
        if company_id is not None:
            stmt = stmt.where(RoutingEvalRun.company_id == company_id)
        return [(r[0], r[1], r[2]) for r in (await self.session.execute(stmt)).all()]

    async def latest_per_company(self) -> list[dict[str, Any]]:
        rows = await self.list_runs(None, limit=500)
        seen: set[str] = set()
        out = []
        for run, slug, number in rows:
            if slug in seen:
                continue
            seen.add(slug)
            out.append(
                {
                    "company": slug,
                    "version": number,
                    "mode": run.mode,
                    "top1": run.metrics.get("top1_accuracy"),
                    "recall_at_k": run.metrics.get("recall_at_k"),
                    "created_at": run.created_at.isoformat(),
                }
            )
        return out
