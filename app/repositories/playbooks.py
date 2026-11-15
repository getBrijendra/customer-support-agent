"""Compiled playbook rows: sections, intents, examples, company rules, reference facts."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    CompanyRules,
    GuidelineSection,
    Intent,
    IntentExample,
    ReferenceFact,
)


class PlaybookRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_all(self, rows: Sequence[object]) -> None:
        self.session.add_all(list(rows))
        await self.session.flush()

    async def list_intents(self, version_id: uuid.UUID) -> list[Intent]:
        stmt = select(Intent).where(Intent.version_id == version_id).order_by(Intent.key)
        return list((await self.session.execute(stmt)).scalars())

    async def intents_by_keys(self, version_id: uuid.UUID, keys: list[str]) -> list[Intent]:
        if not keys:
            return []
        stmt = select(Intent).where(Intent.version_id == version_id, Intent.key.in_(keys))
        found = {i.key: i for i in (await self.session.execute(stmt)).scalars()}
        return [found[k] for k in keys if k in found]

    async def get_rules(self, version_id: uuid.UUID) -> CompanyRules | None:
        stmt = select(CompanyRules).where(CompanyRules.version_id == version_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_facts(self, version_id: uuid.UUID) -> list[ReferenceFact]:
        stmt = select(ReferenceFact).where(ReferenceFact.version_id == version_id)
        return list((await self.session.execute(stmt)).scalars())

    async def list_sections(self, version_id: uuid.UUID) -> list[GuidelineSection]:
        stmt = (
            select(GuidelineSection)
            .where(GuidelineSection.version_id == version_id)
            .order_by(GuidelineSection.ordinal)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def sections_by_ids(
        self, version_id: uuid.UUID, ids: list[uuid.UUID]
    ) -> list[GuidelineSection]:
        if not ids:
            return []
        stmt = (
            select(GuidelineSection)
            .where(GuidelineSection.version_id == version_id, GuidelineSection.id.in_(ids))
            .order_by(GuidelineSection.ordinal)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def eval_examples(self, version_id: uuid.UUID) -> list[tuple[str, str]]:
        """(message, intent_key) for the held-out examples of a version."""
        stmt = (
            select(IntentExample.text, Intent.key)
            .join(Intent, Intent.id == IntentExample.intent_id)
            .where(IntentExample.version_id == version_id, IntentExample.split == "eval")
        )
        return [(r[0], r[1]) for r in (await self.session.execute(stmt)).all()]
