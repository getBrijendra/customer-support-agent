from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GuidelineVersion


class GuidelineVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def next_version_number(self, company_id: uuid.UUID) -> int:
        """Must be called while holding the company row lock (see CompanyRepository.lock)."""
        stmt = select(func.coalesce(func.max(GuidelineVersion.version_number), 0)).where(
            GuidelineVersion.company_id == company_id
        )
        current = (await self.session.execute(stmt)).scalar_one()
        return int(current) + 1

    async def add(self, version: GuidelineVersion) -> GuidelineVersion:
        self.session.add(version)
        await self.session.flush()
        return version

    async def get(self, version_id: uuid.UUID) -> GuidelineVersion | None:
        return await self.session.get(GuidelineVersion, version_id, populate_existing=True)

    async def get_by_number(
        self, company_id: uuid.UUID, version_number: int
    ) -> GuidelineVersion | None:
        stmt = select(GuidelineVersion).where(
            GuidelineVersion.company_id == company_id,
            GuidelineVersion.version_number == version_number,
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_for_company(self, company_id: uuid.UUID) -> list[GuidelineVersion]:
        stmt = (
            select(GuidelineVersion)
            .where(GuidelineVersion.company_id == company_id)
            .order_by(GuidelineVersion.version_number.desc())
        )
        return list((await self.session.execute(stmt)).scalars())

    async def delete(self, version_id: uuid.UUID) -> None:
        await self.session.execute(
            delete(GuidelineVersion).where(GuidelineVersion.id == version_id)
        )
        await self.session.flush()
