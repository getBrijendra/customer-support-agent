from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import CompanyAlreadyExists
from app.db.models import Company, GuidelineVersion, utcnow


@dataclass
class CompanyWithActive:
    company: Company
    active_version: GuidelineVersion | None


class CompanyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, *, slug: str, name: str, description: str | None) -> Company:
        company = Company(id=uuid.uuid4(), slug=slug, name=name, description=description)
        self.session.add(company)
        try:
            await self.session.flush()
        except IntegrityError as exc:
            await self.session.rollback()
            raise CompanyAlreadyExists(f"Company '{slug}' already exists") from exc
        return company

    async def get_by_slug(self, slug: str) -> Company | None:
        result = await self.session.execute(select(Company).where(Company.slug == slug))
        return result.scalar_one_or_none()

    async def get_by_id(self, company_id: uuid.UUID) -> Company | None:
        return await self.session.get(Company, company_id)

    async def get_by_slug_for_update(self, slug: str) -> Company | None:
        """Row-locks the company (Postgres `SELECT ... FOR UPDATE`) until the transaction ends."""
        result = await self.session.execute(
            select(Company).where(Company.slug == slug).with_for_update()
        )
        return result.scalar_one_or_none()

    async def lock(self, company_id: uuid.UUID) -> Company | None:
        result = await self.session.execute(
            select(Company)
            .where(Company.id == company_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def get_with_active(self, slug: str) -> CompanyWithActive | None:
        stmt = (
            select(Company, GuidelineVersion)
            .outerjoin(GuidelineVersion, GuidelineVersion.id == Company.active_version_id)
            .where(Company.slug == slug)
        )
        row = (await self.session.execute(stmt)).first()
        if row is None:
            return None
        return CompanyWithActive(company=row[0], active_version=row[1])

    async def list_with_active(self) -> list[CompanyWithActive]:
        stmt = (
            select(Company, GuidelineVersion)
            .outerjoin(GuidelineVersion, GuidelineVersion.id == Company.active_version_id)
            .order_by(Company.slug)
        )
        rows = (await self.session.execute(stmt)).all()
        return [CompanyWithActive(company=r[0], active_version=r[1]) for r in rows]

    async def set_active_version(self, company: Company, version: GuidelineVersion) -> None:
        now = utcnow()
        company.active_version_id = version.id
        company.updated_at = now
        version.activated_at = now
        await self.session.flush()
