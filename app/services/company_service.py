from __future__ import annotations

from app.core.errors import CompanyNotFound
from app.repositories.companies import CompanyWithActive
from app.services.common import UowFactory


class CompanyService:
    def __init__(self, uow_factory: UowFactory) -> None:
        self._uow = uow_factory

    async def create(self, *, slug: str, name: str, description: str | None) -> CompanyWithActive:
        async with self._uow() as uow:
            company = await uow.companies.create(slug=slug, name=name, description=description)
            await uow.commit()
            return CompanyWithActive(company=company, active_version=None)

    async def list(self) -> list[CompanyWithActive]:
        async with self._uow() as uow:
            result: list[CompanyWithActive] = await uow.companies.list_with_active()
            return result

    async def get(self, slug: str) -> CompanyWithActive:
        async with self._uow() as uow:
            found: CompanyWithActive | None = await uow.companies.get_with_active(slug)
        if found is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        return found
