"""U8 — company repository basic CRUD on SQLite (the only SQLite-backed unit test)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import CompanyAlreadyExists
from app.repositories.companies import CompanyRepository


async def test_create_and_get_by_slug(sqlite_session: AsyncSession) -> None:
    repo = CompanyRepository(sqlite_session)
    created = await repo.create(slug="bookshop", name="Bookshop", description="Books")
    await sqlite_session.commit()

    found = await repo.get_by_slug("bookshop")
    assert found is not None
    assert found.id == created.id
    assert found.name == "Bookshop"
    assert found.active_version_id is None
    assert await repo.get_by_slug("missing") is None
    assert (await repo.get_by_id(created.id)) is not None


async def test_duplicate_slug_raises(sqlite_session: AsyncSession) -> None:
    repo = CompanyRepository(sqlite_session)
    await repo.create(slug="vpn-saas", name="VPN SaaS", description=None)
    await sqlite_session.commit()
    with pytest.raises(CompanyAlreadyExists) as exc:
        await repo.create(slug="vpn-saas", name="Other", description=None)
    assert exc.value.status_code == 409


async def test_list_with_active(sqlite_session: AsyncSession) -> None:
    repo = CompanyRepository(sqlite_session)
    await repo.create(slug="b-co", name="B", description=None)
    await repo.create(slug="a-co", name="A", description=None)
    await sqlite_session.commit()
    rows = await repo.list_with_active()
    assert [r.company.slug for r in rows] == ["a-co", "b-co"]
    assert all(r.active_version is None for r in rows)
