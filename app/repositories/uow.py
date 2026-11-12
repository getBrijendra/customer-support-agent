"""Unit of work: one DB session + the repositories bound to it.

Services depend on this shape (not on SQLAlchemy directly), so unit tests can pass in-memory fakes.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.chats import ChatRepository
from app.repositories.companies import CompanyRepository
from app.repositories.guideline_versions import GuidelineVersionRepository
from app.repositories.metrics import MetricsRepository


class SqlUnitOfWork:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.companies = CompanyRepository(session)
        self.versions = GuidelineVersionRepository(session)
        self.chats = ChatRepository(session)
        self.metrics = MetricsRepository(session)

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()
