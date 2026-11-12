from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, Company


@dataclass
class ChatFilters:
    company_id: uuid.UUID | None = None
    status: str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None


@dataclass
class ChatRow:
    chat: Chat
    company_slug: str


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, chat: Chat) -> Chat:
        self.session.add(chat)
        await self.session.flush()
        return chat

    async def get(self, chat_id: uuid.UUID) -> ChatRow | None:
        stmt = (
            select(Chat, Company.slug)
            .join(Company, Company.id == Chat.company_id)
            .where(Chat.id == chat_id)
        )
        row = (await self.session.execute(stmt)).first()
        return ChatRow(chat=row[0], company_slug=row[1]) if row else None

    async def list(
        self,
        filters: ChatFilters,
        *,
        limit: int,
        after: tuple[datetime, uuid.UUID] | None = None,
    ) -> list[ChatRow]:
        """Keyset pagination ordered by (created_at DESC, id DESC)."""
        stmt = select(Chat, Company.slug).join(Company, Company.id == Chat.company_id)
        if filters.company_id is not None:
            stmt = stmt.where(Chat.company_id == filters.company_id)
        if filters.status is not None:
            stmt = stmt.where(Chat.status == filters.status)
        if filters.created_from is not None:
            stmt = stmt.where(Chat.created_at >= filters.created_from)
        if filters.created_to is not None:
            stmt = stmt.where(Chat.created_at < filters.created_to)
        if after is not None:
            stmt = stmt.where(tuple_(Chat.created_at, Chat.id) < tuple_(after[0], after[1]))
        stmt = stmt.order_by(Chat.created_at.desc(), Chat.id.desc()).limit(limit)
        rows = (await self.session.execute(stmt)).all()
        return [ChatRow(chat=r[0], company_slug=r[1]) for r in rows]
