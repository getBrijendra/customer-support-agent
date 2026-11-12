"""Read side for chats: cursor-paginated listing and detail."""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from datetime import datetime

from app.core.errors import ChatNotFound, CompanyNotFound, ValidationFailed
from app.repositories.chats import ChatFilters, ChatRow
from app.services.common import UowFactory

MAX_LIMIT = 100


def encode_cursor(created_at: datetime, chat_id: uuid.UUID) -> str:
    raw = json.dumps([created_at.isoformat(), str(chat_id)]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created_raw, id_raw = json.loads(base64.urlsafe_b64decode(padded.encode()))
        created_at = datetime.fromisoformat(created_raw)
        if created_at.tzinfo is None:
            raise ValueError("cursor timestamp must be timezone-aware")
        return created_at, uuid.UUID(id_raw)
    except (ValueError, TypeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ValidationFailed("Invalid pagination cursor") from exc


class ChatHistoryService:
    def __init__(self, uow_factory: UowFactory) -> None:
        self._uow = uow_factory

    async def list(
        self,
        *,
        company: str | None,
        status: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[ChatRow], str | None]:
        limit = max(1, min(limit, MAX_LIMIT))
        after = decode_cursor(cursor) if cursor else None
        async with self._uow() as uow:
            filters = ChatFilters(status=status, created_from=created_from, created_to=created_to)
            if company:
                found = await uow.companies.get_by_slug(company)
                if found is None:
                    raise CompanyNotFound(f"Company '{company}' not found")
                filters.company_id = found.id
            rows: list[ChatRow] = await uow.chats.list(filters, limit=limit + 1, after=after)
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            last = rows[-1].chat
            next_cursor = encode_cursor(last.created_at, last.id)
        return rows, next_cursor

    async def get(self, chat_id: uuid.UUID) -> ChatRow:
        async with self._uow() as uow:
            row: ChatRow | None = await uow.chats.get(chat_id)
        if row is None:
            raise ChatNotFound(f"Chat {chat_id} not found")
        return row
