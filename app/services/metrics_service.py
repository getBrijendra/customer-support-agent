from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.errors import CompanyNotFound, ValidationFailed
from app.services.common import UowFactory

DEFAULT_WINDOW = timedelta(days=7)


def resolve_window(
    start: datetime | None, end: datetime | None, now: datetime | None = None
) -> tuple[datetime, datetime]:
    now = now or datetime.now(UTC)
    end = end or now
    start = start or end - DEFAULT_WINDOW
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    if start >= end:
        raise ValidationFailed("'from' must be before 'to'")
    return start, end


class MetricsService:
    def __init__(self, uow_factory: UowFactory) -> None:
        self._uow = uow_factory

    async def summary(
        self, *, start: datetime | None, end: datetime | None, company: str | None
    ) -> dict[str, Any]:
        start, end = resolve_window(start, end)
        async with self._uow() as uow:
            company_id = None
            if company:
                found = await uow.companies.get_by_slug(company)
                if found is None:
                    raise CompanyNotFound(f"Company '{company}' not found")
                company_id = found.id
            repo = uow.metrics
            return {
                "window": {"from": start.isoformat(), "to": end.isoformat()},
                "company": company,
                "chats": await repo.chat_totals(start, end, company_id),
                "by_company": await repo.by_company(start, end, company_id),
                "daily_volume": await repo.daily_volume(start, end, company_id),
                "ingestion": await repo.ingestion(start, end, company_id),
            }
