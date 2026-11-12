"""Aggregate queries for the metrics summary. Postgres-only (percentile_cont, date_trunc, JSONB)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Float, and_, case, cast, extract, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, ChatStatus, Company, GuidelineVersion, VersionStatus


def _r(value: Any) -> float | None:
    return None if value is None else round(float(value), 1)


class MetricsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _chat_window(self, start: datetime, end: datetime, company_id: uuid.UUID | None) -> Any:
        cond = and_(Chat.created_at >= start, Chat.created_at < end)
        if company_id is not None:
            cond = and_(cond, Chat.company_id == company_id)
        return cond

    async def chat_totals(
        self, start: datetime, end: datetime, company_id: uuid.UUID | None
    ) -> dict[str, Any]:
        is_success = Chat.status == ChatStatus.SUCCESS
        escalated = cast(Chat.guidance.op("->>")("requires_escalation"), Boolean)
        stmt = select(
            func.count().label("total"),
            func.count().filter(is_success).label("success"),
            func.count().filter(Chat.status == ChatStatus.ERROR).label("error"),
            func.avg(Chat.latency_ms_total).label("avg"),
            func.percentile_cont(0.5).within_group(Chat.latency_ms_total).label("p50"),
            func.percentile_cont(0.95).within_group(Chat.latency_ms_total).label("p95"),
            func.coalesce(func.sum(Chat.input_tokens), 0).label("tokens_in"),
            func.coalesce(func.sum(Chat.output_tokens), 0).label("tokens_out"),
            func.count().filter(and_(is_success, escalated.is_(True))).label("escalated"),
        ).where(self._chat_window(start, end, company_id))
        row = (await self.session.execute(stmt)).one()

        confidence = Chat.guidance.op("->>")("confidence")
        conf_stmt = (
            select(confidence.label("confidence"), func.count())
            .where(self._chat_window(start, end, company_id), Chat.status == ChatStatus.SUCCESS)
            .group_by(confidence)
        )
        breakdown = {"high": 0, "medium": 0, "low": 0}
        for level, count in (await self.session.execute(conf_stmt)).all():
            if level is not None:
                breakdown[str(level)] = int(count)

        total, success = int(row.total), int(row.success)
        return {
            "total": total,
            "success": success,
            "error": int(row.error),
            "error_rate": round(row.error / total, 4) if total else 0.0,
            "latency_ms": {"avg": _r(row.avg), "p50": _r(row.p50), "p95": _r(row.p95)},
            "tokens": {"input": int(row.tokens_in), "output": int(row.tokens_out)},
            "escalation_rate": round(row.escalated / success, 4) if success else 0.0,
            "confidence_breakdown": breakdown,
        }

    async def by_company(
        self, start: datetime, end: datetime, company_id: uuid.UUID | None
    ) -> list[dict[str, Any]]:
        total = func.count(Chat.id)
        errors = func.count(Chat.id).filter(Chat.status == ChatStatus.ERROR)
        stmt = (
            select(
                Company.slug,
                total.label("chats"),
                func.avg(Chat.latency_ms_total).label("avg_latency"),
                (cast(errors, Float) / func.nullif(total, 0)).label("error_rate"),
            )
            .join(Chat, Chat.company_id == Company.id)
            .where(self._chat_window(start, end, company_id))
            .group_by(Company.slug)
            .order_by(total.desc(), Company.slug)
        )
        return [
            {
                "company": r.slug,
                "chats": int(r.chats),
                "avg_latency_ms": _r(r.avg_latency),
                "error_rate": round(float(r.error_rate or 0), 4),
            }
            for r in (await self.session.execute(stmt)).all()
        ]

    async def daily_volume(
        self, start: datetime, end: datetime, company_id: uuid.UUID | None
    ) -> list[dict[str, Any]]:
        day = func.date_trunc("day", func.timezone("UTC", Chat.created_at))
        stmt = (
            select(day.label("day"), func.count().label("chats"))
            .where(self._chat_window(start, end, company_id))
            .group_by(day)
            .order_by(day)
        )
        return [
            {"date": r.day.date().isoformat(), "chats": int(r.chats)}
            for r in (await self.session.execute(stmt)).all()
        ]

    async def ingestion(
        self, start: datetime, end: datetime, company_id: uuid.UUID | None
    ) -> dict[str, Any]:
        cond = and_(GuidelineVersion.created_at >= start, GuidelineVersion.created_at < end)
        if company_id is not None:
            cond = and_(cond, GuidelineVersion.company_id == company_id)
        duration_ms = (
            extract("epoch", GuidelineVersion.ready_at - GuidelineVersion.created_at) * 1000
        )
        stmt = select(
            func.count().label("uploads"),
            func.count().filter(GuidelineVersion.status == VersionStatus.READY).label("ready"),
            func.count().filter(GuidelineVersion.status == VersionStatus.FAILED).label("failed"),
            func.avg(
                case((GuidelineVersion.status == VersionStatus.READY, duration_ms), else_=None)
            ).label("avg_ms"),
        ).where(cond)
        r = (await self.session.execute(stmt)).one()
        return {
            "uploads": int(r.uploads),
            "ready": int(r.ready),
            "failed": int(r.failed),
            "avg_duration_ms": None if r.avg_ms is None else round(float(r.avg_ms)),
        }
