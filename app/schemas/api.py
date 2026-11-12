"""Request/response models for the public API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ai.schemas import GuidanceOutput
from app.db.models import Chat, Company, GuidelineVersion

SLUG_PATTERN = r"^[a-z0-9-]{2,50}$"


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ----------------------------------------------------------------- companies
class CompanyCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"slug": "vpn-saas", "name": "VPN SaaS", "description": "Consumer VPN"}]
        }
    )
    slug: str = Field(pattern=SLUG_PATTERN, description="URL-safe, lowercase id")
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


class VersionSummary(BaseModel):
    id: uuid.UUID
    version_number: int
    status: str
    file_name: str
    created_at: datetime
    activated_at: datetime | None = None


class CompanyOut(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    active_version_number: int | None
    active_version: VersionSummary | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def build(cls, company: Company, active: GuidelineVersion | None) -> CompanyOut:
        return cls(
            id=company.id,
            slug=company.slug,
            name=company.name,
            description=company.description,
            active_version_number=active.version_number if active else None,
            active_version=VersionSummary.model_validate(active, from_attributes=True)
            if active
            else None,
            created_at=company.created_at,
            updated_at=company.updated_at,
        )


# ---------------------------------------------------------------- guidelines
class GuidelineVersionOut(BaseModel):
    id: uuid.UUID
    company: str
    version_number: int
    status: Literal["processing", "ready", "failed"]
    is_active: bool
    file_name: str
    content_sha256: str
    byte_size: int
    chunk_count: int | None
    vector_collection: str | None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    notes: str | None
    error_message: str | None
    uploaded_by: str | None
    created_at: datetime
    ready_at: datetime | None
    activated_at: datetime | None
    raw_text: str | None = Field(default=None, description="Only with ?include_text=true")
    info: str | None = Field(default=None, description="Informational message, e.g. duplicates")

    @classmethod
    def build(
        cls,
        v: GuidelineVersion,
        *,
        company: Company,
        include_text: bool = False,
        info: str | None = None,
    ) -> GuidelineVersionOut:
        return cls(
            id=v.id,
            company=company.slug,
            version_number=v.version_number,
            status=v.status,
            is_active=company.active_version_id == v.id,
            file_name=v.file_name,
            content_sha256=v.content_sha256,
            byte_size=v.byte_size,
            chunk_count=v.chunk_count,
            vector_collection=v.vector_collection,
            embedding_model=v.embedding_model,
            chunk_size=v.chunk_size,
            chunk_overlap=v.chunk_overlap,
            notes=v.notes,
            error_message=v.error_message,
            uploaded_by=v.uploaded_by,
            created_at=v.created_at,
            ready_at=v.ready_at,
            activated_at=v.activated_at,
            raw_text=v.raw_text if include_text else None,
            info=info,
        )


# --------------------------------------------------------------------- chats
class ChatCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "company": "vpn-saas",
                    "customer_message": "I was charged twice this month and the VPN won't "
                    "connect on my iPhone.",
                    "agent_notes": "Customer is on annual plan",
                    "channel": "email",
                }
            ]
        }
    )
    company: str = Field(pattern=SLUG_PATTERN)
    customer_message: str = Field(min_length=1, max_length=20_000)
    agent_notes: str | None = Field(default=None, max_length=5_000)
    channel: str | None = Field(default=None, max_length=32)


class ChatCreated(BaseModel):
    chat_id: uuid.UUID
    company: str
    guideline_version: int | None
    guidance: GuidanceOutput
    latency_ms: int | None
    created_at: datetime


def _truncate(text: str | None, n: int = 200) -> str | None:
    if text is None or len(text) <= n:
        return text
    return text[: n - 1] + "…"


class ChatListItem(BaseModel):
    chat_id: uuid.UUID
    company: str
    created_at: datetime
    status: str
    customer_message: str
    recommended_response: str | None
    guideline_version: int | None

    @classmethod
    def build(cls, chat: Chat, company_slug: str) -> ChatListItem:
        return cls(
            chat_id=chat.id,
            company=company_slug,
            created_at=chat.created_at,
            status=chat.status,
            customer_message=_truncate(chat.customer_message) or "",
            recommended_response=_truncate(chat.recommended_response),
            guideline_version=chat.guideline_version_number,
        )


class ChatList(BaseModel):
    items: list[ChatListItem]
    next_cursor: str | None


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    score: float | None
    section: str | None
    text: str
    company_slug: str | None = None
    version_number: int | None = None


class LatencyOut(BaseModel):
    total: int | None
    retrieval: int | None
    llm: int | None


class TokensOut(BaseModel):
    input: int | None
    output: int | None


class ChatDetail(BaseModel):
    chat_id: uuid.UUID
    company: str
    status: str
    created_at: datetime
    customer_message: str
    agent_notes: str | None
    channel: str | None
    guideline_version: int | None
    guideline_version_id: uuid.UUID | None
    guidance: GuidanceOutput | None
    retrieved_chunks: list[RetrievedChunkOut]
    llm_model: str | None
    prompt_version: str | None
    latency_ms: LatencyOut
    tokens: TokensOut
    error_message: str | None
    request_id: str | None

    @classmethod
    def build(cls, chat: Chat, company_slug: str) -> ChatDetail:
        return cls(
            chat_id=chat.id,
            company=company_slug,
            status=chat.status,
            created_at=chat.created_at,
            customer_message=chat.customer_message,
            agent_notes=chat.agent_notes,
            channel=chat.channel,
            guideline_version=chat.guideline_version_number,
            guideline_version_id=chat.guideline_version_id,
            guidance=GuidanceOutput.model_validate(chat.guidance) if chat.guidance else None,
            retrieved_chunks=[
                RetrievedChunkOut.model_validate(c) for c in chat.retrieved_chunks or []
            ],
            llm_model=chat.llm_model,
            prompt_version=chat.prompt_version,
            latency_ms=LatencyOut(
                total=chat.latency_ms_total,
                retrieval=chat.latency_ms_retrieval,
                llm=chat.latency_ms_llm,
            ),
            tokens=TokensOut(input=chat.input_tokens, output=chat.output_tokens),
            error_message=chat.error_message,
            request_id=chat.request_id,
        )


# ------------------------------------------------------------------- metrics
class LatencyStats(BaseModel):
    avg: float | None
    p50: float | None
    p95: float | None


class ChatStats(BaseModel):
    total: int
    success: int
    error: int
    error_rate: float
    latency_ms: LatencyStats
    tokens: dict[str, int]
    escalation_rate: float
    confidence_breakdown: dict[str, int]


class CompanyStats(BaseModel):
    company: str
    chats: int
    avg_latency_ms: float | None
    error_rate: float


class DailyVolume(BaseModel):
    date: str
    chats: int


class IngestionStats(BaseModel):
    uploads: int
    ready: int
    failed: int
    avg_duration_ms: int | None


class MetricsSummary(BaseModel):
    window: dict[str, str]
    company: str | None = None
    chats: ChatStats
    by_company: list[CompanyStats]
    daily_volume: list[DailyVolume]
    ingestion: IngestionStats


class HealthOut(BaseModel):
    status: str
    checks: dict[str, bool] | None = None
