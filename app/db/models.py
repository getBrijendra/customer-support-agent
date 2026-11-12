"""SQLAlchemy ORM models. Postgres is the target; SQLite only needs to load them for unit tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

JsonType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class VersionStatus:
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ChatStatus:
    SUCCESS = "success"
    ERROR = "error"


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # Single source of truth for the active version => exactly one active version per company.
    active_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey(
            "guideline_versions.id",
            use_alter=True,
            ondelete="SET NULL",
            name="fk_companies_active_version_id_guideline_versions",
        ),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class GuidelineVersion(Base):
    __tablename__ = "guideline_versions"
    __table_args__ = (
        UniqueConstraint("company_id", "version_number", name="uq_guideline_versions_company_ver"),
        Index("ix_guideline_versions_company_sha", "company_id", "content_sha256"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    vector_collection: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    uploaded_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Chat(Base):
    __tablename__ = "chats"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    guideline_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("guideline_versions.id", ondelete="SET NULL")
    )
    # Denormalised so history stays readable even if the version row is later deleted.
    guideline_version_number: Mapped[int | None] = mapped_column(Integer)
    customer_message: Mapped[str] = mapped_column(Text, nullable=False)
    agent_notes: Mapped[str | None] = mapped_column(Text)
    channel: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    recommended_response: Mapped[str | None] = mapped_column(Text)
    guidance: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    retrieved_chunks: Mapped[list[dict[str, Any]] | None] = mapped_column(JsonType)
    llm_model: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    latency_ms_total: Mapped[int | None] = mapped_column(Integer)
    latency_ms_retrieval: Mapped[int | None] = mapped_column(Integer)
    latency_ms_llm: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


Index("ix_chats_company_created", Chat.company_id, Chat.created_at.desc())
