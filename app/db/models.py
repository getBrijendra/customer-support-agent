"""SQLAlchemy ORM models. Postgres (+ pgvector) is the target; SQLite only needs to load them for
the few unit tests that use it (vector columns fall back to JSON there).

`tsvector` columns are Postgres GENERATED columns created by the migration; they are not mapped
here and are only queried from `repositories/search.py`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
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

JsonType = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")

# Fixed by migration 0002. Settings.EMBEDDING_DIMENSIONS must match (checked at startup).
EMBEDDING_DIMENSIONS = 768
VectorType = Vector(EMBEDDING_DIMENSIONS).with_variant(JSON(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(UTC)


class VersionStatus:
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ReviewStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ChatStatus:
    SUCCESS = "success"
    ERROR = "error"


class BindingStatus:
    DRAFT = "draft"
    TESTED = "tested"
    LIVE = "live"


class ActionStatus:
    PROPOSED = "proposed"
    DENIED = "denied"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"


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
    review_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ReviewStatus.PENDING
    )
    activate_on_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_note: Mapped[str | None] = mapped_column(Text)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_format: Mapped[str] = mapped_column(String(16), nullable=False, default="txt")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    # Where the original uploaded bytes live (NULL when file storage is disabled / pre-0003 rows)
    storage_backend: Mapped[str | None] = mapped_column(String(32))
    storage_bucket: Mapped[str | None] = mapped_column(Text)
    storage_key: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(
        Integer, nullable=False, default=EMBEDDING_DIMENSIONS
    )
    compiler_model: Mapped[str | None] = mapped_column(Text)
    compile_prompt_version: Mapped[str | None] = mapped_column(String(32))
    section_count: Mapped[int | None] = mapped_column(Integer)
    intent_count: Mapped[int | None] = mapped_column(Integer)
    fact_count: Mapped[int | None] = mapped_column(Integer)
    coverage: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    validation_warnings: Mapped[list[str] | None] = mapped_column(JsonType)
    routing_eval_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    compile_duration_ms: Mapped[int | None] = mapped_column(Integer)
    ingest_duration_ms: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    uploaded_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --------------------------------------------------------------- compiled playbook
def _version_fk() -> Any:
    return ForeignKey("guideline_versions.id", ondelete="CASCADE")


def _company_fk() -> Any:
    return ForeignKey("companies.id", ondelete="CASCADE")


class GuidelineSection(Base):
    __tablename__ = "guideline_sections"
    __table_args__ = (Index("ix_guideline_sections_company_version", "company_id", "version_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(Uuid, _version_fk(), nullable=False)
    ref: Mapped[str] = mapped_column(String(16), nullable=False)  # "S1", stable within a version
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    heading: Mapped[str | None] = mapped_column(Text)
    path: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any] = mapped_column(VectorType)


class Intent(Base):
    __tablename__ = "intents"
    __table_args__ = (
        UniqueConstraint("version_id", "key", name="uq_intents_version_key"),
        Index("ix_intents_company_version", "company_id", "version_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(Uuid, _version_fk(), nullable=False)
    key: Mapped[str] = mapped_column(String(80), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    conditions: Mapped[list[str] | None] = mapped_column(JsonType)
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, nullable=False)
    reply_template: Mapped[str | None] = mapped_column(Text)
    placeholders: Mapped[list[str] | None] = mapped_column(JsonType)
    source_section_ids: Mapped[list[str] | None] = mapped_column(JsonType)


class IntentExample(Base):
    __tablename__ = "intent_examples"
    __table_args__ = (Index("ix_intent_examples_company_version", "company_id", "version_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(Uuid, _version_fk(), nullable=False)
    intent_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("intents.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    split: Mapped[str] = mapped_column(String(8), nullable=False)  # index | eval
    embedding: Mapped[Any] = mapped_column(VectorType)


class CompanyRules(Base):
    __tablename__ = "company_rules"
    __table_args__ = (UniqueConstraint("version_id", name="uq_company_rules_version_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(Uuid, _version_fk(), nullable=False)
    rules: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False)
    source_section_ids: Mapped[list[str] | None] = mapped_column(JsonType)


class ReferenceFact(Base):
    __tablename__ = "reference_facts"
    __table_args__ = (Index("ix_reference_facts_company_version", "company_id", "version_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(Uuid, _version_fk(), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_section_ids: Mapped[list[str] | None] = mapped_column(JsonType)
    embedding: Mapped[Any] = mapped_column(VectorType)


# ----------------------------------------------------------------------- evals
class RoutingEvalCase(Base):
    __tablename__ = "routing_eval_cases"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    expected_intent_keys: Mapped[list[str]] = mapped_column(JsonType, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    created_from_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("guideline_versions.id", ondelete="SET NULL")
    )
    chat_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("chats.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RoutingEvalRun(Base):
    __tablename__ = "routing_eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False, index=True)
    version_id: Mapped[uuid.UUID] = mapped_column(Uuid, _version_fk(), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False)
    results: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------- integrations
class CompanySecret(Base):
    __tablename__ = "company_secrets"
    __table_args__ = (UniqueConstraint("company_id", "name", name="uq_company_secrets_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    ciphertext: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class McpServer(Base):
    __tablename__ = "mcp_servers"
    __table_args__ = (UniqueConstraint("company_id", "name", name="uq_mcp_servers_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    transport: Mapped[str] = mapped_column(String(32), nullable=False, default="streamable_http")
    auth_secret_name: Mapped[str | None] = mapped_column(String(100))
    discovered_tools: Mapped[list[dict[str, Any]] | None] = mapped_column(JsonType)
    last_discovered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CapabilityBinding(Base):
    __tablename__ = "capability_bindings"
    __table_args__ = (
        UniqueConstraint("company_id", "capability", name="uq_capability_bindings_capability"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    capability: Mapped[str] = mapped_column(String(80), nullable=False)
    adapter_type: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False)
    policy: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=BindingStatus.DRAFT)
    last_test_result: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# ------------------------------------------------------------------------ chats
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
    primary_intent: Mapped[str | None] = mapped_column(String(80))
    recommended_response: Mapped[str | None] = mapped_column(Text)
    guidance: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    routing: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    references: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    feedback: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    llm_model: Mapped[str | None] = mapped_column(Text)
    router_model: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    router_prompt_version: Mapped[str | None] = mapped_column(String(32))
    latency_ms_total: Mapped[int | None] = mapped_column(Integer)
    latency_ms_routing: Mapped[int | None] = mapped_column(Integer)
    latency_ms_tools: Mapped[int | None] = mapped_column(Integer)
    latency_ms_llm: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    router_input_tokens: Mapped[int | None] = mapped_column(Integer)
    router_output_tokens: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


Index("ix_chats_company_created", Chat.company_id, Chat.created_at.desc())


class ProposedAction(Base):
    __tablename__ = "proposed_actions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    capability: Mapped[str] = mapped_column(String(80), nullable=False)
    args: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    step_ref: Mapped[str | None] = mapped_column(String(120))
    policy_check: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    decided_by: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(Uuid, _company_fk(), nullable=False)
    chat_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    proposed_action_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("proposed_actions.id", ondelete="SET NULL")
    )
    capability: Mapped[str] = mapped_column(String(80), nullable=False)
    adapter_type: Mapped[str | None] = mapped_column(String(20))
    binding_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)
    args: Mapped[dict[str, Any] | None] = mapped_column(JsonType)
    result: Mapped[Any] = mapped_column(JsonType)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
