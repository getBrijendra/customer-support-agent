"""Request/response models for the public API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ai.schemas import GuidanceOutput
from app.db.models import (
    CapabilityBinding,
    Chat,
    Company,
    CompanySecret,
    GuidelineVersion,
    McpServer,
    ProposedAction,
    RoutingEvalCase,
    RoutingEvalRun,
    ToolCall,
)

SLUG_PATTERN = r"^[a-z0-9-]{2,50}$"
CAPABILITY_PATTERN = r"^[a-z_]+\.[a-z_]+$"
SECRET_NAME_PATTERN = r"^[A-Za-z0-9_.-]{1,100}$"


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
    review_status: str
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
    review_status: Literal["pending", "approved", "rejected"]
    review_note: str | None
    reviewed_by: str | None
    activate_on_approval: bool
    is_active: bool
    file_name: str
    source_format: str
    content_sha256: str
    byte_size: int
    content_type: str | None = Field(default=None, description="Of the original upload")
    storage_backend: str | None = Field(
        default=None,
        description="Where the original file is kept (`supabase`); null = not stored. "
        "Download it from `GET .../guidelines/{n}/file`.",
    )
    storage_bucket: str | None = None
    storage_key: str | None = None
    embedding_model: str
    embedding_dimensions: int
    compiler_model: str | None
    compile_prompt_version: str | None
    section_count: int | None
    intent_count: int | None
    fact_count: int | None
    coverage: dict[str, Any] | None
    validation_warnings: list[str] | None
    routing_eval_run_id: uuid.UUID | None
    compile_duration_ms: int | None
    ingest_duration_ms: int | None
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
        data = {
            name: getattr(v, name)
            for name in cls.model_fields
            if name not in {"company", "is_active", "raw_text", "info"}
        }
        return cls(
            **data,
            company=company.slug,
            is_active=company.active_version_id == v.id,
            raw_text=v.raw_text if include_text else None,
            info=info,
        )


class ReviewIn(BaseModel):
    note: str | None = Field(default=None, max_length=2000)


class PlaybookStepOut(BaseModel):
    step_id: str
    type: str
    text: str
    condition: str | None = None
    capability: str | None = None


class IntentOut(BaseModel):
    key: str
    name: str
    description: str
    conditions: list[str]
    steps: list[PlaybookStepOut]
    reply_template: str | None
    placeholders: list[str]
    source_section_ids: list[str]


class FactOut(BaseModel):
    id: uuid.UUID
    title: str
    text: str
    source_section_ids: list[str]


class SectionOut(BaseModel):
    id: uuid.UUID
    ref: str
    heading: str | None
    path: str | None
    char_start: int
    char_end: int


class PlaybookOut(BaseModel):
    company: str
    version_number: int
    compiler_model: str | None
    intents: list[IntentOut]
    company_rules: dict[str, Any]
    facts: list[FactOut]
    sections: list[SectionOut]
    coverage: dict[str, Any] | None
    validation_warnings: list[str] | None


# --------------------------------------------------------------------- chats
class ChatCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "company": "vpn-saas",
                    "customer_message": "I was charged twice this month and the VPN won't "
                    "connect on my iPhone.",
                    "agent_notes": "Customer email: jane@example.com, annual plan",
                    "channel": "email",
                }
            ]
        }
    )
    company: str = Field(pattern=SLUG_PATTERN)
    customer_message: str = Field(min_length=1, max_length=20_000)
    agent_notes: str | None = Field(default=None, max_length=5_000)
    channel: str | None = Field(default=None, max_length=32)


class ActionOut(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    capability: str
    args: dict[str, Any]
    reason: str | None
    step_ref: str | None
    status: str
    policy_check: dict[str, Any] | None
    result: Any | None
    error: str | None
    decided_by: str | None
    decided_at: datetime | None
    executed_at: datetime | None
    created_at: datetime

    @classmethod
    def build(cls, a: ProposedAction) -> ActionOut:
        return cls.model_validate(a, from_attributes=True)


class ToolCallOut(BaseModel):
    id: uuid.UUID
    capability: str
    adapter_type: str | None
    trigger: str
    status: str
    args: dict[str, Any] | None
    result: Any | None
    error: str | None
    latency_ms: int | None
    created_at: datetime

    @classmethod
    def build(cls, t: ToolCall) -> ToolCallOut:
        return cls.model_validate(t, from_attributes=True)


class ChatCreated(BaseModel):
    chat_id: uuid.UUID
    company: str
    guideline_version: int | None
    guidance: GuidanceOutput
    proposed_actions: list[ActionOut]
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
    primary_intent: str | None
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
            primary_intent=chat.primary_intent,
            customer_message=_truncate(chat.customer_message) or "",
            recommended_response=_truncate(chat.recommended_response),
            guideline_version=chat.guideline_version_number,
        )


class ChatList(BaseModel):
    items: list[ChatListItem]
    next_cursor: str | None


class LatencyOut(BaseModel):
    total: int | None
    routing: int | None
    tools: int | None
    llm: int | None


class TokensOut(BaseModel):
    input: int | None
    output: int | None
    router_input: int | None
    router_output: int | None


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
    primary_intent: str | None
    guidance: GuidanceOutput | None
    routing: dict[str, Any] | None
    references: dict[str, Any] | None
    feedback: dict[str, Any] | None
    tool_calls: list[ToolCallOut]
    proposed_actions: list[ActionOut]
    llm_model: str | None
    router_model: str | None
    prompt_version: str | None
    router_prompt_version: str | None
    latency_ms: LatencyOut
    tokens: TokensOut
    error_message: str | None
    request_id: str | None

    @classmethod
    def build(
        cls,
        chat: Chat,
        company_slug: str,
        tool_calls: list[ToolCall],
        actions: list[ProposedAction],
    ) -> ChatDetail:
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
            primary_intent=chat.primary_intent,
            guidance=GuidanceOutput.model_validate(chat.guidance) if chat.guidance else None,
            routing=chat.routing,
            references=chat.references,
            feedback=chat.feedback,
            tool_calls=[ToolCallOut.build(t) for t in tool_calls],
            proposed_actions=[ActionOut.build(a) for a in actions],
            llm_model=chat.llm_model,
            router_model=chat.router_model,
            prompt_version=chat.prompt_version,
            router_prompt_version=chat.router_prompt_version,
            latency_ms=LatencyOut(
                total=chat.latency_ms_total,
                routing=chat.latency_ms_routing,
                tools=chat.latency_ms_tools,
                llm=chat.latency_ms_llm,
            ),
            tokens=TokensOut(
                input=chat.input_tokens,
                output=chat.output_tokens,
                router_input=chat.router_input_tokens,
                router_output=chat.router_output_tokens,
            ),
            error_message=chat.error_message,
            request_id=chat.request_id,
        )


class FeedbackIn(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"correct_intent_keys": ["refund"]}]})
    correct_intent_keys: list[str] = Field(
        max_length=10, description="The intents the message really needed (empty = none fit)"
    )


class RejectIn(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


# -------------------------------------------------------------- integrations
class SecretIn(BaseModel):
    name: str = Field(pattern=SECRET_NAME_PATTERN)
    value: str = Field(min_length=1, max_length=10_000)


class SecretOut(BaseModel):
    name: str
    created_by: str | None
    created_at: datetime

    @classmethod
    def build(cls, s: CompanySecret) -> SecretOut:
        return cls(name=s.name, created_by=s.created_by, created_at=s.created_at)


class McpServerIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=8, max_length=2000)
    auth_secret_name: str | None = Field(default=None, pattern=SECRET_NAME_PATTERN)


class McpServerPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    url: str | None = Field(default=None, min_length=8, max_length=2000)
    auth_secret_name: str | None = Field(default=None, pattern=SECRET_NAME_PATTERN)


class McpServerOut(BaseModel):
    id: uuid.UUID
    name: str
    url: str
    transport: str
    auth_secret_name: str | None
    discovered_tools: list[dict[str, Any]] | None
    last_discovered_at: datetime | None

    @classmethod
    def build(cls, s: McpServer) -> McpServerOut:
        return cls.model_validate(s, from_attributes=True)


class BindingIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"adapter_type": "mock", "config": {"store": "bookshop_orders"}, "policy": {}},
                {
                    "adapter_type": "http",
                    "config": {
                        "method": "GET",
                        "url": "https://api.example.com/orders/{order_id}",
                        "auth_secret": "orders_api_key",
                        "response_path": "data",
                    },
                    "policy": {},
                },
                {
                    "adapter_type": "sql_readonly",
                    "config": {
                        "dsn_secret": "orders_db_dsn",
                        "statement": "SELECT id, status FROM orders WHERE id = :order_id",
                    },
                    "policy": {},
                },
            ]
        }
    )
    adapter_type: Literal["mcp", "http", "sql_readonly", "mock"]
    config: dict[str, Any]
    policy: dict[str, Any] = Field(default_factory=dict)


class BindingOut(BaseModel):
    id: uuid.UUID
    capability: str
    adapter_type: str
    config: dict[str, Any]
    policy: dict[str, Any]
    status: str
    last_test_result: dict[str, Any] | None
    updated_at: datetime

    @classmethod
    def build(cls, b: CapabilityBinding) -> BindingOut:
        return cls.model_validate(b, from_attributes=True)


class BindingTestIn(BaseModel):
    sample_args: dict[str, Any] = Field(default_factory=dict)
    execute_write: bool = Field(
        default=False, description="Actually execute a WRITE capability (side effects!)"
    )


class BindingTestOut(BaseModel):
    binding: BindingOut
    result: dict[str, Any]


# --------------------------------------------------------------------- evals
class EvalCaseIn(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    expected_intent_keys: list[str] = Field(max_length=10)


class EvalCaseOut(BaseModel):
    id: uuid.UUID
    message: str
    expected_intent_keys: list[str]
    source: str
    created_from_version_id: uuid.UUID | None
    chat_id: uuid.UUID | None
    created_at: datetime

    @classmethod
    def build(cls, c: RoutingEvalCase) -> EvalCaseOut:
        return cls.model_validate(c, from_attributes=True)


class EvalRunIn(BaseModel):
    company: str = Field(pattern=SLUG_PATTERN)
    version_number: int | None = Field(default=None, ge=1)
    mode: Literal["retrieval", "full"] = "retrieval"


class EvalRunOut(BaseModel):
    id: uuid.UUID
    company: str
    version_number: int
    mode: str
    metrics: dict[str, Any]
    results: list[dict[str, Any]] | None = None
    created_at: datetime

    @classmethod
    def build(
        cls, run: RoutingEvalRun, company: str, version: int, *, include_results: bool = True
    ) -> EvalRunOut:
        return cls(
            id=run.id,
            company=company,
            version_number=version,
            mode=run.mode,
            metrics=run.metrics,
            results=run.results if include_results else None,
            created_at=run.created_at,
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


class RoutingStats(BaseModel):
    no_match_rate: float
    multi_intent_rate: float
    top_intents: list[dict[str, Any]]
    feedback_count: int
    feedback_correction_rate: float
    latest_eval: list[dict[str, Any]]


class ToolStats(BaseModel):
    calls: list[dict[str, Any]]
    p95_latency_ms: float | None
    prefetch_share: float
    proposals: dict[str, int]


class MetricsSummary(BaseModel):
    window: dict[str, str]
    company: str | None = None
    chats: ChatStats
    by_company: list[CompanyStats]
    daily_volume: list[DailyVolume]
    ingestion: IngestionStats
    routing: RoutingStats
    tools: ToolStats


class HealthOut(BaseModel):
    status: str
    checks: dict[str, bool] | None = None
