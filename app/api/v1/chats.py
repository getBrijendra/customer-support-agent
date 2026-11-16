from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.deps import (
    chat_history_service,
    chat_rate_limiter,
    eval_service,
    get_request_id,
    guidance_service,
)
from app.core.security import Principal, require_agent
from app.schemas.api import (
    SLUG_PATTERN,
    ActionOut,
    ChatCreate,
    ChatCreated,
    ChatDetail,
    ChatList,
    ChatListItem,
    ErrorResponse,
    FeedbackIn,
)
from app.services.chat_history_service import ChatHistoryService
from app.services.eval_service import EvalService
from app.services.guidance_service import ChatRequestData, GuidanceService
from app.services.rate_limiter import ChatRateLimiter, DisabledRateLimiter

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post(
    "",
    response_model=ChatCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Generate guidance for a customer message",
    description=(
        "Routes the message to the company's playbook intents (several when needed), always "
        "applies company rules, retrieves reference facts, runs read-only lookups against bound "
        "company systems, and returns structured guidance plus proposed actions (writes) for the "
        "agent to approve. Every attempt (success or failure) is stored. Rate limited per "
        "API key per company and per company overall (429 `RATE_LIMITED` with `Retry-After`; "
        "rejected requests are not stored as chats)."
    ),
    responses={
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "RATE_LIMITED"},
        409: {"model": ErrorResponse, "description": "NO_ACTIVE_GUIDELINES"},
        502: {"model": ErrorResponse, "description": "LLM_ERROR"},
    },
)
async def create_chat(
    body: ChatCreate,
    response: Response,
    principal: Principal = Depends(require_agent),
    service: GuidanceService = Depends(guidance_service),
    limiter: ChatRateLimiter | DisabledRateLimiter = Depends(chat_rate_limiter),
    request_id: str | None = Depends(get_request_id),
) -> ChatCreated:
    # Before any embedding/LLM work, so a rejected request costs nothing.
    limit_status = await limiter.check_chat(body.company, principal.label)
    if limit_status is not None:
        response.headers.update(limit_status.headers())
    outcome = await service.handle(
        ChatRequestData(
            company=body.company,
            customer_message=body.customer_message,
            agent_notes=body.agent_notes,
            channel=body.channel,
        ),
        request_id=request_id,
    )
    chat = outcome.chat
    return ChatCreated(
        chat_id=chat.id,
        company=outcome.company_slug,
        guideline_version=outcome.guideline_version,
        guidance=outcome.guidance,
        proposed_actions=[ActionOut.build(a) for a in outcome.actions],
        latency_ms=chat.latency_ms_total,
        created_at=chat.created_at,
    )


@router.get("", response_model=ChatList, summary="List previous chats (cursor paginated)")
async def list_chats(
    company: str | None = Query(None, pattern=SLUG_PATTERN),
    status_: Literal["success", "error"] | None = Query(None, alias="status"),
    intent: str | None = Query(None, max_length=80, description="Primary intent key"),
    from_: datetime | None = Query(None, alias="from", description="ISO-8601, inclusive"),
    to: datetime | None = Query(None, description="ISO-8601, exclusive"),
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None, description="Opaque cursor from `next_cursor`"),
    _: Principal = Depends(require_agent),
    service: ChatHistoryService = Depends(chat_history_service),
) -> ChatList:
    rows, next_cursor = await service.list(
        company=company,
        status=status_,
        intent=intent,
        created_from=from_,
        created_to=to,
        limit=limit,
        cursor=cursor,
    )
    return ChatList(
        items=[ChatListItem.build(r.chat, r.company_slug) for r in rows], next_cursor=next_cursor
    )


@router.get(
    "/{chat_id}",
    response_model=ChatDetail,
    summary="Full chat exchange incl. routing trace, tool calls and proposed actions",
    responses={404: {"model": ErrorResponse}},
)
async def get_chat(
    chat_id: uuid.UUID,
    _: Principal = Depends(require_agent),
    service: ChatHistoryService = Depends(chat_history_service),
) -> ChatDetail:
    d = await service.get(chat_id)
    return ChatDetail.build(d.row.chat, d.row.company_slug, d.tool_calls, d.actions)


@router.post(
    "/{chat_id}/feedback",
    summary="Correct the routed intents; becomes a routing eval case",
    responses={404: {"model": ErrorResponse}},
)
async def chat_feedback(
    chat_id: uuid.UUID,
    body: FeedbackIn,
    principal: Principal = Depends(require_agent),
    service: EvalService = Depends(eval_service),
) -> dict[str, Any]:
    return await service.add_feedback(chat_id, body.correct_intent_keys, principal.label)
