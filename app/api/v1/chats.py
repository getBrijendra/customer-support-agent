from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query, status

from app.ai.schemas import GuidanceOutput
from app.api.deps import chat_history_service, get_request_id, guidance_service
from app.core.security import Principal, require_agent
from app.schemas.api import (
    SLUG_PATTERN,
    ChatCreate,
    ChatCreated,
    ChatDetail,
    ChatList,
    ChatListItem,
    ErrorResponse,
)
from app.services.chat_history_service import ChatHistoryService
from app.services.guidance_service import ChatRequestData, GuidanceService

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post(
    "",
    response_model=ChatCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Generate guidance for a customer message",
    description="Retrieves from the company's ACTIVE guideline version and returns structured "
    "guidance. Every attempt (success or failure) is stored.",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "NO_ACTIVE_GUIDELINES"},
        502: {"model": ErrorResponse, "description": "LLM_ERROR"},
    },
)
async def create_chat(
    body: ChatCreate,
    _: Principal = Depends(require_agent),
    service: GuidanceService = Depends(guidance_service),
    request_id: str | None = Depends(get_request_id),
) -> ChatCreated:
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
        guidance=GuidanceOutput.model_validate(chat.guidance),
        latency_ms=chat.latency_ms_total,
        created_at=chat.created_at,
    )


@router.get("", response_model=ChatList, summary="List previous chats (cursor paginated)")
async def list_chats(
    company: str | None = Query(None, pattern=SLUG_PATTERN),
    status_: Literal["success", "error"] | None = Query(None, alias="status"),
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
    summary="Full chat exchange",
    responses={404: {"model": ErrorResponse}},
)
async def get_chat(
    chat_id: uuid.UUID,
    _: Principal = Depends(require_agent),
    service: ChatHistoryService = Depends(chat_history_service),
) -> ChatDetail:
    row = await service.get(chat_id)
    return ChatDetail.build(row.chat, row.company_slug)
