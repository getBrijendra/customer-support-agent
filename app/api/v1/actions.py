from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends

from app.api.deps import action_service
from app.core.security import Principal, require_agent
from app.schemas.api import ActionOut, ErrorResponse, RejectIn
from app.services.action_service import ActionService

router = APIRouter(prefix="/chats/{chat_id}/actions", tags=["actions"])
ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
}


@router.get("", response_model=list[ActionOut], summary="Proposed actions of a chat")
async def list_actions(
    chat_id: uuid.UUID,
    _: Principal = Depends(require_agent),
    service: ActionService = Depends(action_service),
) -> list[ActionOut]:
    return [ActionOut.build(a) for a in await service.list(chat_id)]


@router.post(
    "/{action_id}/approve",
    response_model=ActionOut,
    summary="Approve and execute a proposed action (exactly once)",
    description="Re-checks the binding and policy, then executes through the capability gateway "
    "with the action's idempotency key. Approving an executed action again returns its result.",
    responses={**ERRORS, 502: {"model": ErrorResponse}},
)
async def approve_action(
    chat_id: uuid.UUID,
    action_id: uuid.UUID,
    principal: Principal = Depends(require_agent),
    service: ActionService = Depends(action_service),
) -> ActionOut:
    return ActionOut.build(await service.approve(chat_id, action_id, by=principal.label))


@router.post(
    "/{action_id}/reject",
    response_model=ActionOut,
    summary="Reject a proposed action",
    responses=ERRORS,
)
async def reject_action(
    chat_id: uuid.UUID,
    action_id: uuid.UUID,
    body: RejectIn | None = Body(None),
    principal: Principal = Depends(require_agent),
    service: ActionService = Depends(action_service),
) -> ActionOut:
    return ActionOut.build(
        await service.reject(
            chat_id, action_id, by=principal.label, reason=body.reason if body else None
        )
    )
