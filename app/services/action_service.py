"""Proposed actions: the agent approves or rejects; the harness executes once.

Approval never re-runs the graph or calls an LLM: lock the row, re-check binding + policy,
mark `executing` (commit releases the lock so a concurrent approve sees it), execute through the
gateway with the action's idempotency key, then record `executed` / `failed` + the audit row.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.capabilities.gateway import CapabilityGateway
from app.capabilities.policy import check_policy
from app.capabilities.registry import get_capability
from app.core.errors import (
    ActionNotPending,
    CapabilityNotBound,
    ChatNotFound,
    NotFound,
    PolicyViolation,
)
from app.core.logging import get_logger
from app.db.models import ActionStatus, ProposedAction, utcnow
from app.services.common import AppMetrics, UowFactory
from app.services.integration_service import IntegrationService

logger = get_logger(__name__)


class ActionService:
    def __init__(
        self,
        uow_factory: UowFactory,
        gateway: CapabilityGateway,
        integrations: IntegrationService,
        metrics: AppMetrics,
    ) -> None:
        self._uow = uow_factory
        self._gateway = gateway
        self._integrations = integrations
        self._metrics = metrics

    async def list(self, chat_id: uuid.UUID) -> list[ProposedAction]:
        async with self._uow() as uow:
            if await uow.chats.get_for_company(chat_id) is None:
                raise ChatNotFound(f"Chat {chat_id} not found")
            rows: list[ProposedAction] = await uow.capabilities.actions_for_chat(chat_id)
            return rows

    async def reject(
        self, chat_id: uuid.UUID, action_id: uuid.UUID, *, by: str | None, reason: str | None
    ) -> ProposedAction:
        async with self._uow() as uow:
            action = await uow.capabilities.lock_action(chat_id, action_id)
            if action is None:
                raise NotFound(f"Action {action_id} not found")
            if action.status != ActionStatus.PROPOSED:
                raise ActionNotPending(f"Action is '{action.status}'")
            action.status = ActionStatus.REJECTED
            action.decided_by = by
            action.decided_at = utcnow()
            action.error = reason
            await uow.commit()
        self._metrics.proposed_actions.labels(status="rejected").inc()
        logger.info("action.rejected", extra={"action_id": str(action_id)})
        return action

    async def approve(
        self, chat_id: uuid.UUID, action_id: uuid.UUID, *, by: str | None
    ) -> ProposedAction:
        async with self._uow() as uow:
            action: ProposedAction | None = await uow.capabilities.lock_action(chat_id, action_id)
            if action is None:
                raise NotFound(f"Action {action_id} not found")
            if action.status == ActionStatus.EXECUTED:
                return action  # idempotent: approving again returns the existing result
            if action.status != ActionStatus.PROPOSED:
                raise ActionNotPending(
                    f"Action is '{action.status}'", details={"status": action.status}
                )
            company_id = action.company_id
            bindings = await self._integrations.live_bindings(company_id, uow=uow)
            binding = bindings.get(action.capability)
            cap = get_capability(action.capability)
            if binding is None or cap is None:
                raise CapabilityNotBound(f"'{action.capability}' has no live binding")
            check = check_policy(cap, action.args, binding.policy)
            if not check.ok:
                action.status = ActionStatus.DENIED
                action.policy_check = check.to_dict()
                await uow.commit()
                raise PolicyViolation("; ".join(check.violations))
            action.status = ActionStatus.EXECUTING
            action.decided_by = by
            action.decided_at = utcnow()
            await uow.commit()
            capability, args, key = action.capability, dict(action.args), action.idempotency_key

        logger.info(
            "action.approved", extra={"action_id": str(action_id), "capability": capability}
        )
        record = await self._gateway.call(
            company_id=company_id,
            capability=capability,
            args=args,
            binding=binding,
            get_secret=self._integrations.secret_getter(company_id),
            trigger="execution",
            idempotency_key=key,
        )
        async with self._uow() as uow:
            action = await uow.capabilities.lock_action(chat_id, action_id)
            assert action is not None
            action.status = ActionStatus.EXECUTED if record.ok else ActionStatus.FAILED
            action.result = record.result
            action.error = record.error
            action.executed_at = utcnow()
            if row := record.to_model(company_id, chat_id=chat_id, proposed_action_id=action_id):
                await uow.capabilities.add_tool_calls([row])
            await uow.commit()
        self._metrics.proposed_actions.labels(status=action.status).inc()
        self._metrics.tool_calls.labels(
            capability=capability, adapter=record.adapter_type or "none", status=record.status
        ).inc()
        logger.info(
            "action.executed",
            extra={"action_id": str(action_id), "capability": capability, "status": record.status},
        )
        return action

    @staticmethod
    def summary(a: ProposedAction) -> dict[str, Any]:
        return {"id": str(a.id), "capability": a.capability, "status": a.status, "args": a.args}
