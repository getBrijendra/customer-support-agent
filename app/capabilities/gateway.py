"""Capability gateway: the single path from the harness/services to company systems.

For every call: resolve the live binding (the company comes from the authenticated request,
never from the model) -> validate args against the schema -> (writes) check policy in code ->
inject secrets -> enforce the timeout -> truncate + redact the output -> return an audit record.
The caller persists the record (`tool_calls` row) in its own transaction.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.capabilities.adapters.base import (
    AdapterError,
    BindingSnapshot,
    CallContext,
    CapabilityAdapter,
    SecretGetter,
)
from app.capabilities.policy import PolicyCheck, check_policy
from app.capabilities.registry import ArgumentError, Capability, coerce_args, get_capability
from app.capabilities.secrets import redact
from app.db.models import BindingStatus, ToolCall

CallStatus = Literal["ok", "error", "timeout", "policy_denied", "not_bound", "invalid_args"]
Trigger = Literal["prefetch", "agent", "execution", "test"]


@dataclass
class ToolCallRecord:
    id: uuid.UUID
    capability: str
    trigger: Trigger
    status: CallStatus
    args: dict[str, Any]
    result: Any = None
    error: str | None = None
    latency_ms: int = 0
    adapter_type: str | None = None
    binding_id: uuid.UUID | None = None
    policy: PolicyCheck | None = None
    step_ref: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_model(
        self,
        company_id: uuid.UUID,
        *,
        chat_id: uuid.UUID | None = None,
        proposed_action_id: uuid.UUID | None = None,
    ) -> ToolCall | None:
        if self.status in {"not_bound", "invalid_args"}:
            return None  # nothing was called; not an audit event
        return ToolCall(
            id=self.id,
            company_id=company_id,
            chat_id=chat_id,
            proposed_action_id=proposed_action_id,
            capability=self.capability,
            adapter_type=self.adapter_type,
            binding_id=self.binding_id,
            trigger=self.trigger,
            args=self.args,
            result=self.result,
            status=self.status,
            latency_ms=self.latency_ms,
            error=self.error,
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "capability": self.capability,
            "trigger": self.trigger,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "step_ref": self.step_ref,
        }


def truncate_result(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, default=str)
    if len(text) <= max_chars:
        return json.loads(text)
    return {"truncated": True, "preview": text[:max_chars]}


class CapabilityGateway:
    def __init__(
        self,
        adapters: dict[str, CapabilityAdapter],
        *,
        timeout: float,
        max_result_chars: int,
        allow_private_network: bool,
    ) -> None:
        self.adapters = adapters
        self._timeout = timeout
        self._max_chars = max_result_chars
        self._allow_private = allow_private_network

    def adapter(self, adapter_type: str) -> CapabilityAdapter:
        try:
            return self.adapters[adapter_type]
        except KeyError as exc:
            raise ValueError(f"Unknown adapter type '{adapter_type}'") from exc

    async def call(
        self,
        *,
        company_id: uuid.UUID,
        capability: str,
        args: dict[str, Any],
        binding: BindingSnapshot | None,
        get_secret: SecretGetter,
        trigger: Trigger,
        idempotency_key: str | None = None,
        require_live: bool = True,
        step_ref: str | None = None,
    ) -> ToolCallRecord:
        record_id = uuid.uuid4()
        cap: Capability | None = get_capability(capability)

        def rec(status: CallStatus, **kw: Any) -> ToolCallRecord:
            return ToolCallRecord(
                id=record_id,
                capability=capability,
                trigger=trigger,
                status=status,
                args=kw.pop("args", args),
                adapter_type=binding.adapter_type if binding else None,
                binding_id=binding.id if binding else None,
                step_ref=step_ref,
                **kw,
            )

        if cap is None:
            return rec("invalid_args", error=f"Unknown capability '{capability}'")
        if (
            binding is None
            or binding.company_id != company_id  # tenant isolation, belt and braces
            or binding.capability != capability
            or (require_live and binding.status != BindingStatus.LIVE)
        ):
            return rec("not_bound", error="No live binding for this capability")
        try:
            clean = coerce_args(cap, args)
        except ArgumentError as exc:
            return rec("invalid_args", error=str(exc))
        policy = check_policy(cap, clean, binding.policy or {})
        if not policy.ok:
            return rec(
                "policy_denied", args=clean, error="; ".join(policy.violations), policy=policy
            )

        ctx = CallContext(
            company_id=company_id,
            get_secret=get_secret,
            timeout=self._timeout,
            allow_private_network=self._allow_private,
            idempotency_key=idempotency_key,
        )
        adapter = self.adapter(binding.adapter_type)
        started = time.perf_counter()
        try:
            data = await asyncio.wait_for(adapter.invoke(binding, clean, ctx), self._timeout)
            status: CallStatus = "ok"
            error = None
        except TimeoutError:
            data, status, error = None, "timeout", f"Timed out after {self._timeout}s"
        except AdapterError as exc:
            data, status, error = None, "error", str(exc)[:500]
        except Exception as exc:  # adapter bug or unexpected failure: degrade, never crash
            data, status, error = None, "error", f"{type(exc).__name__}: {exc}"[:500]
        latency = int((time.perf_counter() - started) * 1000)
        secrets = ctx.used_secrets
        return rec(
            status,
            args=redact(clean, secrets),
            result=redact(truncate_result(data, self._max_chars), secrets)
            if data is not None
            else None,
            error=redact(error, secrets) if error else None,
            latency_ms=latency,
            policy=policy,
        )
