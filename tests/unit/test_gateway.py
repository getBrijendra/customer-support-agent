"""U14 — capability gateway: degradation, validation, policy, secrets redaction, idempotency."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from app.capabilities.adapters.base import AdapterError, BindingSnapshot, CallContext
from app.capabilities.adapters.mock import MockAdapter
from app.capabilities.gateway import CapabilityGateway

COMPANY = uuid.uuid4()


def binding(
    capability: str,
    *,
    status: str = "live",
    adapter_type: str = "mock",
    policy: dict[str, Any] | None = None,
    **cfg: Any,
) -> BindingSnapshot:
    return BindingSnapshot(
        id=uuid.uuid4(),
        company_id=COMPANY,
        capability=capability,
        adapter_type=adapter_type,
        config={"store": "bookshop_orders", **cfg},
        policy=policy or {},
        status=status,
    )


async def no_secret(name: str) -> str:
    raise AssertionError("no secrets expected")


class EchoSecretAdapter:
    adapter_type = "echo"

    def validate_config(self, config: dict[str, Any]) -> None: ...

    async def invoke(self, b: BindingSnapshot, args: dict[str, Any], ctx: CallContext) -> Any:
        token = await ctx.secret("api_key")
        raise AdapterError(f"401 for token {token}")


@pytest.fixture
def gw() -> CapabilityGateway:
    return CapabilityGateway(
        {"mock": MockAdapter(), "echo": EchoSecretAdapter()},  # type: ignore[dict-item]
        timeout=0.2,
        max_result_chars=200,
        allow_private_network=True,
    )


async def call(gw: CapabilityGateway, cap: str, args: dict[str, Any], b: Any, **kw: Any) -> Any:
    return await gw.call(
        company_id=COMPANY,
        capability=cap,
        args=args,
        binding=b,
        get_secret=kw.pop("get_secret", no_secret),
        trigger="prefetch",
        **kw,
    )


async def test_read_ok_and_truncated(gw: CapabilityGateway) -> None:
    rec = await call(gw, "order.get", {"order_id": "1001"}, binding("order.get"))
    assert rec.ok and rec.status == "ok"
    assert rec.result == {"truncated": True, "preview": rec.result["preview"]}  # > 200 chars
    assert len(rec.result["preview"]) == 200
    model = rec.to_model(COMPANY)
    assert model is not None and model.status == "ok" and model.trigger == "prefetch"


async def test_unbound_draft_and_foreign_bindings_degrade(gw: CapabilityGateway) -> None:
    assert (await call(gw, "order.get", {"order_id": "1"}, None)).status == "not_bound"
    draft = binding("order.get", status="tested")
    assert (await call(gw, "order.get", {"order_id": "1"}, draft)).status == "not_bound"
    foreign = binding("order.get")
    foreign.company_id = uuid.uuid4()
    assert (await call(gw, "order.get", {"order_id": "1"}, foreign)).status == "not_bound"
    rec = await call(gw, "order.get", {"order_id": "1"}, None)
    assert rec.to_model(COMPANY) is None  # nothing was called -> no audit row


async def test_bad_args_and_unknown_capability(gw: CapabilityGateway) -> None:
    rec = await call(gw, "subscription.get", {}, binding("subscription.get"))
    assert rec.status == "invalid_args" and "Missing" in (rec.error or "")
    rec = await call(gw, "nope.get", {}, binding("nope.get"))
    assert rec.status == "invalid_args"


async def test_adapter_error_and_timeout(gw: CapabilityGateway) -> None:
    rec = await call(gw, "order.get", {"order_id": "999"}, binding("order.get"))
    assert rec.status == "error" and "not found" in (rec.error or "")
    slow = binding("order.get", delay_seconds=1)
    rec = await call(gw, "order.get", {"order_id": "1001"}, slow)
    assert rec.status == "timeout"


async def test_policy_denied_for_writes(gw: CapabilityGateway) -> None:
    b = binding("refund.create", policy={"max_amount": 10})
    rec = await call(gw, "refund.create", {"amount": "50"}, b)
    assert rec.status == "policy_denied" and rec.policy and not rec.policy.ok


async def test_secrets_redacted(gw: CapabilityGateway) -> None:
    async def secret(name: str) -> str:
        return "s3cr3t-value"

    b = binding("order.get", adapter_type="echo")
    rec = await call(gw, "order.get", {"order_id": "1"}, b, get_secret=secret)
    assert rec.status == "error"
    assert "s3cr3t-value" not in (rec.error or "") and "***" in (rec.error or "")


async def test_mock_writes_are_idempotent(gw: CapabilityGateway) -> None:
    b = binding("refund.create", policy={"max_amount": 100})
    first = await call(gw, "refund.create", {"amount": 5}, b, idempotency_key="k1")
    again = await call(gw, "refund.create", {"amount": 5}, b, idempotency_key="k1")
    assert first.ok and again.ok
    assert again.result["replayed"] is True
    assert first.result["reference"] == again.result["reference"]


async def test_concurrent_calls_independent(gw: CapabilityGateway) -> None:
    b = binding("order.get")
    recs = await asyncio.gather(
        *(call(gw, "order.get", {"order_id": oid}, b) for oid in ("1001", "1002", "nope"))
    )
    assert [r.status for r in recs] == ["ok", "ok", "error"]
