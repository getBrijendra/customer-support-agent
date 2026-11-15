"""Capability registry: abstract read/write operations that playbook steps can map to.

Guidelines reference capabilities (`order.get`), never vendor APIs. Each company binds a
capability to an adapter (MCP / HTTP / read-only SQL / mock) — see `gateway.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["read", "write"]


@dataclass(frozen=True)
class Capability:
    name: str
    kind: Kind
    vertical: str
    description: str
    params: dict[str, str]  # param -> JSON type: string | number | integer | boolean
    required: tuple[str, ...] = ()
    policy_keys: tuple[str, ...] = ()

    @property
    def tool_name(self) -> str:
        """Function-calling safe name (`order.get` -> `order_get`)."""
        return self.name.replace(".", "_")

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {p: {"type": t} for p, t in self.params.items()},
            "required": list(self.required),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "vertical": self.vertical,
            "description": self.description,
            "input_schema": self.input_schema(),
            "policy_keys": list(self.policy_keys),
        }


def _c(
    name: str,
    kind: Kind,
    vertical: str,
    description: str,
    params: dict[str, str],
    required: tuple[str, ...] = (),
    policy_keys: tuple[str, ...] = (),
) -> Capability:
    return Capability(name, kind, vertical, description, params, required, policy_keys)


_EMAIL = {"email": "string"}
CAPABILITIES: dict[str, Capability] = {
    c.name: c
    for c in [
        # --- commerce -----------------------------------------------------------
        _c(
            "order.get",
            "read",
            "commerce",
            "Look up an order: items, status, totals, dates.",
            {"order_id": "string", "email": "string"},
        ),
        _c(
            "shipment.track",
            "read",
            "commerce",
            "Courier status, tracking number and ETA for an order.",
            {"order_id": "string"},
            ("order_id",),
        ),
        _c(
            "return_label.create",
            "write",
            "commerce",
            "Create a prepaid return label.",
            {"order_id": "string", "reason": "string"},
            ("order_id",),
        ),
        _c(
            "replacement.create",
            "write",
            "commerce",
            "Ship a replacement item for an order.",
            {"order_id": "string", "item": "string"},
            ("order_id",),
        ),
        _c(
            "discount.issue",
            "write",
            "commerce",
            "Issue a discount code to a customer.",
            {"email": "string", "order_id": "string", "percent": "number"},
            ("percent",),
            ("max_pct",),
        ),
        # --- subscription SaaS --------------------------------------------------
        _c(
            "account.get",
            "read",
            "saas",
            "Account status, suspensions and security holds for a customer.",
            _EMAIL,
            ("email",),
        ),
        _c(
            "subscription.get",
            "read",
            "saas",
            "Plan, renewal date, payment status and recent charges.",
            _EMAIL,
            ("email",),
        ),
        _c(
            "device.list",
            "read",
            "saas",
            "Devices connected to an account and the plan's device limit.",
            _EMAIL,
            ("email",),
        ),
        _c("status.get", "read", "saas", "Service status page: outages and maintenance.", {}),
        _c(
            "device.revoke",
            "write",
            "saas",
            "Disconnect a device from an account.",
            {"email": "string", "device_id": "string"},
            ("email", "device_id"),
        ),
        _c(
            "password.reset",
            "write",
            "saas",
            "Send a password reset and unlock the account.",
            _EMAIL,
            ("email",),
        ),
        _c(
            "payment.retry",
            "write",
            "saas",
            "Retry a failed renewal payment.",
            _EMAIL,
            ("email",),
        ),
        # --- shared -------------------------------------------------------------
        _c(
            "refund.create",
            "write",
            "shared",
            "Refund an amount for an order or a subscription charge.",
            {"order_id": "string", "email": "string", "amount": "number", "reason": "string"},
            ("amount",),
            ("max_amount",),
        ),
    ]
}


def get_capability(name: str) -> Capability | None:
    return CAPABILITIES.get(name)


def by_tool_name(tool_name: str) -> Capability | None:
    return next((c for c in CAPABILITIES.values() if c.tool_name == tool_name), None)


def registry_prompt_listing() -> str:
    """Compact listing for the compiler prompt."""
    return "\n".join(f"- {c.name} ({c.kind}): {c.description}" for c in CAPABILITIES.values())


class ArgumentError(ValueError):
    pass


def coerce_args(cap: Capability, raw: dict[str, Any]) -> dict[str, Any]:
    """Validate + coerce arguments against the capability's schema (LLMs often send strings)."""
    unknown = set(raw) - set(cap.params)
    if unknown:
        raise ArgumentError(f"Unknown argument(s) for {cap.name}: {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for name, typ in cap.params.items():
        if name not in raw or raw[name] is None or raw[name] == "":
            continue
        value = raw[name]
        try:
            if typ == "number":
                out[name] = float(value)
            elif typ == "integer":
                out[name] = int(value)
            elif typ == "boolean":
                out[name] = value if isinstance(value, bool) else str(value).lower() == "true"
            else:
                out[name] = str(value)
        except (TypeError, ValueError) as exc:
            raise ArgumentError(f"Argument '{name}' must be a {typ}") from exc
    missing = [r for r in cap.required if r not in out]
    if missing:
        raise ArgumentError(f"Missing required argument(s) for {cap.name}: {', '.join(missing)}")
    return out
