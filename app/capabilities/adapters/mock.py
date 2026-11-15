"""In-process mock company systems for the demo and tests.

Config: {"store": "bookshop_orders" | "vpn_accounts", "delay_seconds"?: float, "fail"?: bool}
Writes are recorded per idempotency key, so the same key never executes twice.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from app.capabilities.adapters.base import AdapterError, BindingSnapshot, CallContext, ConfigError

BOOKSHOP_ORDERS: dict[str, dict[str, Any]] = {
    "1001": {
        "order_id": "1001",
        "email": "maria@example.com",
        "items": [{"title": "The Silent Sea", "qty": 1, "price": 18.99}],
        "status": "in_transit",
        "placed_at": "2026-11-03",
        "total": 18.99,
        "shipment": {"carrier": "DHL", "tracking": "DHL123456", "eta": "2026-11-15"},
    },
    "1002": {
        "order_id": "1002",
        "email": "tom@example.com",
        "items": [{"title": "Gardening Basics", "qty": 1, "price": 24.5}],
        "status": "delivered",
        "placed_at": "2026-10-28",
        "total": 24.5,
        "shipment": {"carrier": "UPS", "tracking": "1Z999", "eta": "2026-11-01"},
    },
}
VPN_ACCOUNTS: dict[str, dict[str, Any]] = {
    "jane@example.com": {
        "email": "jane@example.com",
        "status": "active",
        "security_hold": False,
        "subscription": {
            "plan": "Annual",
            "renewal_date": "2027-03-01",
            "payment_status": "paid",
            "charges": [
                {"date": "2026-11-02", "amount": 59.99, "id": "ch_1"},
                {"date": "2026-11-02", "amount": 59.99, "id": "ch_2"},
            ],
        },
        "devices": {"limit": 5, "connected": [{"id": "d1", "name": "iPhone 15"}]},
    },
}
SERVICE_STATUS = {"status": "operational", "incidents": []}

STORES = {"bookshop_orders", "vpn_accounts"}


class MockAdapter:
    adapter_type = "mock"

    def __init__(self) -> None:
        self.executed: dict[str, dict[str, Any]] = {}  # idempotency key -> result

    def validate_config(self, config: dict[str, Any]) -> None:
        if config.get("store") not in STORES:
            raise ConfigError(f"mock config.store must be one of {sorted(STORES)}")

    async def invoke(self, binding: BindingSnapshot, args: dict[str, Any], ctx: CallContext) -> Any:
        cfg = binding.config
        if cfg.get("delay_seconds"):
            await asyncio.sleep(float(cfg["delay_seconds"]))
        if cfg.get("fail"):
            raise AdapterError("mock system unavailable")
        cap = binding.capability
        if cap in {"order.get", "shipment.track"}:
            order = self._order(args)
            return (
                order
                if cap == "order.get"
                else {"order_id": order["order_id"], **order["shipment"]}
            )
        if cap in {"account.get", "subscription.get", "device.list"}:
            acct = VPN_ACCOUNTS.get(str(args.get("email", "")).lower())
            if acct is None:
                raise AdapterError("account not found")
            if cap == "subscription.get":
                return acct["subscription"]
            if cap == "device.list":
                return acct["devices"]
            return {
                "email": acct["email"],
                "status": acct["status"],
                "security_hold": acct["security_hold"],
            }
        if cap == "status.get":
            return SERVICE_STATUS
        return self._write(cap, args, ctx)

    @staticmethod
    def _order(args: dict[str, Any]) -> dict[str, Any]:
        if oid := args.get("order_id"):
            order = BOOKSHOP_ORDERS.get(str(oid).lstrip("#"))
        else:
            email = str(args.get("email", "")).lower()
            order = next((o for o in BOOKSHOP_ORDERS.values() if o["email"] == email), None)
        if order is None:
            raise AdapterError("order not found")
        return order

    def _write(self, cap: str, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        key = ctx.idempotency_key or ""
        if key and key in self.executed:
            return {**self.executed[key], "replayed": True}
        ref = hashlib.sha256(f"{cap}:{key}".encode()).hexdigest()[:10]
        result = {"capability": cap, "reference": f"mock_{ref}", "status": "done", "args": args}
        if key:
            self.executed[key] = result
        return result
