"""AgriFarm demo company system as an in-process MCP server (no port, no process to run).

Data comes from `data/mcp/agrifarm.json` (re-read on every call, so edits apply immediately).
`as_of` in that file is the server's "today", so every result is deterministic.

Reads return records from the file. Writes validate against the data (amount <= refundable,
order not yet packed, slot has capacity, ...) and return deterministic references derived from
their arguments. They are recorded in memory per idempotency key (never written back to the
file): the same key returns the original result with `replayed: true`.

Reach it from a binding by registering an MCP server with URL `inproc://agrifarm`.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

DEFAULT_DATA_FILE = Path(__file__).resolve().parents[4] / "data" / "mcp" / "agrifarm.json"
DATA_FILE_ENV = "AGRIFARM_MCP_DATA_FILE"
CANCELLABLE = {"confirmed"}
RESCHEDULABLE = {"confirmed", "delayed"}


def _load(path: Path) -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError(f"AgriFarm data unavailable: {type(exc).__name__}") from exc
    return data


def _ref(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"{prefix}-{digest[:8].upper()}"


def _money(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def build_server(data_file: Path | None = None) -> MCPServer:
    path = data_file or Path(os.environ.get(DATA_FILE_ENV) or DEFAULT_DATA_FILE)
    server = MCPServer(
        "agrifarm",
        instructions="AgriFarm order, delivery, product and Farm Box systems (demo data).",
    )
    ledger: dict[str, dict[str, Any]] = {}  # idempotency key -> first result

    def data() -> dict[str, Any]:
        return _load(path)

    def today(d: dict[str, Any]) -> date:
        return date.fromisoformat(d["as_of"])

    def order(d: dict[str, Any], order_id: str | None, email: str | None = None) -> tuple[str, Any]:
        orders: dict[str, Any] = d["orders"]
        if order_id:
            oid = order_id.strip().lstrip("#").upper()
            if not oid.startswith("AF-") and oid.isdigit():
                oid = f"AF-{oid}"
            if oid in orders:
                return oid, orders[oid]
            raise ToolError(f"Order {order_id} not found")
        if email:
            matches = [(k, o) for k, o in orders.items() if o["email"] == email.strip().lower()]
            if matches:
                return max(matches, key=lambda kv: kv[1]["placed_at"])
        raise ToolError("Order not found (give order_id or the customer's email)")

    def zone_of(d: dict[str, Any], postcode: str) -> str:
        for name, zone in d["zones"].items():
            if postcode.strip() in zone["postcodes"]:
                return str(name)
        raise ToolError(f"Postcode {postcode} is outside the AgriFarm delivery area")

    def open_slots(d: dict[str, Any], zone: str) -> list[dict[str, Any]]:
        now = today(d)
        return [
            s
            for s in d["slots"]
            if s["zone"] == zone and s["capacity_left"] > 0 and date.fromisoformat(s["date"]) > now
        ]

    def find_product(d: dict[str, Any], query: str) -> dict[str, Any]:
        q = query.strip().lower()
        products: list[dict[str, Any]] = d["products"]
        for p in products:
            if q == p["sku"].lower() or q == p["name"].lower():
                return p
        hits = [p for p in products if q in p["name"].lower() or q in p["sku"].lower()]
        if not hits:  # every word present ("strawberries organic")
            words = q.split()
            hits = [p for p in products if all(w.rstrip("s") in p["name"].lower() for w in words)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            names = ", ".join(p["name"] for p in hits)
            raise ToolError(f"'{query}' matches several products: {names}")
        raise ToolError(f"No product matches '{query}'")

    def farm_box(d: dict[str, Any], email: str) -> dict[str, Any]:
        box = d["farm_boxes"].get(email.strip().lower())
        if box is None:
            raise ToolError(f"No Farm Box subscription for {email}")
        return dict(box)

    def once(key: str | None, make: Any) -> dict[str, Any]:
        if key and key in ledger:
            return {**ledger[key], "replayed": True}
        result: dict[str, Any] = make()
        if key:
            ledger[key] = result
        return result

    # ------------------------------------------------------------------ reads
    @server.tool()
    def get_order(order_id: str | None = None, email: str | None = None) -> dict[str, Any]:
        """Look up an order by id (e.g. AF-10421) or the customer's latest order by email:
        status, items, totals, delivery date and slot."""
        d = data()
        oid, o = order(d, order_id, email)
        return {
            "order_id": oid,
            "email": o["email"],
            "status": o["status"],
            "placed_at": o["placed_at"],
            "delivery_date": o["delivery"]["date"],
            "delivery_window": o["delivery"]["window"],
            "items": o["items"],
            "delivery_fee": o["delivery_fee"],
            "total": o["total"],
            "refunded": o["refunded"],
            "currency": d["currency"],
            "can_cancel": o["status"] in CANCELLABLE,
            "can_reschedule": o["status"] in RESCHEDULABLE,
            "packing_cutoff": d["packing_cutoff"],
        }

    @server.tool()
    def track_delivery(order_id: str) -> dict[str, Any]:
        """Courier status, tracking, ETA, driver notes and the cold-chain temperature log."""
        _, o = order(data(), order_id)
        dl = o["delivery"]
        return {
            "order_id": order_id.strip().lstrip("#").upper(),
            "status": o["status"],
            "courier": dl["courier"],
            "tracking": dl["tracking"],
            "scheduled_date": dl["date"],
            "window": dl["window"],
            "delivered_at": dl.get("delivered_at"),
            "eta": dl.get("new_eta") or (None if dl.get("delivered_at") else dl["date"]),
            "driver_note": dl.get("driver_note"),
            "cold_chain": o.get("cold_chain"),
        }

    @server.tool()
    def list_delivery_slots(
        order_id: str | None = None, postcode: str | None = None
    ) -> dict[str, Any]:
        """Open delivery slots (with capacity) for an order's postcode or a given postcode."""
        d = data()
        if not postcode:
            if not order_id:
                raise ToolError("Give order_id or postcode")
            postcode = order(d, order_id)[1]["postcode"]
        zone = zone_of(d, postcode)
        return {
            "postcode": postcode,
            "zone": zone,
            "delivery_days": d["zones"][zone]["delivery_days"],
            "slots": [
                {k: s[k] for k in ("slot_id", "date", "window", "capacity_left")}
                for s in open_slots(d, zone)
            ],
        }

    @server.tool()
    def check_product_availability(product: str, postcode: str | None = None) -> dict[str, Any]:
        """Stock, season and next harvest/restock date for a product (name or SKU), with
        in-season alternatives when it is unavailable."""
        d = data()
        p = find_product(d, product)
        alternatives = [
            {"sku": a["sku"], "name": a["name"], "price": a["price"]}
            for a in d["products"]
            if a["sku"] in (p.get("alternatives") or []) and a["stock"] > 0
        ]
        result = {
            "sku": p["sku"],
            "name": p["name"],
            "price": p["price"],
            "currency": d["currency"],
            "in_stock": p["stock"] > 0,
            "stock": p["stock"],
            "in_season": p["in_season"],
            "next_available": p["next_available"],
            "note": p["note"],
            "alternatives": alternatives if p["stock"] == 0 else [],
        }
        if postcode:
            result["delivers_to_postcode"] = any(
                postcode.strip() in z["postcodes"] for z in d["zones"].values()
            )
        return result

    @server.tool()
    def get_farm_box(email: str) -> dict[str, Any]:
        """Farm Box subscription: plan, weekly price, status, next delivery, pause and charges."""
        d = data()
        box = farm_box(d, email)
        return {"email": email.strip().lower(), "currency": d["currency"], **box}

    # ----------------------------------------------------------------- writes
    @server.tool()
    def issue_refund(
        amount: float,
        order_id: str | None = None,
        email: str | None = None,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Refund an amount to the original payment method, for an order or a Farm Box charge.
        The amount may not exceed what is still refundable."""
        d = data()
        if order_id:
            oid, o = order(d, order_id)
            refundable = _money(o["total"] - o["refunded"])
            target = {"order_id": oid}
        elif email:
            box = farm_box(d, email)
            last = box["charges"][0]
            oid, refundable = last["charge_id"], _money(last["amount"])
            target = {"charge_id": oid, "email": email.strip().lower()}
        else:
            raise ToolError("Give order_id or the Farm Box customer's email")
        if amount <= 0 or _money(amount) > refundable:
            raise ToolError(f"Amount {amount} exceeds the refundable {refundable} for {oid}")

        def make() -> dict[str, Any]:
            return {
                "refund_id": _ref("RF", oid, _money(amount), idempotency_key),
                **target,
                "amount": _money(amount),
                "currency": d["currency"],
                "method": "original_payment_method",
                "expected_settlement": (today(d) + timedelta(days=5)).isoformat(),
                "status": "processed",
                "reason": reason,
            }

        return once(idempotency_key, make)

    @server.tool()
    def create_replacement_delivery(
        order_id: str, item: str | None = None, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Send replacement items free of charge in the next open delivery slot."""
        d = data()
        oid, o = order(d, order_id)
        items = o["items"]
        if item:
            wanted = item.strip().lower()
            items = [i for i in items if wanted in i["name"].lower() or wanted == i["sku"].lower()]
            if not items:
                raise ToolError(f"'{item}' is not part of order {oid}")
        slots = open_slots(d, zone_of(d, o["postcode"]))
        if not slots:
            raise ToolError("No open delivery slot for this area")
        slot = min(slots, key=lambda s: (s["date"], s["window"]))

        def make() -> dict[str, Any]:
            return {
                "replacement_order_id": _ref("AF-R", oid, item, idempotency_key),
                "original_order_id": oid,
                "items": [{"name": i["name"], "qty": i["qty"]} for i in items],
                "delivery_date": slot["date"],
                "delivery_window": slot["window"],
                "slot_id": slot["slot_id"],
                "charge": 0.0,
                "status": "scheduled",
            }

        return once(idempotency_key, make)

    @server.tool()
    def issue_discount_code(
        percent: float,
        email: str | None = None,
        order_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create a single-use percentage discount code valid for 30 days."""
        d = data()
        if not 0 < percent <= 50:
            raise ToolError("percent must be between 0 and 50")
        who = (email or "").strip().lower() or (order(d, order_id)[1]["email"] if order_id else "")
        if not who:
            raise ToolError("Give the customer's email or an order_id")

        def make() -> dict[str, Any]:
            code = _ref(f"FRESH{int(percent)}", who, percent, idempotency_key)
            return {
                "code": code,
                "email": who,
                "percent": percent,
                "single_use": True,
                "expires_on": (today(d) + timedelta(days=30)).isoformat(),
            }

        return once(idempotency_key, make)

    @server.tool()
    def reschedule_delivery(
        order_id: str, slot_id: str, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Move a confirmed or delayed order to another open slot in its delivery zone."""
        d = data()
        oid, o = order(d, order_id)
        if o["status"] not in RESCHEDULABLE:
            raise ToolError(f"Order {oid} is '{o['status']}' and can no longer be rescheduled")
        zone = zone_of(d, o["postcode"])
        slot = next((s for s in open_slots(d, zone) if s["slot_id"] == slot_id.strip()), None)
        if slot is None:
            raise ToolError(f"Slot {slot_id} is full, past or not in zone '{zone}'")

        def make() -> dict[str, Any]:
            return {
                "order_id": oid,
                "previous_date": o["delivery"]["date"],
                "new_date": slot["date"],
                "new_window": slot["window"],
                "slot_id": slot["slot_id"],
                "confirmation": _ref("RS", oid, slot["slot_id"], idempotency_key),
                "status": "rescheduled",
            }

        return once(idempotency_key, make)

    @server.tool()
    def cancel_order(
        order_id: str, reason: str | None = None, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Cancel an order that has not started packing and return the full payment."""
        d = data()
        oid, o = order(d, order_id)
        if o["status"] not in CANCELLABLE:
            raise ToolError(
                f"Order {oid} is '{o['status']}'; only orders that have not started packing "
                "can be cancelled"
            )

        def make() -> dict[str, Any]:
            return {
                "order_id": oid,
                "status": "cancelled",
                "refund_amount": _money(o["total"] - o["refunded"]),
                "currency": d["currency"],
                "refund_id": _ref("RF", oid, "cancel", idempotency_key),
                "expected_settlement": (today(d) + timedelta(days=5)).isoformat(),
                "reason": reason,
            }

        return once(idempotency_key, make)

    @server.tool()
    def issue_store_credit(
        amount: float,
        email: str | None = None,
        order_id: str | None = None,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Add store credit to the customer's account; usable immediately on the next order."""
        d = data()
        who = (email or "").strip().lower() or (order(d, order_id)[1]["email"] if order_id else "")
        customer = d["customers"].get(who)
        if customer is None:
            raise ToolError("Customer not found (give email or order_id)")
        if amount <= 0:
            raise ToolError("amount must be positive")

        def make() -> dict[str, Any]:
            return {
                "credit_id": _ref("CR", who, _money(amount), idempotency_key),
                "email": who,
                "amount": _money(amount),
                "currency": d["currency"],
                "new_balance": _money(customer["store_credit"] + amount),
                "reason": reason,
            }

        return once(idempotency_key, make)

    @server.tool()
    def pause_farm_box(
        email: str, weeks: int, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Pause a Farm Box for 1-4 weeks; no charges or deliveries until it resumes."""
        d = data()
        box = farm_box(d, email)
        if not 1 <= weeks <= 4:
            raise ToolError("A Farm Box can be paused for 1 to 4 weeks")
        if box["status"] == "paused":
            raise ToolError(f"Farm Box is already paused until {box['paused_until']}")
        first = date.fromisoformat(box["next_delivery"])
        skipped = [(first + timedelta(weeks=i)).isoformat() for i in range(weeks)]

        def make() -> dict[str, Any]:
            return {
                "email": email.strip().lower(),
                "plan": box["plan"],
                "skipped_deliveries": skipped,
                "resumes_on": (first + timedelta(weeks=weeks)).isoformat(),
                "confirmation": _ref("PZ", email.strip().lower(), weeks, idempotency_key),
                "status": "paused",
            }

        return once(idempotency_key, make)

    @server.tool()
    def create_restock_alert(
        email: str, product: str, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Email the customer when an out-of-stock or out-of-season product is available."""
        p = find_product(data(), product)

        def make() -> dict[str, Any]:
            return {
                "alert_id": _ref("AL", email.strip().lower(), p["sku"], idempotency_key),
                "email": email.strip().lower(),
                "sku": p["sku"],
                "name": p["name"],
                "expected_available": p["next_available"],
                "status": "active",
            }

        return once(idempotency_key, make)

    # Present on the server but never mapped to a capability: invisible to the model.
    @server.tool()
    def export_customer_list() -> dict[str, Any]:
        """Bulk export of every customer (internal back-office tool)."""
        raise ToolError("Not permitted for support integrations")

    return server
