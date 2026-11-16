"""AgriFarm in-process MCP server, reached through the real McpAdapter via `inproc://agrifarm`."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from app.ai.heuristic import HeuristicPlaybookCompiler
from app.capabilities.adapters.base import AdapterError, BindingSnapshot, CallContext
from app.capabilities.adapters.local_mcp import get_local_server
from app.capabilities.adapters.mcp import McpAdapter
from app.capabilities.policy import check_policy
from app.capabilities.registry import get_capability
from app.documents.txt_parser import TxtParser

pytestmark = pytest.mark.unit

COMPANY = uuid.uuid4()
SEED = Path(__file__).resolve().parents[2] / "data" / "seed_guidelines" / "agrifarm.txt"


async def _no_secret(name: str) -> str:
    raise AssertionError("no secrets expected")


def _binding(capability: str, tool: str, **cfg: Any) -> BindingSnapshot:
    return BindingSnapshot(
        id=uuid.uuid4(),
        company_id=COMPANY,
        capability=capability,
        adapter_type="mcp",
        config={"server_id": str(uuid.uuid4()), "tool_name": tool, "arg_map": {}, **cfg},
        policy={},
        status="live",
        server={"url": "inproc://agrifarm", "auth_secret_name": None},
    )


def _ctx(key: str | None = None) -> CallContext:
    return CallContext(
        company_id=COMPANY,
        get_secret=_no_secret,
        timeout=5,
        allow_private_network=False,  # in-process servers bypass the network guard entirely
        idempotency_key=key,
    )


async def _call(cap: str, tool: str, args: dict[str, Any], key: str | None = None) -> Any:
    binding = _binding(cap, tool, idempotency_arg="idempotency_key")
    return await McpAdapter().invoke(binding, args, _ctx(key))


async def test_discovery_lists_all_tools() -> None:
    tools = await McpAdapter().discover({"url": "inproc://agrifarm"}, _ctx())
    names = {t["name"] for t in tools}
    assert {"get_order", "track_delivery", "issue_refund", "pause_farm_box"} <= names
    assert "export_customer_list" in names  # present on the server, but never bound


async def test_reads_return_file_data() -> None:
    order = await _call("order.get", "get_order", {"order_id": "af-10421"})
    assert order["order_id"] == "AF-10421" and order["total"] == 30.1
    track = await _call("shipment.track", "track_delivery", {"order_id": "AF-10463"})
    assert track["status"] == "delayed" and track["eta"] == "2026-11-17T10:00:00"
    slots = await _call("delivery_slot.list", "list_delivery_slots", {"order_id": "AF-10463"})
    assert slots["zone"] == "north"
    assert "N-2026-11-17-AM" not in {s["slot_id"] for s in slots["slots"]}  # full slot hidden
    prod = await _call(
        "product.availability", "check_product_availability", {"product": "strawberries"}
    )
    assert prod["in_stock"] is False and prod["next_available"] == "2027-05-10"
    assert {a["sku"] for a in prod["alternatives"]} == {"FRU-APP-HONEY-1KG", "FRU-PEAR-1KG"}
    box = await _call("farmbox.get", "get_farm_box", {"email": "marcus.bell@example.com"})
    assert box["status"] == "paused" and box["plan"] == "Large"


async def test_writes_are_deterministic_and_idempotent() -> None:
    args = {"order_id": "AF-10421", "amount": 12.1, "reason": "spoiled spinach"}
    first = await _call("refund.create", "issue_refund", args, key="k-refund-1")
    again = await _call("refund.create", "issue_refund", args, key="k-refund-1")
    assert first["refund_id"].startswith("RF-") and first["amount"] == 12.1
    assert again["refund_id"] == first["refund_id"] and again["replayed"] is True
    other = await _call("refund.create", "issue_refund", args, key="k-refund-2")
    assert other["refund_id"] != first["refund_id"]

    pause = await _call(
        "farmbox.pause", "pause_farm_box", {"email": "priya.sharma@example.com", "weeks": 2}, "k-p"
    )
    assert pause["skipped_deliveries"] == ["2026-11-21", "2026-11-28"]
    assert pause["resumes_on"] == "2026-12-05"
    moved = await _call(
        "delivery.reschedule",
        "reschedule_delivery",
        {"order_id": "AF-10463", "slot_id": "N-2026-11-19-AM"},
        "k-r",
    )
    assert moved["new_date"] == "2026-11-19" and moved["status"] == "rescheduled"
    repl = await _call(
        "replacement.create",
        "create_replacement_delivery",
        {"order_id": "AF-10421", "item": "spinach"},
        "k-x",
    )
    assert repl["slot_id"] == "N-2026-11-17-PM" and repl["items"] == [
        {"name": "Baby Spinach 250g", "qty": 1}
    ]
    cancel = await _call("order.cancel", "cancel_order", {"order_id": "AF-10458"}, "k-c")
    assert cancel["status"] == "cancelled" and cancel["refund_amount"] == 39.4


@pytest.mark.parametrize(
    ("cap", "tool", "args", "message"),
    [
        ("refund.create", "issue_refund", {"order_id": "AF-10421", "amount": 99}, "exceeds"),
        ("order.cancel", "cancel_order", {"order_id": "AF-10470"}, "packed"),
        (
            "delivery.reschedule",
            "reschedule_delivery",
            {"order_id": "AF-10463", "slot_id": "N-2026-11-17-AM"},
            "full",
        ),
        (
            "farmbox.pause",
            "pause_farm_box",
            {"email": "marcus.bell@example.com", "weeks": 1},
            "already paused",
        ),
        ("order.get", "get_order", {"order_id": "AF-99999"}, "not found"),
    ],
)
async def test_business_rules_raise_adapter_errors(
    cap: str, tool: str, args: dict[str, Any], message: str
) -> None:
    with pytest.raises(AdapterError, match=message):
        await _call(cap, tool, args, key=f"k-{tool}")


async def test_unknown_inproc_server_is_an_adapter_error() -> None:
    binding = _binding("order.get", "get_order")
    binding.server = {"url": "inproc://nope", "auth_secret_name": None}
    with pytest.raises(AdapterError, match="Unknown in-process MCP server"):
        await McpAdapter().invoke(binding, {"order_id": "AF-10421"}, _ctx())


def test_local_server_is_a_process_singleton() -> None:
    assert get_local_server("agrifarm") is get_local_server("agrifarm")


def test_max_weeks_policy() -> None:
    cap = get_capability("farmbox.pause")
    assert cap is not None
    assert check_policy(cap, {"email": "a@b.c", "weeks": 4}, {"max_weeks": 4}).ok
    denied = check_policy(cap, {"email": "a@b.c", "weeks": 6}, {"max_weeks": 4})
    assert not denied.ok and "max_weeks" in denied.violations[0]


async def test_heuristic_compiler_maps_agrifarm_actions() -> None:
    doc = TxtParser().parse(SEED.read_text(encoding="utf-8"))
    result = await HeuristicPlaybookCompiler().compile(
        doc, company_name="AgriFarm", previous_intents=[]
    )
    pb = result.playbook
    assert len(pb.intents) == 5 and len(pb.facts) >= 10
    caps = {st.capability for it in pb.intents for st in it.steps if st.capability}
    assert {
        "order.get", "shipment.track", "delivery_slot.list", "product.availability",
        "farmbox.get", "refund.create", "replacement.create", "discount.issue",
        "delivery.reschedule", "order.cancel", "credit.issue", "farmbox.pause",
        "restock_alert.create",
    } == caps  # fmt: skip
    assert pb.rules.sign_off and pb.rules.prohibitions
