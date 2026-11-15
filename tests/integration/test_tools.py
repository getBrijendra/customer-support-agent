"""Integration cases 23-30: capability bindings (mock / MCP / read-only SQL), prefetch, manual
fallback, proposals + approval exactly once, policy denial, tenant isolation, timeouts."""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import httpx
import pytest
from mcp.server.mcpserver import MCPServer
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.capabilities.adapters.mcp import McpAdapter
from app.db.models import CapabilityBinding, ToolCall
from tests.conftest import (
    ADMIN_HEADERS,
    AGENT_HEADERS,
    FakeGuidanceGenerator,
    ScriptedSelector,
)
from tests.integration.conftest import (
    BOOKSHOP_TEXT,
    VPN_TEXT,
    ClientFactory,
    _plain_dsn,
    app_of,
    bind_live,
    chat,
    create_company,
    upload,
)

SHIPPING = ScriptedSelector(lambda m, c: ["shipping_delays"])
DAMAGED = ScriptedSelector(lambda m, c: ["damaged_books"])
BINDINGS = "/api/v1/companies/bookshop/capability-bindings"


def _step(body: dict[str, Any], ref: str) -> dict[str, Any]:
    steps = [s for i in body["guidance"]["detected_intents"] for s in i["steps"]]
    return next(s for s in steps if s["step_ref"] == ref)


async def _tool_calls(sm: async_sessionmaker[AsyncSession]) -> list[ToolCall]:
    async with sm() as s:
        return list((await s.execute(select(ToolCall).order_by(ToolCall.created_at))).scalars())


async def _bookshop(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    r = await upload(client, "bookshop", BOOKSHOP_TEXT)
    assert r.json()["is_active"], r.text


async def test_bound_read_is_verified_then_manual_when_unbound(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Cases 23 + 24, plus capability coverage."""
    async with client_factory(selector=SHIPPING) as client:
        await _bookshop(client)
        cov = (
            await client.get(
                "/api/v1/companies/bookshop/guidelines/1/capability-coverage",
                headers=ADMIN_HEADERS,
            )
        ).json()
        assert cov["live"] == 0 and cov["manual"] == cov["referenced"] > 0

        await bind_live(
            client,
            "bookshop",
            "shipment.track",
            {"store": "bookshop_orders"},
            sample={"order_id": "1001"},
        )
        cov = (
            await client.get(
                "/api/v1/companies/bookshop/guidelines/1/capability-coverage",
                headers=ADMIN_HEADERS,
            )
        ).json()
        row = next(c for c in cov["capabilities"] if c["capability"] == "shipment.track")
        assert row["handled_by"] == "ai" and row["steps"] == ["shipping_delays#s1"]

        r = await chat(client, "bookshop", "My order #1001 is late, where is it?")
        assert r.status_code == 201, r.text
        step = _step(r.json(), "shipping_delays#s1")
        assert step["performed_by"] == "ai_verified" and step["tool_call_id"]
        calls = await _tool_calls(db_sessionmaker)
        assert [(c.trigger, c.status) for c in calls] == [("test", "ok"), ("prefetch", "ok")]
        assert str(calls[1].id) == step["tool_call_id"]
        assert calls[1].result["eta"] == "2026-11-15"
        detail = (
            await client.get(f"/api/v1/chats/{r.json()['chat_id']}", headers=AGENT_HEADERS)
        ).json()
        assert detail["tool_calls"][0]["capability"] == "shipment.track"
        gen = app_of(client).state.resources.generator
        assert gen.calls[-1]["context"].tool_results[0].result["carrier"] == "DHL"

        # Case 24: remove the binding -> the same chat degrades to a manual step.
        assert (
            await client.delete(f"{BINDINGS}/shipment.track", headers=ADMIN_HEADERS)
        ).status_code == 204
        r = await chat(client, "bookshop", "My order #1001 is late, where is it?")
        assert _step(r.json(), "shipping_delays#s1")["performed_by"] == "agent"
        assert len(await _tool_calls(db_sessionmaker)) == 2


async def test_binding_lifecycle_rules(client: httpx.AsyncClient) -> None:
    await _bookshop(client)
    r = await client.put(
        f"{BINDINGS}/shipment.track",
        json={"adapter_type": "mock", "config": {"store": "bookshop_orders"}},
        headers=ADMIN_HEADERS,
    )
    assert r.json()["status"] == "draft"
    r = await client.post(f"{BINDINGS}/shipment.track/go-live", headers=ADMIN_HEADERS)
    assert r.status_code == 409 and r.json()["error"]["code"] == "BINDING_NOT_TESTED"
    r = await client.post(
        f"{BINDINGS}/shipment.track/test",
        json={"sample_args": {"order_id": "nope"}},
        headers=ADMIN_HEADERS,
    )
    assert r.json()["result"]["ok"] is False and r.json()["binding"]["status"] == "draft"
    for bad in (
        {"adapter_type": "mock", "config": {"store": "nope"}},
        {"adapter_type": "http", "config": {"url": "ftp://x"}},
        {"adapter_type": "mock", "config": {"store": "bookshop_orders"}, "policy": {"max_pct": 5}},
    ):
        r = await client.put(f"{BINDINGS}/refund.create", json=bad, headers=ADMIN_HEADERS)
        assert r.status_code == 422, bad
    r = await client.put(
        f"{BINDINGS}/unknown.thing",
        json={"adapter_type": "mock", "config": {}},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 422
    caps = (await client.get("/api/v1/capabilities", headers=ADMIN_HEADERS)).json()
    assert {"order.get", "refund.create", "subscription.get"} <= {c["name"] for c in caps}


async def test_write_proposal_approved_exactly_once(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 25."""
    gen = FakeGuidanceGenerator(propose=[("refund.create", {"amount": 20, "order_id": "1001"})])
    async with client_factory(selector=DAMAGED, generator=gen) as client:
        await _bookshop(client)
        await bind_live(
            client,
            "bookshop",
            "refund.create",
            {"store": "bookshop_orders"},
            policy={"max_amount": 50},
            sample={"amount": 10},
        )
        r = await chat(client, "bookshop", "My book arrived damaged, order #1001")
        assert r.status_code == 201, r.text
        body = r.json()
        [action] = body["proposed_actions"]
        assert action["status"] == "proposed" and action["args"] == {
            "amount": 20.0,
            "order_id": "1001",
        }
        step = _step(body, "damaged_books#s3")
        assert (
            step["performed_by"] == "needs_approval" and step["proposed_action_id"] == action["id"]
        )
        assert await _tool_calls(db_sessionmaker) == []  # proposing never executes

        url = f"/api/v1/chats/{body['chat_id']}/actions/{action['id']}"
        first, second = await asyncio.gather(
            client.post(f"{url}/approve", headers=AGENT_HEADERS),
            client.post(f"{url}/approve", headers=AGENT_HEADERS),
        )
        statuses = sorted([first.status_code, second.status_code])
        assert statuses in ([200, 200], [200, 409]), (first.text, second.text)
        done = (await client.post(f"{url}/approve", headers=AGENT_HEADERS)).json()
        assert done["status"] == "executed" and done["result"]["status"] == "done"
        assert done["decided_by"] and done["executed_at"]
        executions = [c for c in await _tool_calls(db_sessionmaker) if c.trigger == "execution"]
        assert len(executions) == 1 and str(executions[0].proposed_action_id) == action["id"]
        mock = app_of(client).state.resources.gateway.adapters["mock"]
        assert len(mock.executed) == 1

        listed = (
            await client.get(f"/api/v1/chats/{body['chat_id']}/actions", headers=AGENT_HEADERS)
        ).json()
        assert [a["status"] for a in listed] == ["executed"]
        r = await client.post(f"{url}/reject", headers=AGENT_HEADERS)
        assert r.status_code == 409 and r.json()["error"]["code"] == "ACTION_NOT_PENDING"


async def test_policy_violation_denied_and_reject_flow(client_factory: ClientFactory) -> None:
    """Case 26."""
    gen = FakeGuidanceGenerator(
        propose=[("refund.create", {"amount": 80}), ("return_label.create", {"order_id": "1"})]
    )
    async with client_factory(selector=DAMAGED, generator=gen) as client:
        await _bookshop(client)
        await bind_live(
            client,
            "bookshop",
            "refund.create",
            {"store": "bookshop_orders"},
            policy={"max_amount": 50},
            sample={"amount": 10},
        )
        body = (await chat(client, "bookshop", "Damaged book, I want 80 back")).json()
        [action] = body["proposed_actions"]  # return_label isn't referenced/bound -> dropped
        assert action["status"] == "denied"
        assert "exceeds max_amount" in action["policy_check"]["violations"][0]
        assert _step(body, "damaged_books#s3")["performed_by"] == "agent"
        url = f"/api/v1/chats/{body['chat_id']}/actions/{action['id']}"
        r = await client.post(f"{url}/approve", headers=AGENT_HEADERS)
        assert r.status_code == 409 and r.json()["error"]["code"] == "ACTION_NOT_PENDING"

        gen.propose = [("refund.create", {"amount": 5})]
        body = (await chat(client, "bookshop", "Damaged book")).json()
        url = f"/api/v1/chats/{body['chat_id']}/actions/{body['proposed_actions'][0]['id']}"
        r = await client.post(
            f"{url}/reject", json={"reason": "store credit"}, headers=AGENT_HEADERS
        )
        assert r.status_code == 200 and r.json()["status"] == "rejected"
        assert (await client.post(f"{url}/approve", headers=AGENT_HEADERS)).status_code == 409


async def test_mcp_binding_end_to_end(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 27: discover -> bind -> test -> go-live -> chat, against an in-process MCP server."""
    server = MCPServer("orders")
    seen: list[str] = []

    @server.tool()
    def lookup_order(id: str) -> dict[str, Any]:
        """Look up an order by id."""
        seen.append(id)
        return {"id": id, "status": "in_transit", "eta": "2026-11-20"}

    @server.tool()
    def delete_everything() -> str:
        """Dangerous tool that must never be exposed."""
        raise AssertionError("must not be called")

    adapter = McpAdapter(connector=lambda url, headers: server)
    async with client_factory(selector=SHIPPING, adapters={"mcp": adapter}) as client:
        await _bookshop(client)
        await client.post(
            "/api/v1/companies/bookshop/secrets",
            json={"name": "mcp_token", "value": "tok-123"},
            headers=ADMIN_HEADERS,
        )
        r = await client.post(
            "/api/v1/companies/bookshop/mcp-servers",
            json={
                "name": "orders",
                "url": "https://mcp.bookshop.test/mcp",
                "auth_secret_name": "mcp_token",
            },
            headers=ADMIN_HEADERS,
        )
        assert r.status_code == 201, r.text
        server_id = r.json()["id"]
        r = await client.post(
            f"/api/v1/companies/bookshop/mcp-servers/{server_id}/discover", headers=ADMIN_HEADERS
        )
        assert {t["name"] for t in r.json()["discovered_tools"]} == {
            "lookup_order",
            "delete_everything",
        }
        await bind_live(
            client,
            "bookshop",
            "shipment.track",
            {"server_id": server_id, "tool_name": "lookup_order", "arg_map": {"order_id": "id"}},
            adapter_type="mcp",
            sample={"order_id": "1001"},
        )
        r = await chat(client, "bookshop", "Where is order #1001?")
        step = _step(r.json(), "shipping_delays#s1")
        assert step["performed_by"] == "ai_verified"
        assert seen == ["1001", "1001"]  # test call + chat prefetch
        calls = await _tool_calls(db_sessionmaker)
        assert calls[-1].adapter_type == "mcp" and calls[-1].result["eta"] == "2026-11-20"


async def test_sql_readonly_binding(
    client_factory: ClientFactory,
    db_sessionmaker: async_sessionmaker[AsyncSession],
    pg_url: str,
) -> None:
    """Case 28: an admin-defined SELECT against a company database; writes are rejected."""
    dsn = _plain_dsn(pg_url)
    conn = await asyncpg.connect(dsn)
    await conn.execute(
        "DROP TABLE IF EXISTS ext_orders; CREATE TABLE ext_orders (id text, status text, eta text);"
        "INSERT INTO ext_orders VALUES ('1001', 'shipped', '2026-11-18')"
    )
    try:
        async with client_factory(selector=SHIPPING) as client:
            await _bookshop(client)
            await client.post(
                "/api/v1/companies/bookshop/secrets",
                json={"name": "orders_db", "value": dsn},
                headers=ADMIN_HEADERS,
            )
            secrets = (
                await client.get("/api/v1/companies/bookshop/secrets", headers=ADMIN_HEADERS)
            ).json()
            assert secrets[0]["name"] == "orders_db" and "value" not in secrets[0]
            for bad in (
                "DELETE FROM ext_orders",
                "SELECT 1; DROP TABLE ext_orders",
                "UPDATE ext_orders SET status = 'x'",
            ):
                r = await client.put(
                    f"{BINDINGS}/shipment.track",
                    json={
                        "adapter_type": "sql_readonly",
                        "config": {"dsn_secret": "orders_db", "statement": bad},
                    },
                    headers=ADMIN_HEADERS,
                )
                assert r.status_code == 422, bad
            await bind_live(
                client,
                "bookshop",
                "shipment.track",
                {
                    "dsn_secret": "orders_db",
                    "statement": "SELECT id, status, eta FROM ext_orders WHERE id = :order_id",
                },
                adapter_type="sql_readonly",
                sample={"order_id": "1001"},
            )
            r = await chat(client, "bookshop", "Where is order #1001?")
            assert _step(r.json(), "shipping_delays#s1")["performed_by"] == "ai_verified"
            call = (await _tool_calls(db_sessionmaker))[-1]
            assert call.result == {
                "rows": [{"id": "1001", "status": "shipped", "eta": "2026-11-18"}],
                "row_count": 1,
            }
            assert "postgres:postgres" not in str(call.args) + str(call.result)

        # Defence in depth: even a config that slipped past validation runs READ ONLY.
        from app.capabilities.adapters.base import BindingSnapshot, CallContext
        from app.capabilities.adapters.sql_readonly import SqlReadOnlyAdapter

        async def secret(_: str) -> str:
            return dsn

        b = BindingSnapshot(
            None,
            None,
            "x",
            "sql_readonly",  # type: ignore[arg-type]
            {"dsn_secret": "s", "statement": "SELECT 1"},
            {},
            "live",
        )
        adapter = SqlReadOnlyAdapter()
        b.config["statement"] = "WITH d AS (DELETE FROM ext_orders RETURNING *) SELECT * FROM d"
        with pytest.raises(Exception, match="not allowed"):
            await adapter.invoke(b, {}, CallContext(None, secret, 5, True))  # type: ignore[arg-type]
        assert await conn.fetchval("SELECT count(*) FROM ext_orders") == 1
    finally:
        await conn.execute("DROP TABLE IF EXISTS ext_orders")
        await conn.close()


async def test_integrations_are_tenant_isolated(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 29: company A never uses company B's binding, server or secret."""
    selector = ScriptedSelector(
        lambda m, c: (
            ["shipping_delays"]
            if any(x.key == "shipping_delays" for x in c)
            else ["billing_and_subscription_issues"]
        )
    )
    async with client_factory(selector=selector) as client:
        await _bookshop(client)
        await create_company(client, "vpn-saas")
        await upload(client, "vpn-saas", VPN_TEXT)
        await bind_live(
            client,
            "vpn-saas",
            "shipment.track",  # deliberately the capability bookshop's playbook uses
            {"store": "bookshop_orders"},
            sample={"order_id": "1001"},
        )
        await client.post(
            "/api/v1/companies/vpn-saas/secrets",
            json={"name": "vpn_secret", "value": "x"},
            headers=ADMIN_HEADERS,
        )
        r = await client.post(
            "/api/v1/companies/vpn-saas/mcp-servers",
            json={"name": "vpn", "url": "https://mcp.vpn.test/mcp"},
            headers=ADMIN_HEADERS,
        )
        vpn_server = r.json()["id"]

        r = await chat(client, "bookshop", "Where is order #1001?")
        assert _step(r.json(), "shipping_delays#s1")["performed_by"] == "agent"
        assert [c.trigger for c in await _tool_calls(db_sessionmaker)] == ["test"]

        assert (
            await client.get("/api/v1/companies/bookshop/secrets", headers=ADMIN_HEADERS)
        ).json() == []
        r = await client.put(
            f"{BINDINGS}/shipment.track",
            json={"adapter_type": "mcp", "config": {"server_id": vpn_server, "tool_name": "x"}},
            headers=ADMIN_HEADERS,
        )
        assert r.status_code == 404
        r = await client.post(
            f"/api/v1/companies/bookshop/mcp-servers/{vpn_server}/discover", headers=ADMIN_HEADERS
        )
        assert r.status_code == 404


async def test_tool_timeout_degrades_without_failing(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 30."""
    async with client_factory(
        selector=SHIPPING, TOOL_TIMEOUT_SECONDS=0.3, COMPANY_CACHE_TTL_SECONDS=0
    ) as client:
        await _bookshop(client)
        await bind_live(
            client,
            "bookshop",
            "shipment.track",
            {"store": "bookshop_orders"},
            sample={"order_id": "1001"},
        )
        async with db_sessionmaker() as s:
            await s.execute(
                text(
                    "UPDATE capability_bindings SET config = "
                    "config || '{\"delay_seconds\": 2}'::jsonb"
                )
            )
            await s.commit()
            assert (await s.execute(select(CapabilityBinding.status))).scalar_one() == "live"
        r = await chat(client, "bookshop", "Where is order #1001?")
        assert r.status_code == 201
        assert _step(r.json(), "shipping_delays#s1")["performed_by"] == "agent"
        assert (await _tool_calls(db_sessionmaker))[-1].status == "timeout"
