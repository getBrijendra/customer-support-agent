"""AgriFarm end to end against the in-process MCP server (`inproc://agrifarm`): register ->
discover -> bind -> chat with a verified lookup -> approve a write that executes exactly once."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ToolCall
from tests.conftest import ADMIN_HEADERS, AGENT_HEADERS, FakeGuidanceGenerator, ScriptedSelector
from tests.integration.conftest import ClientFactory, bind_live, chat, create_company, upload

AGRIFARM_TEXT = (
    Path(__file__).resolve().parents[2] / "data" / "seed_guidelines" / "agrifarm.txt"
).read_text(encoding="utf-8")
FARM_BOX = ScriptedSelector(lambda m, c: ["farm_box_subscription_changes"])
MARCUS = "marcus.bell@example.com"


def _step(body: dict[str, Any], ref: str) -> dict[str, Any]:
    steps = [s for i in body["guidance"]["detected_intents"] for s in i["steps"]]
    return next(s for s in steps if s["step_ref"] == ref)


async def test_agrifarm_inproc_mcp_end_to_end(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    gen = FakeGuidanceGenerator(propose=[("refund.create", {"email": MARCUS, "amount": 54})])
    async with client_factory(selector=FARM_BOX, generator=gen) as client:
        await create_company(client, "agrifarm", "AgriFarm")
        r = await upload(client, "agrifarm", AGRIFARM_TEXT, filename="agrifarm.txt")
        assert r.json()["is_active"], r.text
        playbook = (
            await client.get(
                "/api/v1/companies/agrifarm/guidelines/1/playbook", headers=AGENT_HEADERS
            )
        ).json()
        assert len(playbook["intents"]) == 5

        servers = "/api/v1/companies/agrifarm/mcp-servers"
        r = await client.post(
            servers, json={"name": "bad", "url": "inproc://nope"}, headers=ADMIN_HEADERS
        )
        assert r.status_code == 422, r.text
        r = await client.post(
            servers,
            json={"name": "agrifarm-systems", "url": "inproc://agrifarm"},
            headers=ADMIN_HEADERS,
        )
        assert r.status_code == 201, r.text
        assert r.json()["transport"] == "in_process"
        server_id = r.json()["id"]
        r = await client.post(f"{servers}/{server_id}/discover", headers=ADMIN_HEADERS)
        assert {"get_farm_box", "issue_refund", "export_customer_list"} <= {
            t["name"] for t in r.json()["discovered_tools"]
        }

        await bind_live(
            client,
            "agrifarm",
            "farmbox.get",
            {"server_id": server_id, "tool_name": "get_farm_box"},
            adapter_type="mcp",
            sample={"email": "priya.sharma@example.com"},
        )
        await bind_live(
            client,
            "agrifarm",
            "refund.create",
            {
                "server_id": server_id,
                "tool_name": "issue_refund",
                "idempotency_arg": "idempotency_key",
            },
            adapter_type="mcp",
            policy={"max_amount": 60},
            sample={"email": MARCUS, "amount": 10},
        )

        r = await chat(
            client,
            "agrifarm",
            "I paused my Farm Box on Wednesday but you still charged me $54 on Thursday.",
            agent_notes=f"Customer email: {MARCUS}",
        )
        assert r.status_code == 201, r.text
        body = r.json()
        lookup = _step(body, "farm_box_subscription_changes#s1")
        assert lookup["performed_by"] == "ai_verified" and lookup["tool_call_id"]
        ctx = gen.calls[-1]["context"]
        assert ctx.tool_results[0].result["status"] == "paused"

        [action] = body["proposed_actions"]
        assert action["status"] == "proposed"
        url = f"/api/v1/chats/{body['chat_id']}/actions/{action['id']}"
        done = (await client.post(f"{url}/approve", headers=AGENT_HEADERS)).json()
        assert done["status"] == "executed", done
        assert done["result"]["refund_id"].startswith("RF-")
        assert done["result"]["charge_id"] == "ch_fb_7802" and done["result"]["amount"] == 54.0
        again = (await client.post(f"{url}/approve", headers=AGENT_HEADERS)).json()
        assert again["result"]["refund_id"] == done["result"]["refund_id"]

    async with db_sessionmaker() as s:
        calls = list((await s.execute(select(ToolCall))).scalars())
    executions = [c for c in calls if c.trigger == "execution"]
    assert len(executions) == 1 and executions[0].adapter_type == "mcp"
