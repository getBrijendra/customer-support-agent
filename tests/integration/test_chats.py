"""Integration cases 8-12 and 18-20: isolation, chat lifecycle, persistence, listing, detail,
multi-intent, reference facts, no-match fallback."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.prompts import PROMPT_VERSION, ROUTER_PROMPT_VERSION
from app.db.models import Chat, Company
from app.search.base import Query
from tests.conftest import AGENT_HEADERS, FailingGuidanceGenerator, ScriptedSelector
from tests.integration.conftest import (
    BOOKSHOP_TEXT,
    VPN_TEXT,
    ClientFactory,
    app_of,
    chat,
    create_company,
    upload,
)


async def _setup_both(client: httpx.AsyncClient) -> tuple[dict[str, Any], dict[str, Any]]:
    await create_company(client, "bookshop", "Bookshop")
    await create_company(client, "vpn-saas", "VPN SaaS")
    b = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
    v = (await upload(client, "vpn-saas", VPN_TEXT)).json()
    assert b["is_active"] and v["is_active"], (b, v)
    return b, v


async def test_tenant_and_version_isolation(client: httpx.AsyncClient) -> None:
    """Case 8: hybrid search never returns rows from another company or version."""
    b, _ = await _setup_both(client)
    resources = app_of(client).state.resources
    emb = resources.embeddings_for("fake-embedding")
    company_ids = {}
    async with resources.sessionmaker() as s:
        for slug in ("bookshop", "vpn-saas"):
            company_ids[slug] = (
                await s.execute(select(Company.id).where(Company.slug == slug))
            ).scalar_one()
    vpn_words = "subscription billing refunds duplicate charges WireGuard device connections"
    q = Query(company_ids["bookshop"], uuid.UUID(b["id"]), emb.embed_query(vpn_words), vpn_words)
    intents = await resources.search.search_intents(q, 50)
    facts = await resources.search.search_facts(q, 50)
    sections = await resources.search.search_sections(q, 50)
    assert {c.key for c in intents} <= {"shipping_delays", "damaged_books", "returns"}
    assert intents and not any("WireGuard" in f.text for f in facts)
    assert not any("subscription" in s.text.lower() for s in sections)
    # wrong company/version combination returns nothing
    q_wrong = Query(company_ids["vpn-saas"], uuid.UUID(b["id"]), q.vector, q.text)
    assert await resources.search.search_intents(q_wrong, 10) == []

    r = await chat(client, "bookshop", "I was charged twice for my VPN subscription")
    assert r.status_code == 201
    detail = (
        await client.get(f"/api/v1/chats/{r.json()['chat_id']}", headers=AGENT_HEADERS)
    ).json()
    keys = {c["key"] for c in detail["routing"]["candidates"]}
    assert keys <= {"shipping_delays", "damaged_books", "returns"}


async def test_chat_happy_path(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 9."""
    b, _ = await _setup_both(client)
    r = await chat(
        client, "bookshop", "My book arrived damaged", agent_notes="order #42", channel="email"
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body) == {
        "chat_id",
        "company",
        "guideline_version",
        "guidance",
        "proposed_actions",
        "latency_ms",
        "created_at",
    }
    assert body["company"] == "bookshop" and body["guideline_version"] == 1
    g = body["guidance"]
    assert set(g) == {
        "detected_intents",
        "recommended_response",
        "requires_escalation",
        "escalation_reason",
        "references",
        "proposed_actions",
        "confidence",
        "missing_information",
    }
    assert [i["intent_key"] for i in g["detected_intents"]] == ["damaged_books"]
    steps = g["detected_intents"][0]["steps"]
    assert [s["performed_by"] for s in steps] == ["ai", "agent", "agent"]  # no bindings yet
    assert "intent:damaged_books#s1" in g["references"]
    assert "made-up:9999" not in g["references"]
    assert r.headers["X-Request-ID"]

    async with db_sessionmaker() as s:
        row = (await s.execute(select(Chat))).scalar_one()
    assert row.status == "success" and row.primary_intent == "damaged_books"
    assert row.customer_message == "My book arrived damaged"
    assert row.agent_notes == "order #42" and row.channel == "email"
    assert (
        row.prompt_version == PROMPT_VERSION and row.router_prompt_version == ROUTER_PROMPT_VERSION
    )
    assert row.llm_model == "fake-llm" and row.router_model == "heuristic-router"
    assert row.latency_ms_total is not None and row.latency_ms_routing is not None
    assert row.latency_ms_llm is not None and row.latency_ms_tools is not None
    assert row.input_tokens == 120 and row.output_tokens == 40
    assert row.request_id == r.headers["X-Request-ID"]
    assert str(row.guideline_version_id) == b["id"]
    assert row.routing and row.routing["entities"] == {"order_id": "42"}


async def test_chat_without_active_guidelines_409(client: httpx.AsyncClient) -> None:
    """Case 10."""
    await create_company(client, "bookshop")
    r = await chat(client, "bookshop")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "NO_ACTIVE_GUIDELINES"
    assert (await chat(client, "unknown-co")).json()["error"]["code"] == "COMPANY_NOT_FOUND"


async def test_llm_failure_502_and_row_saved(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 11."""
    async with client_factory(generator=FailingGuidanceGenerator("upstream 503")) as client:
        await create_company(client, "bookshop")
        await upload(client, "bookshop", BOOKSHOP_TEXT)
        r = await chat(client, "bookshop")
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "LLM_ERROR"
        assert r.json()["error"]["request_id"] == r.headers["X-Request-ID"]

        listing = (
            await client.get("/api/v1/chats", params={"status": "error"}, headers=AGENT_HEADERS)
        ).json()
        assert len(listing["items"]) == 1
        detail = (
            await client.get(
                f"/api/v1/chats/{listing['items'][0]['chat_id']}", headers=AGENT_HEADERS
            )
        ).json()
        assert detail["status"] == "error"
        assert "upstream 503" in detail["error_message"]
        assert detail["guidance"] is None
        assert detail["routing"]["selected"]  # routing succeeded before the LLM failed
        assert detail["latency_ms"]["total"] is not None

    async with db_sessionmaker() as s:
        assert (await s.execute(select(Chat.status))).scalars().all() == ["error"]


async def _insert_chats(
    sm: async_sessionmaker[AsyncSession], slug: str, timestamps: list[datetime], **fields: Any
) -> list[uuid.UUID]:
    async with sm() as s:
        company_id = (await s.execute(select(Company.id).where(Company.slug == slug))).scalar_one()
        ids = []
        for ts in timestamps:
            row = Chat(
                id=uuid.uuid4(),
                company_id=company_id,
                customer_message="x" * 300,
                status="success",
                created_at=ts,
                recommended_response="ok",
                **fields,
            )
            s.add(row)
            ids.append(row.id)
        await s.commit()
    return ids


async def test_list_filters_pagination_and_detail(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 12."""
    await _setup_both(client)
    same_ts = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
    tied = await _insert_chats(db_sessionmaker, "bookshop", [same_ts] * 5, primary_intent="returns")
    await _insert_chats(db_sessionmaker, "bookshop", [datetime(2026, 11, 1, 9, tzinfo=UTC)])
    await _insert_chats(db_sessionmaker, "vpn-saas", [same_ts, datetime(2026, 11, 7, tzinfo=UTC)])

    r = await client.get(
        "/api/v1/chats", params={"company": "vpn-saas", "limit": 100}, headers=AGENT_HEADERS
    )
    assert {i["company"] for i in r.json()["items"]} == {"vpn-saas"} and len(r.json()["items"]) == 2
    r = await client.get("/api/v1/chats", params={"intent": "returns"}, headers=AGENT_HEADERS)
    assert len(r.json()["items"]) == 5

    # date range with tz offsets: 13:00+05:30 == 07:30Z ; 07:00-06:00 == 13:00Z (2026-11-05)
    params = {
        "company": "bookshop",
        "from": datetime(
            2026, 11, 5, 13, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
        ).isoformat(),
        "to": datetime(2026, 11, 5, 7, 0, tzinfo=timezone(timedelta(hours=-6))).isoformat(),
        "limit": 2,
    }
    seen: list[str] = []
    pages = 0
    cursor = None
    while True:
        q = dict(params, **({"cursor": cursor} if cursor else {}))
        r = await client.get("/api/v1/chats", params=q, headers=AGENT_HEADERS)
        assert r.status_code == 200, r.text
        page = r.json()
        pages += 1
        seen += [i["chat_id"] for i in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert pages == 3
    assert len(seen) == len(set(seen)) == 5
    assert set(seen) == {str(i) for i in tied}

    item = (await client.get("/api/v1/chats", params={"limit": 1}, headers=AGENT_HEADERS)).json()[
        "items"
    ][0]
    assert set(item) == {
        "chat_id",
        "company",
        "created_at",
        "status",
        "primary_intent",
        "customer_message",
        "recommended_response",
        "guideline_version",
    }
    assert len(item["customer_message"]) <= 200

    r = await client.get("/api/v1/chats", params={"cursor": "garbage"}, headers=AGENT_HEADERS)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"

    real = (await chat(client, "bookshop", "refund for a damaged book please")).json()
    d = (await client.get(f"/api/v1/chats/{real['chat_id']}", headers=AGENT_HEADERS)).json()
    assert d["guidance"]["detected_intents"][0]["intent_key"] == "damaged_books"
    assert d["routing"]["candidates"][0]["key"] and "score" in d["routing"]["candidates"][0]
    assert d["references"]["used"] and d["references"]["dropped"] == ["made-up:9999"]
    assert d["prompt_version"] == PROMPT_VERSION and d["llm_model"] == "fake-llm"
    assert d["tokens"]["input"] == 120 and d["tokens"]["output"] == 40
    assert d["guideline_version"] == 1 and d["tool_calls"] == [] and d["proposed_actions"] == []

    r = await client.get(f"/api/v1/chats/{uuid.uuid4()}", headers=AGENT_HEADERS)
    assert r.status_code == 404


async def test_multi_intent_chat(client_factory: ClientFactory) -> None:
    """Case 18: a message with two requests gets a plan for each."""
    selector = ScriptedSelector(
        lambda m, c: ["billing_and_subscription_issues", "connection_failures"]
    )
    async with client_factory(selector=selector) as client:
        await create_company(client, "vpn-saas")
        await upload(client, "vpn-saas", VPN_TEXT)
        r = await chat(
            client, "vpn-saas", "I was charged twice this month and the VPN won't connect"
        )
        assert r.status_code == 201, r.text
        intents = r.json()["guidance"]["detected_intents"]
        assert [i["intent_key"] for i in intents] == [
            "billing_and_subscription_issues",
            "connection_failures",
        ]
        assert all(i["steps"] for i in intents)
        detail = (
            await client.get(f"/api/v1/chats/{r.json()['chat_id']}", headers=AGENT_HEADERS)
        ).json()
        assert len(detail["routing"]["selected"]) == 2
        assert detail["primary_intent"] == "billing_and_subscription_issues"


async def test_reference_fact_cited(client: httpx.AsyncClient) -> None:
    """Case 19."""
    await create_company(client, "vpn-saas")
    await upload(client, "vpn-saas", VPN_TEXT)
    r = await chat(client, "vpn-saas", "Which protocols do you support? WireGuard?")
    g = r.json()["guidance"]
    facts = [x for x in g["references"] if x.startswith("fact:")]
    assert facts
    resources = app_of(client).state.resources
    ctx = resources.generator.calls[-1]["context"]
    assert ctx.facts[0].ref == facts[0] and "WireGuard" in ctx.facts[0].text


async def test_no_match_uses_section_fallback(client: httpx.AsyncClient) -> None:
    """Case 20."""
    await create_company(client, "bookshop")
    await upload(client, "bookshop", BOOKSHOP_TEXT)
    r = await chat(client, "bookshop", "Can you recommend a good pizza place?")
    assert r.status_code == 201, r.text
    g = r.json()["guidance"]
    assert g["detected_intents"] == [] and g["confidence"] == "low" and g["requires_escalation"]
    assert any(x.startswith("section:") for x in g["references"])
    detail = (
        await client.get(f"/api/v1/chats/{r.json()['chat_id']}", headers=AGENT_HEADERS)
    ).json()
    assert detail["routing"]["no_match"] is True and detail["primary_intent"] is None
