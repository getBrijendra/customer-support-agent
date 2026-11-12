"""Integration cases 8-12: tenant isolation, chat lifecycle, persistence, listing, detail."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.prompts import PROMPT_VERSION
from app.db.models import Chat, Company
from tests.conftest import AGENT_HEADERS, FailingGuidanceGenerator
from tests.integration.conftest import (
    BOOKSHOP_TEXT,
    VPN_TEXT,
    ClientFactory,
    create_company,
    upload,
)


async def _chat(
    client: httpx.AsyncClient, company: str, msg: str = "I want a refund", **extra: Any
) -> httpx.Response:
    return await client.post(
        "/api/v1/chats",
        json={"company": company, "customer_message": msg, **extra},
        headers=AGENT_HEADERS,
    )


async def _setup_both(client: httpx.AsyncClient) -> tuple[dict[str, Any], dict[str, Any]]:
    await create_company(client, "bookshop", "Bookshop")
    await create_company(client, "vpn-saas", "VPN SaaS")
    b = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
    v = (await upload(client, "vpn-saas", VPN_TEXT)).json()
    return b, v


async def test_tenant_isolation(client: httpx.AsyncClient) -> None:
    """Case 8."""
    b, v = await _setup_both(client)
    r = await _chat(client, "bookshop", "I was charged twice for my VPN subscription")
    assert r.status_code == 201
    detail = (
        await client.get(f"/api/v1/chats/{r.json()['chat_id']}", headers=AGENT_HEADERS)
    ).json()
    chunks = detail["retrieved_chunks"]
    assert chunks
    assert {c["company_slug"] for c in chunks} == {"bookshop"}
    assert all(c["chunk_id"].startswith(b["id"]) for c in chunks)
    assert not any(c["chunk_id"].startswith(v["id"]) for c in chunks)
    assert not any("VPN profile" in c["text"] for c in chunks)


async def test_chat_happy_path(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 9."""
    b, _ = await _setup_both(client)
    r = await _chat(
        client, "bookshop", "My book arrived damaged", agent_notes="order #42", channel="email"
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body) == {
        "chat_id",
        "company",
        "guideline_version",
        "guidance",
        "latency_ms",
        "created_at",
    }
    assert body["company"] == "bookshop" and body["guideline_version"] == 1
    g = body["guidance"]
    assert set(g) == {
        "recommended_response",
        "handling_instructions",
        "requires_escalation",
        "escalation_reason",
        "detected_intent",
        "guideline_references",
        "confidence",
        "missing_information",
    }
    assert g["guideline_references"] and all(
        ref.startswith(b["id"]) for ref in g["guideline_references"]
    )
    assert "made-up:9999" not in g["guideline_references"]
    assert isinstance(body["latency_ms"], int)
    assert r.headers["X-Request-ID"]

    async with db_sessionmaker() as s:
        chat = (await s.execute(select(Chat))).scalar_one()
    assert chat.status == "success"
    assert chat.customer_message == "My book arrived damaged"
    assert chat.agent_notes == "order #42" and chat.channel == "email"
    assert chat.prompt_version == PROMPT_VERSION
    assert chat.llm_model == "fake-llm"
    assert chat.latency_ms_total is not None and chat.latency_ms_retrieval is not None
    assert chat.latency_ms_llm is not None
    assert chat.input_tokens == 120 and chat.output_tokens == 40
    assert chat.request_id == r.headers["X-Request-ID"]
    assert str(chat.guideline_version_id) == b["id"]


async def test_chat_without_active_guidelines_409(client: httpx.AsyncClient) -> None:
    """Case 10."""
    await create_company(client, "bookshop")
    r = await _chat(client, "bookshop")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "NO_ACTIVE_GUIDELINES"
    assert (await _chat(client, "unknown-co")).json()["error"]["code"] == "COMPANY_NOT_FOUND"


async def test_llm_failure_502_and_row_saved(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 11."""
    async with client_factory(generator=FailingGuidanceGenerator("upstream 503")) as client:
        await create_company(client, "bookshop")
        await upload(client, "bookshop", BOOKSHOP_TEXT)
        r = await _chat(client, "bookshop")
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
        assert detail["retrieved_chunks"]  # retrieval succeeded before the LLM failed
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
            chat = Chat(
                id=uuid.uuid4(),
                company_id=company_id,
                customer_message="x" * 300,
                status="success",
                created_at=ts,
                recommended_response="ok",
                **fields,
            )
            s.add(chat)
            ids.append(chat.id)
        await s.commit()
    return ids


async def test_list_filters_pagination_and_detail(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 12."""
    await _setup_both(client)
    same_ts = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
    tied = await _insert_chats(db_sessionmaker, "bookshop", [same_ts] * 5)
    await _insert_chats(db_sessionmaker, "bookshop", [datetime(2026, 11, 1, 9, tzinfo=UTC)])
    await _insert_chats(db_sessionmaker, "vpn-saas", [same_ts, datetime(2026, 11, 7, tzinfo=UTC)])

    # company filter
    r = await client.get(
        "/api/v1/chats", params={"company": "vpn-saas", "limit": 100}, headers=AGENT_HEADERS
    )
    assert {i["company"] for i in r.json()["items"]} == {"vpn-saas"} and len(r.json()["items"]) == 2

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

    # list items are compact and truncated
    item = (await client.get("/api/v1/chats", params={"limit": 1}, headers=AGENT_HEADERS)).json()[
        "items"
    ][0]
    assert set(item) == {
        "chat_id",
        "company",
        "created_at",
        "status",
        "customer_message",
        "recommended_response",
        "guideline_version",
    }
    assert len(item["customer_message"]) <= 200

    # bad cursor
    r = await client.get("/api/v1/chats", params={"cursor": "garbage"}, headers=AGENT_HEADERS)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"

    # detail incl. JSONB fields from a real chat
    real = (await _chat(client, "bookshop", "refund please")).json()
    d = (await client.get(f"/api/v1/chats/{real['chat_id']}", headers=AGENT_HEADERS)).json()
    assert d["guidance"]["detected_intent"] == "refund_request"
    assert d["retrieved_chunks"][0]["chunk_id"] and "score" in d["retrieved_chunks"][0]
    assert d["prompt_version"] == PROMPT_VERSION and d["llm_model"] == "fake-llm"
    assert d["tokens"] == {"input": 120, "output": 40}
    assert d["guideline_version"] == 1

    r = await client.get(f"/api/v1/chats/{uuid.uuid4()}", headers=AGENT_HEADERS)
    assert r.status_code == 404
