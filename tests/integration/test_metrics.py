"""Integration case 14: metrics summary computed in Postgres matches seeded rows."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chat, Company, GuidelineVersion
from tests.conftest import ADMIN_HEADERS, ADMIN_KEY
from tests.integration.conftest import create_company

DAY1 = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
DAY2 = datetime(2026, 10, 2, 23, 30, tzinfo=UTC)
WINDOW = {"from": "2026-10-01T00:00:00Z", "to": "2026-10-03T00:00:00Z"}


def _guidance(escalate: bool, confidence: str) -> dict[str, object]:
    return {
        "recommended_response": "ok",
        "handling_instructions": [],
        "requires_escalation": escalate,
        "escalation_reason": "x" if escalate else None,
        "detected_intent": "x",
        "guideline_references": [],
        "confidence": confidence,
        "missing_information": [],
    }


async def _seed(sm: async_sessionmaker[AsyncSession]) -> None:
    async with sm() as s:
        ids = dict((await s.execute(select(Company.slug, Company.id))).all())
        book, vpn = ids["bookshop"], ids["vpn-saas"]
        # bookshop: 10 successes on DAY1, latencies 100..1000; 2 escalated; confidence mix
        for i in range(10):
            s.add(
                Chat(
                    id=uuid.uuid4(),
                    company_id=book,
                    customer_message="m",
                    status="success",
                    latency_ms_total=(i + 1) * 100,
                    input_tokens=100,
                    output_tokens=10,
                    guidance=_guidance(escalate=i < 2, confidence="high" if i < 7 else "low"),
                    created_at=DAY1 + timedelta(minutes=i),
                )
            )
        # vpn-saas: 3 successes + 2 errors on DAY2 (all latency 2000)
        for i in range(5):
            ok = i < 3
            s.add(
                Chat(
                    id=uuid.uuid4(),
                    company_id=vpn,
                    customer_message="m",
                    status="success" if ok else "error",
                    latency_ms_total=2000,
                    input_tokens=50 if ok else None,
                    output_tokens=5 if ok else None,
                    guidance=_guidance(False, "medium") if ok else None,
                    error_message=None if ok else "boom",
                    created_at=DAY2,
                )
            )
        # outside the window: must be ignored
        s.add(
            Chat(
                id=uuid.uuid4(),
                company_id=book,
                customer_message="m",
                status="error",
                latency_ms_total=99999,
                created_at=DAY1 - timedelta(days=3),
            )
        )
        # ingestion: 2 ready (1s and 3s), 1 failed, in window
        for n, (status, secs) in enumerate([("ready", 1), ("ready", 3), ("failed", None)], start=1):
            s.add(
                GuidelineVersion(
                    id=uuid.uuid4(),
                    company_id=book,
                    version_number=n,
                    status=status,
                    file_name="g.txt",
                    raw_text="t",
                    content_sha256=str(n),
                    byte_size=1,
                    embedding_model="m",
                    chunk_size=1,
                    chunk_overlap=0,
                    created_at=DAY1,
                    ready_at=DAY1 + timedelta(seconds=secs) if secs else None,
                )
            )
        await s.commit()


async def test_metrics_summary_matches_seed(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    await create_company(client, "bookshop")
    await create_company(client, "vpn-saas")
    await _seed(db_sessionmaker)

    r = await client.get("/api/v1/internal/metrics/summary", params=WINDOW, headers=ADMIN_HEADERS)
    assert r.status_code == 200, r.text
    m = r.json()
    c = m["chats"]
    assert c["total"] == 15 and c["success"] == 13 and c["error"] == 2
    assert c["error_rate"] == round(2 / 15, 4)
    # latencies: 100..1000 (10 rows) + 2000 x5 -> sorted list of 15
    lat = sorted([(i + 1) * 100 for i in range(10)] + [2000] * 5)
    assert c["latency_ms"]["avg"] == round(sum(lat) / 15, 1)
    assert c["latency_ms"]["p50"] == 800.0  # percentile_cont(0.5) of 15 values = 8th value
    # p95: position 0.95 * 14 = 13.3 -> between 2000 and 2000
    assert c["latency_ms"]["p95"] == 2000.0
    assert c["tokens"] == {"input": 10 * 100 + 3 * 50, "output": 10 * 10 + 3 * 5}
    assert c["escalation_rate"] == round(2 / 13, 4)
    assert c["confidence_breakdown"] == {"high": 7, "medium": 3, "low": 3}

    by = {row["company"]: row for row in m["by_company"]}
    assert by["bookshop"]["chats"] == 10 and by["bookshop"]["avg_latency_ms"] == 550.0
    assert by["bookshop"]["error_rate"] == 0.0
    assert by["vpn-saas"]["chats"] == 5 and by["vpn-saas"]["error_rate"] == 0.4

    assert m["daily_volume"] == [
        {"date": "2026-10-01", "chats": 10},
        {"date": "2026-10-02", "chats": 5},
    ]
    assert m["ingestion"] == {"uploads": 3, "ready": 2, "failed": 1, "avg_duration_ms": 2000}

    # company filter + exact percentiles on a known set (100..1000)
    r = await client.get(
        "/api/v1/internal/metrics/summary",
        params={**WINDOW, "company": "bookshop"},
        headers=ADMIN_HEADERS,
    )
    b = r.json()["chats"]
    assert b["total"] == 10
    assert b["latency_ms"] == {"avg": 550.0, "p50": 550.0, "p95": 955.0}

    # dashboard renders the same data
    r = await client.get("/api/v1/internal/dashboard", params={**WINDOW, "key": ADMIN_KEY})
    assert r.status_code == 200
    assert "chart.umd.min.js" in r.text and "2026-10-02" in r.text


async def test_metrics_defaults_and_validation(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/internal/metrics/summary", headers=ADMIN_HEADERS)
    assert r.status_code == 200
    w = r.json()["window"]
    span = datetime.fromisoformat(w["to"]) - datetime.fromisoformat(w["from"])
    assert span == timedelta(days=7)
    assert r.json()["chats"]["total"] == 0 and r.json()["chats"]["latency_ms"]["p50"] is None

    r = await client.get(
        "/api/v1/internal/metrics/summary",
        params={"from": "2026-10-03T00:00:00Z", "to": "2026-10-01T00:00:00Z"},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 422
    r = await client.get(
        "/api/v1/internal/metrics/summary", params={"company": "nope"}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 404


async def test_prometheus_custom_metrics(client: httpx.AsyncClient) -> None:
    from tests.integration.conftest import BOOKSHOP_TEXT, upload

    await create_company(client, "bookshop")
    await upload(client, "bookshop", BOOKSHOP_TEXT)
    from tests.conftest import AGENT_HEADERS

    await client.post(
        "/api/v1/chats",
        json={"company": "bookshop", "customer_message": "hi"},
        headers=AGENT_HEADERS,
    )
    text = (await client.get("/metrics", headers=ADMIN_HEADERS)).text
    assert 'chat_requests_total{company="bookshop",status="success"} 1.0' in text
    assert "chat_latency_seconds_bucket" in text
    assert 'llm_tokens_total{direction="input"} 120.0' in text
    assert 'ingestion_total{status="ready"} 1.0' in text
    assert "http_requests_total" in text
