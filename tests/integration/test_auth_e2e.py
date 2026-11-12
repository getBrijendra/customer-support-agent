"""Integration case 13: real routers reject missing / wrong-role keys."""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import ADMIN_HEADERS, ADMIN_KEY, AGENT_HEADERS
from tests.integration.conftest import BOOKSHOP_TEXT, create_company, upload

ADMIN_ONLY = [
    ("POST", "/api/v1/companies", {"json": {"slug": "x-co", "name": "X"}}),
    (
        "POST",
        "/api/v1/companies/bookshop/guidelines",
        {"files": {"file": ("g.txt", b"hi", "text/plain")}},
    ),
    ("POST", "/api/v1/companies/bookshop/guidelines/1/activate", {}),
    ("DELETE", "/api/v1/companies/bookshop/guidelines/1", {}),
    ("GET", "/api/v1/internal/metrics/summary", {}),
    ("GET", "/api/v1/internal/dashboard", {}),
    ("GET", "/metrics", {}),
]
AGENT_OK = [
    ("GET", "/api/v1/companies"),
    ("GET", "/api/v1/companies/bookshop"),
    ("GET", "/api/v1/companies/bookshop/guidelines"),
    ("GET", "/api/v1/companies/bookshop/guidelines/active"),
    ("GET", "/api/v1/companies/bookshop/guidelines/1"),
    ("GET", "/api/v1/chats"),
]


@pytest.fixture
async def seeded(client: httpx.AsyncClient) -> httpx.AsyncClient:
    await create_company(client, "bookshop")
    await upload(client, "bookshop", BOOKSHOP_TEXT)
    return client


async def test_missing_key_rejected_everywhere(seeded: httpx.AsyncClient) -> None:
    for method, path, kwargs in ADMIN_ONLY:
        r = await seeded.request(method, path, **kwargs)
        assert r.status_code == 401, (method, path, r.status_code)
    for method, path in AGENT_OK:
        assert (await seeded.request(method, path)).status_code == 401, path
    r = await seeded.post("/api/v1/chats", json={"company": "bookshop", "customer_message": "hi"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


async def test_agent_key_forbidden_on_admin_routes(seeded: httpx.AsyncClient) -> None:
    for method, path, kwargs in ADMIN_ONLY:
        r = await seeded.request(method, path, headers=AGENT_HEADERS, **kwargs)
        assert r.status_code == 403, (method, path, r.status_code)
        if path != "/metrics":
            assert r.json()["error"]["code"] == "FORBIDDEN"


async def test_agent_key_allowed_on_agent_routes(seeded: httpx.AsyncClient) -> None:
    for method, path in AGENT_OK:
        r = await seeded.request(method, path, headers=AGENT_HEADERS)
        assert r.status_code == 200, (path, r.text)
    r = await seeded.post(
        "/api/v1/chats",
        json={"company": "bookshop", "customer_message": "hi"},
        headers=AGENT_HEADERS,
    )
    assert r.status_code == 201


async def test_admin_key_reaches_admin_routes(seeded: httpx.AsyncClient) -> None:
    assert (
        await seeded.get("/api/v1/internal/metrics/summary", headers=ADMIN_HEADERS)
    ).status_code == 200
    assert (await seeded.get("/metrics", headers=ADMIN_HEADERS)).status_code == 200
    r = await seeded.get("/api/v1/internal/dashboard", params={"key": ADMIN_KEY})
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


async def test_health_endpoints_are_public(seeded: httpx.AsyncClient) -> None:
    assert (await seeded.get("/api/v1/health")).status_code == 200
    assert (await seeded.get("/api/v1/ready")).status_code == 200
