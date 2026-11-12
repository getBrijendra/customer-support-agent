"""Readiness against real Postgres + company endpoints."""

from __future__ import annotations

import httpx

from tests.conftest import ADMIN_HEADERS, AGENT_HEADERS
from tests.integration.conftest import create_company


async def test_ready_returns_200_against_postgres(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/ready")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok", "checks": {"database": True, "vector_store": True}}


async def test_create_list_get_company(client: httpx.AsyncClient) -> None:
    created = await create_company(client, "bookshop", "Bookshop")
    assert created["slug"] == "bookshop" and created["active_version_number"] is None

    r = await client.get("/api/v1/companies", headers=AGENT_HEADERS)
    assert [c["slug"] for c in r.json()] == ["bookshop"]

    r = await client.get("/api/v1/companies/bookshop", headers=AGENT_HEADERS)
    assert r.status_code == 200 and r.json()["name"] == "Bookshop"


async def test_duplicate_slug_409(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    r = await client.post(
        "/api/v1/companies", json={"slug": "bookshop", "name": "Again"}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 409


async def test_invalid_slug_and_missing_company(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/api/v1/companies", json={"slug": "Bad Slug!", "name": "x"}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = await client.get("/api/v1/companies/nope", headers=AGENT_HEADERS)
    assert r.status_code == 404 and r.json()["error"]["code"] == "COMPANY_NOT_FOUND"
