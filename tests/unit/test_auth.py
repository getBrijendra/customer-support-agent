"""U5 — auth dependencies on a tiny app with no DB."""

from __future__ import annotations

import httpx
import pytest
from fastapi import Depends, FastAPI

from app.core.config import _parse_keys
from app.core.errors import register_exception_handlers
from app.core.security import (
    Principal,
    require_admin,
    require_admin_header_or_query,
    require_agent,
)
from tests.conftest import ADMIN_HEADERS, ADMIN_KEY, AGENT_HEADERS, make_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:
    app = FastAPI()
    app.state.settings = make_settings()
    register_exception_handlers(app)

    @app.get("/agent")
    async def agent(p: Principal = Depends(require_agent)) -> dict[str, str]:
        return {"role": p.role, "label": p.label}

    @app.get("/admin")
    async def admin(p: Principal = Depends(require_admin)) -> dict[str, str]:
        return {"role": p.role, "label": p.label}

    @app.get("/dash")
    async def dash(p: Principal = Depends(require_admin_header_or_query)) -> dict[str, str]:
        return {"role": p.role}

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_missing_key_401(client: httpx.AsyncClient) -> None:
    r = await client.get("/agent")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


async def test_wrong_key_401(client: httpx.AsyncClient) -> None:
    r = await client.get("/admin", headers={"X-API-Key": "nope"})
    assert r.status_code == 401


async def test_agent_key_on_admin_route_403(client: httpx.AsyncClient) -> None:
    r = await client.get("/admin", headers=AGENT_HEADERS)
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "FORBIDDEN"


async def test_agent_key_on_agent_route(client: httpx.AsyncClient) -> None:
    r = await client.get("/agent", headers=AGENT_HEADERS)
    assert r.status_code == 200 and r.json()["role"] == "agent"


async def test_admin_key_passes_everywhere(client: httpx.AsyncClient) -> None:
    assert (await client.get("/agent", headers=ADMIN_HEADERS)).json()["role"] == "admin"
    r = await client.get("/admin", headers=ADMIN_HEADERS)
    assert r.status_code == 200
    assert r.json() == {"role": "admin", "label": "tester"}


async def test_query_key_for_dashboard(client: httpx.AsyncClient) -> None:
    assert (await client.get("/dash", params={"key": ADMIN_KEY})).status_code == 200
    assert (await client.get("/dash")).status_code == 401
    assert (await client.get("/admin", params={"key": ADMIN_KEY})).status_code == 401


def test_key_parsing_labels() -> None:
    keys = _parse_keys("alice:k1, k2 ,,", "agent")
    assert keys[0].label == "alice" and keys[0].key == "k1"
    assert keys[1].key == "k2" and keys[1].label.startswith("agent-") and "k2" not in keys[1].label
