"""App wiring that needs no database: health, request ids, error format, cache, OpenAPI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.logging import JsonFormatter, request_id_var, text_fingerprint
from app.main import create_app
from app.services.common import CompanyContext, CompanyContextCache
from tests.conftest import (
    FakeEmbeddingsFactory,
    FakeGuidanceGenerator,
    make_settings,
    running_client,
)


def _app(chroma_dir: Path):  # type: ignore[no-untyped-def]
    settings = make_settings(
        CHROMA_PERSIST_DIR=str(chroma_dir),
        DATABASE_URL="postgresql+asyncpg://u:p@127.0.0.1:1/never_connected",
    )
    return create_app(
        settings, generator=FakeGuidanceGenerator(), embeddings_for=FakeEmbeddingsFactory()
    )


async def test_health_and_request_id(chroma_dir: Path) -> None:
    async with running_client(_app(chroma_dir)) as client:
        r = await client.get("/api/v1/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        assert len(r.headers["X-Request-ID"]) == 32

        r = await client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
        assert r.headers["X-Request-ID"] == "abc-123"

        # invalid incoming ids are replaced, not echoed
        r = await client.get("/api/v1/health", headers={"X-Request-ID": "bad id <script>"})
        assert r.headers["X-Request-ID"] != "bad id <script>"


async def test_error_format_includes_request_id(chroma_dir: Path) -> None:
    async with running_client(_app(chroma_dir)) as client:
        r = await client.get("/api/v1/companies", headers={"X-Request-ID": "rid-1"})
        assert r.status_code == 401
        assert r.json() == {
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Missing or invalid API key",
                "request_id": "rid-1",
            }
        }


async def test_openapi_lists_every_endpoint(chroma_dir: Path) -> None:
    async with running_client(_app(chroma_dir)) as client:
        paths = (await client.get("/openapi.json")).json()["paths"]
    expected = {
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/companies",
        "/api/v1/companies/{slug}",
        "/api/v1/companies/{slug}/guidelines",
        "/api/v1/companies/{slug}/guidelines/active",
        "/api/v1/companies/{slug}/guidelines/{version_number}",
        "/api/v1/companies/{slug}/guidelines/{version_number}/activate",
        "/api/v1/chats",
        "/api/v1/chats/{chat_id}",
        "/api/v1/internal/metrics/summary",
        "/api/v1/internal/dashboard",
        "/metrics",
    }
    assert expected <= set(paths)


def test_cache_ttl_and_invalidate() -> None:
    now = [0.0]
    cache = CompanyContextCache(ttl_seconds=30, clock=lambda: now[0])
    ctx = CompanyContext(
        company_id=__import__("uuid").uuid4(),
        slug="s",
        name="S",
        active_version_id=None,
        active_version_number=None,
        active_version_ready=False,
        collection=None,
        embedding_model=None,
    )
    cache.set(ctx)
    assert cache.get("s") is ctx
    now[0] = 31
    assert cache.get("s") is None
    cache.set(ctx)
    cache.invalidate("s")
    assert cache.get("s") is None


def test_json_logs_never_contain_raw_message() -> None:
    fp = text_fingerprint("my secret customer text")
    assert fp["len"] == 23 and "secret" not in json.dumps(fp)
    token = request_id_var.set("rid-9")
    try:
        record = logging.makeLogRecord(
            {"msg": "chat.completed", "levelname": "INFO", "company": "x"}
        )
        out = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert (
        out["event"] == "chat.completed" and out["request_id"] == "rid-9" and out["company"] == "x"
    )
