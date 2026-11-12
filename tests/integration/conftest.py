"""Integration tier: the real app against real Postgres (local Supabase or a CI Postgres service).

Session start: drop + recreate the test database, then `alembic upgrade head` (never create_all).
Between tests: TRUNCATE all app tables.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.main import create_app
from tests.conftest import (
    ADMIN_HEADERS,
    FakeEmbeddingsFactory,
    FakeGuidanceGenerator,
    make_settings,
    running_client,
)

ROOT = Path(__file__).resolve().parents[2]
APP_TABLES = ("chats", "guideline_versions", "companies")
SKIP_HINT = (
    "start local Supabase with `make db-up` (or set TEST_DATABASE_URL to a reachable Postgres)"
)


def _configured_urls() -> tuple[str | None, str | None]:
    env_settings = Settings()  # reads env vars + .env
    return (
        os.environ.get("TEST_DATABASE_URL") or env_settings.TEST_DATABASE_URL,
        env_settings.DATABASE_URL,
    )


def _plain_dsn(url: str, database: str | None = None) -> str:
    u = make_url(url).set(drivername="postgresql")
    if database:
        u = u.set(database=database)
    return u.render_as_string(hide_password=False)


async def _recreate_database(url: str) -> None:
    name = make_url(url).database
    conn = await asyncpg.connect(_plain_dsn(url, "postgres"), timeout=5)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


def alembic_config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["configure_logging"] = False
    return cfg


def check_safe_test_url(test_url: str, app_url: str | None) -> None:
    name = make_url(test_url).database or ""
    if not name.endswith("_test"):
        pytest.exit(f"Refusing to run: test database name '{name}' must end in '_test'", 2)
    if app_url and make_url(test_url) == make_url(app_url):
        pytest.exit("Refusing to run: TEST_DATABASE_URL equals DATABASE_URL", 2)


@pytest.fixture(scope="session")
def pg_url() -> str:
    """Fresh, fully migrated test database (once per session)."""
    test_url, app_url = _configured_urls()
    if not test_url:
        pytest.skip(f"TEST_DATABASE_URL is not set — {SKIP_HINT}")
    check_safe_test_url(test_url, app_url)
    try:
        asyncio.run(_recreate_database(test_url))
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"Postgres at TEST_DATABASE_URL is unreachable ({exc!r}) — {SKIP_HINT}")
    command.upgrade(alembic_config(test_url), "head")
    return test_url


@pytest.fixture
async def db_engine(pg_url: str) -> AsyncIterator[Any]:
    engine = create_async_engine(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(APP_TABLES)} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
def db_sessionmaker(db_engine: Any) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


ClientFactory = Callable[..., AbstractAsyncContextManager[httpx.AsyncClient]]


@pytest.fixture
def client_factory(
    pg_url: str, db_engine: Any, tmp_path: Path, fake_embeddings: FakeEmbeddingsFactory
) -> ClientFactory:
    """Build a running app + client. Override generator/settings per test when needed."""

    @asynccontextmanager
    async def _make(
        generator: Any = None, **settings_overrides: Any
    ) -> AsyncIterator[httpx.AsyncClient]:
        settings = make_settings(
            DATABASE_URL=pg_url,
            CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
            **settings_overrides,
        )
        app = create_app(
            settings,
            generator=generator or FakeGuidanceGenerator(),
            embeddings_for=fake_embeddings,
        )
        async with running_client(app) as client:
            client.app = app  # type: ignore[attr-defined]
            yield client

    return _make


@pytest.fixture
async def client(client_factory: ClientFactory) -> AsyncIterator[httpx.AsyncClient]:
    async with client_factory() as c:
        yield c


def app_of(client: httpx.AsyncClient) -> FastAPI:
    return client.app  # type: ignore[attr-defined, no-any-return]


# ------------------------------------------------------------------- helpers
BOOKSHOP_TEXT = """\
TONE
Greet the customer by first name and sign off with "Warm regards, The Bookshop Team".

REFUNDS:
Books can be returned within 30 days of delivery for a full refund. Damaged books are refunded
including shipping costs. E-books are refundable only if the file is corrupted.

SHIPPING:
Standard shipping takes 3-5 business days; express takes 1-2 business days.

ESCALATION:
Escalate to a team lead if the customer threatens legal action.
"""

VPN_TEXT = """\
BILLING
Duplicate charges are refunded within 5 business days after verification in the billing console.
Annual plans can be cancelled with a prorated refund within the first 30 days.

CONNECTION ISSUES
Ask for device, OS version and protocol. On iOS, reinstall the VPN profile first.

ESCALATION
Escalate to tier 2 if the issue persists after the standard troubleshooting steps.
"""


async def create_company(
    client: httpx.AsyncClient, slug: str, name: str | None = None
) -> dict[str, Any]:
    r = await client.post(
        "/api/v1/companies",
        json={"slug": slug, "name": name or slug.title()},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


async def upload(
    client: httpx.AsyncClient,
    slug: str,
    content: str | bytes,
    *,
    filename: str = "guide.txt",
    activate: bool | None = None,
    notes: str | None = None,
) -> httpx.Response:
    data: dict[str, str] = {}
    if activate is not None:
        data["activate"] = "true" if activate else "false"
    if notes:
        data["notes"] = notes
    body = content.encode() if isinstance(content, str) else content
    return await client.post(
        f"/api/v1/companies/{slug}/guidelines",
        files={"file": (filename, body, "text/plain")},
        data=data,
        headers=ADMIN_HEADERS,
    )


async def active_version_ids(
    sessionmaker: async_sessionmaker[AsyncSession], slug: str
) -> list[Any]:
    """Raw DB check of the active pointer (exactly one value, possibly None)."""
    async with sessionmaker() as s:
        rows = (
            await s.execute(
                text("SELECT active_version_id FROM companies WHERE slug = :s"), {"s": slug}
            )
        ).all()
    return [r[0] for r in rows]
