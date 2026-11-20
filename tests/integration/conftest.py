"""Integration tier against ephemeral PGlite (default) or an explicit real Postgres service.

Session start: reset the isolated database, then `alembic upgrade head` (never create_all).
Between tests: TRUNCATE all app tables.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterator
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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.heuristic import HeuristicExampleGenerator, HeuristicPlaybookCompiler
from app.core.config import Settings
from app.db.session import create_engine
from app.main import create_app
from tests.conftest import (
    ADMIN_HEADERS,
    FakeEmbeddingsFactory,
    FakeGuidanceGenerator,
    ScriptedSelector,
    make_settings,
    running_client,
)

ROOT = Path(__file__).resolve().parents[2]
APP_TABLES = (
    "tool_calls",
    "proposed_actions",
    "capability_bindings",
    "mcp_servers",
    "company_secrets",
    "routing_eval_runs",
    "routing_eval_cases",
    "reference_facts",
    "company_rules",
    "intent_examples",
    "intents",
    "guideline_sections",
    "chats",
    "guideline_versions",
    "companies",
)
SKIP_HINT = "set TEST_DATABASE_URL and pass --integration-db=postgres"
PGLITE_SERVER = (
    ROOT / "node_modules" / "@electric-sql" / "pglite-socket" / "dist" / "scripts" / "server.js"
)
PGLITE_START_TIMEOUT_SECONDS = 60


@pytest.fixture(scope="session")
def integration_db_backend(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--integration-db"))


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


async def _pglite_version(url: str, connect_timeout: float = 1) -> str:
    conn = await asyncpg.connect(
        _plain_dsn(url),
        timeout=connect_timeout,
        statement_cache_size=0,
    )
    try:
        return str(await conn.fetchval("SELECT version()"))
    finally:
        # PGlite's wire server does not complete asyncpg's graceful close handshake.
        conn.terminate()


def _wait_for_pglite(url: str, process: subprocess.Popen[Any], log: Any) -> None:
    deadline = time.monotonic() + PGLITE_START_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            version = asyncio.run(_pglite_version(url))
            if "PGlite" not in version:
                raise RuntimeError(f"unexpected database server: {version}")
            return
        except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.05)
    log.flush()
    log.seek(0)
    output = log.read().strip()
    pytest.exit(
        f"PGlite test server failed to start ({last_error!r}).\n{output}",
        returncode=2,
    )


@pytest.fixture(scope="session")
def pglite_url(integration_db_backend: str) -> Iterator[str | None]:
    """Start one isolated in-memory PGlite socket server for the integration session."""
    if integration_db_backend != "pglite":
        yield None
        return
    node = shutil.which("node")
    if not node or not PGLITE_SERVER.is_file():
        pytest.exit("PGlite integration tests require Node.js and `npm install`", returncode=2)
    port = _available_port()
    url = f"postgresql+asyncpg://postgres:postgres@127.0.0.1:{port}/postgres"
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [
                node,
                str(PGLITE_SERVER),
                "--db=memory://",
                "--host=127.0.0.1",
                f"--port={port}",
                "--extensions=@electric-sql/pglite-pgvector:vector",
                "--max-connections=256",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        try:
            _wait_for_pglite(url, process, log)
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


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


async def _reset_pglite_database(url: str) -> None:
    conn = await asyncpg.connect(_plain_dsn(url), timeout=5, statement_cache_size=0)
    try:
        version = str(await conn.fetchval("SELECT version()"))
        if "PGlite" not in version:
            raise RuntimeError(f"refusing to reset non-PGlite server: {version}")
        await conn.execute("DROP SCHEMA public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        conn.terminate()


def alembic_config(url: str, *, pglite: bool = False) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["configure_logging"] = False
    cfg.attributes["pglite"] = pglite
    return cfg


def check_safe_test_url(test_url: str, app_url: str | None) -> None:
    name = make_url(test_url).database or ""
    if not name.endswith("_test"):
        pytest.exit(f"Refusing to run: test database name '{name}' must end in '_test'", 2)
    if app_url and make_url(test_url) == make_url(app_url):
        pytest.exit("Refusing to run: TEST_DATABASE_URL equals DATABASE_URL", 2)


@pytest.fixture(scope="session")
def pg_url(integration_db_backend: str, pglite_url: str | None) -> str:
    """Fresh, fully migrated test database (once per session)."""
    if integration_db_backend == "pglite":
        if not pglite_url:  # pragma: no cover - fixture invariant
            pytest.exit("PGlite test server did not provide a database URL", returncode=2)
        asyncio.run(_reset_pglite_database(pglite_url))
        command.upgrade(alembic_config(pglite_url, pglite=True), "head")
        return pglite_url

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
async def db_engine(pg_url: str, integration_db_backend: str) -> AsyncIterator[Any]:
    pglite = integration_db_backend == "pglite"
    settings = make_settings(
        DATABASE_URL=pg_url,
        PGLITE_MODE=pglite,
        PGLITE_DATABASE_URL=pg_url,
        PGLITE_APP_POOL_SIZE=1,
    )
    engine = create_engine(settings)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(APP_TABLES)} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose(close=not pglite)


@pytest.fixture
def db_sessionmaker(db_engine: Any) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


ClientFactory = Callable[..., AbstractAsyncContextManager[httpx.AsyncClient]]


@pytest.fixture
def client_factory(
    pg_url: str,
    db_engine: Any,
    integration_db_backend: str,
    fake_embeddings: FakeEmbeddingsFactory,
) -> ClientFactory:
    """Build a running app + client. Override collaborators/settings per test when needed."""

    @asynccontextmanager
    async def _make(
        generator: Any = None,
        selector: Any = None,
        tool_model: Any = None,
        adapters: Any = None,
        compiler: Any = None,
        file_store: Any = None,
        **settings_overrides: Any,
    ) -> AsyncIterator[httpx.AsyncClient]:
        pglite = integration_db_backend == "pglite"
        values: dict[str, Any] = {
            "DATABASE_URL": pg_url,
            "PGLITE_MODE": pglite,
            "PGLITE_DATABASE_URL": pg_url,
            "PGLITE_APP_POOL_SIZE": 1,
        }
        values.update(settings_overrides)
        settings = make_settings(**values)
        app = create_app(
            settings,
            generator=generator or FakeGuidanceGenerator(),
            selector=selector or ScriptedSelector(),
            compiler=compiler or HeuristicPlaybookCompiler(),
            examples=HeuristicExampleGenerator(),
            tool_model=tool_model,
            embeddings_for=fake_embeddings,
            adapters=adapters,
            file_store=file_store,
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
BOOKSHOP_TEXT = """TONE
- Greet the customer by first name.
- Sign off with "Warm regards, The Bookshop Team".
- Never promise delivery dates that the courier has not confirmed.

1. Shipping Delays
Approach to solve:
- Verify order details and tracking with the courier.
- Apologize for the delay.
- Escalate to the logistics team if the parcel is lost.

Sample reply:
Hello [Name], I'm sorry your order is late. It is now expected on [Date].

2. Damaged Books
Approach to solve:
- Ask the customer for a photo of the damage.
- Verify the order.
- Process a refund including shipping costs.

Sample reply:
Hello [Name], I'm sorry your book arrived damaged.

3. Returns
Approach to solve:
- Check the order date: returns are accepted within 30 days of delivery.
- Issue a prepaid return label.

REFERENCE
- Standard shipping: 3-5 business days
- Express shipping: 1-2 business days
"""

VPN_TEXT = """1. Billing and Subscription Issues
Approach to solve:
- Check subscription status, renewal date, and plan type.
- Process refunds for duplicate charges per policy.
- Escalate complex disputes to the finance department.

Sample reply:
Hello [Name], I have checked your subscription and billing.

2. Connection Failures
Approach to solve:
- Ask which server location and VPN protocol the user selected.
- Suggest switching to the WireGuard protocol.

Sample reply:
Hello [Name], I'm sorry you can't connect.

3. Multi-Device Connection Limit
Approach to solve:
- Check the number of active device connections on the account.
- Advise disconnecting unused devices.

REFERENCE
- Supported protocols: WireGuard and OpenVPN
- Device limit: 5 simultaneous devices on every plan
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


async def chat(
    client: httpx.AsyncClient, company: str, msg: str = "My book arrived damaged", **extra: Any
) -> httpx.Response:
    from tests.conftest import AGENT_HEADERS

    return await client.post(
        "/api/v1/chats",
        json={"company": company, "customer_message": msg, **extra},
        headers=AGENT_HEADERS,
    )


async def bind_live(
    client: httpx.AsyncClient,
    slug: str,
    capability: str,
    config: dict[str, Any],
    *,
    adapter_type: str = "mock",
    policy: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """PUT binding -> test -> go-live; returns the live binding."""
    base = f"/api/v1/companies/{slug}/capability-bindings/{capability}"
    r = await client.put(
        base,
        json={"adapter_type": adapter_type, "config": config, "policy": policy or {}},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 200, r.text
    r = await client.post(f"{base}/test", json={"sample_args": sample or {}}, headers=ADMIN_HEADERS)
    assert r.status_code == 200 and r.json()["result"]["ok"], r.text
    r = await client.post(f"{base}/go-live", headers=ADMIN_HEADERS)
    assert r.status_code == 200 and r.json()["status"] == "live", r.text
    return r.json()  # type: ignore[no-any-return]


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
