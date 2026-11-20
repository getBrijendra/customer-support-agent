"""Integration case 15: upgrade -> downgrade base -> upgrade on an empty database."""

from __future__ import annotations

import asyncio

import asyncpg
from alembic import command
from sqlalchemy.engine import make_url

from tests.integration.conftest import (
    _plain_dsn,
    _recreate_database,
    _reset_pglite_database,
    alembic_config,
)


async def _tables(url: str, *, pglite: bool = False) -> set[str]:
    conn = await asyncpg.connect(_plain_dsn(url), statement_cache_size=0 if pglite else 100)
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {r["tablename"] for r in rows}
    finally:
        if pglite:
            conn.terminate()
        else:
            await conn.close()


async def _drop(url: str) -> None:
    conn = await asyncpg.connect(_plain_dsn(url, "postgres"))
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{make_url(url).database}" WITH (FORCE)')
    finally:
        await conn.close()


V2_TABLES = {
    "companies",
    "guideline_versions",
    "chats",
    "guideline_sections",
    "intents",
    "intent_examples",
    "company_rules",
    "reference_facts",
    "routing_eval_cases",
    "routing_eval_runs",
    "company_secrets",
    "mcp_servers",
    "capability_bindings",
    "tool_calls",
    "proposed_actions",
}


def test_upgrade_downgrade_upgrade(pg_url: str, integration_db_backend: str) -> None:
    pglite = integration_db_backend == "pglite"
    if pglite:
        # PGlite exposes one database. This suite is sequential, so reset it in place and restore
        # the head revision before returning it to the remaining integration tests.
        url = pg_url
        asyncio.run(_reset_pglite_database(url))
    else:
        # Real Postgres gets a separate scratch database so the shared test DB stays migrated.
        scratch = make_url(pg_url).set(database="support_assistant_migrations_test")
        url = scratch.render_as_string(hide_password=False)
        asyncio.run(_recreate_database(url))
    try:
        cfg = alembic_config(url, pglite=pglite)
        command.upgrade(cfg, "head")
        assert {
            "companies",
            "guideline_versions",
            "chats",
            "alembic_version",
        } <= asyncio.run(_tables(url, pglite=pglite))
        command.downgrade(cfg, "base")
        assert asyncio.run(_tables(url, pglite=pglite)) <= {"alembic_version"}
        command.upgrade(cfg, "head")
        assert asyncio.run(_tables(url, pglite=pglite)) >= V2_TABLES
        command.downgrade(cfg, "0001")  # v2 -> v1 schema
        tables = asyncio.run(_tables(url, pglite=pglite))
        assert {"companies", "guideline_versions", "chats"} <= tables and not V2_TABLES & (
            tables - {"companies", "guideline_versions", "chats"}
        )
        command.upgrade(cfg, "head")
    finally:
        if pglite:
            asyncio.run(_reset_pglite_database(url))
            command.upgrade(alembic_config(url, pglite=True), "head")
        else:
            asyncio.run(_drop(url))
