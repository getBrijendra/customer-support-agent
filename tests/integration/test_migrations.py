"""Integration case 15: upgrade -> downgrade base -> upgrade on an empty database."""

from __future__ import annotations

import asyncio

import asyncpg
from alembic import command
from sqlalchemy.engine import make_url

from tests.integration.conftest import _plain_dsn, _recreate_database, alembic_config


async def _tables(url: str) -> set[str]:
    conn = await asyncpg.connect(_plain_dsn(url))
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {r["tablename"] for r in rows}
    finally:
        await conn.close()


async def _drop(url: str) -> None:
    conn = await asyncpg.connect(_plain_dsn(url, "postgres"))
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{make_url(url).database}" WITH (FORCE)')
    finally:
        await conn.close()


def test_upgrade_downgrade_upgrade(pg_url: str) -> None:
    # Separate scratch database so the shared test DB is never left half-migrated.
    scratch = make_url(pg_url).set(database="support_assistant_migrations_test")
    url = scratch.render_as_string(hide_password=False)
    asyncio.run(_recreate_database(url))
    try:
        cfg = alembic_config(url)
        command.upgrade(cfg, "head")
        assert {"companies", "guideline_versions", "chats", "alembic_version"} <= asyncio.run(
            _tables(url)
        )
        command.downgrade(cfg, "base")
        assert asyncio.run(_tables(url)) <= {"alembic_version"}
        command.upgrade(cfg, "head")
        assert {"companies", "guideline_versions", "chats"} <= asyncio.run(_tables(url))
    finally:
        asyncio.run(_drop(url))
