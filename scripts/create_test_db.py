"""Create the integration-test database on local Supabase if it doesn't exist.

Uses asyncpg directly so `psql` isn't needed. Usage: python scripts/create_test_db.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg
from sqlalchemy.engine import make_url

from app.core.config import get_settings


def _dsn(url: str, database: str) -> str:
    u = make_url(url).set(drivername="postgresql", database=database)
    return u.render_as_string(hide_password=False)


async def ensure_database(url: str) -> bool:
    """Create the database named in `url` if missing. Returns True if it was created."""
    name = make_url(url).database
    if not name:
        raise SystemExit("TEST_DATABASE_URL has no database name")
    conn = await asyncpg.connect(_dsn(url, "postgres"))
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if exists:
            return False
        await conn.execute(f'CREATE DATABASE "{name}"')
        return True
    finally:
        await conn.close()


async def main() -> None:
    url = get_settings().TEST_DATABASE_URL
    if not url:
        print("TEST_DATABASE_URL is not set; nothing to do.")
        return
    created = await ensure_database(url)
    name = make_url(url).database
    print(f"Test database '{name}' {'created' if created else 'already exists'}.")


if __name__ == "__main__":
    asyncio.run(main())
