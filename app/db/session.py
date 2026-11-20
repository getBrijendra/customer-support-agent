"""Async engine + session factory + FastAPI session dependency."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import Settings


def create_engine(settings: Settings, url: str | None = None) -> AsyncEngine:
    url = url or settings.DATABASE_URL
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if settings.PGLITE_MODE and url == settings.DATABASE_URL:
        # PGlite executes through one WASM database but its socket server multiplexes clients.
        # Disable named prepared statements (their names collide across logical connections) and
        # keep a bounded pool, leaving the remaining socket slots available to browser tools.
        kwargs["connect_args"] = {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        }
        kwargs["pool_size"] = settings.PGLITE_APP_POOL_SIZE
        kwargs["max_overflow"] = 0
    elif settings.uses_transaction_pooler and url == settings.DATABASE_URL:
        # Supabase transaction pooler (PgBouncer, port 6543): prepared statements break,
        # and the pooler already pools connections, so don't pool on our side.
        kwargs["connect_args"] = {"statement_cache_size": 0, "prepared_statement_cache_size": 0}
        kwargs["poolclass"] = NullPool
        kwargs.pop("pool_pre_ping")
    else:
        kwargs["pool_size"] = settings.DB_POOL_SIZE
        kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW
    return create_async_engine(url, **kwargs)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.resources.sessionmaker
    async with sessionmaker() as session:
        yield session


async def ping(engine: AsyncEngine) -> bool:
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
