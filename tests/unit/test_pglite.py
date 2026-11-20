from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.db import session as db_session


def test_pglite_mode_selects_local_url_and_engine_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_create(url: str, **kwargs: Any) -> object:
        captured.update(url=url, **kwargs)
        return sentinel

    monkeypatch.setattr(db_session, "create_async_engine", fake_create)
    settings = Settings(
        _env_file=None,
        PGLITE_MODE=True,
        PGLITE_DATABASE_URL="postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/postgres",
        PGLITE_APP_POOL_SIZE=1,
        DATABASE_URL="postgresql+asyncpg://real:secret@db.example.com:5432/postgres",
    )

    assert db_session.create_engine(settings) is sentinel
    assert captured["url"] == settings.PGLITE_DATABASE_URL
    assert captured["pool_size"] == 1
    assert captured["max_overflow"] == 0
    assert captured["connect_args"] == {
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    }


def test_pglite_is_rejected_in_production() -> None:
    with pytest.raises(ValueError, match="PGLITE_MODE cannot be enabled in production"):
        Settings(_env_file=None, APP_ENV="prod", PGLITE_MODE=True)


def test_fake_llm_mode_selects_offline_models() -> None:
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="fake",
        GOOGLE_API_KEY="must-not-be-used",
        LANGSMITH_TRACING=True,
    )

    assert settings.LLM_MODEL == "heuristic-generator"
    assert settings.COMPILER_MODEL == "heuristic-compiler"
    assert settings.ROUTER_MODEL == "heuristic-router"
    assert settings.EMBEDDING_MODEL == "fake-embedding"
    assert settings.LANGSMITH_TRACING is False


def test_fake_llm_is_rejected_in_production() -> None:
    with pytest.raises(ValueError, match="fake LLM provider cannot be enabled in production"):
        Settings(_env_file=None, APP_ENV="prod", LLM_PROVIDER="fake")
