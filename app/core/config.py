"""Application settings, loaded from environment variables / `.env`."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


@dataclass(frozen=True)
class ApiKey:
    """An API key plus a human-readable label (stored as `uploaded_by`, safe to log)."""

    key: str
    label: str


def _parse_keys(raw: str, role: str) -> list[ApiKey]:
    """Parse `key1,key2` or `label:key1,label2:key2` into ApiKey entries."""
    keys: list[ApiKey] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        if ":" in item:
            label, key = item.split(":", 1)
            label, key = label.strip(), key.strip()
        else:
            key = item
            label = f"{role}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"
        if key:
            keys.append(ApiKey(key=key, label=label))
    return keys


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    APP_ENV: Literal["local", "test", "prod"] = "local"

    # --- Database -------------------------------------------------------------
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:55422/postgres"
    TEST_DATABASE_URL: str | None = None
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # --- AI provider ----------------------------------------------------------
    LLM_PROVIDER: Literal["google"] = "google"
    GOOGLE_API_KEY: SecretStr | None = None
    LLM_MODEL: str = "gemini-3.1-pro-preview"
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: float = 60
    LLM_MAX_RETRIES: int = 2
    EMBEDDING_MODEL: str = "gemini-embedding-2"

    # --- Vector store ---------------------------------------------------------
    VECTOR_BACKEND: Literal["chroma_local", "chroma_remote"] = "chroma_local"
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_HOST: str | None = None
    CHROMA_PORT: int | None = None
    CHROMA_API_KEY: SecretStr | None = None
    CHROMA_TENANT: str | None = None
    CHROMA_DATABASE: str | None = None

    # --- RAG ------------------------------------------------------------------
    CHUNK_SIZE: int = Field(default=800, gt=50)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)
    RAG_TOP_K: int = Field(default=6, ge=1, le=50)
    RAG_SEARCH_TYPE: Literal["similarity", "mmr"] = "mmr"

    # --- Ingestion ------------------------------------------------------------
    MAX_UPLOAD_BYTES: int = 2_000_000
    INGESTION_SYNC: bool = False

    # --- Auth -----------------------------------------------------------------
    AGENT_API_KEYS: str = ""
    ADMIN_API_KEYS: str = ""

    # --- Misc -----------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    COMPANY_CACHE_TTL_SECONDS: float = 30.0

    @property
    def agent_keys(self) -> list[ApiKey]:
        return _parse_keys(self.AGENT_API_KEYS, "agent")

    @property
    def admin_keys(self) -> list[ApiKey]:
        return _parse_keys(self.ADMIN_API_KEYS, "admin")

    @property
    def uses_transaction_pooler(self) -> bool:
        """Supabase transaction pooler (port 6543) cannot use asyncpg prepared statements."""
        return make_url(self.DATABASE_URL).port == 6543


@lru_cache
def get_settings() -> Settings:
    return Settings()
