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
    LLM_MODEL: str = "gemini-3.1-pro-preview"  # generator
    COMPILER_MODEL: str = "gemini-3.1-pro-preview"  # guideline -> playbook compiler
    ROUTER_MODEL: str = "gemini-3.5-flash-lite"  # intent selector
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT_SECONDS: float = 60
    LLM_MAX_RETRIES: int = 2
    EMBEDDING_MODEL: str = "gemini-embedding-2"
    # Fixed by the migration (vector(768) columns); changing it needs a migration + re-embed.
    EMBEDDING_DIMENSIONS: int = 768

    # --- Search & routing -----------------------------------------------------
    TEXT_SEARCH_CONFIG: str = "english"
    HNSW_EF_SEARCH: int = Field(default=100, ge=10, le=1000)
    ROUTING_CANDIDATES: int = Field(default=20, ge=1, le=100)
    ROUTING_MAX_INTENTS: int = Field(default=3, ge=1, le=10)
    # No-match when no candidate has a text match AND the best cosine similarity is below this.
    ROUTING_MIN_SCORE: float = Field(default=0.3, ge=0, le=1)
    REFERENCE_TOP_K: int = Field(default=5, ge=0, le=50)
    FALLBACK_SECTION_TOP_K: int = Field(default=4, ge=1, le=50)

    # --- Ingestion & compilation ---------------------------------------------
    MAX_UPLOAD_BYTES: int = 2_000_000
    INGESTION_SYNC: bool = False
    SUPPORTED_GUIDELINE_FORMATS: str = "txt"
    COMPILE_SINGLE_PASS_MAX_CHARS: int = 30_000
    INTENT_INDEX_EXAMPLES: int = Field(default=8, ge=1, le=50)
    INTENT_EVAL_EXAMPLES: int = Field(default=3, ge=0, le=20)
    AUTO_APPROVE_GUIDELINES: bool = True
    AUTO_APPROVE_MIN_ROUTING_TOP1: float = Field(default=0.8, ge=0, le=1)

    # --- Original-file storage ------------------------------------------------
    # `supabase`: keep the uploaded bytes in a private Supabase Storage bucket.
    # `none`: keep only the decoded text in Postgres. `memory`: tests / offline demos.
    GUIDELINE_STORAGE_BACKEND: Literal["none", "supabase", "memory"] = "none"
    SUPABASE_URL: str | None = (
        None  # API URL from `supabase status` (local: http://127.0.0.1:55421)
    )
    SUPABASE_SERVICE_ROLE_KEY: SecretStr | None = None  # service_role / sb_secret_ key; server only
    GUIDELINE_STORAGE_BUCKET: str = Field(
        default="guideline-files", pattern=r"^[a-z0-9][a-z0-9._-]{1,62}$"
    )
    STORAGE_TIMEOUT_SECONDS: float = 30

    # --- Tools / capabilities -------------------------------------------------
    TOOLS_ENABLED: bool = True
    AGENT_MAX_STEPS: int = Field(default=4, ge=0, le=20)
    TOOL_TIMEOUT_SECONDS: float = 10
    AGENT_TOTAL_TIMEOUT_SECONDS: float = 30
    TOOL_RESULT_MAX_CHARS: int = 4000
    SECRETS_ENCRYPTION_KEY: SecretStr | None = None
    OUTBOUND_ALLOW_PRIVATE_NETWORK: bool = True
    SQL_ADAPTER_ROW_LIMIT: int = 50
    SQL_ADAPTER_TIMEOUT_SECONDS: float = 5

    # --- Optional tracing (sends data to LangSmith when enabled) --------------
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str | None = None

    # --- Optional Sentry error reporting + log capture (off unless DSN set) ---
    SENTRY_DSN: SecretStr | None = None
    SENTRY_ENVIRONMENT: str | None = None  # defaults to APP_ENV
    SENTRY_RELEASE: str | None = None
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.0, ge=0, le=1)
    SENTRY_LOG_EVENT_LEVEL: str = "ERROR"  # log records at/above this become Sentry issues
    SENTRY_LOG_BREADCRUMB_LEVEL: str = "INFO"  # ...and these are attached as breadcrumbs
    SENTRY_ENABLE_LOGS: bool = False  # also ship records to Sentry Logs
    SENTRY_LOGS_LEVEL: str = "INFO"  # minimum level sent to Sentry Logs

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
    def supported_formats(self) -> set[str]:
        return {f.strip().lower().lstrip(".") for f in self.SUPPORTED_GUIDELINE_FORMATS.split(",")}

    @property
    def uses_transaction_pooler(self) -> bool:
        """Supabase transaction pooler (port 6543) cannot use asyncpg prepared statements."""
        return make_url(self.DATABASE_URL).port == 6543


@lru_cache
def get_settings() -> Settings:
    return Settings()
