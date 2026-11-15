"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1 import (
    actions,
    chats,
    companies,
    evals,
    guidelines,
    health,
    integrations,
    internal,
)
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.security import require_admin
from app.resources import build_resources
from app.services.common import AppMetrics

logger = get_logger(__name__)

API_PREFIX = "/api/v1"

DESCRIPTION = """\
AI guidance for customer-support agents, grounded in each company's own versioned support
guidelines. Uploads are **compiled** into playbooks (intents + typed steps), company-wide rules
and reference facts; chats are **routed** to intents with hybrid search (pgvector + full-text) and
a router LLM, can **look up** data in the company's own systems (MCP / HTTP / read-only SQL) and
**propose actions** that the agent approves.

**Auth:** send `X-API-Key`. Agent keys: chats, actions, feedback, read companies/guidelines.
Admin keys: everything (uploads, review, activation, integrations, evals, metrics).
"""

TAGS = [
    {"name": "guidelines", "description": "Versioned uploads, compiled playbooks, review"},
    {"name": "chats", "description": "Guidance for a customer message + history + feedback"},
    {"name": "actions", "description": "Approve / reject proposed writes"},
    {"name": "integrations", "description": "Secrets, MCP servers, capability bindings"},
    {"name": "evals", "description": "Routing eval cases and runs"},
]


def create_app(settings: Settings | None = None, **overrides: Any) -> FastAPI:
    """`overrides` are passed to `build_resources` (generator, selector, compiler, examples,
    tool_model, embeddings_for, adapters) — tests inject deterministic fakes this way."""
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)
    metrics = AppMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = build_resources(settings, metrics=metrics, **overrides)
        app.state.resources = resources
        logger.info(
            "app.started",
            extra={
                "env": settings.APP_ENV,
                "llm_model": settings.LLM_MODEL,
                "router_model": settings.ROUTER_MODEL,
                "compiler_model": settings.COMPILER_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "tools_enabled": settings.TOOLS_ENABLED,
            },
        )
        try:
            yield
        finally:
            await resources.aclose()
            logger.info("app.stopped")

    app = FastAPI(
        title="Support Guidance Assistant",
        version="2.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=TAGS,
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    for module in (health, companies, guidelines, chats, actions, integrations, evals, internal):
        app.include_router(module.router, prefix=API_PREFIX)

    Instrumentator(
        registry=metrics.registry,
        excluded_handlers=["/metrics", f"{API_PREFIX}/health", f"{API_PREFIX}/ready"],
    ).instrument(app).expose(
        app,
        endpoint="/metrics",
        tags=["internal"],
        dependencies=[Depends(require_admin)],
        summary="Prometheus metrics (admin)",
    )
    return app


app = create_app()
