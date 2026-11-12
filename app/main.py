"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.ai.guidance_chain import GuidanceGenerator
from app.api.v1 import chats, companies, guidelines, health, internal
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.security import require_admin
from app.resources import build_resources
from app.services.common import AppMetrics
from app.vectorstore.base import EmbeddingsFactory, VectorStoreGateway

logger = get_logger(__name__)

API_PREFIX = "/api/v1"

DESCRIPTION = """\
AI guidance for customer-support agents, grounded in each company's own versioned support
guidelines (RAG over the **active** guideline version).

**Auth:** send `X-API-Key`. Agent keys can create/read chats and read companies/guidelines;
admin keys can do everything (uploads, activation, deletion, metrics).
"""


def create_app(
    settings: Settings | None = None,
    *,
    vector_store: VectorStoreGateway | None = None,
    generator: GuidanceGenerator | None = None,
    embeddings_for: EmbeddingsFactory | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)
    metrics = AppMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = build_resources(
            settings,
            vector_store=vector_store,
            generator=generator,
            embeddings_for=embeddings_for,
            metrics=metrics,
        )
        app.state.resources = resources
        logger.info(
            "app.started",
            extra={
                "env": settings.APP_ENV,
                "vector_backend": settings.VECTOR_BACKEND,
                "llm_model": settings.LLM_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
            },
        )
        try:
            yield
        finally:
            await resources.aclose()
            logger.info("app.stopped")

    app = FastAPI(
        title="Support Guidance Assistant",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(companies.router, prefix=API_PREFIX)
    app.include_router(guidelines.router, prefix=API_PREFIX)
    app.include_router(chats.router, prefix=API_PREFIX)
    app.include_router(internal.router, prefix=API_PREFIX)

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
