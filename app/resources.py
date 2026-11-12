"""Long-lived collaborators created once at startup (lifespan) and shared by all requests."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.ai.guidance_chain import (
    GenerationError,
    GenerationResult,
    GuidanceGenerator,
    LangChainGuidanceGenerator,
)
from app.ai.providers import EmbeddingsRegistry, get_chat_model
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.session import create_engine, create_sessionmaker
from app.services.chat_history_service import ChatHistoryService
from app.services.common import (
    AppMetrics,
    CompanyContextCache,
    UowFactory,
    sql_uow_factory,
)
from app.services.company_service import CompanyService
from app.services.guidance_service import GuidanceService
from app.services.ingestion_service import IngestionService
from app.services.metrics_service import MetricsService
from app.vectorstore.base import EmbeddingsFactory, RetrievedChunk, VectorStoreGateway
from app.vectorstore.factory import create_vector_store

logger = get_logger(__name__)


class UnavailableGenerator:
    """Used when the LLM client can't be built (e.g. no API key): chats fail with LLM_ERROR."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self._reason = reason

    async def generate(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        raise GenerationError(f"LLM is not configured: {self._reason}")


def default_generator(settings: Settings) -> GuidanceGenerator:
    try:
        llm = get_chat_model(settings)
    except Exception as exc:
        logger.warning("llm.unavailable", extra={"error": f"{type(exc).__name__}: {exc}"})
        return UnavailableGenerator(settings.LLM_MODEL, str(exc))
    return LangChainGuidanceGenerator(llm, settings.LLM_MODEL, settings.LLM_TIMEOUT_SECONDS)


@dataclass
class Resources:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    vector_store: VectorStoreGateway
    generator: GuidanceGenerator
    cache: CompanyContextCache
    metrics: AppMetrics = field(default_factory=AppMetrics)

    @property
    def uow_factory(self) -> UowFactory:
        return sql_uow_factory(self.sessionmaker)

    def company_service(self) -> CompanyService:
        return CompanyService(self.uow_factory)

    def ingestion_service(self) -> IngestionService:
        return IngestionService(
            self.uow_factory, self.vector_store, self.settings, self.cache, self.metrics
        )

    def guidance_service(self) -> GuidanceService:
        return GuidanceService(
            self.uow_factory,
            self.vector_store,
            self.generator,
            self.settings,
            self.cache,
            self.metrics,
        )

    def chat_history_service(self) -> ChatHistoryService:
        return ChatHistoryService(self.uow_factory)

    def metrics_service(self) -> MetricsService:
        return MetricsService(self.uow_factory)

    async def aclose(self) -> None:
        await self.engine.dispose()


def build_resources(
    settings: Settings,
    *,
    vector_store: VectorStoreGateway | None = None,
    generator: GuidanceGenerator | None = None,
    embeddings_for: EmbeddingsFactory | None = None,
    metrics: AppMetrics | None = None,
) -> Resources:
    engine = create_engine(settings)
    embeddings_for = embeddings_for or EmbeddingsRegistry(settings)
    return Resources(
        settings=settings,
        engine=engine,
        sessionmaker=create_sessionmaker(engine),
        vector_store=vector_store or create_vector_store(settings, embeddings_for),
        generator=generator or default_generator(settings),
        cache=CompanyContextCache(settings.COMPANY_CACHE_TTL_SECONDS),
        metrics=metrics or AppMetrics(),
    )
