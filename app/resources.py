"""Long-lived collaborators created once at startup (lifespan) and shared by all requests.

Every AI collaborator can be injected (tests use deterministic fakes; no Gemini key needed).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.ai.compiler import CompileResult, LLMPlaybookCompiler, PlaybookCompiler
from app.ai.examples import ExampleGenerator, ExampleResult, LLMExampleGenerator
from app.ai.guidance_chain import (
    GenerationError,
    GenerationResult,
    GuidanceGenerator,
    LangChainGuidanceGenerator,
    LangChainToolCallingModel,
    ToolCallingModel,
)
from app.ai.providers import EmbeddingsRegistry, LLMCallError, Role, get_chat_model, model_name_for
from app.ai.router import IntentSelector, LLMIntentSelector, SelectionResult
from app.capabilities.adapters.base import CapabilityAdapter
from app.capabilities.adapters.http import HttpAdapter
from app.capabilities.adapters.mcp import McpAdapter
from app.capabilities.adapters.mock import MockAdapter
from app.capabilities.adapters.sql_readonly import SqlReadOnlyAdapter
from app.capabilities.gateway import CapabilityGateway
from app.capabilities.secrets import FernetSecretStore, SecretStore
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models import EMBEDDING_DIMENSIONS
from app.db.session import create_engine, create_sessionmaker
from app.harness.graph import build_graph
from app.harness.nodes import HarnessDeps, HarnessNodes
from app.search.base import SearchGateway
from app.search.pgvector import PgVectorSearchGateway
from app.services.action_service import ActionService
from app.services.chat_history_service import ChatHistoryService
from app.services.common import AppMetrics, CompanyContextCache, UowFactory, sql_uow_factory
from app.services.company_service import CompanyService
from app.services.eval_service import EvalService
from app.services.guidance_service import GuidanceService
from app.services.ingestion_service import IngestionService
from app.services.integration_service import IntegrationService
from app.services.metrics_service import MetricsService
from app.services.rate_limiter import ChatRateLimiter, DisabledRateLimiter
from app.services.review_service import ReviewService
from app.storage.base import FileStore, StorageUnavailable
from app.storage.factory import build_file_store

logger = get_logger(__name__)
EmbeddingsFactory = Callable[[str], Embeddings]


class _Unavailable:
    """Stands in for an LLM collaborator that couldn't be built (e.g. no API key)."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self._reason = reason

    def _fail(self) -> Exception:
        return LLMCallError(f"LLM is not configured: {self._reason}")

    async def generate(self, **_: Any) -> GenerationResult:
        raise GenerationError(str(self._fail()))

    async def select(self, **_: Any) -> SelectionResult:
        raise self._fail()

    async def compile(self, *_: Any, **__: Any) -> CompileResult:
        raise self._fail()

    async def step(self, *_: Any, **__: Any) -> Any:
        raise self._fail()


class _UnavailableExamples:
    async def generate(self, *_: Any, **__: Any) -> ExampleResult:
        raise LLMCallError("LLM is not configured")


def _llm(settings: Settings, role: Role) -> BaseChatModel | None:
    try:
        return get_chat_model(settings, role)
    except Exception as exc:
        logger.warning("llm.unavailable", extra={"role": role, "error": f"{type(exc).__name__}"})
        return None


def default_adapters(settings: Settings) -> dict[str, CapabilityAdapter]:
    return {
        "mock": MockAdapter(),
        "http": HttpAdapter(),
        "sql_readonly": SqlReadOnlyAdapter(
            row_limit=settings.SQL_ADAPTER_ROW_LIMIT,
            statement_timeout_s=settings.SQL_ADAPTER_TIMEOUT_SECONDS,
        ),
        "mcp": McpAdapter(),
    }


def configure_tracing(settings: Settings) -> None:
    """LangSmith tracing is opt-in: it sends customer messages and tool data to LangSmith."""
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY.get_secret_value()
        if settings.LANGSMITH_PROJECT:
            os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        logger.warning("tracing.enabled", extra={"provider": "langsmith"})
    else:
        os.environ["LANGSMITH_TRACING"] = "false"


@dataclass
class Resources:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    embeddings_for: EmbeddingsFactory
    search: SearchGateway
    generator: GuidanceGenerator
    selector: IntentSelector
    compiler: PlaybookCompiler
    examples: ExampleGenerator
    tool_model: ToolCallingModel | None
    gateway: CapabilityGateway
    secret_store: SecretStore
    cache: CompanyContextCache
    file_store: FileStore | None = None
    metrics: AppMetrics = field(default_factory=AppMetrics)
    rate_limiter: ChatRateLimiter | DisabledRateLimiter = field(default_factory=DisabledRateLimiter)
    graph: Any = None

    @property
    def uow_factory(self) -> UowFactory:
        return sql_uow_factory(self.sessionmaker)

    def company_service(self) -> CompanyService:
        return CompanyService(self.uow_factory)

    def eval_service(self) -> EvalService:
        return EvalService(
            self.uow_factory, self.search, self.embeddings_for, self.selector, self.settings
        )

    def ingestion_service(self) -> IngestionService:
        return IngestionService(
            self.uow_factory,
            self.settings,
            self.cache,
            self.metrics,
            compiler=self.compiler,
            examples=self.examples,
            embeddings_for=self.embeddings_for,
            evals=self.eval_service(),
            file_store=self.file_store,
        )

    def review_service(self) -> ReviewService:
        return ReviewService(self.uow_factory, self.cache, self.file_store)

    def integration_service(self) -> IntegrationService:
        return IntegrationService(
            self.uow_factory,
            self.gateway,
            self.secret_store,
            self.cache,
            self.metrics,
            self.settings,
        )

    def guidance_service(self) -> GuidanceService:
        return GuidanceService(
            self.uow_factory,
            self.graph,
            self.settings,
            self.cache,
            self.integration_service().live_bindings,
        )

    def action_service(self) -> ActionService:
        return ActionService(
            self.uow_factory, self.gateway, self.integration_service(), self.metrics
        )

    def chat_history_service(self) -> ChatHistoryService:
        return ChatHistoryService(self.uow_factory)

    def metrics_service(self) -> MetricsService:
        return MetricsService(self.uow_factory)

    async def startup(self) -> None:
        """Async setup after construction. A storage outage must not stop the API from serving
        chats, so bucket creation failures are logged (uploads then fail with STORAGE_ERROR)."""
        if self.file_store is not None:
            try:
                await self.file_store.ensure_bucket()
            except StorageUnavailable as exc:
                logger.error(
                    "storage.bucket_unavailable",
                    extra={"bucket": self.file_store.bucket, "error": str(exc)[:300]},
                )

    async def aclose(self) -> None:
        if self.file_store is not None:
            await self.file_store.aclose()
        await self.engine.dispose()


def build_rate_limiter(
    settings: Settings, metrics: AppMetrics | None = None
) -> ChatRateLimiter | DisabledRateLimiter:
    if not settings.RATE_LIMIT_ENABLED:
        return DisabledRateLimiter()
    return ChatRateLimiter.from_url(
        settings.RATE_LIMIT_STORAGE_URL,
        per_agent=settings.CHAT_RATE_LIMIT_PER_AGENT,
        per_company=settings.CHAT_RATE_LIMIT_PER_COMPANY,
        strategy=settings.RATE_LIMIT_STRATEGY,
        fail_open=settings.RATE_LIMIT_FAIL_OPEN,
        metrics=metrics,
    )


def build_resources(
    settings: Settings,
    *,
    generator: GuidanceGenerator | None = None,
    selector: IntentSelector | None = None,
    compiler: PlaybookCompiler | None = None,
    examples: ExampleGenerator | None = None,
    tool_model: ToolCallingModel | None = None,
    embeddings_for: EmbeddingsFactory | None = None,
    adapters: dict[str, CapabilityAdapter] | None = None,
    metrics: AppMetrics | None = None,
    file_store: FileStore | None = None,
) -> Resources:
    if settings.EMBEDDING_DIMENSIONS != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"EMBEDDING_DIMENSIONS={settings.EMBEDDING_DIMENSIONS} but the schema uses "
            f"vector({EMBEDDING_DIMENSIONS}); changing it needs a migration and re-embedding"
        )
    configure_tracing(settings)
    engine = create_engine(settings)
    sessionmaker = create_sessionmaker(engine)
    embeddings_for = embeddings_for or EmbeddingsRegistry(settings)
    timeout = settings.LLM_TIMEOUT_SECONDS

    def role_or_unavailable(role: Role) -> tuple[BaseChatModel | None, Any]:
        llm = _llm(settings, role)
        return llm, (
            None if llm is not None else _Unavailable(model_name_for(settings, role), "no client")
        )

    if generator is None:
        llm, fallback = role_or_unavailable("generator")
        generator = (
            LangChainGuidanceGenerator(llm, settings.LLM_MODEL, timeout) if llm else fallback
        )
        if tool_model is None and llm is not None:
            tool_model = LangChainToolCallingModel(llm, settings.LLM_MODEL, timeout)
    if selector is None:
        llm, fallback = role_or_unavailable("router")
        selector = (
            LLMIntentSelector(llm, settings.ROUTER_MODEL, timeout=timeout) if llm else fallback
        )
    if compiler is None or examples is None:
        llm, fallback = role_or_unavailable("compiler")
        if compiler is None:
            compiler = (
                LLMPlaybookCompiler(
                    llm,
                    settings.COMPILER_MODEL,
                    timeout=timeout * 4,
                    single_pass_max_chars=settings.COMPILE_SINGLE_PASS_MAX_CHARS,
                )
                if llm
                else fallback
            )
        if examples is None:
            examples = (
                LLMExampleGenerator(llm, timeout=timeout * 3) if llm else _UnavailableExamples()
            )
    assert generator is not None and selector is not None
    assert compiler is not None and examples is not None

    all_adapters = default_adapters(settings)
    all_adapters.update(adapters or {})
    gateway = CapabilityGateway(
        all_adapters,
        timeout=settings.TOOL_TIMEOUT_SECONDS,
        max_result_chars=settings.TOOL_RESULT_MAX_CHARS,
        allow_private_network=settings.OUTBOUND_ALLOW_PRIVATE_NETWORK,
    )
    key = (
        settings.SECRETS_ENCRYPTION_KEY.get_secret_value()
        if settings.SECRETS_ENCRYPTION_KEY
        else None
    )
    resources = Resources(
        settings=settings,
        engine=engine,
        sessionmaker=sessionmaker,
        embeddings_for=embeddings_for,
        search=PgVectorSearchGateway(
            sessionmaker,
            ts_config=settings.TEXT_SEARCH_CONFIG,
            ef_search=settings.HNSW_EF_SEARCH,
            examples_per_intent=settings.INTENT_INDEX_EXAMPLES + 2,
        ),
        generator=generator,
        selector=selector,
        compiler=compiler,
        examples=examples,
        tool_model=tool_model,
        gateway=gateway,
        secret_store=FernetSecretStore(key),
        cache=CompanyContextCache(settings.COMPANY_CACHE_TTL_SECONDS),
        file_store=file_store if file_store is not None else build_file_store(settings),
        metrics=metrics or AppMetrics(),
    )
    resources.rate_limiter = build_rate_limiter(settings, resources.metrics)
    integrations = resources.integration_service()
    resources.graph = build_graph(
        HarnessNodes(
            HarnessDeps(
                uow=resources.uow_factory,
                search=resources.search,
                embeddings_for=embeddings_for,
                selector=selector,
                generator=generator,
                tool_model=tool_model,
                gateway=gateway,
                secret_getter=integrations.secret_getter,
                settings=settings,
                metrics=resources.metrics,
            )
        )
    )
    return resources
