"""Plumbing shared by services: unit-of-work typing, company context cache, Prometheus metrics."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from prometheus_client import CollectorRegistry, Counter, Histogram
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.capabilities.adapters.base import BindingSnapshot
from app.repositories.uow import SqlUnitOfWork


class UnitOfWork(Protocol):
    session: Any
    companies: Any
    versions: Any
    chats: Any
    metrics: Any
    playbooks: Any
    capabilities: Any
    evals: Any

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


UowFactory = Callable[[], AbstractAsyncContextManager[UnitOfWork]]


def sql_uow_factory(sessionmaker: async_sessionmaker[AsyncSession]) -> UowFactory:
    @asynccontextmanager
    async def _factory() -> Any:
        async with sessionmaker() as session:
            yield SqlUnitOfWork(session)

    return _factory


@dataclass(frozen=True)
class CompanyContext:
    """What a chat request needs to know about a company, cached briefly per process."""

    company_id: uuid.UUID
    slug: str
    name: str
    active_version_id: uuid.UUID | None
    active_version_number: int | None
    active_version_usable: bool  # ready + approved
    embedding_model: str | None
    embedding_dimensions: int | None
    rules: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, BindingSnapshot] = field(default_factory=dict)  # live only


class CompanyContextCache:
    """Tiny in-process TTL cache: slug -> CompanyContext. Invalidated on upload/activation."""

    def __init__(self, ttl_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl_seconds
        self._clock = clock
        self._data: dict[str, tuple[float, CompanyContext]] = {}
        self._lock = threading.Lock()

    def get(self, slug: str) -> CompanyContext | None:
        with self._lock:
            item = self._data.get(slug)
            if item is None:
                return None
            expires, ctx = item
            if self._clock() >= expires:
                del self._data[slug]
                return None
            return ctx

    def set(self, ctx: CompanyContext) -> None:
        if self._ttl <= 0:
            return
        with self._lock:
            self._data[ctx.slug] = (self._clock() + self._ttl, ctx)

    def invalidate(self, slug: str) -> None:
        with self._lock:
            self._data.pop(slug, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class AppMetrics:
    """Custom Prometheus metrics, bound to a registry (one per app instance)."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.chat_requests = Counter(
            "chat_requests_total",
            "Chat guidance requests",
            ["company", "status"],
            registry=self.registry,
        )
        self.chat_latency = Histogram(
            "chat_latency_seconds",
            "End-to-end chat guidance latency",
            buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "llm_tokens_total", "LLM tokens used", ["role", "direction"], registry=self.registry
        )
        self.ingestions = Counter(
            "ingestion_total", "Guideline ingestions", ["status"], registry=self.registry
        )
        self.compile_seconds = Histogram(
            "ingestion_compile_seconds",
            "Guideline compilation duration",
            buckets=(1, 2, 5, 10, 20, 40, 80, 160),
            registry=self.registry,
        )
        self.no_match = Counter(
            "routing_no_match_total",
            "Chats with no matching intent",
            ["company"],
            registry=self.registry,
        )
        self.intents_selected = Histogram(
            "routing_intents_selected",
            "Intents selected per chat",
            buckets=(0, 1, 2, 3, 5),
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "tool_calls_total",
            "Capability calls",
            ["capability", "adapter", "status"],
            registry=self.registry,
        )
        self.tool_latency = Histogram(
            "tool_latency_seconds",
            "Capability call latency",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.proposed_actions = Counter(
            "proposed_actions_total",
            "Proposed actions by status",
            ["status"],
            registry=self.registry,
        )
