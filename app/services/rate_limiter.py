"""Rate limiting for `POST /chats`, per agent per company and per company overall.

Built on the `limits` library (async API). Storage is chosen by URL: `memory://` for local runs
and tests (per process), `redis://…` / `rediss://…` in production so every API replica shares
the same counters. No FastAPI imports: the router calls `check_chat` and maps the result.

Keys:
- agent:   ("chat", "agent", <company slug>, <API-key label>)
- company: ("chat", "company", <company slug>)

If the storage is unreachable the limiter fails open (or closed, per `RATE_LIMIT_FAIL_OPEN`) so a
Redis outage does not take the chat API down with it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

from limits import RateLimitItem, parse_many
from limits.aio.storage import Storage
from limits.aio.strategies import (
    FixedWindowRateLimiter,
    MovingWindowRateLimiter,
    RateLimiter,
    SlidingWindowCounterRateLimiter,
)
from limits.storage import storage_from_string

from app.core.errors import RateLimited
from app.core.logging import get_logger
from app.services.common import AppMetrics

logger = get_logger(__name__)

Strategy = Literal["sliding-window-counter", "moving-window", "fixed-window"]
Scope = Literal["agent", "company"]

_STRATEGIES: dict[str, type[RateLimiter]] = {
    "sliding-window-counter": SlidingWindowCounterRateLimiter,
    "moving-window": MovingWindowRateLimiter,
    "fixed-window": FixedWindowRateLimiter,
}


def parse_limits(spec: str) -> list[RateLimitItem]:
    """`"20/minute;300/hour"` -> items. Empty string -> no limit."""
    spec = spec.strip()
    return list(parse_many(spec)) if spec else []


def async_storage_uri(uri: str) -> str:
    """Accept `memory://` / `redis://…` and return the async variant `limits` expects."""
    return uri if uri.startswith("async+") else f"async+{uri}"


@dataclass(frozen=True)
class RateLimitStatus:
    """The tightest agent-level limit after a successful hit (for `X-RateLimit-*` headers)."""

    limit: int
    remaining: int
    reset_at: float  # epoch seconds

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(self.remaining, 0)),
            "X-RateLimit-Reset": str(math.ceil(self.reset_at)),
        }


class ChatRateLimiter:
    def __init__(
        self,
        storage: Storage,
        *,
        per_agent: list[RateLimitItem],
        per_company: list[RateLimitItem],
        strategy: Strategy = "sliding-window-counter",
        fail_open: bool = True,
        metrics: AppMetrics | None = None,
    ) -> None:
        self._storage = storage
        self._limiter = _STRATEGIES[strategy](storage)
        self._per_agent = per_agent
        self._per_company = per_company
        self._fail_open = fail_open
        self._metrics = metrics

    @classmethod
    def from_url(
        cls,
        storage_url: str,
        *,
        per_agent: str,
        per_company: str,
        strategy: Strategy = "sliding-window-counter",
        fail_open: bool = True,
        metrics: AppMetrics | None = None,
    ) -> ChatRateLimiter:
        storage = storage_from_string(async_storage_uri(storage_url), wrap_exceptions=True)
        assert isinstance(storage, Storage)
        return cls(
            storage,
            per_agent=parse_limits(per_agent),
            per_company=parse_limits(per_company),
            strategy=strategy,
            fail_open=fail_open,
            metrics=metrics,
        )

    async def check_chat(self, company_slug: str, agent_label: str) -> RateLimitStatus | None:
        """Consume one chat from the agent's and the company's quotas.

        Raises `RateLimited` (429) when either is exhausted. The company quota is only tested
        (not consumed) before the agent hit, so a request rejected at one level doesn't use up
        quota at the other. Returns the agent-level status, or None when there is no agent limit
        or the storage failed open.
        """
        agent_ids = ("chat", "agent", company_slug, agent_label)
        company_ids = ("chat", "company", company_slug)
        try:
            for item in self._per_company:
                if not await self._limiter.test(item, *company_ids):
                    await self._reject("company", item, company_ids, company_slug)
            for item in self._per_agent:
                if not await self._limiter.hit(item, *agent_ids):
                    await self._reject("agent", item, agent_ids, company_slug)
            for item in self._per_company:
                if not await self._limiter.hit(item, *company_ids):
                    await self._reject("company", item, company_ids, company_slug)
            return await self._status(agent_ids)
        except RateLimited:
            raise
        except Exception as exc:
            logger.warning(
                "rate_limit.storage_error",
                extra={"error_type": type(exc).__name__, "fail_open": self._fail_open},
            )
            if self._fail_open:
                return None
            raise RateLimited("Rate limiter unavailable; try again shortly", retry_after=5) from exc

    async def _reject(
        self, scope: Scope, item: RateLimitItem, ids: tuple[str, ...], company_slug: str
    ) -> None:
        stats = await self._limiter.get_window_stats(item, *ids)
        retry_after = max(1, math.ceil(stats.reset_time - time.time()))
        if self._metrics is not None:
            self._metrics.rate_limited.labels(company=company_slug, scope=scope).inc()
        logger.info(
            "rate_limit.exceeded",
            extra={"company": company_slug, "scope": scope, "limit": str(item)},
        )
        raise RateLimited(
            f"Too many chat requests ({scope} limit {item}); retry in {retry_after}s",
            retry_after=retry_after,
            details={"scope": scope, "limit": str(item), "retry_after_seconds": retry_after},
        )

    async def _status(self, agent_ids: tuple[str, ...]) -> RateLimitStatus | None:
        tightest: RateLimitStatus | None = None
        for item in self._per_agent:
            stats = await self._limiter.get_window_stats(item, *agent_ids)
            status = RateLimitStatus(item.amount, stats.remaining, stats.reset_time)
            if tightest is None or status.remaining < tightest.remaining:
                tightest = status
        return tightest

    async def healthcheck(self) -> bool:
        try:
            return bool(await self._storage.check())
        except Exception:
            return False

    async def reset(self) -> None:
        """Clear all counters (tests / admin tooling)."""
        await self._storage.reset()


class DisabledRateLimiter:
    """Used when `RATE_LIMIT_ENABLED=false`."""

    async def check_chat(self, company_slug: str, agent_label: str) -> RateLimitStatus | None:
        return None

    async def healthcheck(self) -> bool:
        return True

    async def reset(self) -> None:
        return None
