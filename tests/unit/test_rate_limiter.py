"""Chat rate limiting: per agent per company, per company overall, fail-open, and the 429 API."""

from __future__ import annotations

from typing import Any

import pytest
from limits.aio.storage import MemoryStorage

from app.api.deps import guidance_service
from app.core.config import Settings
from app.core.errors import CompanyNotFound, RateLimited
from app.main import create_app
from app.services.common import AppMetrics
from app.services.rate_limiter import (
    ChatRateLimiter,
    DisabledRateLimiter,
    async_storage_uri,
    parse_limits,
)
from tests.conftest import (
    ADMIN_HEADERS,
    AGENT_HEADERS,
    FakeEmbeddingsFactory,
    FakeGuidanceGenerator,
    ScriptedSelector,
    make_settings,
    running_client,
)

pytestmark = pytest.mark.unit


def _limiter(per_agent: str = "2/minute", per_company: str = "", **kw: Any) -> ChatRateLimiter:
    return ChatRateLimiter(
        MemoryStorage(),
        per_agent=parse_limits(per_agent),
        per_company=parse_limits(per_company),
        **kw,
    )


def test_parse_and_storage_uri() -> None:
    assert parse_limits("") == []
    assert [i.amount for i in parse_limits("20/minute;300/hour")] == [20, 300]
    assert async_storage_uri("memory://") == "async+memory://"
    assert async_storage_uri("redis://r:6379/0") == "async+redis://r:6379/0"
    assert async_storage_uri("async+redis://r") == "async+redis://r"


def test_settings_reject_bad_limit() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, CHAT_RATE_LIMIT_PER_AGENT="lots")  # type: ignore[call-arg]


async def test_agent_limit_is_per_agent_and_per_company() -> None:
    limiter = _limiter("2/minute")
    s1 = await limiter.check_chat("bookshop", "alice")
    s2 = await limiter.check_chat("bookshop", "alice")
    assert s1 is not None and s1.remaining == 1 and s1.limit == 2
    assert s2 is not None and s2.remaining == 0
    with pytest.raises(RateLimited) as exc:
        await limiter.check_chat("bookshop", "alice")
    assert exc.value.status_code == 429 and exc.value.code == "RATE_LIMITED"
    assert exc.value.details["scope"] == "agent"
    assert 1 <= exc.value.retry_after <= 60
    assert exc.value.headers == {"Retry-After": str(exc.value.retry_after)}

    # another agent of the same company, and the same agent on another company, are unaffected
    assert await limiter.check_chat("bookshop", "bob") is not None
    assert await limiter.check_chat("vpn-saas", "alice") is not None


async def test_company_limit_caps_all_agents() -> None:
    metrics = AppMetrics()
    limiter = _limiter("10/minute", "3/minute", metrics=metrics)
    for agent in ("a", "b", "c"):
        await limiter.check_chat("bookshop", agent)
    with pytest.raises(RateLimited) as exc:
        await limiter.check_chat("bookshop", "d")
    assert exc.value.details["scope"] == "company"
    assert metrics.rate_limited.labels(company="bookshop", scope="company")._value.get() == 1
    await limiter.check_chat("vpn-saas", "d")  # other tenants unaffected


async def test_agent_rejection_does_not_consume_company_quota() -> None:
    limiter = _limiter("1/minute", "2/minute")
    await limiter.check_chat("bookshop", "alice")
    for _ in range(3):
        with pytest.raises(RateLimited):
            await limiter.check_chat("bookshop", "alice")
    # company used 1 of 2, so another agent still gets through
    await limiter.check_chat("bookshop", "bob")


async def test_multiple_windows_report_tightest() -> None:
    limiter = _limiter("5/minute;3/hour")
    status = await limiter.check_chat("bookshop", "alice")
    assert status is not None and status.limit == 3 and status.remaining == 2
    headers = status.headers()
    assert headers["X-RateLimit-Limit"] == "3" and headers["X-RateLimit-Remaining"] == "2"


class _BrokenStorage(MemoryStorage):
    async def incr(self, *_: Any, **__: Any) -> int:
        raise ConnectionError("redis down")

    async def get(self, *_: Any, **__: Any) -> int:
        raise ConnectionError("redis down")


async def test_storage_failure_fails_open_or_closed() -> None:
    open_ = ChatRateLimiter(
        _BrokenStorage(),
        per_agent=parse_limits("1/minute"),
        per_company=[],
        strategy="fixed-window",
    )
    assert await open_.check_chat("bookshop", "alice") is None
    assert await open_.check_chat("bookshop", "alice") is None

    closed = ChatRateLimiter(
        _BrokenStorage(),
        per_agent=parse_limits("1/minute"),
        per_company=[],
        strategy="fixed-window",
        fail_open=False,
    )
    with pytest.raises(RateLimited):
        await closed.check_chat("bookshop", "alice")


async def test_disabled_limiter() -> None:
    limiter = DisabledRateLimiter()
    for _ in range(100):
        assert await limiter.check_chat("bookshop", "alice") is None


async def test_chat_endpoint_returns_429_with_retry_after() -> None:
    settings = make_settings(
        DATABASE_URL="postgresql+asyncpg://u:p@127.0.0.1:1/never_connected",
        RATE_LIMIT_ENABLED=True,
        RATE_LIMIT_STORAGE_URL="memory://",
        CHAT_RATE_LIMIT_PER_AGENT="1/minute",
        CHAT_RATE_LIMIT_PER_COMPANY="",
    )
    app = create_app(
        settings,
        generator=FakeGuidanceGenerator(),
        selector=ScriptedSelector(),
        embeddings_for=FakeEmbeddingsFactory(),
    )
    handled: list[str] = []

    class _Service:
        async def handle(self, data: Any, request_id: str | None = None) -> Any:
            handled.append(data.company)
            raise CompanyNotFound()

    app.dependency_overrides[guidance_service] = lambda: _Service()
    body = {"company": "bookshop", "customer_message": "Where is my order?"}

    async with running_client(app) as client:
        r = await client.post("/api/v1/chats", json=body, headers=AGENT_HEADERS)
        assert r.status_code == 404  # passed the limiter, reached the service

        r = await client.post("/api/v1/chats", json=body, headers=AGENT_HEADERS)
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "RATE_LIMITED"
        assert r.json()["error"]["details"]["scope"] == "agent"
        assert int(r.headers["Retry-After"]) >= 1
        assert handled == ["bookshop"]  # the rejected request never reached the service

        # a different key (label) has its own quota
        r = await client.post("/api/v1/chats", json=body, headers=ADMIN_HEADERS)
        assert r.status_code == 404

        metrics = await client.get("/metrics", headers=ADMIN_HEADERS)
        assert 'chat_rate_limited_total{company="bookshop",scope="agent"} 1.0' in metrics.text
