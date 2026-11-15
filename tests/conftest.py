"""Shared test setup for every tier: fake embeddings, deterministic compiler/router, fake
generators and tool-calling models, settings factory.

No test needs a Gemini key, hosted Supabase or external network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.messages import AIMessage, BaseMessage

from app.ai.context import GuidanceContext
from app.ai.guidance_chain import GenerationError, GenerationResult
from app.ai.heuristic import HeuristicIntentSelector, extract_entities
from app.ai.router import SelectionResult, sanitize_selection
from app.ai.schemas import (
    ActionArgument,
    GuidanceOutput,
    IntentGuidance,
    IntentSelection,
    PlanStep,
    ProposedActionOut,
    SelectedIntent,
)
from app.core.config import Settings
from app.db.models import EMBEDDING_DIMENSIONS
from app.search.base import IntentCandidate

AGENT_KEY = "agent-test-key"
ADMIN_KEY = "admin-test-key"
AGENT_HEADERS = {"X-API-Key": AGENT_KEY}
ADMIN_HEADERS = {"X-API-Key": ADMIN_KEY}
TEST_FERNET_KEY = Fernet.generate_key().decode()

_TIERS = {"unit": "unit", "integration": "integration", "live": "live"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag every test with its tier marker based on its folder (tests/unit, ...)."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        for folder, marker in _TIERS.items():
            if folder in parts:
                item.add_marker(getattr(pytest.mark, marker))


# --------------------------------------------------------------------- fakes
class FakeEmbeddingsFactory:
    """Same deterministic fake for every model name; records which models were requested."""

    def __init__(self, size: int = EMBEDDING_DIMENSIONS) -> None:
        self.size = size
        self.requested: list[str] = []
        self._emb = DeterministicFakeEmbedding(size=size)

    def __call__(self, model: str) -> Embeddings:
        self.requested.append(model)
        return self._emb


class FakeGuidanceGenerator:
    """Builds a plausible GuidanceOutput from the context it receives (one entry per playbook,
    every step echoed, a made-up reference that must be stripped, and optional proposals)."""

    model_name = "fake-llm"

    def __init__(
        self,
        *,
        propose: list[tuple[str, dict[str, Any]]] | None = None,
        input_tokens: int = 120,
        output_tokens: int = 40,
    ) -> None:
        self.propose = propose or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        context: GuidanceContext,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
    ) -> GenerationResult:
        self.calls.append(
            {"context": context, "customer_message": customer_message, "agent_notes": agent_notes}
        )
        intents = [
            IntentGuidance(
                intent_key=p.key,
                confidence="high",
                steps=[
                    PlanStep(
                        step_ref=f"{p.key}#{s['step_id']}",
                        type=s["type"],
                        instruction=s["text"],
                        performed_by="ai_verified",  # must be overridden by the gate
                    )
                    for s in p.steps
                ],
            )
            for p in context.playbooks
        ]
        refs = [r for p in context.playbooks for r in [f"intent:{x}" for x in p.step_refs()][:2]]
        refs += [f.ref for f in context.facts[:1]]
        refs += [t.ref for t in context.tool_results if t.status == "ok"]
        refs += [s.ref for s in context.sections[:1]]
        refs.append("made-up:9999")
        actions = []
        available = {c["name"] for c in context.write_capabilities}
        for cap, args in self.propose:
            if cap in available or cap.startswith("!"):
                step_ref = next(
                    (
                        f"{p.key}#{s['step_id']}"
                        for p in context.playbooks
                        for s in p.steps
                        if s.get("capability") == cap.lstrip("!")
                    ),
                    None,
                )
                actions.append(
                    ProposedActionOut(
                        capability=cap.lstrip("!"),
                        arguments=[ActionArgument(name=k, value=str(v)) for k, v in args.items()],
                        reason="test proposal",
                        step_ref=step_ref,
                    )
                )
        greeting = context.rules.get("greeting") or "Hi"
        output = GuidanceOutput(
            detected_intents=intents,
            recommended_response=f"{greeting} - thanks for reaching out. Here is what we can do.",
            requires_escalation=False,
            escalation_reason=None,
            references=refs,
            proposed_actions=actions,
            confidence="high",
            missing_information=["Order number"],
        )
        return GenerationResult(output, self.model_name, self.input_tokens, self.output_tokens)


class FailingGuidanceGenerator:
    model_name = "fake-llm"

    def __init__(self, message: str = "provider exploded (503)") -> None:
        self.message = message

    async def generate(self, **_: Any) -> GenerationResult:
        raise GenerationError(self.message)


class ScriptedSelector:
    """Router fake: `script(message, candidates) -> list[keys] | None` (None = heuristic)."""

    model_name = "fake-router"

    def __init__(
        self, script: Callable[[str, list[IntentCandidate]], list[str] | None] | None = None
    ) -> None:
        self._script = script
        self._fallback = HeuristicIntentSelector()

    async def select(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        candidates: list[IntentCandidate],
        max_intents: int,
    ) -> SelectionResult:
        keys = self._script(customer_message, candidates) if self._script else None
        if keys is None:
            return await self._fallback.select(
                company_name=company_name,
                customer_message=customer_message,
                agent_notes=agent_notes,
                candidates=candidates,
                max_intents=max_intents,
            )
        sel = IntentSelection(
            intents=[SelectedIntent(key=k, confidence="high") for k in keys],
            entities=extract_entities(customer_message, agent_notes),
        )
        return SelectionResult(
            sanitize_selection(sel, candidates, max_intents), self.model_name, 30, 5
        )


class ScriptedToolModel:
    """Tool-calling fake: returns the scripted AIMessages in order, then 'DONE'."""

    model_name = "fake-tool-llm"

    def __init__(self, script: list[AIMessage] | None = None, *, loop_forever: bool = False):
        self.script = list(script or [])
        self.loop_forever = loop_forever
        self.calls: list[list[BaseMessage]] = []
        self.tools_seen: list[list[str]] = []

    async def step(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        self.calls.append(list(messages))
        self.tools_seen.append([t["function"]["name"] for t in tools])
        if self.loop_forever and tools:
            name = tools[0]["function"]["name"]
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": {"email": "x@example.com"}, "id": "loop"}],
            )
        if self.script:
            return self.script.pop(0)
        return AIMessage(content="DONE")


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ------------------------------------------------------------------ settings
def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "APP_ENV": "test",
        "INGESTION_SYNC": True,
        "AGENT_API_KEYS": AGENT_KEY,
        "ADMIN_API_KEYS": f"tester:{ADMIN_KEY}",
        "EMBEDDING_MODEL": "fake-embedding",
        "LLM_MODEL": "fake-llm",
        "ROUTER_MODEL": "fake-router",
        "COMPILER_MODEL": "heuristic-compiler",
        "LOG_LEVEL": "WARNING",
        "COMPANY_CACHE_TTL_SECONDS": 30,
        "AUTO_APPROVE_MIN_ROUTING_TOP1": 0.5,
        "SECRETS_ENCRYPTION_KEY": TEST_FERNET_KEY,
        "TOOL_TIMEOUT_SECONDS": 2,
        "LANGSMITH_TRACING": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@asynccontextmanager
async def running_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app's lifespan and yield an httpx client bound to it (no network)."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingsFactory:
    return FakeEmbeddingsFactory()


@pytest.fixture
def fake_generator() -> FakeGuidanceGenerator:
    return FakeGuidanceGenerator()
