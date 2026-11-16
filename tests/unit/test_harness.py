"""U7 + U15 — the LangGraph harness end to end with in-memory fakes (no DB, no network):
routing (single, multi, no-match), citation stripping, performed_by gating, prefetch, the
bounded tool loop, proposals + policy, and persistence of failures."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from app.capabilities.adapters.base import BindingSnapshot
from app.capabilities.adapters.mock import MockAdapter
from app.capabilities.gateway import CapabilityGateway
from app.core.errors import LLMError, NoActiveGuidelines, RetrievalError
from app.db.models import Chat, Intent, ProposedAction, ToolCall
from app.harness.graph import build_graph
from app.harness.nodes import HarnessDeps, HarnessNodes
from app.search.base import FactHit, IntentCandidate, Query, SectionHit
from app.services.common import AppMetrics, CompanyContext, CompanyContextCache
from app.services.guidance_service import ChatRequestData, GuidanceService
from tests.conftest import (
    FailingGuidanceGenerator,
    FakeEmbeddingsFactory,
    FakeGuidanceGenerator,
    ScriptedSelector,
    ScriptedToolModel,
    make_settings,
    tool_call,
)

COMPANY_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()

INTENTS = {
    "shipping_issue": Intent(
        id=uuid.uuid4(),
        company_id=COMPANY_ID,
        version_id=VERSION_ID,
        key="shipping_issue",
        name="Shipping Issues",
        description="Late or lost parcels",
        conditions=[],
        steps=[
            {"step_id": "s1", "type": "say", "text": "Apologize for the delay"},
            {"step_id": "s2", "type": "lookup", "text": "Check order", "capability": "order.get"},
            {"step_id": "s3", "type": "escalate", "text": "Escalate if lost"},
        ],
        reply_template="Hello [Name],",
        placeholders=["Name"],
        source_section_ids=[],
    ),
    "billing_issue": Intent(
        id=uuid.uuid4(),
        company_id=COMPANY_ID,
        version_id=VERSION_ID,
        key="billing_issue",
        name="Billing",
        description="Charges and refunds",
        conditions=[],
        steps=[
            {
                "step_id": "s1",
                "type": "lookup",
                "text": "Check subscription",
                "capability": "subscription.get",
            },
            {
                "step_id": "s2",
                "type": "action",
                "text": "Refund duplicate charge",
                "capability": "refund.create",
            },
            {"step_id": "s3", "type": "ask", "text": "Ask for the invoice"},
        ],
        reply_template=None,
        placeholders=[],
        source_section_ids=[],
    ),
}


class FakeSearch:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.section_calls = 0

    async def search_intents(self, q: Query, k: int) -> list[IntentCandidate]:
        if self.fail:
            raise RuntimeError("search down")
        assert q.company_id == COMPANY_ID and q.version_id == VERSION_ID
        return [
            IntentCandidate(i.id, i.key, i.name, i.description, 0.03, 0.2, 0.1)
            for i in INTENTS.values()
        ]

    async def search_facts(self, q: Query, k: int) -> list[FactHit]:
        return [FactHit(uuid.uuid4(), "Express", "Express shipping takes 1-2 days", 0.02)]

    async def search_sections(self, q: Query, k: int) -> list[SectionHit]:
        self.section_calls += 1
        return [SectionHit(uuid.uuid4(), "S9", "Misc", "Other policies")]

    async def healthcheck(self) -> bool:
        return True


class Store:
    def __init__(self) -> None:
        self.chats: list[Chat] = []
        self.actions: list[ProposedAction] = []
        self.tool_calls: list[ToolCall] = []


def uow_factory(store: Store) -> Any:
    class Playbooks:
        async def intents_by_keys(self, version_id: uuid.UUID, keys: list[str]) -> list[Intent]:
            return [INTENTS[k] for k in keys if k in INTENTS]

        async def sections_by_ids(self, version_id: uuid.UUID, ids: list[Any]) -> list[Any]:
            return []

    class Chats:
        async def add(self, chat: Chat) -> Chat:
            store.chats.append(chat)
            return chat

    class Caps:
        async def add_actions(self, rows: list[ProposedAction]) -> None:
            store.actions.extend(rows)

        async def add_tool_calls(self, rows: list[ToolCall]) -> None:
            store.tool_calls.extend(rows)

    class Uow:
        playbooks, chats, capabilities = Playbooks(), Chats(), Caps()

        async def commit(self) -> None: ...

    @asynccontextmanager
    async def factory() -> AsyncIterator[Uow]:
        yield Uow()

    return factory


def live(capability: str, policy: dict[str, Any] | None = None) -> BindingSnapshot:
    return BindingSnapshot(
        id=uuid.uuid4(),
        company_id=COMPANY_ID,
        capability=capability,
        adapter_type="mock",
        config={"store": "bookshop_orders" if "order" in capability else "vpn_accounts"},
        policy=policy or {},
        status="live",
    )


def build(
    *,
    selector: Any = None,
    generator: Any = None,
    tool_model: Any = None,
    search: FakeSearch | None = None,
    bindings: dict[str, BindingSnapshot] | None = None,
    usable: bool = True,
    **settings: Any,
) -> tuple[GuidanceService, Store, FakeSearch]:
    s = make_settings(**settings)
    store = Store()
    search = search or FakeSearch()
    cache = CompanyContextCache(60)
    cache.set(
        CompanyContext(
            company_id=COMPANY_ID,
            slug="acme",
            name="Acme",
            active_version_id=VERSION_ID,
            active_version_number=3,
            active_version_usable=usable,
            embedding_model="fake-embedding",
            embedding_dimensions=768,
            rules={"greeting": "Hello [Name],", "sign_off": "Best, Acme"},
            bindings=bindings or {},
        )
    )
    gateway = CapabilityGateway(
        {"mock": MockAdapter()}, timeout=1, max_result_chars=4000, allow_private_network=True
    )

    def secrets(_: uuid.UUID) -> Any:
        async def get(name: str) -> str:
            return "x"

        return get

    nodes = HarnessNodes(
        HarnessDeps(
            uow=uow_factory(store),
            search=search,
            embeddings_for=FakeEmbeddingsFactory(),
            selector=selector or ScriptedSelector(lambda m, c: ["shipping_issue"]),
            generator=generator or FakeGuidanceGenerator(),
            tool_model=tool_model,
            gateway=gateway,
            secret_getter=secrets,
            settings=s,
            metrics=AppMetrics(),
        )
    )

    async def no_bindings(_: uuid.UUID) -> dict[str, BindingSnapshot]:
        return {}

    service = GuidanceService(uow_factory(store), build_graph(nodes), s, cache, no_bindings)
    return service, store, search


def req(msg: str = "Where is my parcel?", notes: str | None = None) -> ChatRequestData:
    return ChatRequestData(company="acme", customer_message=msg, agent_notes=notes)


async def test_single_intent_success_and_gating() -> None:
    service, store, _ = build()
    out = await service.handle(req(), request_id="rid-1")
    g = out.guidance
    assert [i.intent_key for i in g.detected_intents] == ["shipping_issue"]
    steps = {s.step_ref: s for s in g.detected_intents[0].steps}
    assert steps["shipping_issue#s1"].performed_by == "ai"
    # the fake claims ai_verified; no tool ran, so the gate overrides it
    assert steps["shipping_issue#s2"].performed_by == "agent"
    assert steps["shipping_issue#s3"].performed_by == "agent"
    assert "made-up:9999" not in g.references
    assert any(r.startswith("fact:") for r in g.references)
    assert g.recommended_response.startswith("Hello [Name],")  # company rules reached the LLM
    chat = store.chats[0]
    assert chat.status == "success" and chat.primary_intent == "shipping_issue"
    assert chat.routing and chat.routing["selected"][0]["key"] == "shipping_issue"
    assert len(chat.routing["candidates"]) == 2 and chat.routing["no_match"] is False
    assert chat.references and chat.references["dropped"] == ["made-up:9999"]
    assert chat.input_tokens == 120 and chat.router_input_tokens == 30
    assert chat.latency_ms_routing is not None and chat.latency_ms_llm is not None
    assert chat.prompt_version == "v3" and chat.request_id == "rid-1"
    assert chat.guideline_version_number == 3


async def test_multi_intent() -> None:
    service, store, _ = build(
        selector=ScriptedSelector(lambda m, c: ["billing_issue", "shipping_issue"])
    )
    out = await service.handle(req("Charged twice and my parcel is late"), request_id=None)
    assert [i.intent_key for i in out.guidance.detected_intents] == [
        "billing_issue",
        "shipping_issue",
    ]
    assert store.chats[0].primary_intent == "billing_issue"


async def test_no_match_uses_section_fallback() -> None:
    service, store, search = build(selector=ScriptedSelector(lambda m, c: []))
    out = await service.handle(req("Do you sell pizza?"), request_id=None)
    assert search.section_calls == 1
    g = out.guidance
    assert g.detected_intents == [] and g.confidence == "low" and g.requires_escalation
    assert any(r.startswith("section:") for r in g.references)
    assert store.chats[0].routing and store.chats[0].routing["no_match"] is True


async def test_llm_failure_persisted_and_mapped() -> None:
    service, store, _ = build(generator=FailingGuidanceGenerator("upstream 503"))
    with pytest.raises(LLMError):
        await service.handle(req(), request_id=None)
    chat = store.chats[0]
    assert chat.status == "error" and "upstream 503" in (chat.error_message or "")
    assert chat.guidance is None and chat.routing is not None  # routing ran before the failure


async def test_search_failure_persisted_and_mapped() -> None:
    service, store, _ = build(search=FakeSearch(fail=True))
    with pytest.raises(RetrievalError):
        await service.handle(req(), request_id=None)
    assert store.chats[0].status == "error" and "search down" in (
        store.chats[0].error_message or ""
    )


async def test_no_usable_version_409() -> None:
    service, store, _ = build(usable=False)
    with pytest.raises(NoActiveGuidelines):
        await service.handle(req(), request_id=None)
    assert store.chats == []


async def test_prefetch_skips_the_llm_when_args_are_known() -> None:
    tools = ScriptedToolModel()
    service, store, _ = build(bindings={"order.get": live("order.get")}, tool_model=tools)
    out = await service.handle(req("Where is order #1001?"), request_id=None)
    assert tools.calls == []  # everything was prefetched deterministically
    step = next(
        s for s in out.guidance.detected_intents[0].steps if s.step_ref == "shipping_issue#s2"
    )
    assert step.performed_by == "ai_verified" and step.tool_call_id
    assert [t.trigger for t in store.tool_calls] == ["prefetch"]
    assert store.tool_calls[0].status == "ok" and store.tool_calls[0].chat_id == out.chat.id
    assert f"tool:{step.tool_call_id}" in out.guidance.references


async def test_tool_agent_looks_up_with_args_it_finds() -> None:
    tools = ScriptedToolModel([tool_call("subscription_get", {"email": "jane@example.com"})])
    service, store, _ = build(
        selector=ScriptedSelector(lambda m, c: ["billing_issue"]),
        bindings={"subscription.get": live("subscription.get")},
        tool_model=tools,
    )
    out = await service.handle(req("I was charged twice"), request_id=None)
    assert tools.tools_seen[0] == ["subscription_get"]  # only bound READ tools are offered
    assert [t.trigger for t in store.tool_calls] == ["agent"]
    step = out.guidance.detected_intents[0].steps[0]
    assert step.performed_by == "ai_verified"


async def test_tool_loop_is_bounded() -> None:
    tools = ScriptedToolModel(loop_forever=True)
    service, store, _ = build(
        selector=ScriptedSelector(lambda m, c: ["billing_issue"]),
        bindings={"subscription.get": live("subscription.get")},
        tool_model=tools,
        AGENT_MAX_STEPS=2,
    )
    out = await service.handle(req("charged twice"), request_id=None)
    assert len(tools.calls) == 2  # stopped at AGENT_MAX_STEPS
    assert all(t.status == "error" for t in store.tool_calls)  # unknown account
    assert out.guidance.detected_intents[0].steps[0].performed_by == "agent"  # degraded
    assert out.chat.status == "success"


async def test_proposals_are_policy_gated() -> None:
    gen = FakeGuidanceGenerator(propose=[("refund.create", {"amount": 20, "email": "a@b.co"})])
    service, store, _ = build(
        selector=ScriptedSelector(lambda m, c: ["billing_issue"]),
        generator=gen,
        bindings={"refund.create": live("refund.create", {"max_amount": 50})},
    )
    out = await service.handle(req("refund please"), request_id=None)
    assert [a.status for a in store.actions] == ["proposed"]
    step = next(s for s in out.guidance.detected_intents[0].steps if s.type == "action")
    assert step.performed_by == "needs_approval" and step.proposed_action_id == str(
        store.actions[0].id
    )
    assert out.actions[0].idempotency_key


async def test_policy_violation_is_denied_and_step_stays_manual() -> None:
    gen = FakeGuidanceGenerator(propose=[("refund.create", {"amount": 80})])
    service, store, _ = build(
        selector=ScriptedSelector(lambda m, c: ["billing_issue"]),
        generator=gen,
        bindings={"refund.create": live("refund.create", {"max_amount": 50})},
    )
    out = await service.handle(req("refund please"), request_id=None)
    assert [a.status for a in store.actions] == ["denied"]
    assert "exceeds max_amount" in str(store.actions[0].policy_check)
    step = next(s for s in out.guidance.detected_intents[0].steps if s.type == "action")
    assert step.performed_by == "agent"


async def test_unbound_write_proposal_is_dropped() -> None:
    gen = FakeGuidanceGenerator(propose=[("!refund.create", {"amount": 5})])
    service, store, _ = build(
        selector=ScriptedSelector(lambda m, c: ["billing_issue"]), generator=gen
    )
    await service.handle(req("refund please"), request_id=None)
    assert store.actions == []


async def test_tools_disabled() -> None:
    tools = ScriptedToolModel()
    service, store, _ = build(
        bindings={"order.get": live("order.get")}, tool_model=tools, TOOLS_ENABLED=False
    )
    await service.handle(req("Where is order #1001?"), request_id=None)
    assert store.tool_calls == [] and tools.calls == []
