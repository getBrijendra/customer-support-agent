"""Graph nodes. Each node is a plain async function `(state) -> partial state`, so it can be
unit-tested without LangGraph. Any exception is captured into `errors`; later nodes skip and
the run ends in `persist`, so every attempt is stored."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai.context import GuidanceContext, PlaybookView, ToolResultView
from app.ai.guidance_chain import GuidanceGenerator, ToolCallingModel, tool_schema
from app.ai.heuristic import extract_entities
from app.ai.prompts import PROMPT_VERSION, ROUTER_PROMPT_VERSION, TOOL_AGENT_SYSTEM
from app.ai.router import IntentSelector
from app.ai.schemas import GuidanceOutput, IntentSelection
from app.capabilities.gateway import CapabilityGateway, ToolCallRecord
from app.capabilities.policy import check_policy
from app.capabilities.registry import (
    ArgumentError,
    by_tool_name,
    coerce_args,
    get_capability,
)
from app.core.config import Settings
from app.core.logging import get_logger, text_fingerprint
from app.db.models import ActionStatus, Chat, ChatStatus, ProposedAction, utcnow
from app.harness.state import GuidanceState
from app.search.base import Query, SearchGateway
from app.services.common import AppMetrics, UowFactory

logger = get_logger(__name__)

Node = Callable[[GuidanceState], Awaitable[dict[str, Any]]]
SecretGetterFactory = Callable[[uuid.UUID], Callable[[str], Awaitable[str]]]


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class NodeFailure(Exception):
    """Raised inside a node to report a specific error kind (search / router / llm)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def guarded(name: str, kind: str = "internal") -> Callable[[Node], Node]:
    """Skip when an earlier node failed; turn exceptions into `errors` entries."""

    def deco(fn: Node) -> Node:
        async def wrapper(state: GuidanceState) -> dict[str, Any]:
            if state.get("errors"):
                return {}
            started = time.perf_counter()
            try:
                out = await fn(state)
            except Exception as exc:
                logger.warning(
                    "harness.node_failed",
                    extra={"node": name, "error": f"{type(exc).__name__}: {exc}"[:300]},
                )
                err_kind = exc.kind if isinstance(exc, NodeFailure) else kind
                return {
                    "errors": [
                        {"node": name, "kind": err_kind, "message": f"{exc}"[:2000] or name}
                    ],
                    "timings": {name: _ms(started)},
                }
            out.setdefault("timings", {})
            out["timings"] = {**out["timings"], name: _ms(started)}
            logger.debug("harness.node", extra={"node": name, "duration_ms": _ms(started)})
            return out

        wrapper.__name__ = name
        return wrapper

    return deco


@dataclass
class HarnessDeps:
    uow: UowFactory
    search: SearchGateway
    embeddings_for: Callable[[str], Embeddings]
    selector: IntentSelector
    generator: GuidanceGenerator
    tool_model: ToolCallingModel | None
    gateway: CapabilityGateway
    secret_getter: SecretGetterFactory
    settings: Settings
    metrics: AppMetrics


class HarnessNodes:
    def __init__(self, deps: HarnessDeps) -> None:
        self.d = deps
        s = deps.settings
        self.load_context = guarded("load_context", "search")(self._load_context)
        self.route = guarded("route", "router")(self._route)
        self.retrieve_facts = guarded("retrieve_facts", "search")(self._retrieve_facts)
        self.fallback_sections = guarded("fallback_sections", "search")(self._fallback)
        self.prefetch = guarded("prefetch")(self._prefetch)
        self.tool_agent = guarded("tool_agent")(self._tool_agent)
        self.tools = guarded("tools")(self._tools)
        self.generate = guarded("generate", "llm")(self._generate)
        self.validate_and_gate = guarded("validate_and_gate")(self._validate_and_gate)
        self._max_steps = s.AGENT_MAX_STEPS

    # ------------------------------------------------------------------ helpers
    def _query(self, state: GuidanceState) -> Query:
        c = state["company"]
        assert c.active_version_id is not None
        text = state["customer_message"]
        if state.get("agent_notes"):
            text = f"{text}\n\n{state['agent_notes']}"
        return Query(c.company_id, c.active_version_id, state["query_vector"], text)

    def _call(
        self,
        state: GuidanceState,
        capability: str,
        args: dict[str, Any],
        trigger: str,
        step_ref: str | None,
    ) -> Awaitable[ToolCallRecord]:
        c = state["company"]
        return self.d.gateway.call(
            company_id=c.company_id,
            capability=capability,
            args=args,
            binding=c.bindings.get(capability),
            get_secret=self.d.secret_getter(c.company_id),
            trigger=trigger,  # type: ignore[arg-type]
            step_ref=step_ref,
        )

    def _observe(self, records: list[ToolCallRecord]) -> None:
        for r in records:
            if r.status in {"not_bound", "invalid_args"}:
                continue
            self.d.metrics.tool_calls.labels(
                capability=r.capability, adapter=r.adapter_type or "none", status=r.status
            ).inc()
            self.d.metrics.tool_latency.observe(r.latency_ms / 1000)
            logger.info(
                "tool.called",
                extra={
                    "capability": r.capability,
                    "trigger": r.trigger,
                    "status": r.status,
                    "latency_ms": r.latency_ms,
                },
            )

    # ------------------------------------------------------------------- nodes
    async def _load_context(self, state: GuidanceState) -> dict[str, Any]:
        c = state["company"]
        assert c.embedding_model
        text = state["customer_message"]
        if state.get("agent_notes"):
            text = f"{text}\n\n{state['agent_notes']}"
        emb = self.d.embeddings_for(c.embedding_model)
        vector = await asyncio.to_thread(emb.embed_query, text)
        return {"query_vector": list(vector)}

    async def _route(self, state: GuidanceState) -> dict[str, Any]:
        s = self.d.settings
        c = state["company"]
        try:
            candidates = await self.d.search.search_intents(
                self._query(state), s.ROUTING_CANDIDATES
            )
        except Exception as exc:
            raise NodeFailure("search", f"{type(exc).__name__}: {exc}") from exc
        result = await self.d.selector.select(
            company_name=c.name,
            customer_message=state["customer_message"],
            agent_notes=state.get("agent_notes"),
            candidates=candidates,
            max_intents=s.ROUTING_MAX_INTENTS,
        )
        sel: IntentSelection = result.selection
        playbooks: list[PlaybookView] = []
        if sel.intents:
            assert c.active_version_id is not None
            async with self.d.uow() as uow:
                rows = await uow.playbooks.intents_by_keys(
                    c.active_version_id, [i.key for i in sel.intents]
                )
            by_key = {i.key: i for i in sel.intents}
            for r in rows:
                picked = by_key[r.key]
                playbooks.append(
                    PlaybookView(
                        key=r.key,
                        name=r.name,
                        description=r.description,
                        conditions=list(r.conditions or []),
                        steps=list(r.steps),
                        reply_template=r.reply_template,
                        placeholders=list(r.placeholders or []),
                        source_section_ids=list(r.source_section_ids or []),
                        confidence=picked.confidence,
                        message_span=picked.message_span,
                    )
                )
        entities = {
            **{
                e.name: e.value
                for e in extract_entities(state["customer_message"], state.get("agent_notes"))
            },
            **sel.entity_map(),
        }
        no_match = not playbooks
        self.d.metrics.intents_selected.observe(len(playbooks))
        if no_match:
            self.d.metrics.no_match.labels(company=c.slug).inc()
        logger.info(
            "routing.completed",
            extra={
                "company": c.slug,
                "candidates": len(candidates),
                "selected": [p.key for p in playbooks],
                "no_match": no_match,
            },
        )
        return {
            "candidates": candidates,
            "selection": sel,
            "router_model": result.model,
            "router_tokens": (result.input_tokens, result.output_tokens),
            "playbooks": playbooks,
            "no_match": no_match,
            "entities": entities,
        }

    async def _retrieve_facts(self, state: GuidanceState) -> dict[str, Any]:
        facts = await self.d.search.search_facts(
            self._query(state), self.d.settings.REFERENCE_TOP_K
        )
        return {"facts": facts}

    async def _fallback(self, state: GuidanceState) -> dict[str, Any]:
        sections = await self.d.search.search_sections(
            self._query(state), self.d.settings.FALLBACK_SECTION_TOP_K
        )
        return {"sections": sections}

    def _lookup_steps(self, state: GuidanceState) -> list[tuple[str, str]]:
        """(step_ref, capability) for lookup steps of selected playbooks with a live binding."""
        c = state["company"]
        out = []
        for p in state.get("playbooks", []):
            for st in p.steps:
                cap = st.get("capability")
                if st.get("type") == "lookup" and cap and cap in c.bindings:
                    out.append((f"{p.key}#{st['step_id']}", cap))
        return out

    async def _prefetch(self, state: GuidanceState) -> dict[str, Any]:
        """Deterministic lookups: when the required args are already known (order id, email),
        call the capability without asking the LLM."""
        if not self.d.settings.TOOLS_ENABLED:
            return {}
        entities = state.get("entities", {})
        done: set[tuple[str, str]] = set()
        calls = []
        for step_ref, cap_name in self._lookup_steps(state):
            cap = get_capability(cap_name)
            if cap is None:
                continue
            args = {p: entities[p] for p in cap.params if p in entities}
            if any(r not in args for r in cap.required) or (cap.params and not args):
                continue
            sig = (cap_name, json.dumps(args, sort_keys=True))
            if sig in done:
                continue
            done.add(sig)
            calls.append(self._call(state, cap_name, args, "prefetch", step_ref))
        records = list(await asyncio.gather(*calls)) if calls else []
        self._observe(records)
        return {"tool_records": records}

    def _pending_lookups(self, state: GuidanceState) -> list[tuple[str, str]]:
        ok = {r.capability for r in state.get("tool_records", []) if r.ok}
        tried = {r.capability for r in state.get("tool_records", [])}
        return [
            (ref, cap)
            for ref, cap in self._lookup_steps(state)
            if cap not in ok and cap not in tried
        ]

    def tool_loop_should_continue(self, state: GuidanceState) -> bool:
        if state.get("errors") or state.get("agent_done"):
            return False
        messages = state.get("agent_messages") or []
        last = messages[-1] if messages else None
        return isinstance(last, AIMessage) and bool(last.tool_calls)

    async def _tool_agent(self, state: GuidanceState) -> dict[str, Any]:
        steps = state.get("agent_steps", 0)
        if (
            not self.d.settings.TOOLS_ENABLED
            or self.d.tool_model is None
            or steps >= self._max_steps
            or time.monotonic() >= state["deadline"]
        ):
            return {"agent_done": True}
        pending = self._pending_lookups(state)
        if not pending and steps == 0:
            return {"agent_done": True}
        caps = {cap for _, cap in self._lookup_steps(state)}
        tools = []
        for name in sorted(caps):
            cap = get_capability(name)
            if cap is not None and cap.kind == "read":
                tools.append(tool_schema(cap.tool_name, cap.description, cap.input_schema()))
        messages = list(state.get("agent_messages") or [])
        if not messages:
            pending_txt = "\n".join(f"- {ref}: {cap}" for ref, cap in pending)
            messages = [
                SystemMessage(
                    TOOL_AGENT_SYSTEM.format(
                        company_name=state["company"].name, pending=pending_txt
                    )
                ),
                HumanMessage(
                    f"<agent_notes>\n{state.get('agent_notes') or ''}\n</agent_notes>\n"
                    f"<customer_message>\n{state['customer_message']}\n</customer_message>"
                ),
            ]
        remaining = state["deadline"] - time.monotonic()
        try:
            ai = await asyncio.wait_for(self.d.tool_model.step(messages, tools), remaining)
        except Exception as exc:  # tool loop failures degrade to manual steps
            logger.warning("harness.tool_agent_failed", extra={"error": type(exc).__name__})
            return {"agent_done": True, "agent_messages": messages}
        return {
            "agent_messages": [*messages, ai],
            "agent_steps": steps + 1,
            "agent_done": not ai.tool_calls,
        }

    async def _tools(self, state: GuidanceState) -> dict[str, Any]:
        messages = list(state["agent_messages"])
        last = messages[-1]
        assert isinstance(last, AIMessage)
        allowed = {cap for _, cap in self._lookup_steps(state)}
        step_for = {cap: ref for ref, cap in self._lookup_steps(state)}
        records: list[ToolCallRecord] = []
        for call in last.tool_calls:
            cap = by_tool_name(call["name"])
            if cap is None or cap.name not in allowed or cap.kind != "read":
                messages.append(
                    ToolMessage(
                        content=json.dumps({"error": "tool not available"}),
                        tool_call_id=call.get("id") or "",
                    )
                )
                continue
            rec = await self._call(
                state, cap.name, dict(call["args"]), "agent", step_for.get(cap.name)
            )
            records.append(rec)
            payload = rec.result if rec.ok else {"error": rec.error, "status": rec.status}
            messages.append(
                ToolMessage(
                    content=json.dumps(payload, default=str), tool_call_id=call.get("id") or ""
                )
            )
        self._observe(records)
        return {"agent_messages": messages, "tool_records": records}

    async def _generate(self, state: GuidanceState) -> dict[str, Any]:
        c = state["company"]
        playbooks = state.get("playbooks", [])
        sections = list(state.get("sections", []))
        cited = {sid for p in playbooks for sid in p.source_section_ids}
        if cited and c.active_version_id:
            async with self.d.uow() as uow:
                rows = await uow.playbooks.sections_by_ids(
                    c.active_version_id, [uuid.UUID(s) for s in cited]
                )
            from app.search.base import SectionHit

            have = {s.id for s in sections}
            sections += [
                SectionHit(id=r.id, ref_label=r.ref, heading=r.heading, text=r.text)
                for r in rows
                if r.id not in have
            ]
        referenced_writes: set[str] = {
            str(st["capability"])
            for p in playbooks
            for st in p.steps
            if st.get("type") == "action" and st.get("capability")
        }
        write_caps = []
        if self.d.settings.TOOLS_ENABLED:
            for name in sorted(referenced_writes):
                cap = get_capability(str(name))
                if cap and cap.kind == "write" and cap.name in c.bindings:
                    write_caps.append(cap.summary())
        context = GuidanceContext(
            company_name=c.name,
            rules=c.rules,
            playbooks=playbooks,
            facts=state.get("facts", []),
            sections=sections,
            tool_results=[
                ToolResultView(
                    call_id=str(r.id),
                    capability=r.capability,
                    status=r.status,
                    result=r.result,
                    error=r.error,
                    step_ref=r.step_ref,
                )
                for r in state.get("tool_records", [])
                if r.status not in {"not_bound"}
            ],
            write_capabilities=write_caps,
            no_match=state.get("no_match", False),
        )
        generation = await self.d.generator.generate(
            context=context,
            customer_message=state["customer_message"],
            agent_notes=state.get("agent_notes"),
            channel=state.get("channel"),
        )
        return {"context": context, "generation": generation}

    async def _validate_and_gate(self, state: GuidanceState) -> dict[str, Any]:
        c = state["company"]
        context: GuidanceContext = state["context"]
        out: GuidanceOutput = state["generation"].output.model_copy(deep=True)
        valid = context.reference_ids()
        dropped = [r for r in out.references if r not in valid]
        if dropped:
            logger.info("harness.references_dropped", extra={"count": len(dropped)})
        out.references = [r for r in out.references if r in valid]

        playbooks = {p.key: p for p in context.playbooks}
        out.detected_intents = [i for i in out.detected_intents if i.intent_key in playbooks]

        # --- proposed actions: schema + policy in code; unbound -> manual
        actions: list[ProposedAction] = []
        for pa in out.proposed_actions:
            cap = get_capability(pa.capability)
            binding = c.bindings.get(pa.capability)
            if cap is None or cap.kind != "write" or binding is None:
                continue
            policy_check: dict[str, Any]
            try:
                args = coerce_args(cap, pa.args())
                check = check_policy(cap, args, binding.policy)
                policy_check = check.to_dict()
                status = ActionStatus.PROPOSED if check.ok else ActionStatus.DENIED
            except ArgumentError as exc:
                args = pa.args()
                policy_check = {"ok": False, "violations": [str(exc)]}
                status = ActionStatus.DENIED
            actions.append(
                ProposedAction(
                    id=uuid.uuid4(),
                    company_id=c.company_id,
                    chat_id=state["chat_id"],
                    capability=cap.name,
                    args=args,
                    reason=pa.reason,
                    step_ref=pa.step_ref,
                    policy_check=policy_check,
                    status=status,
                    idempotency_key=uuid.uuid4().hex,
                    created_at=utcnow(),
                )
            )
            self.d.metrics.proposed_actions.labels(status=status).inc()
            logger.info("action.proposed", extra={"capability": cap.name, "status": status})
        live_actions = [a for a in actions if a.status == ActionStatus.PROPOSED]

        # --- performed_by is decided in code, never trusted from the model
        ok_calls = {r.capability: r for r in state.get("tool_records", []) if r.ok}
        for ig in out.detected_intents:
            pb = playbooks[ig.intent_key]
            for st in ig.steps:
                known = pb.step(st.step_ref.split("#", 1)[1]) if "#" in st.step_ref else None
                if known:
                    st.type = known["type"]
                st.tool_call_id = None
                st.proposed_action_id = None
                cap_name = known.get("capability") if known else None
                if st.type in {"say", "ask"}:
                    st.performed_by = "ai"
                elif st.type == "lookup" and cap_name in ok_calls:
                    st.performed_by = "ai_verified"
                    st.tool_call_id = str(ok_calls[cap_name].id)
                elif st.type == "action" and (
                    act := next(
                        (
                            a
                            for a in live_actions
                            if a.step_ref == st.step_ref or a.capability == cap_name
                        ),
                        None,
                    )
                ):
                    st.performed_by = "needs_approval"
                    st.proposed_action_id = str(act.id)
                else:
                    st.performed_by = "agent"

        if state.get("no_match"):
            out.confidence = "low"
            out.requires_escalation = True
            out.escalation_reason = out.escalation_reason or (
                "No guideline playbook covers this request"
            )
        references = {
            "used": out.references,
            "dropped": dropped,
            "facts": [str(f.id) for f in context.facts],
            "sections": [str(s.id) for s in context.sections],
            "tool_calls": [str(r.id) for r in state.get("tool_records", []) if r.ok],
        }
        return {"output": out, "actions": actions, "references": references}

    # ------------------------------------------------------------------ persist
    async def persist(self, state: GuidanceState) -> dict[str, Any]:
        """Always runs: stores the chat (success or error) + tool calls + proposed actions."""
        c = state["company"]
        errors = state.get("errors", [])
        timings = state.get("timings", {})
        sel = state.get("selection")
        gen = state.get("generation")
        out = state.get("output")
        records = state.get("tool_records", [])
        router_in, router_out = state.get("router_tokens", (None, None))
        chat = Chat(
            id=state["chat_id"],
            company_id=c.company_id,
            guideline_version_id=c.active_version_id,
            guideline_version_number=c.active_version_number,
            customer_message=state["customer_message"],
            agent_notes=state.get("agent_notes"),
            channel=state.get("channel"),
            status=ChatStatus.ERROR if errors or out is None else ChatStatus.SUCCESS,
            llm_model=gen.model if gen else self.d.generator.model_name,
            router_model=state.get("router_model") or self.d.selector.model_name,
            prompt_version=PROMPT_VERSION,
            router_prompt_version=ROUTER_PROMPT_VERSION,
            request_id=state.get("request_id"),
            created_at=utcnow(),
            routing=None
            if sel is None
            else {
                "candidates": [x.to_record() for x in state.get("candidates", [])],
                "selected": [i.model_dump() for i in sel.intents],
                "no_match": bool(state.get("no_match")),
                "entities": state.get("entities", {}),
            },
            references=state.get("references"),
            latency_ms_routing=(timings.get("load_context", 0) + timings["route"])
            if "route" in timings
            else None,
            latency_ms_tools=(
                timings.get("prefetch", 0) + timings.get("tool_agent", 0) + timings.get("tools", 0)
            )
            if "prefetch" in timings
            else None,
            latency_ms_llm=timings.get("generate"),
            latency_ms_total=_ms(state["started_at"]),
            router_input_tokens=router_in,
            router_output_tokens=router_out,
        )
        if out is not None and not errors:
            chat.guidance = out.model_dump(mode="json")
            chat.recommended_response = out.recommended_response
            chat.primary_intent = (
                out.detected_intents[0].intent_key
                if out.detected_intents
                else (sel.intents[0].key if sel and sel.intents else None)
            )
        elif sel and sel.intents:
            chat.primary_intent = sel.intents[0].key
        if gen is not None:
            chat.input_tokens = gen.input_tokens
            chat.output_tokens = gen.output_tokens
        if errors:
            chat.error_message = "; ".join(f"{e['node']}: {e['message']}" for e in errors)[:2000]

        tool_rows = [r.to_model(c.company_id, chat_id=chat.id) for r in records]
        actions = state.get("actions", []) if not errors else []
        try:
            async with self.d.uow() as uow:
                await uow.chats.add(chat)
                if actions:
                    await uow.capabilities.add_actions(actions)
                rows = [r for r in tool_rows if r is not None]
                if rows:
                    await uow.capabilities.add_tool_calls(rows)
                await uow.commit()
        except Exception:
            logger.exception("chat.persist_failed", extra={"company": c.slug})
            raise
        m = self.d.metrics
        m.chat_requests.labels(company=c.slug, status=chat.status).inc()
        m.chat_latency.observe((chat.latency_ms_total or 0) / 1000)
        for role, tin, tout in (
            ("generator", chat.input_tokens, chat.output_tokens),
            ("router", router_in, router_out),
        ):
            if tin:
                m.llm_tokens.labels(role=role, direction="input").inc(tin)
            if tout:
                m.llm_tokens.labels(role=role, direction="output").inc(tout)
        log = {
            "chat_id": str(chat.id),
            "company": c.slug,
            "version": c.active_version_number,
            "intents": [p.key for p in state.get("playbooks", [])],
            "no_match": bool(state.get("no_match")),
            "tool_calls": len(tool_rows),
            "proposed_actions": len(actions),
            "latency_ms_total": chat.latency_ms_total,
            "latency_ms_routing": chat.latency_ms_routing,
            "latency_ms_tools": chat.latency_ms_tools,
            "latency_ms_llm": chat.latency_ms_llm,
            "input_tokens": chat.input_tokens,
            "output_tokens": chat.output_tokens,
            "customer_message": text_fingerprint(chat.customer_message),
        }
        if errors:
            logger.warning(
                "chat.failed",
                extra={**log, "error": chat.error_message[:300] if chat.error_message else None},
            )
        else:
            logger.info("chat.completed", extra=log)
        return {"chat": chat}
