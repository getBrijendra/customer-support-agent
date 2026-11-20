"""Deterministic, offline implementations of every AI collaborator used by local development.

The compiler understands the "N. Title / Approach to solve: / - bullets / Sample reply:" layout
of the seed guidelines, plus TONE/RULES and REFERENCE/FACTS sections. The router and generator
are deliberately simple and predictable; they are a functional demo path, not quality substitutes
for production models.
"""

from __future__ import annotations

import re

from app.ai.compiler import CompileResult
from app.ai.context import GuidanceContext
from app.ai.examples import ExampleResult
from app.ai.guidance_chain import GenerationResult
from app.ai.router import SelectionResult, sanitize_selection
from app.ai.schemas import (
    CompiledFact,
    CompiledIntent,
    CompiledPlaybook,
    CompiledRules,
    CompiledStep,
    Entity,
    ExampleSet,
    GuidanceOutput,
    IntentExamples,
    IntentGuidance,
    IntentSelection,
    PlanStep,
    SectionMapping,
    SelectedIntent,
    to_key,
)
from app.documents.base import ParsedDocument, Section
from app.search.base import IntentCandidate

RULE_HEADINGS = ("tone", "style", "general rules", "rules", "communication", "voice")
FACT_HEADINGS = ("reference", "facts", "faq", "pricing", "plans", "product info")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
_PLACEHOLDER = re.compile(r"\[([^\]]+)\]")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ORDER = re.compile(
    r"(?:order\s*(?:id|number|no\.?)?\s*[:#]?\s*|#)([A-Za-z0-9-]*\d[A-Za-z0-9-]*)", re.I
)

_LOOKUP_CAPS = [
    # specific grocery phrases first ("Farm Box plan" must not fall through to "plan")
    (("farm box",), "farmbox.get"),
    (("delivery slot",), "delivery_slot.list"),
    (("stock level", "harvest", "in stock"), "product.availability"),
    (("tracking", "courier", "shipment", "carrier", "eta"), "shipment.track"),
    (("order",), "order.get"),
    (("subscription", "payment", "billing", "transaction", "renewal", "plan"), "subscription.get"),
    (("device", "connections"), "device.list"),
    (("status page", "outage", "maintenance"), "status.get"),
    (("account", "credential", "email address"), "account.get"),
]
_ACTION_CAPS = [
    (("refund",), "refund.create"),
    (("store credit",), "credit.issue"),
    (("reschedule",), "delivery.reschedule"),
    (("cancel the order",), "order.cancel"),
    (("pause",), "farmbox.pause"),
    (("back-in-stock alert", "restock alert"), "restock_alert.create"),
    (("return label", "prepaid label"), "return_label.create"),
    (("replacement", "ship correct", "replace", "reship"), "replacement.create"),
    (("discount", "compensation", "coupon"), "discount.issue"),
    (("password", "unlock"), "password.reset"),
    (("retry", "retrying payment"), "payment.retry"),
    (("disconnect", "remove device", "revoke"), "device.revoke"),
]
_ACTION_VERBS = (
    "process", "issue", "ship", "refund", "reset", "unlock", "apply", "cancel", "replace",
    "arrange", "retry", "reship", "offer a discount", "offer compensation", "reschedule",
    "pause", "set up",
)  # fmt: skip
_LOOKUP_VERBS = ("verify", "check", "look up", "review", "confirm", "inspect", "validate")
_ASK_VERBS = ("ask", "request", "collect", "find out", "get the customer", "obtain")


def classify_step(text: str) -> tuple[str, str | None]:
    """Return (step type, capability or None) for one guideline bullet."""
    t = text.lower()
    if "escalat" in t or ("forward" in t and "team" in t):
        return "escalate", None
    if t.startswith(_ASK_VERBS):
        return "ask", None
    if t.startswith(_LOOKUP_VERBS):
        return "lookup", _match(t, _LOOKUP_CAPS)
    if t.startswith(_ACTION_VERBS) or any(f" {v} " in f" {t} " for v in ("refund", "reship")):
        return "action", _match(t, _ACTION_CAPS)
    return "say", None


def _match(text: str, table: list[tuple[tuple[str, ...], str]]) -> str | None:
    for words, cap in table:
        if any(w in text for w in words):
            return cap
    return None


def _bullets(text: str) -> list[str]:
    return [m.group(1).strip() for line in text.split("\n") if (m := _BULLET.match(line))]


def _split_reply(body: str) -> tuple[str, str | None]:
    m = re.search(r"^\s*sample (?:reply|response)\s*:?\s*$", body, re.I | re.M)
    if not m:
        return body, None
    return body[: m.start()], body[m.end() :].strip() or None


def _rules_from(section: Section, rules: CompiledRules) -> None:
    lines = _bullets(section.body) or [ln.strip() for ln in section.body.split("\n") if ln.strip()]
    for line in lines:
        low = line.lower()
        if "greet" in low:
            rules.greeting = rules.greeting or line
        elif "sign off" in low or "sign-off" in low or "regards" in low:
            rules.sign_off = rules.sign_off or line
        elif low.startswith(("never", "do not", "don't")):
            rules.prohibitions.append(line)
        elif "escalat" in low:
            rules.escalation_defaults.append(line)
        elif "empath" in low or "apolog" in low:
            rules.empathy = rules.empathy or line
        else:
            rules.tone = f"{rules.tone} {line}".strip() if rules.tone else line
    rules.source_sections.append(section.ref)


def _facts_from(section: Section) -> list[CompiledFact]:
    lines = _bullets(section.body) or [ln.strip() for ln in section.body.split("\n") if ln.strip()]
    facts = []
    for line in lines:
        title = line.split(":", 1)[0].strip() if ":" in line else " ".join(line.split()[:5])
        facts.append(CompiledFact(title=title, text=line, source_sections=[section.ref]))
    return facts


def _intent_from(section: Section, previous: dict[str, str]) -> CompiledIntent:
    heading = section.heading or f"Section {section.ref}"
    procedure, reply = _split_reply(section.body)
    bullets = _bullets(procedure)
    if not bullets:
        bullets = [s.strip() for s in re.split(r"(?<=[.!?])\s+", procedure) if len(s.strip()) > 3]
        bullets = [b for b in bullets if not b.lower().startswith("approach to solve")]
    steps = []
    for n, b in enumerate(bullets or [f"Handle: {heading}"], start=1):
        typ, cap = classify_step(b)
        steps.append(CompiledStep(step_id=f"s{n}", type=typ, text=b, capability=cap))
    key = to_key(heading)
    for prev_key, prev_name in previous.items():  # reuse stable keys
        if to_key(prev_name) == key or prev_key == key:
            key = prev_key
            break
    first = bullets[0] if bullets else heading
    return CompiledIntent(
        key=key,
        name=heading,
        description=f"Customer messages about {heading.lower()}. First step: {first}",
        steps=steps,
        reply_template=reply,
        placeholders=list(dict.fromkeys(_PLACEHOLDER.findall(reply or ""))),
        source_sections=[section.ref],
    )


class HeuristicPlaybookCompiler:
    model_name = "heuristic-compiler"

    async def compile(
        self, doc: ParsedDocument, *, company_name: str, previous_intents: list[dict[str, str]]
    ) -> CompileResult:
        previous = {p["key"]: p.get("name") or p["key"] for p in previous_intents}
        rules = CompiledRules()
        intents: list[CompiledIntent] = []
        facts: list[CompiledFact] = []
        kinds: list[SectionMapping] = []
        for s in doc.sections:
            h = (s.heading or "").lower()
            if s.heading is None and not s.body.strip():
                kinds.append(SectionMapping(section_ref=s.ref, kind="ignored"))
            elif s.heading is None or any(w in h for w in RULE_HEADINGS):
                _rules_from(s, rules)
                kinds.append(SectionMapping(section_ref=s.ref, kind="rule"))
            elif any(w in h for w in FACT_HEADINGS):
                facts.extend(_facts_from(s))
                kinds.append(SectionMapping(section_ref=s.ref, kind="fact"))
            else:
                intents.append(_intent_from(s, previous))
                kinds.append(SectionMapping(section_ref=s.ref, kind="intent"))
        # Rules implied by the sample replies ("Hello [Name]," on every template).
        greetings = {
            t.reply_template.split("\n", 1)[0].strip() for t in intents if t.reply_template
        }
        if not rules.greeting and len(greetings) == 1:
            rules.greeting = f"Start replies with: {greetings.pop()}"
        pb = CompiledPlaybook(intents=intents, rules=rules, facts=facts, section_kinds=kinds)
        return CompileResult(pb, self.model_name)


class HeuristicExampleGenerator:
    NEGATIVES = (
        "Can you recommend a good pizza place near me?",
        "What will the weather be like tomorrow?",
        "Tell me a joke about cats.",
        "Who won the football match last night?",
    )

    async def generate(
        self,
        intents: list[CompiledIntent],
        *,
        company_name: str,
        n_index: int,
        n_eval: int,
        n_negatives: int,
    ) -> ExampleResult:
        out = []
        for it in intents:
            name = it.name.lower()
            index = [it.name, f"I need help with {name}", it.description]
            index += [s.text for s in it.steps]
            evals = [
                f"Hi, I have a problem: {name}.",
                f"Hello, question about {name}, can you help?",
                f"{name.capitalize()} - what should I do?",
            ]
            out.append(
                IntentExamples(
                    intent_key=it.key, index_examples=index[:n_index], eval_examples=evals[:n_eval]
                )
            )
        return ExampleResult(ExampleSet(intents=out, negatives=list(self.NEGATIVES[:n_negatives])))


def extract_entities(*texts: str | None) -> list[Entity]:
    joined = "\n".join(t for t in texts if t)
    entities: list[Entity] = []
    if m := _EMAIL.search(joined):
        entities.append(Entity(name="email", value=m.group(0)))
    if m := _ORDER.search(joined):
        entities.append(Entity(name="order_id", value=m.group(1)))
    return entities


class HeuristicIntentSelector:
    """Top candidate if it has a keyword match (or high similarity); otherwise no match."""

    model_name = "heuristic-router"

    def __init__(self, min_vector_score: float = 0.3) -> None:
        self._min = min_vector_score

    async def select(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        candidates: list[IntentCandidate],
        max_intents: int,
    ) -> SelectionResult:
        text_matches = [candidate for candidate in candidates if candidate.text_score]
        best = (
            max(text_matches, key=lambda candidate: candidate.text_score or 0)
            if text_matches
            else (candidates[0] if candidates else None)
        )
        picked = []
        if best is not None and (best.text_score or (best.vector_score or 0) >= self._min):
            picked.append(SelectedIntent(key=best.key, confidence="medium"))
        sel = IntentSelection(
            intents=picked,
            no_match=not picked,
            entities=extract_entities(customer_message, agent_notes),
        )
        return SelectionResult(sanitize_selection(sel, candidates, max_intents), self.model_name)


class HeuristicGuidanceGenerator:
    """Build deterministic, guideline-grounded output without an external model call."""

    model_name = "heuristic-generator"

    async def generate(
        self,
        *,
        context: GuidanceContext,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
    ) -> GenerationResult:
        del customer_message, agent_notes, channel
        detected: list[IntentGuidance] = []
        missing: list[str] = []
        references: list[str] = []
        needs_escalation = context.no_match

        for playbook in context.playbooks:
            steps = []
            intent_missing = []
            for step in playbook.steps:
                step_ref = f"{playbook.key}#{step['step_id']}"
                steps.append(
                    PlanStep(
                        step_ref=step_ref,
                        type=step["type"],
                        instruction=step["text"],
                        performed_by="agent",
                    )
                )
                references.append(f"intent:{step_ref}")
                if step["type"] == "ask":
                    intent_missing.append(step["text"])
                if step["type"] == "escalate":
                    needs_escalation = True
            missing.extend(intent_missing)
            detected.append(
                IntentGuidance(
                    intent_key=playbook.key,
                    confidence=playbook.confidence,
                    steps=steps,
                    missing_information=intent_missing,
                )
            )

        references.extend(f.ref for f in context.facts)
        references.extend(s.ref for s in context.sections)
        references.extend(t.ref for t in context.tool_results if t.status == "ok")
        topics = ", ".join(p.name for p in context.playbooks)
        if context.no_match or not topics:
            response = (
                "Hello,\n\nThank you for contacting us. I could not match this request to a "
                "specific support guideline, so I am sending it for manual review.\n\n"
                f"Best regards,\n{context.company_name} Support"
            )
        else:
            response = (
                f"Hello,\n\nThank you for contacting us about {topics}. We will follow the "
                "applicable support steps and verify any account-specific details before taking "
                "action."
            )
            if missing:
                response += f" Please provide the requested information: {missing[0]}"
            response += f"\n\nBest regards,\n{context.company_name} Support"

        escalation_reason = None
        if context.no_match:
            escalation_reason = "No guideline playbook covers this request"
        elif needs_escalation:
            escalation_reason = "The matched guideline includes an escalation step"
        output = GuidanceOutput(
            detected_intents=detected,
            recommended_response=response,
            requires_escalation=needs_escalation,
            escalation_reason=escalation_reason,
            references=list(dict.fromkeys(references)),
            proposed_actions=[],
            confidence="low" if context.no_match else "medium",
            missing_information=list(dict.fromkeys(missing)),
        )
        return GenerationResult(output=output, model=self.model_name)
