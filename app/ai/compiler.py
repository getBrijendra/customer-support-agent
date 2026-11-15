"""Guideline -> playbook compilation (LLM) + deterministic validation and map-reduce merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from app.ai.prompts import compiler_prompt
from app.ai.providers import invoke_structured
from app.ai.schemas import (
    CompiledIntent,
    CompiledPlaybook,
    CompiledRules,
    SectionMapping,
)
from app.capabilities.registry import get_capability, registry_prompt_listing
from app.documents.base import ParsedDocument, Section


@dataclass
class CompileResult:
    playbook: CompiledPlaybook
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class PlaybookCompiler(Protocol):
    model_name: str

    async def compile(
        self, doc: ParsedDocument, *, company_name: str, previous_intents: list[dict[str, str]]
    ) -> CompileResult: ...


def render_sections(sections: list[Section]) -> str:
    return "\n\n".join(
        f'<section ref="{s.ref}" heading="{escape(s.heading or "", quote=True)}">\n'
        f"{escape(s.text)}\n</section>"
        for s in sections
    )


def render_previous(previous: list[dict[str, str]]) -> str:
    if not previous:
        return "(none - this is the first version)"
    return "\n".join(f"- {p['key']}: {p.get('description', '')}" for p in previous)


class LLMPlaybookCompiler:
    """Single pass for small documents; map (per section) + deterministic reduce otherwise."""

    def __init__(
        self, llm: BaseChatModel, model_name: str, *, timeout: float, single_pass_max_chars: int
    ) -> None:
        self._llm = llm
        self.model_name = model_name
        self._timeout = timeout
        self._max_chars = single_pass_max_chars

    async def _run(
        self, sections: list[Section], previous: list[dict[str, str]]
    ) -> tuple[CompiledPlaybook, int, int]:
        res = await invoke_structured(
            self._llm,
            compiler_prompt(),
            CompiledPlaybook,
            {
                "capabilities": registry_prompt_listing(),
                "previous_intents": render_previous(previous),
                "sections": render_sections(sections),
            },
            timeout_s=self._timeout,
        )
        return res.output, res.input_tokens or 0, res.output_tokens or 0

    async def compile(
        self, doc: ParsedDocument, *, company_name: str, previous_intents: list[dict[str, str]]
    ) -> CompileResult:
        if len(doc.text) <= self._max_chars:
            pb, i, o = await self._run(doc.sections, previous_intents)
            return CompileResult(pb, self.model_name, i, o)
        parts: list[CompiledPlaybook] = []
        tin = tout = 0
        batch: list[Section] = []
        size = 0
        for s in doc.sections:  # map: batches of whole sections under the size limit
            if batch and size + len(s.text) > self._max_chars:
                pb, i, o = await self._run(batch, previous_intents)
                parts.append(pb)
                tin, tout, batch, size = tin + i, tout + o, [], 0
            batch.append(s)
            size += len(s.text)
        if batch:
            pb, i, o = await self._run(batch, previous_intents)
            parts.append(pb)
            tin, tout = tin + i, tout + o
        return CompileResult(merge_playbooks(parts), self.model_name, tin, tout)


def merge_playbooks(parts: list[CompiledPlaybook]) -> CompiledPlaybook:
    """Reduce step: merge intents with the same key (steps appended, sources unioned), merge
    rules (first non-empty scalar wins, lists unioned) and concatenate facts."""
    intents: dict[str, CompiledIntent] = {}
    rules = CompiledRules()
    facts = []
    kinds: dict[str, SectionMapping] = {}
    for pb in parts:
        for it in pb.intents:
            cur = intents.get(it.key)
            if cur is None:
                intents[it.key] = it.model_copy(deep=True)
                continue
            offset = len(cur.steps)
            for n, st in enumerate(it.steps, start=1):
                cur.steps.append(st.model_copy(update={"step_id": f"s{offset + n}"}))
            cur.source_sections = _union(cur.source_sections, it.source_sections)
            cur.conditions = _union(cur.conditions, it.conditions)
            cur.placeholders = _union(cur.placeholders, it.placeholders)
            cur.reply_template = cur.reply_template or it.reply_template
        for name in ("tone", "greeting", "sign_off", "empathy"):
            if not getattr(rules, name) and getattr(pb.rules, name):
                setattr(rules, name, getattr(pb.rules, name))
        for name in ("prohibitions", "escalation_defaults", "formatting", "source_sections"):
            setattr(rules, name, _union(getattr(rules, name), getattr(pb.rules, name)))
        facts.extend(pb.facts)
        for m in pb.section_kinds:
            kinds.setdefault(m.section_ref, m)
    return CompiledPlaybook(
        intents=list(intents.values()), rules=rules, facts=facts, section_kinds=list(kinds.values())
    )


def _union(a: list[str], b: list[str]) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


# --------------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    playbook: CompiledPlaybook
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_playbook(pb: CompiledPlaybook, doc: ParsedDocument) -> ValidationReport:
    """Normalise and check a compiled playbook against the parsed document.

    Blocking errors: no intents, duplicate keys, an intent without steps, an intent that cites
    no existing section. Fixable problems (unknown refs/capabilities, kind mismatches, step ids)
    are corrected and reported as warnings. Coverage maps every section to a kind.
    """
    pb = pb.model_copy(deep=True)
    refs = {s.ref for s in doc.sections}
    errors: list[str] = []
    warnings: list[str] = []

    if not pb.intents:
        errors.append("No intents were compiled from the guidelines")
    seen: set[str] = set()
    for it in pb.intents:
        if it.key in seen:
            errors.append(f"Duplicate intent key '{it.key}'")
        seen.add(it.key)
        bad = [r for r in it.source_sections if r not in refs]
        if bad:
            warnings.append(f"Intent '{it.key}' cites unknown sections {bad}; dropped")
            it.source_sections = [r for r in it.source_sections if r in refs]
        if not it.source_sections:
            errors.append(f"Intent '{it.key}' does not cite any existing section")
        if not it.steps:
            errors.append(f"Intent '{it.key}' has no steps")
        for n, st in enumerate(it.steps, start=1):
            st.step_id = f"s{n}"
            if st.capability is None:
                continue
            cap = get_capability(st.capability)
            if st.type not in {"lookup", "action"}:
                st.capability = None
            elif cap is None:
                warnings.append(
                    f"Intent '{it.key}' step s{n}: unknown capability '{st.capability}'; "
                    "treated as manual"
                )
                st.capability = None
            elif (st.type == "lookup") != (cap.kind == "read"):
                warnings.append(
                    f"Intent '{it.key}' step s{n}: {st.type} step mapped to {cap.kind} "
                    f"capability '{cap.name}'; treated as manual"
                )
                st.capability = None

    for f in pb.facts:
        f.source_sections = [r for r in f.source_sections if r in refs]
    pb.rules.source_sections = [r for r in pb.rules.source_sections if r in refs]

    kinds: dict[str, str] = {
        m.section_ref: m.kind for m in pb.section_kinds if m.section_ref in refs
    }
    for it in pb.intents:
        for r in it.source_sections:
            kinds.setdefault(r, "intent")
    for f in pb.facts:
        for r in f.source_sections:
            kinds.setdefault(r, "fact")
    for r in pb.rules.source_sections:
        kinds.setdefault(r, "rule")
    unmapped = [s.ref for s in doc.sections if s.ref not in kinds]
    if unmapped:
        warnings.append(f"Sections not covered by any intent, rule or fact: {unmapped}")
    pb.section_kinds = [SectionMapping(section_ref=r, kind=k) for r, k in kinds.items()]
    coverage = {"sections": kinds, "unmapped": unmapped, "total": len(doc.sections)}
    return ValidationReport(pb, errors, warnings, coverage)
