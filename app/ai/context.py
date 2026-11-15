"""Prompt context assembly: everything the generator sees, as tagged, escaped blocks.

Ids rendered here are the only ids the model may cite; `GuidanceContext.reference_ids()` is used
afterwards to strip anything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html import escape
from typing import Any

from app.search.base import FactHit, SectionHit


@dataclass
class PlaybookView:
    key: str
    name: str
    description: str
    conditions: list[str]
    steps: list[dict[str, Any]]  # {step_id, type, text, condition, capability}
    reply_template: str | None
    placeholders: list[str]
    source_section_ids: list[str]
    confidence: str = "medium"  # from the router
    message_span: str | None = None

    def step(self, step_id: str) -> dict[str, Any] | None:
        return next((s for s in self.steps if s.get("step_id") == step_id), None)

    def step_refs(self) -> list[str]:
        return [f"{self.key}#{s['step_id']}" for s in self.steps]


@dataclass
class ToolResultView:
    call_id: str
    capability: str
    status: str  # ok | error | timeout | policy_denied | not_bound
    result: Any = None
    error: str | None = None
    step_ref: str | None = None

    @property
    def ref(self) -> str:
        return f"tool:{self.call_id}"


@dataclass
class GuidanceContext:
    company_name: str
    rules: dict[str, Any]
    playbooks: list[PlaybookView] = field(default_factory=list)
    facts: list[FactHit] = field(default_factory=list)
    sections: list[SectionHit] = field(default_factory=list)
    tool_results: list[ToolResultView] = field(default_factory=list)
    write_capabilities: list[dict[str, Any]] = field(default_factory=list)  # bound + live
    no_match: bool = False

    def reference_ids(self) -> set[str]:
        ids: set[str] = set()
        for p in self.playbooks:
            ids.add(f"intent:{p.key}")
            ids.update(f"intent:{r}" for r in p.step_refs())
        ids.update(f.ref for f in self.facts)
        ids.update(s.ref for s in self.sections)
        ids.update(t.ref for t in self.tool_results if t.status == "ok")
        return ids


def _attr(v: Any) -> str:
    return escape(str(v), quote=True)


def render_rules(rules: dict[str, Any]) -> str:
    lines = []
    for key, value in rules.items():
        if not value:
            continue
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {escape(str(v))}" for v in value)
        else:
            lines.append(f"{key}: {escape(str(value))}")
    body = "\n".join(lines) if lines else "(no company-wide rules were found in the guidelines)"
    return f"<company_rules>\n{body}\n</company_rules>"


def render_playbook(p: PlaybookView) -> str:
    out = [f'<playbook intent="{_attr(p.key)}" name="{_attr(p.name)}">']
    out.append(f"<description>{escape(p.description)}</description>")
    if p.message_span:
        out.append(f"<matched_text>{escape(p.message_span)}</matched_text>")
    for c in p.conditions:
        out.append(f"<condition>{escape(c)}</condition>")
    for s in p.steps:
        attrs = f'id="{_attr(p.key)}#{_attr(s["step_id"])}" type="{_attr(s["type"])}"'
        if s.get("capability"):
            attrs += f' capability="{_attr(s["capability"])}"'
        if s.get("condition"):
            attrs += f' when="{_attr(s["condition"])}"'
        out.append(f"  <step {attrs}>{escape(s['text'])}</step>")
    if p.reply_template:
        out.append(f"<reply_template>\n{escape(p.reply_template)}\n</reply_template>")
    out.append("</playbook>")
    return "\n".join(out)


def render_context(ctx: GuidanceContext) -> str:
    parts = [render_rules(ctx.rules)]
    if ctx.playbooks:
        parts.extend(render_playbook(p) for p in ctx.playbooks)
    else:
        parts.append("<playbooks>(no playbook matched this message)</playbooks>")
    for f in ctx.facts:
        parts.append(
            f'<reference_fact id="{_attr(f.ref)}" title="{_attr(f.title)}">\n'
            f"{escape(f.text)}\n</reference_fact>"
        )
    for s in ctx.sections:
        parts.append(
            f'<guideline_section id="{_attr(s.ref)}" heading="{_attr(s.heading or "")}">\n'
            f"{escape(s.text)}\n</guideline_section>"
        )
    for t in ctx.tool_results:
        payload = t.result if t.status == "ok" else {"error": t.error}
        parts.append(
            f'<tool_result call_id="{_attr(t.ref)}" capability="{_attr(t.capability)}" '
            f'status="{_attr(t.status)}">\n{escape(json.dumps(payload, default=str))}\n'
            f"</tool_result>"
        )
    if ctx.write_capabilities:
        lines = [
            f"- {c['name']}: {c['description']} params={json.dumps(c['input_schema'])}"
            for c in ctx.write_capabilities
        ]
        parts.append("<available_actions>\n" + "\n".join(lines) + "\n</available_actions>")
    else:
        parts.append("<available_actions>(none: do not propose actions)</available_actions>")
    return "\n\n".join(parts)
