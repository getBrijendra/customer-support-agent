"""Structured LLM input/output models.

Gemini structured output handles plain objects/lists best, so free-form maps are modelled as
lists of name/value pairs and converted in code.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

StepType = Literal["say", "ask", "lookup", "action", "escalate"]
Confidence = Literal["high", "medium", "low"]
PerformedBy = Literal["ai", "ai_verified", "needs_approval", "agent"]
SectionKind = Literal["intent", "rule", "fact", "ignored"]

_KEY_RE = re.compile(r"[^a-z0-9]+")


def to_key(text: str) -> str:
    return _KEY_RE.sub("_", text.lower()).strip("_")[:80] or "intent"


# ------------------------------------------------------------------- compiler
class CompiledStep(BaseModel):
    step_id: str = Field(description="s1, s2, ... unique within the intent, in order")
    type: StepType = Field(
        description="say = tell the customer; ask = ask the customer for info; lookup = read "
        "data in a company system; action = change something (refund, reset, ship); "
        "escalate = hand over to another team"
    )
    text: str = Field(description="The step, as written in the guidelines (short)")
    condition: str | None = Field(default=None, description="When the step applies, if stated")
    capability: str | None = Field(
        default=None,
        description="For lookup/action steps only: the matching capability name from the "
        "registry, or null if none fits",
    )


class CompiledIntent(BaseModel):
    key: str = Field(description="stable snake_case key, e.g. refund_request")
    name: str
    description: str = Field(description="One sentence: which customer messages this covers")
    conditions: list[str] = Field(default_factory=list)
    steps: list[CompiledStep]
    reply_template: str | None = Field(default=None, description="Sample reply, verbatim")
    placeholders: list[str] = Field(
        default_factory=list, description="Placeholders in the template, e.g. Name, Date"
    )
    source_sections: list[str] = Field(description="Section refs (S1, S2...) it came from")

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        return to_key(v)


class CompiledRules(BaseModel):
    tone: str | None = None
    greeting: str | None = None
    sign_off: str | None = None
    empathy: str | None = None
    prohibitions: list[str] = Field(default_factory=list)
    escalation_defaults: list[str] = Field(default_factory=list)
    formatting: list[str] = Field(default_factory=list)
    source_sections: list[str] = Field(default_factory=list)

    def as_rules(self) -> dict[str, Any]:
        return self.model_dump(exclude={"source_sections"})

    def is_empty(self) -> bool:
        return not any(v for v in self.as_rules().values())


class CompiledFact(BaseModel):
    title: str
    text: str
    source_sections: list[str] = Field(default_factory=list)


class SectionMapping(BaseModel):
    section_ref: str
    kind: SectionKind


class CompiledPlaybook(BaseModel):
    intents: list[CompiledIntent]
    rules: CompiledRules = Field(default_factory=CompiledRules)
    facts: list[CompiledFact] = Field(default_factory=list)
    section_kinds: list[SectionMapping] = Field(default_factory=list)


class IntentExamples(BaseModel):
    intent_key: str
    index_examples: list[str] = Field(description="Realistic customer messages for the index")
    eval_examples: list[str] = Field(description="Different held-out customer messages")


class ExampleSet(BaseModel):
    intents: list[IntentExamples]
    negatives: list[str] = Field(
        default_factory=list, description="Customer messages none of the intents cover"
    )


# --------------------------------------------------------------------- router
class SelectedIntent(BaseModel):
    key: str
    confidence: Confidence
    message_span: str | None = Field(
        default=None, description="The part of the message this intent covers"
    )


class Entity(BaseModel):
    name: str = Field(description="order_id | email | device_id | amount | other")
    value: str


class IntentSelection(BaseModel):
    intents: list[SelectedIntent] = Field(default_factory=list)
    no_match: bool = False
    entities: list[Entity] = Field(default_factory=list)

    def entity_map(self) -> dict[str, str]:
        return {e.name: e.value for e in self.entities if e.value}


# ------------------------------------------------------------------ generator
class ActionArgument(BaseModel):
    name: str
    value: str


class PlanStep(BaseModel):
    step_ref: str = Field(description='"intent_key#sN" of the playbook step')
    type: StepType
    instruction: str = Field(description="What to do for this case (internal, for the agent)")
    performed_by: PerformedBy = Field(
        default="agent", description="Set by the system after validation"
    )
    tool_call_id: str | None = None
    proposed_action_id: str | None = None


class IntentGuidance(BaseModel):
    intent_key: str
    confidence: Confidence
    steps: list[PlanStep]
    missing_information: list[str] = Field(default_factory=list)


class ProposedActionOut(BaseModel):
    capability: str
    arguments: list[ActionArgument] = Field(default_factory=list)
    reason: str
    step_ref: str | None = None

    def args(self) -> dict[str, Any]:
        return {a.name: a.value for a in self.arguments}


class GuidanceOutput(BaseModel):
    """Structured guidance the generator returns for the support agent."""

    detected_intents: list[IntentGuidance] = Field(default_factory=list)
    recommended_response: str = Field(
        description="One ready-to-send reply covering all intents, in the company's tone"
    )
    requires_escalation: bool
    escalation_reason: str | None = None
    references: list[str] = Field(
        default_factory=list,
        description='Ids used: "intent:key#sN", "fact:<id>", "section:<id>", "tool:<id>"',
    )
    proposed_actions: list[ProposedActionOut] = Field(default_factory=list)
    confidence: Confidence
    missing_information: list[str] = Field(default_factory=list)
