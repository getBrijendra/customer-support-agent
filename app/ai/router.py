"""Intent selection: the router LLM picks 0..N intents from the hybrid-search candidates."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from app.ai.prompts import router_prompt
from app.ai.providers import invoke_structured
from app.ai.schemas import IntentSelection
from app.search.base import IntentCandidate


@dataclass
class SelectionResult:
    selection: IntentSelection
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class IntentSelector(Protocol):
    model_name: str

    async def select(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        candidates: list[IntentCandidate],
        max_intents: int,
    ) -> SelectionResult: ...


def render_candidates(candidates: list[IntentCandidate]) -> str:
    return "\n".join(
        f"- key={c.key} | {escape(c.name)}: {escape(c.description)}" for c in candidates
    )


def sanitize_selection(
    selection: IntentSelection, candidates: list[IntentCandidate], max_intents: int
) -> IntentSelection:
    """Drop keys that weren't candidates, de-duplicate, cap the count."""
    allowed = {c.key for c in candidates}
    picked = []
    seen: set[str] = set()
    for it in selection.intents:
        if it.key in allowed and it.key not in seen:
            picked.append(it)
            seen.add(it.key)
    picked = picked[:max_intents]
    return IntentSelection(intents=picked, no_match=not picked, entities=selection.entities)


def retrieval_no_match(candidates: list[IntentCandidate], min_score: float) -> bool:
    """No match when there are no candidates, or the best one has neither a keyword match nor
    a cosine similarity of at least `min_score`. (RRF scores are rank-based, so they can't be
    used as an absolute threshold.)"""
    if not candidates:
        return True
    best_vector = max((c.vector_score or 0.0) for c in candidates)
    any_text = any(c.text_score for c in candidates)
    return not any_text and best_vector < min_score


class LLMIntentSelector:
    def __init__(self, llm: BaseChatModel, model_name: str, *, timeout: float) -> None:
        self._llm = llm
        self.model_name = model_name
        self._timeout = timeout

    async def select(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        candidates: list[IntentCandidate],
        max_intents: int,
    ) -> SelectionResult:
        if not candidates:
            return SelectionResult(IntentSelection(no_match=True), self.model_name)
        res = await invoke_structured(
            self._llm,
            router_prompt(),
            IntentSelection,
            {
                "company_name": company_name,
                "max_intents": max_intents,
                "candidates": render_candidates(candidates),
                "agent_notes": agent_notes or "",
                "customer_message": customer_message,
            },
            timeout_s=self._timeout,
        )
        return SelectionResult(
            sanitize_selection(res.output, candidates, max_intents),
            self.model_name,
            res.input_tokens,
            res.output_tokens,
        )
