"""Synthetic customer messages per intent: index examples (routing) + held-out eval examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from app.ai.prompts import examples_prompt
from app.ai.providers import invoke_structured
from app.ai.schemas import CompiledIntent, ExampleSet


@dataclass
class ExampleResult:
    examples: ExampleSet
    input_tokens: int | None = None
    output_tokens: int | None = None


class ExampleGenerator(Protocol):
    async def generate(
        self,
        intents: list[CompiledIntent],
        *,
        company_name: str,
        n_index: int,
        n_eval: int,
        n_negatives: int,
    ) -> ExampleResult: ...


def describe_intents(intents: list[CompiledIntent]) -> str:
    lines = []
    for it in intents:
        steps = "; ".join(s.text for s in it.steps[:6])
        lines.append(f"- key={it.key} | {it.name}: {it.description} | steps: {steps}")
    return "\n".join(lines)


class LLMExampleGenerator:
    def __init__(self, llm: BaseChatModel, *, timeout: float) -> None:
        self._llm = llm
        self._timeout = timeout

    async def generate(
        self,
        intents: list[CompiledIntent],
        *,
        company_name: str,
        n_index: int,
        n_eval: int,
        n_negatives: int,
    ) -> ExampleResult:
        res = await invoke_structured(
            self._llm,
            examples_prompt(),
            ExampleSet,
            {
                "company_name": company_name,
                "n_index": n_index,
                "n_eval": n_eval,
                "n_negatives": n_negatives,
                "intents": describe_intents(intents),
            },
            timeout_s=self._timeout,
        )
        return ExampleResult(res.output, res.input_tokens, res.output_tokens)
