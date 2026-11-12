"""LCEL chain: prompt -> LLM structured output, behind a small `GuidanceGenerator` interface.

Retrieval happens just before (see GuidanceService) so it can be timed separately; tests swap
the generator for a fake with the same interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from app.ai.prompts import build_prompt, prompt_inputs
from app.ai.schemas import GuidanceOutput
from app.vectorstore.base import RetrievedChunk


class GenerationError(Exception):
    """Raised when the LLM call fails or returns unparseable output."""


@dataclass
class GenerationResult:
    output: GuidanceOutput
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class GuidanceGenerator(Protocol):
    model_name: str

    async def generate(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult: ...


def build_guidance_chain(llm: BaseChatModel) -> Runnable[dict[str, Any], dict[str, Any]]:
    structured = llm.with_structured_output(GuidanceOutput, include_raw=True)
    return build_prompt() | structured  # type: ignore[return-value]


class LangChainGuidanceGenerator:
    def __init__(self, llm: BaseChatModel, model_name: str, timeout_seconds: float) -> None:
        self.model_name = model_name
        self._chain = build_guidance_chain(llm)
        # Hard ceiling on top of the client's own per-attempt timeout + retries.
        self._timeout = timeout_seconds * 3

    async def generate(
        self,
        *,
        company_name: str,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        inputs = prompt_inputs(
            company_name=company_name,
            customer_message=customer_message,
            agent_notes=agent_notes,
            channel=channel,
            chunks=chunks,
        )
        try:
            result = await asyncio.wait_for(self._chain.ainvoke(inputs), timeout=self._timeout)
        except TimeoutError as exc:
            raise GenerationError("LLM call timed out") from exc
        except Exception as exc:
            raise GenerationError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

        parsed = result.get("parsed")
        raw = result.get("raw")
        if parsed is None:
            err = result.get("parsing_error")
            raise GenerationError(f"LLM returned unparseable output: {err}")
        if isinstance(parsed, dict):
            parsed = GuidanceOutput.model_validate(parsed)

        in_tok = out_tok = None
        if isinstance(raw, AIMessage) and raw.usage_metadata:
            in_tok = raw.usage_metadata.get("input_tokens")
            out_tok = raw.usage_metadata.get("output_tokens")
        return GenerationResult(
            output=parsed, model=self.model_name, input_tokens=in_tok, output_tokens=out_tok
        )
