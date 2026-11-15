"""Generator (structured guidance) and tool-calling model interfaces + LangChain implementations.

Tests swap both for fakes with the same interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from app.ai.context import GuidanceContext, render_context
from app.ai.prompts import generator_prompt
from app.ai.providers import LLMCallError, invoke_structured
from app.ai.schemas import GuidanceOutput


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
        context: GuidanceContext,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
    ) -> GenerationResult: ...


def generator_inputs(
    context: GuidanceContext, customer_message: str, agent_notes: str | None, channel: str | None
) -> dict[str, Any]:
    return {
        "company_name": context.company_name,
        "context": render_context(context),
        "customer_message": customer_message,
        "agent_notes": agent_notes or "",
        "channel": channel or "unspecified",
    }


class LangChainGuidanceGenerator:
    def __init__(self, llm: BaseChatModel, model_name: str, timeout_seconds: float) -> None:
        self.model_name = model_name
        self._llm = llm
        # Hard ceiling on top of the client's own per-attempt timeout + retries.
        self._timeout = timeout_seconds * 3

    async def generate(
        self,
        *,
        context: GuidanceContext,
        customer_message: str,
        agent_notes: str | None,
        channel: str | None,
    ) -> GenerationResult:
        try:
            res = await invoke_structured(
                self._llm,
                generator_prompt(),
                GuidanceOutput,
                generator_inputs(context, customer_message, agent_notes, channel),
                timeout_s=self._timeout,
            )
        except LLMCallError as exc:
            raise GenerationError(str(exc)) from exc
        return GenerationResult(res.output, self.model_name, res.input_tokens, res.output_tokens)


class ToolCallingModel(Protocol):
    """One step of the bounded tool loop: given the conversation so far and the tool schemas,
    return an AIMessage (with `tool_calls`, or without them when done)."""

    model_name: str

    async def step(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage: ...


class LangChainToolCallingModel:
    def __init__(self, llm: BaseChatModel, model_name: str, timeout_seconds: float) -> None:
        self._llm = llm
        self.model_name = model_name
        self._timeout = timeout_seconds

    async def step(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        bound = self._llm.bind_tools(tools)
        msg = await asyncio.wait_for(bound.ainvoke(messages), timeout=self._timeout)
        if not isinstance(msg, AIMessage):
            raise LLMCallError("tool model returned a non-AI message")
        return msg


def tool_schema(name: str, description: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-style function schema accepted by `bind_tools` for every provider."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": input_schema},
    }
