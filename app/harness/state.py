"""Graph state for one chat run. No checkpointer: state lives only for the request."""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

from app.ai.context import GuidanceContext, PlaybookView
from app.ai.guidance_chain import GenerationResult
from app.ai.schemas import GuidanceOutput, IntentSelection
from app.capabilities.gateway import ToolCallRecord
from app.db.models import Chat, ProposedAction
from app.search.base import FactHit, IntentCandidate, SectionHit
from app.services.common import CompanyContext


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


class GuidanceState(TypedDict, total=False):
    # --- inputs
    chat_id: uuid.UUID
    request_id: str | None
    company: CompanyContext
    customer_message: str
    agent_notes: str | None
    channel: str | None
    started_at: float
    deadline: float  # monotonic deadline for the tool agent
    # --- routing
    query_vector: list[float]
    candidates: list[IntentCandidate]
    selection: IntentSelection
    router_model: str | None
    router_tokens: tuple[int | None, int | None]
    playbooks: list[PlaybookView]
    no_match: bool
    facts: list[FactHit]
    sections: list[SectionHit]
    entities: dict[str, str]
    # --- tools
    tool_records: Annotated[list[ToolCallRecord], operator.add]
    agent_messages: list[BaseMessage]
    agent_steps: int
    agent_done: bool
    # --- generation
    context: GuidanceContext
    generation: GenerationResult
    output: GuidanceOutput
    actions: list[ProposedAction]
    references: dict[str, Any]
    # --- bookkeeping
    timings: Annotated[dict[str, int], merge_dicts]
    errors: Annotated[list[dict[str, str]], operator.add]
    chat: Chat
