from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GuidanceOutput(BaseModel):
    """Structured guidance the LLM must return for the support agent."""

    recommended_response: str = Field(
        description="Ready-to-send reply to the customer, in the company's tone and format."
    )
    handling_instructions: list[str] = Field(
        description="Internal steps for the support agent (e.g. verify identity, issue refund)."
    )
    requires_escalation: bool = Field(description="True if the case must be escalated.")
    escalation_reason: str | None = Field(
        default=None, description="Why escalation is needed, or null."
    )
    detected_intent: str = Field(
        description="Short snake_case label(s) for the customer intent, e.g. refund_request."
    )
    guideline_references: list[str] = Field(
        description="chunk_id values of the guideline excerpts that justify the answer."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="high only if the excerpts clearly cover the case; low if they don't."
    )
    missing_information: list[str] = Field(
        description="Information the agent should ask the customer for, if any."
    )
