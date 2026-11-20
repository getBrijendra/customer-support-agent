from __future__ import annotations

import pytest

import app.resources as resource_module
from app.ai.context import GuidanceContext, PlaybookView
from app.ai.heuristic import (
    HeuristicGuidanceGenerator,
    HeuristicIntentSelector,
    HeuristicPlaybookCompiler,
)
from app.ai.providers import DeterministicHashEmbeddings
from app.resources import build_resources
from tests.conftest import make_settings


async def test_fake_provider_builds_every_ai_collaborator_without_google(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("Gemini factory must not be called in fake mode")

    monkeypatch.setattr(resource_module, "_llm", fail_if_called)
    resources = build_resources(
        make_settings(LLM_PROVIDER="fake", GOOGLE_API_KEY="must-not-be-used")
    )
    try:
        assert isinstance(resources.generator, HeuristicGuidanceGenerator)
        assert isinstance(resources.selector, HeuristicIntentSelector)
        assert isinstance(resources.compiler, HeuristicPlaybookCompiler)
        assert resources.tool_model is None
        assert len(resources.embeddings_for("fake-embedding").embed_query("hello")) == 768
    finally:
        await resources.aclose()


async def test_heuristic_generator_returns_grounded_guidance() -> None:
    context = GuidanceContext(
        company_name="Bookshop",
        rules={},
        playbooks=[
            PlaybookView(
                key="shipping_delay",
                name="Shipping Delay",
                description="A parcel is late",
                conditions=[],
                steps=[
                    {
                        "step_id": "s1",
                        "type": "ask",
                        "text": "Ask for the order number",
                        "capability": None,
                    }
                ],
                reply_template=None,
                placeholders=[],
                source_section_ids=[],
            )
        ],
    )

    result = await HeuristicGuidanceGenerator().generate(
        context=context,
        customer_message="Where is my parcel?",
        agent_notes=None,
        channel="email",
    )

    assert result.model == "heuristic-generator"
    assert result.output.detected_intents[0].intent_key == "shipping_delay"
    assert result.output.references == ["intent:shipping_delay#s1"]
    assert "order number" in result.output.recommended_response


def test_hash_embeddings_reward_shared_words() -> None:
    embeddings = DeterministicHashEmbeddings(768)
    query = embeddings.embed_query("refund for damaged book")
    related = embeddings.embed_query("damaged book refund policy")
    unrelated = embeddings.embed_query("vpn connection protocol")

    related_score = sum(a * b for a, b in zip(query, related, strict=True))
    unrelated_score = sum(a * b for a, b in zip(query, unrelated, strict=True))
    assert related_score > unrelated_score
