"""U9 — compiler output validation + coverage, map-reduce merge, and the heuristic compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.compiler import merge_playbooks, validate_playbook
from app.ai.heuristic import HeuristicExampleGenerator, HeuristicPlaybookCompiler, classify_step
from app.ai.schemas import (
    CompiledFact,
    CompiledIntent,
    CompiledPlaybook,
    CompiledRules,
    CompiledStep,
)
from app.documents.txt_parser import TxtParser

DOC = TxtParser().parse(
    "TONE\nBe warm.\n\n1. Refunds\n- Verify the order.\n- Process the refund.\n\n"
    "2. Shipping\n- Check tracking.\n\nREFERENCE\n- Express: 1-2 days\n"
)


def _intent(key: str, refs: list[str], steps: list[CompiledStep] | None = None) -> CompiledIntent:
    return CompiledIntent(
        key=key,
        name=key.title(),
        description=f"About {key}",
        steps=steps or [CompiledStep(step_id="x", type="say", text="Hello")],
        source_sections=refs,
    )


def test_valid_playbook_and_coverage() -> None:
    pb = CompiledPlaybook(
        intents=[
            _intent(
                "refund_request",
                ["S2"],
                [
                    CompiledStep(step_id="a", type="lookup", text="Verify", capability="order.get"),
                    CompiledStep(
                        step_id="b", type="action", text="Refund", capability="refund.create"
                    ),
                ],
            ),
            _intent("shipping", ["S3"]),
        ],
        rules=CompiledRules(tone="warm", source_sections=["S1"]),
        facts=[CompiledFact(title="Express", text="1-2 days", source_sections=["S4"])],
    )
    report = validate_playbook(pb, DOC)
    assert report.ok, report.errors
    assert report.warnings == []
    assert report.coverage["sections"] == {
        "S1": "rule",
        "S2": "intent",
        "S3": "intent",
        "S4": "fact",
    }
    steps = report.playbook.intents[0].steps
    assert [s.step_id for s in steps] == ["s1", "s2"]  # renumbered
    assert steps[0].capability == "order.get" and steps[1].capability == "refund.create"


def test_fixable_problems_become_warnings() -> None:
    pb = CompiledPlaybook(
        intents=[
            _intent(
                "refunds",
                ["S2", "S99"],
                [
                    CompiledStep(step_id="s1", type="lookup", text="x", capability="nope.get"),
                    CompiledStep(step_id="s2", type="lookup", text="y", capability="refund.create"),
                    CompiledStep(step_id="s3", type="say", text="z", capability="order.get"),
                ],
            )
        ]
    )
    report = validate_playbook(pb, DOC)
    assert report.ok
    steps = report.playbook.intents[0].steps
    assert all(s.capability is None for s in steps)
    text = " ".join(report.warnings)
    assert "S99" in text and "unknown capability 'nope.get'" in text
    assert "lookup step mapped to write" in text
    assert "not covered" in text and report.coverage["unmapped"] == ["S1", "S3", "S4"]
    assert report.playbook.intents[0].source_sections == ["S2"]


@pytest.mark.parametrize(
    ("pb", "message"),
    [
        (CompiledPlaybook(intents=[]), "No intents"),
        (CompiledPlaybook(intents=[_intent("a", ["S2"]), _intent("a", ["S3"])]), "Duplicate"),
        (CompiledPlaybook(intents=[_intent("a", ["S42"])]), "does not cite"),
        (
            CompiledPlaybook(
                intents=[
                    CompiledIntent(
                        key="a", name="A", description="d", steps=[], source_sections=["S2"]
                    )
                ]
            ),
            "has no steps",
        ),
    ],
)
def test_blocking_errors(pb: CompiledPlaybook, message: str) -> None:
    report = validate_playbook(pb, DOC)
    assert not report.ok and message in " ".join(report.errors)


def test_merge_playbooks_reduce_step() -> None:
    a = CompiledPlaybook(
        intents=[_intent("refunds", ["S2"])],
        rules=CompiledRules(tone="warm", prohibitions=["never x"]),
    )
    b = CompiledPlaybook(
        intents=[_intent("refunds", ["S3"]), _intent("shipping", ["S3"])],
        rules=CompiledRules(tone="ignored", greeting="Hi", prohibitions=["never x", "never y"]),
        facts=[CompiledFact(title="t", text="x")],
    )
    merged = merge_playbooks([a, b])
    assert [i.key for i in merged.intents] == ["refunds", "shipping"]
    refunds = merged.intents[0]
    assert refunds.source_sections == ["S2", "S3"] and len(refunds.steps) == 2
    assert [s.step_id for s in refunds.steps] == ["x", "s2"]
    assert merged.rules.tone == "warm" and merged.rules.greeting == "Hi"
    assert merged.rules.prohibitions == ["never x", "never y"] and len(merged.facts) == 1


@pytest.mark.parametrize(
    ("text", "typ", "cap"),
    [
        ("Verify order details and tracking.", "lookup", "shipment.track"),
        ("Check subscription status, renewal date, and plan type.", "lookup", "subscription.get"),
        ("Ask which server location and protocol the user selected.", "ask", None),
        ("Process cancellations or refunds per policy.", "action", "refund.create"),
        ("Escalate complex disputes to the finance department.", "escalate", None),
        ("Apologize for the inconvenience.", "say", None),
        ("Assist with password resets and account unlock procedures.", "say", None),
    ],
)
def test_heuristic_step_classification(text: str, typ: str, cap: str | None) -> None:
    assert classify_step(text) == (typ, cap)


SEED = Path(__file__).resolve().parents[2] / "data" / "seed_guidelines" / "vpn_saas.txt"
SEED_TEXT = SEED.read_text(encoding="utf-8", errors="replace") if SEED.is_file() else None


async def test_heuristic_compiler_on_seed_layout() -> None:
    if SEED_TEXT is None:
        pytest.skip("seed file missing")
    doc = TxtParser().parse(SEED_TEXT)
    result = await HeuristicPlaybookCompiler().compile(doc, company_name="VPN", previous_intents=[])
    report = validate_playbook(result.playbook, doc)
    assert report.ok, report.errors
    keys = [i.key for i in report.playbook.intents]
    assert "connection_failures" in keys and "billing_and_subscription_issues" in keys
    billing = next(i for i in report.playbook.intents if i.key == "billing_and_subscription_issues")
    assert billing.reply_template and "Plan Name" in billing.placeholders
    assert report.playbook.rules.greeting  # inferred from the shared "Hello [Name]," opening
    examples = await HeuristicExampleGenerator().generate(
        report.playbook.intents, company_name="VPN", n_index=5, n_eval=2, n_negatives=2
    )
    assert len(examples.examples.intents) == len(keys)
    assert all(len(e.eval_examples) == 2 for e in examples.examples.intents)


async def test_heuristic_compiler_reuses_previous_keys() -> None:
    doc = TxtParser().parse("1. Refunds\n- Process the refund.\n")
    result = await HeuristicPlaybookCompiler().compile(
        doc, company_name="X", previous_intents=[{"key": "refund_request", "name": "Refunds"}]
    )
    assert [i.key for i in result.playbook.intents] == ["refund_request"]
