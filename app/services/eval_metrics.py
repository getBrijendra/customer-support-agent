"""Routing eval metrics (pure functions)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseOutcome:
    message: str
    expected: list[str]  # empty = out of scope (negative)
    predicted: list[str]  # empty = no match
    candidates: list[str] = field(default_factory=list)
    stale: bool = False  # an expected key does not exist in the evaluated version
    source: str = "synthetic"

    @property
    def negative(self) -> bool:
        return not self.expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "expected": self.expected,
            "predicted": self.predicted,
            "candidates": self.candidates[:5],
            "stale": self.stale,
            "source": self.source,
            "correct": None if self.stale else self.correct_top1,
        }

    @property
    def correct_top1(self) -> bool:
        if self.negative:
            return not self.predicted
        return bool(self.predicted) and self.predicted[0] in self.expected


def _ratio(n: float, d: float) -> float | None:
    return round(n / d, 4) if d else None


def compute_metrics(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    scored = [o for o in outcomes if not o.stale]
    positives = [o for o in scored if not o.negative]
    negatives = [o for o in scored if o.negative]

    top1 = sum(1 for o in positives if o.correct_top1)
    recall_k = sum(1 for o in positives if set(o.expected) <= set(o.candidates))
    exact = sum(1 for o in positives if set(o.predicted) == set(o.expected))
    tp = sum(len(set(o.predicted) & set(o.expected)) for o in positives)
    fp = sum(len(set(o.predicted) - set(o.expected)) for o in positives)
    fn = sum(len(set(o.expected) - set(o.predicted)) for o in positives)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall
        else (0.0 if precision is not None and recall is not None else None)
    )

    predicted_none = [o for o in scored if not o.predicted]
    correct_none = [o for o in predicted_none if o.negative]

    per_intent: dict[str, dict[str, int]] = {}
    for o in positives:
        for key in o.expected:
            row = per_intent.setdefault(key, {"cases": 0, "correct": 0})
            row["cases"] += 1
            row["correct"] += int(key in o.predicted)
    confusions = Counter(
        (o.expected[0], o.predicted[0] if o.predicted else "<no_match>")
        for o in positives
        if not o.correct_top1
    )
    return {
        "cases": len(outcomes),
        "positives": len(positives),
        "negatives": len(negatives),
        "stale": len(outcomes) - len(scored),
        "top1_accuracy": _ratio(top1, len(positives)),
        "recall_at_k": _ratio(recall_k, len(positives)),
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": f1,
        "exact_match_rate": _ratio(exact, len(positives)),
        "no_match_precision": _ratio(len(correct_none), len(predicted_none)),
        "no_match_recall": _ratio(len(correct_none), len(negatives)),
        "per_intent": {
            k: {**v, "accuracy": _ratio(v["correct"], v["cases"])}
            for k, v in sorted(per_intent.items())
        },
        "top_confusions": [
            {"expected": e, "predicted": p, "count": n} for (e, p), n in confusions.most_common(5)
        ],
    }


@dataclass
class ApprovalDecision:
    approved: bool
    reason: str


def decide_approval(
    *,
    auto_approve: bool,
    blocking_errors: list[str],
    top1: float | None,
    threshold: float,
) -> ApprovalDecision:
    """Auto-approve only when enabled, compilation is clean and the retrieval eval passes."""
    if blocking_errors:
        return ApprovalDecision(False, "compilation has blocking errors")
    if not auto_approve:
        return ApprovalDecision(False, "auto-approval is disabled; waiting for an admin")
    if top1 is None:
        return ApprovalDecision(True, "auto-approved (no eval cases to score)")
    if top1 < threshold:
        return ApprovalDecision(
            False, f"routing eval top-1 {top1:.2f} is below the {threshold:.2f} gate"
        )
    return ApprovalDecision(True, f"auto-approved (routing eval top-1 {top1:.2f})")
