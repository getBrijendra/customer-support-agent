"""U10 RRF + aggregation, U11 eval metrics, U12 approval decision, U13 policy,
U16 SQL statement validation, U17 SSRF guard."""

from __future__ import annotations

import pytest

from app.ai.router import retrieval_no_match
from app.capabilities.adapters.base import ConfigError, dig, render_template
from app.capabilities.adapters.sql_readonly import to_positional, validate_select
from app.capabilities.policy import check_policy
from app.capabilities.registry import ArgumentError, coerce_args, get_capability
from app.core.net import UnsafeUrl, check_url
from app.search.base import IntentCandidate
from app.search.fusion import ExampleHit, aggregate_by_intent, or_tsquery, rrf
from app.services.eval_metrics import CaseOutcome, compute_metrics, decide_approval


# ------------------------------------------------------------------- U10 fusion
def test_rrf_scores() -> None:
    scores = rrf([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert max(scores, key=scores.__getitem__) == "b"


def test_aggregate_takes_max_per_intent() -> None:
    vec = [
        ExampleHit("e1", "refund", "refund my book", vector_score=0.9),
        ExampleHit("e2", "shipping", "where is my parcel", vector_score=0.8),
        ExampleHit("e3", "refund", "money back", vector_score=0.5),
    ]
    txt = [
        ExampleHit("e2", "shipping", "where is my parcel", text_score=0.4),
        ExampleHit("e3", "refund", "money back", text_score=0.1),
    ]
    out = aggregate_by_intent(vec, txt)
    assert [a.intent_id for a in out] == ["shipping", "refund"]
    shipping, refund = out
    assert shipping.score == pytest.approx(1 / 62 + 1 / 61)
    assert shipping.vector_score == 0.8 and shipping.text_score == 0.4
    assert refund.vector_score == 0.9 and refund.text_score == 0.1  # maxima across examples
    assert refund.best_example == "money back"  # e3 fused higher than e1


def test_or_tsquery_is_safe_and_deduplicated() -> None:
    assert or_tsquery("Hi! My book's cover is torn; book REFUND please?") == (
        "book | cover | torn | refund"
    )
    assert or_tsquery("'; DROP TABLE x; --") == "drop | table"
    assert or_tsquery("the a an") == ""


def test_retrieval_no_match_rule() -> None:
    def cand(v: float | None, t: float | None) -> IntentCandidate:
        return IntentCandidate("id", "k", "n", "d", 0.1, vector_score=v, text_score=t)  # type: ignore[arg-type]

    assert retrieval_no_match([], 0.3)
    assert retrieval_no_match([cand(0.1, None)], 0.3)
    assert not retrieval_no_match([cand(0.1, 0.05)], 0.3)
    assert not retrieval_no_match([cand(0.31, None)], 0.3)


# ---------------------------------------------------------------- U11 metrics
def test_eval_metrics() -> None:
    outcomes = [
        CaseOutcome("m1", ["refund"], ["refund"], ["refund", "shipping"]),
        CaseOutcome("m2", ["refund"], ["shipping"], ["shipping", "refund"]),
        CaseOutcome("m3", ["billing", "connection"], ["billing"], ["billing", "connection"]),
        CaseOutcome("m4", ["shipping"], [], ["refund"]),
        CaseOutcome("neg1", [], [], ["refund"]),
        CaseOutcome("neg2", [], ["refund"], ["refund"]),
        CaseOutcome("stale", ["old_key"], ["refund"], ["refund"], stale=True),
    ]
    m = compute_metrics(outcomes)
    assert (m["cases"], m["positives"], m["negatives"], m["stale"]) == (7, 4, 2, 1)
    assert m["top1_accuracy"] == 0.5  # m1, m3
    assert m["recall_at_k"] == 0.75  # m4's expected key not in candidates
    # tp: m1 1, m3 1 ; fp: m2 1 ; fn: m2 1, m3 1, m4 1
    assert m["set_precision"] == round(2 / 3, 4) and m["set_recall"] == round(2 / 5, 4)
    assert m["set_f1"] == pytest.approx(0.5, abs=1e-3)
    assert m["exact_match_rate"] == 0.25
    # predicted no-match: m4 (wrong), neg1 (right)
    assert m["no_match_precision"] == 0.5 and m["no_match_recall"] == 0.5
    assert m["per_intent"]["refund"] == {"cases": 2, "correct": 1, "accuracy": 0.5}
    assert {"expected": "refund", "predicted": "shipping", "count": 1} in m["top_confusions"]


def test_eval_metrics_empty() -> None:
    m = compute_metrics([])
    assert m["top1_accuracy"] is None and m["cases"] == 0


# --------------------------------------------------------------- U12 approval
@pytest.mark.parametrize(
    ("auto", "errors", "top1", "approved"),
    [
        (True, [], 0.9, True),
        (True, [], 0.5, False),
        (True, [], None, True),
        (False, [], 1.0, False),
        (True, ["boom"], 1.0, False),
    ],
)
def test_approval_decision(
    auto: bool, errors: list[str], top1: float | None, approved: bool
) -> None:
    d = decide_approval(auto_approve=auto, blocking_errors=errors, top1=top1, threshold=0.8)
    assert d.approved is approved and d.reason


# ----------------------------------------------------------------- U13 policy
def test_policy_caps() -> None:
    refund = get_capability("refund.create")
    discount = get_capability("discount.issue")
    assert refund and discount
    assert check_policy(refund, {"amount": 20.0}, {"max_amount": 50}).ok
    over = check_policy(refund, {"amount": 80.0}, {"max_amount": 50})
    assert not over.ok and "exceeds max_amount" in over.violations[0]
    assert not check_policy(refund, {"amount": -1.0}, {}).ok
    assert not check_policy(discount, {"percent": 15.0}, {"max_pct": 10}).ok
    assert not check_policy(refund, {"amount": 1.0}, {"blocked": True}).ok
    assert not check_policy(
        refund, {"amount": 1.0, "reason": "x"}, {"allowed_values": {"reason": ["damaged"]}}
    ).ok
    reads_always_ok = get_capability("order.get")
    assert reads_always_ok and check_policy(reads_always_ok, {}, {"blocked": True}).ok


def test_argument_coercion() -> None:
    refund = get_capability("refund.create")
    assert refund
    assert coerce_args(refund, {"amount": "12.5", "order_id": 42}) == {
        "amount": 12.5,
        "order_id": "42",
    }
    with pytest.raises(ArgumentError, match="Missing"):
        coerce_args(refund, {"order_id": "1"})
    with pytest.raises(ArgumentError, match="must be a number"):
        coerce_args(refund, {"amount": "lots"})
    with pytest.raises(ArgumentError, match="Unknown"):
        coerce_args(refund, {"amount": 1, "admin": True})


# ------------------------------------------------------------ U16 SQL checks
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, status FROM orders WHERE id = :order_id",
        "  select * from t -- trailing comment\n;",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT created_at::text FROM t WHERE email = :email",
    ],
)
def test_select_statements_accepted(sql: str) -> None:
    validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "UPDATE orders SET x = 1",
        "SELECT * INTO backup FROM orders",
        "SELECT pg_sleep(10)",
        "/* sneaky */ INSERT INTO t VALUES (1)",
        "",
        "SET statement_timeout = 0",
    ],
)
def test_non_select_statements_rejected(sql: str) -> None:
    with pytest.raises(ConfigError):
        validate_select(sql)


def test_named_to_positional_params() -> None:
    sql, names = to_positional("SELECT a::text FROM t WHERE x = :email AND y = :id OR z = :email")
    assert sql == "SELECT a::text FROM t WHERE x = $1 AND y = $2 OR z = $1"
    assert names == ["email", "id"]


def test_templates_and_dig() -> None:
    assert render_template({"u": "/orders/{order_id}", "q": ["{missing}"]}, {"order_id": 7}) == {
        "u": "/orders/7",
        "q": [""],
    }
    assert dig({"data": {"items": [{"id": 1}]}}, "data.items.0.id") == 1
    assert dig({"a": 1}, "b.c") is None


# ------------------------------------------------------------------- U17 SSRF
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8080",
        "http://10.0.0.5/x",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "ftp://example.com/file",
        "file:///etc/passwd",
    ],
)
def test_ssrf_guard_blocks_private_and_non_http(url: str) -> None:
    with pytest.raises(UnsafeUrl):
        check_url(url, allow_private=False, resolve=False)


def test_ssrf_guard_allows_public_and_local_when_enabled() -> None:
    check_url("https://93.184.216.34/api", allow_private=False, resolve=False)
    check_url("https://api.example.com/v1", allow_private=False, resolve=False)
    check_url("http://127.0.0.1:9000/mcp", allow_private=True)
