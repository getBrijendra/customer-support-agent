"""Pure ranking helpers: Reciprocal Rank Fusion and max-per-intent aggregation."""

from __future__ import annotations

import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

RRF_K = 60
_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do", "does", "for",
    "from", "had", "has", "have", "hi", "hello", "how", "i", "if", "in", "is", "it", "its", "me",
    "my", "no", "not", "of", "on", "or", "our", "please", "so", "that", "the", "their", "them",
    "there", "this", "to", "was", "we", "what", "when", "why", "will", "with", "would", "you",
    "your",
}  # fmt: skip


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> dict[Hashable, float]:
    """Fuse ranked lists of ids: score(id) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class ExampleHit:
    example_id: Any
    intent_id: Any
    text: str
    vector_score: float | None = None  # cosine similarity
    text_score: float | None = None  # ts_rank


@dataclass
class IntentAggregate:
    intent_id: Any
    score: float
    vector_score: float | None
    text_score: float | None
    best_example: str


def aggregate_by_intent(
    vector_hits: Sequence[ExampleHit], text_hits: Sequence[ExampleHit], k: int = RRF_K
) -> list[IntentAggregate]:
    """RRF over example rankings, then keep the MAX fused score per intent (best example wins).

    Returns intents sorted by score desc (ties broken by intent id for determinism).
    """
    fused = rrf([[h.example_id for h in vector_hits], [h.example_id for h in text_hits]], k)
    by_example: dict[Any, ExampleHit] = {}
    for h in [*vector_hits, *text_hits]:
        cur = by_example.get(h.example_id)
        if cur is None:
            by_example[h.example_id] = ExampleHit(
                h.example_id, h.intent_id, h.text, h.vector_score, h.text_score
            )
        else:
            cur.vector_score = cur.vector_score if h.vector_score is None else h.vector_score
            cur.text_score = cur.text_score if h.text_score is None else h.text_score
    best: dict[Any, IntentAggregate] = {}
    for ex_id, score in fused.items():
        ex = by_example[ex_id]
        agg = best.get(ex.intent_id)
        if agg is None:
            best[ex.intent_id] = IntentAggregate(
                ex.intent_id, score, ex.vector_score, ex.text_score, ex.text
            )
            continue
        if score > agg.score:
            agg.score, agg.best_example = score, ex.text
        agg.vector_score = _max(agg.vector_score, ex.vector_score)
        agg.text_score = _max(agg.text_score, ex.text_score)
    return sorted(best.values(), key=lambda a: (-a.score, str(a.intent_id)))


def _max(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def or_tsquery(text: str, max_terms: int = 32) -> str:
    """Build an OR tsquery string ('refund | damaged | book') from free text.

    Customer messages are long; an AND query (plainto/websearch) would rarely match anything.
    Only alphanumeric tokens are kept, so the result is always safe for to_tsquery().
    """
    seen: list[str] = []
    for word in _WORD.findall(text.lower()):
        if len(word) < 2 or word in _STOP or word in seen:
            continue
        seen.append(word)
        if len(seen) >= max_terms:
            break
    return " | ".join(seen)
