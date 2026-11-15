"""Search & routing index interface. Only `search/pgvector.py` knows about pgvector."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class IntentCandidate:
    intent_id: uuid.UUID
    key: str
    name: str
    description: str
    score: float  # fused RRF score
    vector_score: float | None = None  # best cosine similarity of its examples
    text_score: float | None = None  # best ts_rank of its examples
    best_example: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": round(self.score, 6),
            "vector_score": None if self.vector_score is None else round(self.vector_score, 4),
            "text_score": None if self.text_score is None else round(self.text_score, 4),
            "best_example": self.best_example,
        }


@dataclass
class FactHit:
    id: uuid.UUID
    title: str
    text: str
    score: float

    @property
    def ref(self) -> str:
        return f"fact:{self.id}"


@dataclass
class SectionHit:
    id: uuid.UUID
    ref_label: str  # "S3"
    heading: str | None
    text: str
    score: float = 0.0

    @property
    def ref(self) -> str:
        return f"section:{self.id}"


@dataclass
class Query:
    company_id: uuid.UUID
    version_id: uuid.UUID
    vector: list[float]
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


class SearchGateway(Protocol):
    async def search_intents(self, q: Query, k: int) -> list[IntentCandidate]: ...
    async def search_facts(self, q: Query, k: int) -> list[FactHit]: ...
    async def search_sections(self, q: Query, k: int) -> list[SectionHit]: ...
    async def healthcheck(self) -> bool: ...
