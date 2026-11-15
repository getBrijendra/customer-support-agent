"""Hybrid search SQL: pgvector cosine + Postgres full-text, always filtered by company + version.

Postgres-only. Every query runs inside one transaction that first sets
`hnsw.iterative_scan = relaxed_order` so filtered HNSW scans keep returning rows for small
tenants (pgvector >= 0.8), and `hnsw.ef_search` for recall.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.search.fusion import ExampleHit, or_tsquery

# table -> (text expression returned, extra columns)
_TABLES: dict[str, str] = {
    "intent_examples": "e.text",
    "reference_facts": "e.title || E'\\n' || e.text",
    "guideline_sections": "coalesce(e.heading, '') || E'\\n' || e.text",
}


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


@dataclass
class RowHit:
    id: uuid.UUID
    text: str
    extra: dict[str, Any]
    vector_score: float | None = None
    text_score: float | None = None


class SearchRepository:
    def __init__(self, session: AsyncSession, *, ts_config: str, ef_search: int) -> None:
        self.session = session
        self.ts_config = ts_config
        self.ef_search = int(ef_search)

    async def _prepare(self) -> None:
        await self.session.execute(text(f"SET LOCAL hnsw.ef_search = {self.ef_search}"))
        await self.session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))

    async def _hybrid(
        self,
        table: str,
        company_id: uuid.UUID,
        version_id: uuid.UUID,
        vec: list[float],
        query_text: str,
        n: int,
        extra_cols: str,
        extra_where: str = "",
    ) -> tuple[list[RowHit], list[RowHit]]:
        await self._prepare()
        text_expr = _TABLES[table]
        base = f"FROM {table} e WHERE e.company_id = :c AND e.version_id = :v {extra_where}"
        params: dict[str, Any] = {"c": company_id, "v": version_id, "n": n}
        vrows = (
            await self.session.execute(
                text(
                    f"SELECT e.id, {text_expr} AS t, {extra_cols}, "
                    f"1 - (e.embedding <=> CAST(:q AS vector)) AS score "
                    f"{base} AND e.embedding IS NOT NULL "
                    f"ORDER BY e.embedding <=> CAST(:q AS vector) LIMIT :n"
                ),
                {**params, "q": vector_literal(vec)},
            )
        ).mappings()
        vector_hits = [_row(r, "vector") for r in vrows]

        text_hits: list[RowHit] = []
        tsq = or_tsquery(query_text)
        if tsq:
            trows = (
                await self.session.execute(
                    text(
                        f"SELECT e.id, {text_expr} AS t, {extra_cols}, ts_rank(e.tsv, q) AS score "
                        f"FROM {table} e, to_tsquery(CAST(:cfg AS regconfig), :tq) q "
                        f"WHERE e.company_id = :c AND e.version_id = :v {extra_where} "
                        f"AND e.tsv @@ q ORDER BY score DESC, e.id LIMIT :n"
                    ),
                    {**params, "cfg": self.ts_config, "tq": tsq},
                )
            ).mappings()
            text_hits = [_row(r, "text") for r in trows]
        return vector_hits, text_hits

    async def intent_example_hits(
        self, company_id: uuid.UUID, version_id: uuid.UUID, vec: list[float], q: str, n: int
    ) -> tuple[list[ExampleHit], list[ExampleHit]]:
        v, t = await self._hybrid(
            "intent_examples",
            company_id,
            version_id,
            vec,
            q,
            n,
            "e.intent_id",
            "AND e.split = 'index'",
        )
        return [_example(h) for h in v], [_example(h) for h in t]

    async def fact_hits(
        self, company_id: uuid.UUID, version_id: uuid.UUID, vec: list[float], q: str, n: int
    ) -> tuple[list[RowHit], list[RowHit]]:
        return await self._hybrid(
            "reference_facts", company_id, version_id, vec, q, n, "e.title, e.text AS body"
        )

    async def section_hits(
        self, company_id: uuid.UUID, version_id: uuid.UUID, vec: list[float], q: str, n: int
    ) -> tuple[list[RowHit], list[RowHit]]:
        return await self._hybrid(
            "guideline_sections",
            company_id,
            version_id,
            vec,
            q,
            n,
            "e.ref, e.heading, e.text AS body",
        )

    async def intents_by_id(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, dict[str, Any]]:
        if not ids:
            return {}
        rows = (
            await self.session.execute(
                text("SELECT id, key, name, description FROM intents WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
        ).mappings()
        return {r["id"]: dict(r) for r in rows}

    async def ping(self) -> bool:
        row = (
            await self.session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
        ).first()
        return row is not None


def _row(r: Any, kind: str) -> RowHit:
    extra = {k: v for k, v in r.items() if k not in {"id", "t", "score"}}
    score = None if r["score"] is None else float(r["score"])
    return RowHit(
        id=r["id"],
        text=r["t"],
        extra=extra,
        vector_score=score if kind == "vector" else None,
        text_score=score if kind == "text" else None,
    )


def _example(h: RowHit) -> ExampleHit:
    return ExampleHit(
        example_id=h.id,
        intent_id=h.extra["intent_id"],
        text=h.text,
        vector_score=h.vector_score,
        text_score=h.text_score,
    )
