"""SearchGateway backed by pgvector + Postgres full-text in the application database."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.search import RowHit, SearchRepository
from app.search.base import FactHit, IntentCandidate, Query, SectionHit
from app.search.fusion import aggregate_by_intent, rrf


class PgVectorSearchGateway:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        ts_config: str = "english",
        ef_search: int = 100,
        examples_per_intent: int = 8,
    ) -> None:
        self._sm = sessionmaker
        self._ts = ts_config
        self._ef = ef_search
        self._per_intent = examples_per_intent

    def _repo(self, session: AsyncSession) -> SearchRepository:
        return SearchRepository(session, ts_config=self._ts, ef_search=self._ef)

    async def search_intents(self, q: Query, k: int) -> list[IntentCandidate]:
        n = max(50, k * self._per_intent)
        async with self._sm() as session, session.begin():
            repo = self._repo(session)
            vec_hits, text_hits = await repo.intent_example_hits(
                q.company_id, q.version_id, q.vector, q.text, n
            )
            aggs = aggregate_by_intent(vec_hits, text_hits)[:k]
            meta = await repo.intents_by_id([a.intent_id for a in aggs])
        out: list[IntentCandidate] = []
        for a in aggs:
            m = meta.get(a.intent_id)
            if m is None:
                continue
            out.append(
                IntentCandidate(
                    intent_id=a.intent_id,
                    key=m["key"],
                    name=m["name"],
                    description=m["description"],
                    score=a.score,
                    vector_score=a.vector_score,
                    text_score=a.text_score,
                    best_example=a.best_example,
                )
            )
        return out

    async def search_facts(self, q: Query, k: int) -> list[FactHit]:
        if k <= 0:
            return []
        async with self._sm() as session, session.begin():
            v, t = await self._repo(session).fact_hits(
                q.company_id, q.version_id, q.vector, q.text, max(k * 3, 10)
            )
        ranked = _fuse(v, t)[:k]
        return [
            FactHit(id=h.id, title=h.extra["title"], text=h.extra["body"], score=s)
            for h, s in ranked
        ]

    async def search_sections(self, q: Query, k: int) -> list[SectionHit]:
        async with self._sm() as session, session.begin():
            v, t = await self._repo(session).section_hits(
                q.company_id, q.version_id, q.vector, q.text, max(k * 3, 10)
            )
        ranked = _fuse(v, t)[:k]
        return [
            SectionHit(
                id=h.id,
                ref_label=h.extra["ref"],
                heading=h.extra["heading"],
                text=h.extra["body"],
                score=s,
            )
            for h, s in ranked
        ]

    async def healthcheck(self) -> bool:
        try:
            async with self._sm() as session:
                return await self._repo(session).ping()
        except Exception:
            return False


def _fuse(vector_hits: list[RowHit], text_hits: list[RowHit]) -> list[tuple[RowHit, float]]:
    scores = rrf([[h.id for h in vector_hits], [h.id for h in text_hits]])
    by_id = {h.id: h for h in [*vector_hits, *text_hits]}
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return [(by_id[i], s) for i, s in ordered]  # type: ignore[index]
