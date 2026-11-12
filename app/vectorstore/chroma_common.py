"""Shared Chroma gateway logic, used by both the local and the remote client implementations."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import UUID

import numpy as np
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from pydantic import ConfigDict

from app.core.logging import get_logger
from app.vectorstore.base import EmbeddingsFactory, make_collection_name

logger = get_logger(__name__)

COLLECTION_CONFIGURATION: Any = {"hnsw": {"space": "cosine"}}


class ChromaScoredRetriever(BaseRetriever):
    """Retriever returning documents with `id` and a `score` (cosine similarity) in metadata.

    Supports plain similarity and MMR; the query is embedded once and MMR is computed locally on
    the candidates' embeddings, so scores are available in both modes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any
    collection_name: str
    embeddings: Embeddings
    top_k: int = 6
    search_type: str = "mmr"
    fetch_k: int = 24
    lambda_mult: float = 0.6

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        collection = self.client.get_collection(self.collection_name)
        query_vec = self.embeddings.embed_query(query)
        n = self.fetch_k if self.search_type == "mmr" else self.top_k
        n = max(1, min(n, collection.count()))
        res = collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        order = list(range(len(ids)))
        if self.search_type == "mmr" and ids:
            embs = res["embeddings"][0] if res["embeddings"] is not None else []
            order = maximal_marginal_relevance(
                np.array(query_vec, dtype=np.float32),
                [list(e) for e in embs],
                k=min(self.top_k, len(ids)),
                lambda_mult=self.lambda_mult,
            )
        out: list[Document] = []
        for i in order[: self.top_k]:
            meta = dict(metas[i] or {})
            meta["score"] = 1.0 - float(dists[i])  # cosine distance -> similarity
            out.append(Document(id=ids[i], page_content=docs[i] or "", metadata=meta))
        return out

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._get_relevant_documents, query, run_manager=run_manager.get_sync()
        )


class ChromaGatewayBase:
    """Implements VectorStoreGateway on top of any chromadb client (persistent, HTTP or cloud)."""

    def __init__(
        self, client: ClientAPI, embeddings_for: EmbeddingsFactory, default_embedding_model: str
    ) -> None:
        self._client = client
        self._embeddings_for = embeddings_for
        self._default_model = default_embedding_model

    def collection_name(self, company_id: UUID, version_number: int) -> str:
        return make_collection_name(company_id, version_number)

    def _store(self, collection: str, embedding_model: str | None) -> Chroma:
        return Chroma(
            client=self._client,
            collection_name=collection,
            embedding_function=self._embeddings_for(embedding_model or self._default_model),
            collection_configuration=COLLECTION_CONFIGURATION,
        )

    async def add_chunks(
        self,
        collection: str,
        docs: list[Document],
        ids: list[str],
        *,
        embedding_model: str | None = None,
    ) -> None:
        def _add() -> None:
            # langchain-chroma upserts, so re-running an ingestion is idempotent.
            self._store(collection, embedding_model).add_documents(docs, ids=ids)

        await asyncio.to_thread(_add)

    def as_retriever(
        self,
        collection: str,
        *,
        top_k: int,
        search_type: str,
        embedding_model: str | None = None,
    ) -> BaseRetriever:
        return ChromaScoredRetriever(
            client=self._client,
            collection_name=collection,
            embeddings=self._embeddings_for(embedding_model or self._default_model),
            top_k=top_k,
            search_type=search_type,
            fetch_k=max(top_k * 4, 20),
        )

    async def delete_collection(self, collection: str) -> None:
        def _delete() -> None:
            with contextlib.suppress(NotFoundError, ValueError):
                self._client.delete_collection(collection)

        await asyncio.to_thread(_delete)

    async def collection_exists(self, collection: str) -> bool:
        def _exists() -> bool:
            try:
                self._client.get_collection(collection)
                return True
            except (NotFoundError, ValueError):
                return False

        return await asyncio.to_thread(_exists)

    async def collection_count(self, collection: str) -> int:
        return await asyncio.to_thread(lambda: self._client.get_collection(collection).count())

    async def healthcheck(self) -> bool:
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception:
            logger.warning("vectorstore.healthcheck_failed", exc_info=True)
            return False
