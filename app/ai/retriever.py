"""Builds the retriever for a company's ACTIVE guideline version."""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from app.core.config import Settings
from app.vectorstore.base import RetrievedChunk, VectorStoreGateway


def build_retriever(
    vector_store: VectorStoreGateway,
    *,
    collection: str,
    embedding_model: str,
    settings: Settings,
) -> BaseRetriever:
    # Query-time embeddings must match the model recorded on the version at ingest time.
    return vector_store.as_retriever(
        collection,
        top_k=settings.RAG_TOP_K,
        search_type=settings.RAG_SEARCH_TYPE,
        embedding_model=embedding_model,
    )


def build_query(customer_message: str, agent_notes: str | None) -> str:
    return f"{customer_message}\n\n{agent_notes}" if agent_notes else customer_message


async def retrieve_chunks(retriever: BaseRetriever, query: str) -> list[RetrievedChunk]:
    docs = await retriever.ainvoke(query)
    return [RetrievedChunk.from_document(d) for d in docs]
