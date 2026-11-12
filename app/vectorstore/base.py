"""Vector store abstraction. Nothing outside `app/vectorstore` may import `chromadb`."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

EmbeddingsFactory = Callable[[str], Embeddings]
"""Returns the embeddings client for a model name (query time must use the ingest-time model)."""

# Chroma: 3-512 chars of [a-zA-Z0-9._-], starting and ending with an alphanumeric.
CHROMA_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


def make_collection_name(company_id: UUID, version_number: int) -> str:
    return f"gl_{company_id.hex[:12]}_v{version_number}"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float | None
    section: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(cls, doc: Document) -> RetrievedChunk:
        meta = dict(doc.metadata)
        score = meta.pop("score", None)
        chunk_id = doc.id or str(meta.get("chunk_id", ""))
        return cls(
            chunk_id=chunk_id,
            text=doc.page_content,
            score=None if score is None else round(float(score), 4),
            section=meta.get("section") or None,
            metadata=meta,
        )

    def to_record(self) -> dict[str, Any]:
        """Shape stored in `chats.retrieved_chunks`."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "section": self.section,
            "text": self.text,
            "company_slug": self.metadata.get("company_slug"),
            "version_number": self.metadata.get("version_number"),
        }


@runtime_checkable
class VectorStoreGateway(Protocol):
    def collection_name(self, company_id: UUID, version_number: int) -> str: ...

    async def add_chunks(
        self,
        collection: str,
        docs: list[Document],
        ids: list[str],
        *,
        embedding_model: str | None = None,
    ) -> None: ...

    def as_retriever(
        self,
        collection: str,
        *,
        top_k: int,
        search_type: str,
        embedding_model: str | None = None,
    ) -> BaseRetriever: ...

    async def delete_collection(self, collection: str) -> None: ...

    async def collection_exists(self, collection: str) -> bool: ...

    async def healthcheck(self) -> bool: ...
