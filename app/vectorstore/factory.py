from __future__ import annotations

from app.core.config import Settings
from app.vectorstore.base import EmbeddingsFactory, VectorStoreGateway


def create_vector_store(
    settings: Settings, embeddings_for: EmbeddingsFactory
) -> VectorStoreGateway:
    if settings.VECTOR_BACKEND == "chroma_local":
        from app.vectorstore.chroma_local import LocalChromaGateway

        return LocalChromaGateway(
            settings.CHROMA_PERSIST_DIR, embeddings_for, settings.EMBEDDING_MODEL
        )
    if settings.VECTOR_BACKEND == "chroma_remote":
        from app.vectorstore.chroma_remote import RemoteChromaGateway

        return RemoteChromaGateway(settings, embeddings_for)
    raise ValueError(f"Unknown VECTOR_BACKEND: {settings.VECTOR_BACKEND}")
