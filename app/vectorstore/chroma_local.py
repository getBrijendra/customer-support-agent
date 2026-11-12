"""Local persistent Chroma (single node). Create once at startup and reuse."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.vectorstore.base import EmbeddingsFactory
from app.vectorstore.chroma_common import ChromaGatewayBase


class LocalChromaGateway(ChromaGatewayBase):
    def __init__(
        self,
        persist_dir: str | Path,
        embeddings_for: EmbeddingsFactory,
        default_embedding_model: str,
    ) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        super().__init__(client, embeddings_for, default_embedding_model)
