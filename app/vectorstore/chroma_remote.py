"""Remote Chroma: a self-hosted Chroma server (HttpClient) or Chroma Cloud (CloudClient).

Selected with VECTOR_BACKEND=chroma_remote plus the CHROMA_* settings. Behaviour is identical to the
local gateway because both share ChromaGatewayBase.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.vectorstore.base import EmbeddingsFactory
from app.vectorstore.chroma_common import ChromaGatewayBase


class RemoteChromaGateway(ChromaGatewayBase):
    def __init__(self, settings: Settings, embeddings_for: EmbeddingsFactory) -> None:
        api_key = settings.CHROMA_API_KEY.get_secret_value() if settings.CHROMA_API_KEY else None
        client_settings = ChromaSettings(anonymized_telemetry=False)
        if api_key and not settings.CHROMA_HOST:
            # Chroma Cloud
            client = chromadb.CloudClient(
                tenant=settings.CHROMA_TENANT,
                database=settings.CHROMA_DATABASE,
                api_key=api_key,
                settings=client_settings,
            )
        else:
            if not settings.CHROMA_HOST:
                raise ValueError("CHROMA_HOST (or CHROMA_API_KEY for Chroma Cloud) is required")
            headers = {"x-chroma-token": api_key} if api_key else None
            client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT or 8000,
                headers=headers,
                tenant=settings.CHROMA_TENANT or chromadb.DEFAULT_TENANT,
                database=settings.CHROMA_DATABASE or chromadb.DEFAULT_DATABASE,
                settings=client_settings,
            )
        super().__init__(client, embeddings_for, settings.EMBEDDING_MODEL)
