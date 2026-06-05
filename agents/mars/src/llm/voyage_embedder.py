"""
Voyage AI embeddings backend (voyage-3, 1024 dims by default).

Voyage is Anthropic's recommended embedding provider.  The API is simple:
POST https://api.voyageai.com/v1/embeddings with model + input.

We use the voyageai SDK if available, otherwise fall back to direct HTTP.
"""
from __future__ import annotations

import logging
import os

from mars.config import VOYAGE_API_KEY, VOYAGE_EMBEDDING_MODEL

log = logging.getLogger(__name__)


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str = VOYAGE_API_KEY,
        model: str = VOYAGE_EMBEDDING_MODEL,
    ):
        try:
            import voyageai
            self._client = voyageai.Client(api_key=api_key)
            self._use_sdk = True
        except ImportError:
            import httpx
            self._api_key = api_key
            self._model = model
            self._http = httpx.Client(
                base_url="https://api.voyageai.com/v1",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
            self._use_sdk = False
        self._model = model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._use_sdk:
            result = self._client.embed(texts, model=self._model)
            return [e.embedding for e in result.embeddings]
        # HTTP fallback
        resp = self._http.post(
            "/embeddings",
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]
