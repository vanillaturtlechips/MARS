"""
OpenAI embeddings backend (text-embedding-3-small by default).
Used for pgvector RAG; the LLM backend is Anthropic.
"""
from __future__ import annotations

import logging

import openai

from mars.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

log = logging.getLogger(__name__)


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_EMBEDDING_MODEL,
    ):
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
