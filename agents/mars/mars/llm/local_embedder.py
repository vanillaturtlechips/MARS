"""Local sentence-transformers embedder (default BAAI/bge-small-en-v1.5, 384-dim).

No API key, runs on CPU — for Anthropic(Haiku)-only setups so RAG embeddings
don't require an OpenAI/Voyage key (Anthropic has no embedding API).

To use:
  EMBEDDING_PROVIDER=local
  EMBEDDING_DIM=384                         # must match the pgvector column
  (optional) LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
  uv pip install sentence-transformers

bge models are trained for cosine similarity; we L2-normalize so dot == cosine.
"""
from __future__ import annotations

import logging

from mars.config import LOCAL_EMBEDDING_MODEL

log = logging.getLogger(__name__)


class LocalEmbedder:
    def __init__(self, model: str = LOCAL_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        log.info("[local_embedder] loading %s (CPU)", model)
        self._model = SentenceTransformer(model)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vecs]
