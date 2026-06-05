"""
Provider-agnostic LLM client abstraction.

Two concerns are separated:
  LLMClient   — structured-output chat completions (Anthropic Claude)
  Embedder    — embedding generation (Voyage AI voyage-3, 1024 dims)

Both can be swapped by subclassing and registering a different backend.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM (structured output)
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Return a dict matching the given JSON schema via tool-use / structured output."""

    @abstractmethod
    def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Mock implementations — used in tests; no network calls
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Returns canned output; inject via constructor for deterministic tests."""

    def __init__(self, canned_output: dict[str, Any] | None = None):
        self._canned = canned_output or {}

    def complete_structured(self, system_prompt, user_message, output_schema, *, temperature=0.0):
        log.debug("MockLLMClient returning canned output")
        return self._canned


class MockEmbedder(Embedder):
    """Returns deterministic zero-vectors; dimensionality configurable."""

    def __init__(self, dim: int = 1024):
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.0] * self._dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client(provider: str = "anthropic") -> LLMClient:
    if provider == "anthropic":
        from mars.llm.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient()
    if provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def get_embedder(provider: str = "voyage") -> Embedder:
    if provider == "voyage":
        from mars.llm.voyage_embedder import VoyageEmbedder
        return VoyageEmbedder()
    if provider == "openai":
        from mars.llm.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    if provider == "mock":
        return MockEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider!r}")
