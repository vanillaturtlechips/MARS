"""
Provider-agnostic LLM client abstraction.

Two concerns are separated:
  LLMClient   — structured-output completions (fleet/strategy agents)
  Embedder    — embedding generation (OpenAI text-embedding-3-small, 1536 dims)

The Failure Analysis Investigator uses OpenAIInvestigatorClient directly
(not through LLMClient) because it needs the tool-calling loop API.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-calling data types (defined here to avoid importing openai in tests)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRequest:
    id:        str
    name:      str
    arguments: dict[str, Any]


@dataclass
class ToolCallResponse:
    finish_reason: str                           # "stop" | "tool_calls" | "length"
    content:       str | None
    tool_calls:    list[ToolCallRequest] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM (structured output — fleet/strategy agents)
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Return a dict matching the given JSON schema via structured output."""

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
# Mock implementations — no network calls; used in all tests
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Returns canned output; inject via constructor for deterministic tests."""

    def __init__(self, canned_output: dict[str, Any] | None = None):
        self._canned = canned_output or {}

    def complete_structured(self, system_prompt, user_message, output_schema, *, temperature=0.0):
        log.debug("MockLLMClient returning canned output")
        return dict(self._canned)   # return a copy so callers can mutate safely


class MockEmbedder(Embedder):
    """Returns deterministic zero-vectors; dimensionality configurable."""

    def __init__(self, dim: int = 1536):  # default matches schema vector(1536)
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.0] * self._dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client(provider: str | None = None) -> LLMClient:
    """
    Return the active LLM client.

    provider overrides the LLM_PROVIDER config value if supplied.
    For the investigator (tool-calling loop) use get_investigator_client() instead.
    """
    from mars.config import LLM_PROVIDER
    p = provider or LLM_PROVIDER

    if p == "openai":
        from mars.llm.openai_client import OpenAIStructuredClient
        return OpenAIStructuredClient()
    if p == "anthropic":
        from mars.llm.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient()
    if p == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown LLM provider: {p!r}")


def get_investigator_client(provider: str | None = None):
    """
    Return the investigator client (tool-calling loop).

    Returns OpenAIInvestigatorClient for openai/default, or a mock
    (ToolCallMockClient) for testing.
    """
    from mars.config import LLM_PROVIDER
    p = provider or LLM_PROVIDER

    if p == "openai":
        from mars.llm.openai_client import OpenAIInvestigatorClient
        return OpenAIInvestigatorClient()
    if p in ("mock", "anthropic"):
        return ToolCallMockClient()
    raise ValueError(f"Unknown investigator provider: {p!r}")


def get_embedder(provider: str | None = None) -> Embedder:
    """
    Return the active embedder.  Defaults to OpenAI text-embedding-3-small.
    Dimension MUST match the schema vector(N) column.
    """
    from mars.config import LLM_PROVIDER
    p = provider or (LLM_PROVIDER if LLM_PROVIDER != "anthropic" else "openai")

    if p == "openai":
        from mars.llm.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    if p == "voyage":
        from mars.llm.voyage_embedder import VoyageEmbedder
        return VoyageEmbedder()
    if p == "mock":
        return MockEmbedder()
    raise ValueError(f"Unknown embedding provider: {p!r}")


# ---------------------------------------------------------------------------
# Mock tool-call client — for tests; returns no tool calls (immediate stop)
# ---------------------------------------------------------------------------

class ToolCallMockClient:
    """
    Mock investigator client for tests.

    On the first call to chat_with_tools() it requests one query_failures
    tool call — this populates the transcript so DV grounding checks pass.
    On the second call (or if skip_tool_call=True) it returns stop.
    complete_structured() returns the canned_diagnosis.
    """

    def __init__(
        self,
        canned_diagnosis: dict[str, Any] | None = None,
        skip_tool_call: bool = False,
    ):
        self._canned_diagnosis = canned_diagnosis
        self._skip_tool_call   = skip_tool_call
        self._called           = False

    def chat_with_tools(self, messages, tools, system_prompt=None):
        # Uses ToolCallResponse / ToolCallRequest defined in this module
        # (no openai import needed here — safe for test environments).
        if not self._skip_tool_call and not self._called:
            self._called = True
            return ToolCallResponse(
                finish_reason="tool_calls",
                content=None,
                tool_calls=[
                    ToolCallRequest(id="mock_call_1", name="query_failures", arguments={})
                ],
            )
        return ToolCallResponse(finish_reason="stop", content=None, tool_calls=[])

    def complete_structured(self, system_prompt, user_message, output_schema, *, temperature=0.0):
        if self._canned_diagnosis:
            return dict(self._canned_diagnosis)
        return {
            "cause": "unknown",
            "scope": "isolated",
            "persistence": "persistent",
            "affected_zone": None,
            "confidence": 0.1,
            "evidence": [{"observation": "mock", "refs": ["trigger_event.robot_id"]}],
            "relied_on_precedents": [],
        }
