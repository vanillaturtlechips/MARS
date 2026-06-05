"""
OpenAI LLM backend — structured output (for fleet/strategy agents) and
tool-calling loop (for the Failure Analysis Investigator).

Structured output:
  Uses response_format={"type": "json_schema"} to enforce the diagnosis/
  strategy/fleet schemas.  Parsed from message.content as JSON.

Tool-calling (investigator):
  chat_with_tools() drives one step of the ReAct loop.  The caller manages
  the conversation array; this method appends the assistant turn and
  returns the parsed response so the caller can execute tools and continue.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from mars.config import OPENAI_API_KEY, OPENAI_MODEL, LLM_TEMPERATURE
from mars.llm.client import ToolCallRequest, ToolCallResponse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output client (Operations Strategy + Fleet State agents)
# ---------------------------------------------------------------------------

class OpenAIStructuredClient:
    """
    Calls OpenAI with response_format=json_schema to enforce the output shape.
    Temperature is kept low (default 0) for stable, defensible outputs.
    """

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        self._client = OpenAI(api_key=api_key)
        self._model  = model
        self._temperature = temperature

    def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        temp = temperature if temperature is not None else self._temperature

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temp,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_output",
                    "schema": output_schema,
                    "strict": False,
                },
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            log.error("OpenAI returned non-JSON: %s", content[:200])
            raise RuntimeError(f"OpenAI structured output was not valid JSON: {exc}") from exc


# ToolCallRequest and ToolCallResponse are defined in client.py
# (imported above) to avoid circular imports and to let tests use them
# without loading the openai package.

# ---------------------------------------------------------------------------
# Investigator client (Failure Analysis ReAct loop)
# ---------------------------------------------------------------------------

class OpenAIInvestigatorClient:
    """
    Drives the ReAct tool-calling loop for the Failure Analysis Investigator.

    chat_with_tools() makes ONE API call and returns the response.
    The caller (FailureAnalysisAgent) manages the conversation array,
    executes the returned tool calls, appends results, and calls again.
    """

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        self._client      = OpenAI(api_key=api_key)
        self._model       = model
        self._temperature = temperature

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> ToolCallResponse:
        """
        One step of the tool-calling loop.

        messages: the full conversation so far (user + assistant + tool results).
                  The system prompt is prepended here if provided and not already
                  in messages.
        tools:    OpenAI tool definition dicts (type/function/description/parameters).

        Returns ToolCallResponse with:
          finish_reason="stop"       → model produced a final message; content set
          finish_reason="tool_calls" → model wants to call tools; tool_calls populated
          finish_reason="length"     → context window exceeded (treat as non-convergence)
        """
        full_messages: list[dict] = []
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            tools=tools,
            tool_choice="auto",
            messages=full_messages,
        )

        choice       = response.choices[0]
        finish       = choice.finish_reason
        msg          = choice.message
        tool_calls   = []

        if finish == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return ToolCallResponse(
            finish_reason=finish,
            content=msg.content,
            tool_calls=tool_calls,
        )

    def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Final diagnosis step: produce the structured output from the investigation
        transcript.  Uses json_schema response_format (no tools).
        """
        temp = temperature if temperature is not None else self._temperature

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temp,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "diagnosis",
                    "schema": output_schema,
                    "strict": False,
                },
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            log.error("OpenAI final step non-JSON: %s", content[:200])
            raise RuntimeError(f"OpenAI final structured output was not valid JSON: {exc}") from exc
