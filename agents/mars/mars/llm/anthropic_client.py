"""
Anthropic Claude backend — structured output via tool-use.

Claude does not have a native JSON-mode; we use tool_use with the output schema
as the tool's input_schema, which forces the model to call the tool with a
valid JSON object.  We parse tool_use content blocks and return the arguments.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from mars.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_TEMPERATURE
from mars.llm.client import ToolCallRequest, ToolCallResponse

log = logging.getLogger(__name__)

_TOOL_NAME = "structured_output"


def _structured_via_forced_tool(client, model, system_prompt, user_message,
                                output_schema, temperature) -> dict[str, Any]:
    """Force a single tool call whose input_schema is the desired output schema."""
    tool_def = {
        "name": _TOOL_NAME,
        "description": "Return structured output matching the schema.",
        "input_schema": output_schema,
    }
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system=system_prompt,
        tools=[tool_def],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": user_message}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == _TOOL_NAME:
            return block.input
    raw = json.dumps([b.model_dump() for b in response.content], indent=2)
    log.error("No tool_use block in response:\n%s", raw)
    raise RuntimeError("Anthropic response contained no tool_use block")


class AnthropicLLMClient:
    def __init__(self, api_key: str = ANTHROPIC_API_KEY, model: str = ANTHROPIC_MODEL):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: dict[str, Any],
        *,
        temperature: float = LLM_TEMPERATURE,
    ) -> dict[str, Any]:
        tool_def = {
            "name": _TOOL_NAME,
            "description": "Return structured output matching the schema.",
            "input_schema": output_schema,
        }

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=temperature,
            system=system_prompt,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": user_message}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == _TOOL_NAME:
                return block.input  # already a dict from the SDK

        # Should not reach here with tool_choice forced, but log and raise
        raw = json.dumps([b.model_dump() for b in response.content], indent=2)
        log.error("No tool_use block in response:\n%s", raw)
        raise RuntimeError("Anthropic response contained no tool_use block")


# ---------------------------------------------------------------------------
# Investigator client (Failure Analysis ReAct loop) — Anthropic tool-use
# ---------------------------------------------------------------------------

def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """{type:function, function:{name,description,parameters}} -> Anthropic tool."""
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Translate the OpenAI-style conversation the ReAct loop builds into Anthropic
    content-block format:
      user            -> {role:user, content:str}
      assistant+calls -> {role:assistant, content:[text?, tool_use...]}
      tool result     -> {role:user, content:[tool_result]}
    Adjacent same-role messages are coalesced (Anthropic wants multiple
    tool_results in ONE user turn, and alternating roles).
    """
    raw: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue  # passed via the top-level `system` param
        if role == "user":
            raw.append({"role": "user", "content": [{"type": "text", "text": m.get("content") or ""}]})
        elif role == "assistant":
            content: list[dict[str, Any]] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                fn = tc["function"]
                args = fn["arguments"]
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": json.loads(args) if isinstance(args, str) else (args or {}),
                })
            raw.append({"role": "assistant", "content": content})
        elif role == "tool":
            raw.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content") or "",
            }]})

    # Coalesce adjacent same-role turns
    merged: list[dict[str, Any]] = []
    for msg in raw:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)
    return merged


class AnthropicInvestigatorClient:
    """
    Drives the ReAct tool-calling loop on Anthropic (mirrors
    OpenAIInvestigatorClient's interface so FailureAnalysisAgent is unchanged).
    """

    def __init__(self, api_key: str = ANTHROPIC_API_KEY, model: str = ANTHROPIC_MODEL,
                 temperature: float = LLM_TEMPERATURE):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def chat_with_tools(self, messages, tools, system_prompt=None) -> ToolCallResponse:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=self._temperature,
            system=system_prompt or anthropic.NOT_GIVEN,
            tools=_openai_tools_to_anthropic(tools),
            messages=_openai_messages_to_anthropic(messages),
        )
        tool_calls: list[ToolCallRequest] = []
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id, name=block.name, arguments=dict(block.input)))
            elif block.type == "text":
                text_parts.append(block.text)

        content = "".join(text_parts) or None
        if response.stop_reason == "tool_use" and tool_calls:
            return ToolCallResponse("tool_calls", content, tool_calls)
        return ToolCallResponse("stop", content, [])

    def complete_structured(self, system_prompt, user_message, output_schema,
                            *, temperature=None) -> dict[str, Any]:
        temp = temperature if temperature is not None else self._temperature
        return _structured_via_forced_tool(
            self._client, self._model, system_prompt, user_message, output_schema, temp)
