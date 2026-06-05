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

log = logging.getLogger(__name__)

_TOOL_NAME = "structured_output"


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
