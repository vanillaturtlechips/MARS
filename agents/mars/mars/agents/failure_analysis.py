"""
Failure Analysis Investigator — §1 (ReAct tool-calling loop)

Architecture change from M1-M4:
  Before: single-shot agent; orchestrator pre-assembled the bundle
  After:  read-only ReAct loop; investigator assembles its own evidence
          via tool calls; orchestrator only passes the trigger event

Output contract is UNCHANGED — the same diagnosis dict is produced,
so everything downstream (Decision Validator, disposition, Strategy Trigger)
is untouched.

Loop design:
  1. User message = trigger event (JSON)
  2. Model calls tools (query_failures, get_zone_state, ...) up to budget
  3. When model stops (or budget exhausted), one final structured-output call
     produces the diagnosis JSON
  4. Tool transcript is flattened into the same keys the Decision Validator
     already resolves against (mission_failures, zone_state, etc.)
     so the grounding check requires no changes.

Budgets (all configurable via env):
  max_tool_calls  — total tool executions across all iterations (default 10)
  max_iterations  — conversation turns (default 5)
  timeout_sec     — wall-clock limit for the entire investigation (default 30)

Non-convergence fallback:
  If the budget is exhausted without a confident diagnosis, produce
  cause=unknown, confidence=0.1, so the Decision Validator degrades
  and the orchestrator falls back to a safe default — never a crash.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from mars.config import (
    INVESTIGATOR_MAX_ITERATIONS,
    INVESTIGATOR_MAX_TOOL_CALLS,
    INVESTIGATOR_TIMEOUT_SEC,
)
from mars.agents.tools import TOOL_DEFINITIONS

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Failure Analysis Investigator in a warehouse robot supervisory system.
Your only job is to DIAGNOSE why a robot failure occurred.
You diagnose; you NEVER decide what to do — retry, handoff, reschedule, and policy
are handled by deterministic components downstream.

You have access to read-only tools. Investigate by calling them to gather evidence:
  query_failures     — recent failures fleet-wide or filtered by zone/robot
  get_zone_state     — zone occupancy, health, and failure count
  get_robot_history  — per-robot failure history (distinguish robot vs env causes)
  search_incidents   — similar past incidents with trust scores
  get_active_policies — currently active operational policies

Reason from the PRIMARY data:
  - many robots failing in ONE zone          → likely zone_wide / environmental
  - ONE robot failing across MANY zones      → likely robot_specific
  - single robot, single zone, others fine   → likely isolated
  - fault_flag set on trigger event          → robot-internal established

After gathering sufficient evidence, stop calling tools. The system will then
ask you to produce the final diagnosis as a structured JSON object.\
"""

_FINAL_PROMPT = """\
Based on your investigation, produce the diagnosis as a JSON object.
Cite only facts you actually retrieved from the tools — never invent
counts, incidents, or precedents.  Evidence refs must point to entries
in the data you collected (e.g. 'mission_failures[0]', 'zone_state.occupancy').\
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cause": {
            "type": "string",
            "enum": [
                "transient_obstacle", "robot_internal_fault", "low_battery",
                "localization_failure", "zone_congestion", "zone_blocked",
                "fleet_overload", "unknown",
            ],
        },
        "scope": {
            "type": "string",
            "enum": ["isolated", "robot_specific", "zone_wide", "fleet_wide"],
        },
        "persistence": {"type": "string", "enum": ["transient", "persistent"]},
        "affected_zone": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "observation": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["observation", "refs"],
            },
        },
        "relied_on_precedents": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["cause", "scope", "persistence", "confidence", "evidence"],
}

# Non-convergence fallback — safe default when budget is exhausted
_FALLBACK_DIAGNOSIS: dict[str, Any] = {
    "cause":         "unknown",
    "scope":         "isolated",
    "persistence":   "persistent",
    "affected_zone": None,
    "confidence":    0.1,
    "evidence":      [{"observation": "investigation budget exhausted without convergence",
                       "refs":        ["trigger_event.robot_id"]}],
    "relied_on_precedents": [],
}


class FailureAnalysisAgent:
    """
    Read-only ReAct investigator.

    Constructor args:
      llm_client — must implement chat_with_tools() and complete_structured()
                   (use get_investigator_client() from llm.client)
      tools      — InvestigatorTools or MockInvestigatorTools instance
    """

    def __init__(self, llm_client, tools):
        self._llm   = llm_client
        self._tools = tools

    def analyze(self, trigger_event: dict[str, Any]) -> dict[str, Any]:
        """
        Run the investigation loop and return the diagnosis dict.

        The returned dict has the same shape as the old single-shot output
        PLUS a '_tool_transcript' key containing the flattened evidence
        dict used by the Decision Validator for grounding checks.
        """
        robot_id = trigger_event.get("robot_id", "?")
        zone     = trigger_event.get("zone",     "?")
        log.info("[investigator] starting  robot=%s zone=%s", robot_id, zone)

        deadline    = time.monotonic() + INVESTIGATOR_TIMEOUT_SEC
        tool_count  = 0
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": json.dumps(trigger_event, default=str)},
        ]

        # Accumulated tool results — used to build the transcript
        tool_results: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # ReAct loop
        # ------------------------------------------------------------------
        converged = False
        for _iteration in range(INVESTIGATOR_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                log.warning("[investigator] timeout after %d tool calls", tool_count)
                break
            if tool_count >= INVESTIGATOR_MAX_TOOL_CALLS:
                log.warning("[investigator] max_tool_calls reached (%d)", tool_count)
                break

            try:
                response = self._llm.chat_with_tools(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    system_prompt=_SYSTEM_PROMPT,
                )
            except Exception:
                log.exception("[investigator] chat_with_tools failed")
                break

            if response.finish_reason == "stop" or not response.tool_calls:
                converged = True
                break

            # Append assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in response.tool_calls
                ],
            })

            for tc in response.tool_calls:
                if tool_count >= INVESTIGATOR_MAX_TOOL_CALLS:
                    break
                try:
                    result = self._tools.dispatch(tc.name, tc.arguments)
                    tool_count += 1
                    # Store most-recent result per tool name (last call wins)
                    tool_results[tc.name] = result
                    log.debug("[investigator] tool=%s args=%s", tc.name, tc.arguments)
                except Exception:
                    log.exception("[investigator] tool dispatch failed: %s", tc.name)
                    result = {"error": "tool execution failed"}
                # Append tool result
                messages.append({
                    "role":         "tool",
                    "content":      json.dumps(result, default=str),
                    "tool_call_id": tc.id,
                })

        log.info("[investigator] loop ended: converged=%s tool_calls=%d", converged, tool_count)

        # ------------------------------------------------------------------
        # Final structured-output call
        # ------------------------------------------------------------------
        transcript = _build_transcript(trigger_event, tool_results)

        if not converged and tool_count == 0:
            # No tools called, no convergence — return fallback immediately
            diagnosis = dict(_FALLBACK_DIAGNOSIS)
        else:
            # Ask model to produce the diagnosis from what it learned
            transcript_summary = json.dumps(transcript, default=str)
            try:
                diagnosis = self._llm.complete_structured(
                    system_prompt=_FINAL_PROMPT,
                    user_message=transcript_summary,
                    output_schema=_OUTPUT_SCHEMA,
                )
            except Exception:
                log.exception("[investigator] final structured output failed — using fallback")
                diagnosis = dict(_FALLBACK_DIAGNOSIS)

        # Attach transcript (same key conventions as old _input_bundle)
        diagnosis["_tool_transcript"] = transcript
        return diagnosis


# ---------------------------------------------------------------------------
# Transcript builder — maps tool results to the keys the DV resolves against
# ---------------------------------------------------------------------------

def _build_transcript(
    trigger_event: dict[str, Any],
    tool_results:  dict[str, Any],
) -> dict[str, Any]:
    """
    Flatten tool results into a dict whose top-level keys match the ref
    strings used in evidence.refs.

    Key mapping (preserves backward compatibility with existing tests):
      trigger_event            → trigger_event.{field}
      query_failures result    → mission_failures[N].{field}
      get_zone_state result    → zone_state.{field}
      get_robot_history result → robot_history[N].{field}
      search_incidents result  → retrieved_precedents[N].{field}
      get_active_policies      → active_policies[N].{field}
    """
    return {
        "trigger_event":       trigger_event,
        "mission_failures":    tool_results.get("query_failures",  []),
        "zone_state":          tool_results.get("get_zone_state",  {}),
        "robot_history":       tool_results.get("get_robot_history", []),
        "retrieved_precedents": tool_results.get("search_incidents", []),
        "active_policies":     tool_results.get("get_active_policies", []),
    }
