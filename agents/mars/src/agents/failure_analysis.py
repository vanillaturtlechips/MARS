"""
Failure Analysis Agent — §1 of mars_agent_contracts.md

System prompt, input bundle assembly, and structured-output call.
Output validation is done by the caller (Decision Validator).

Invariants enforced here:
  - distribution scalars are NOT passed in the bundle (agent derives scope
    from raw mission_failures)
  - bundle is assembled from blackboard queries, not from router signals
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Failure Analysis Agent in a warehouse robot supervisory system.
Your only job is to DIAGNOSE why a robot failure occurred. You diagnose; you
NEVER decide what to do about it — retry, handoff, reschedule, and policy are
handled by deterministic components downstream.

You receive a JSON bundle: the trigger failure event, a window of recent
failures across the whole fleet (mission_failures), the focal robot and zone
state, and retrieved historical precedents each carrying a trust score.

Reason over the PRIMARY data. In particular, DERIVE scope yourself from how the
failures are distributed across robots and zones in mission_failures:
  - one robot failing across many different zones  -> likely robot_specific
  - many robots failing in one zone                -> likely zone_wide / fleet_wide
  - a single robot in a single zone, others fine   -> likely isolated
Do not assume scope; compute it from the failures you are given. A reported
fault_flag on the trigger event establishes a robot-internal cause directly.

Produce:
  - cause          one value from failure_cause
  - scope          one value from scope
  - persistence    transient | persistent (only transient causes justify retry)
  - affected_zone  the zone if scope is zone/fleet-wide, else null
  - confidence     0.0-1.0, honest; bounded by evidence strength AND precedent
                   trust. LOW-trust precedents may not raise it. If the data is
                   thin or contradictory, return low confidence or cause=unknown.
  - evidence       list of observations; each references the input data it came
                   from (refs). Cite only facts present in the input. Never
                   invent counts, incidents, or precedents.
  - relied_on_precedents  ids of any retrieved precedents you used.

Output ONLY the JSON object. No prose.\
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


class FailureAnalysisAgent:
    def __init__(self, llm_client):
        self._llm = llm_client

    def analyze(
        self,
        trigger_event: dict[str, Any],
        mission_failures: list[dict[str, Any]],
        robot_state: dict[str, Any],
        zone_state: dict[str, Any],
        retrieved_precedents: list[dict[str, Any]],
        retrieval_trust: dict[str, Any],
        robot_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Build input bundle and call the LLM.  Returns the raw agent output dict.
        Decision Validator must be run on the result before acting.
        """
        bundle = {
            "trigger_event": trigger_event,
            "mission_failures": mission_failures,
            "robot_events": robot_events or [],
            "robot_state": robot_state,
            "zone_state": zone_state,
            "retrieved_precedents": retrieved_precedents,
            "retrieval_trust": retrieval_trust,
        }

        log.info(
            "[failure_analysis_agent] analyzing robot=%s zone=%s  mission_failures=%d precedents=%d",
            trigger_event.get("robot_id"), trigger_event.get("zone"),
            len(mission_failures), len(retrieved_precedents),
        )

        output = self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_message=json.dumps(bundle, default=str),
            output_schema=_OUTPUT_SCHEMA,
        )

        # Attach the input bundle so the Decision Validator can resolve refs
        output["_input_bundle"] = bundle
        return output
