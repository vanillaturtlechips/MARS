"""
Fleet State Analysis Agent — §3 of mars_agent_contracts.md (thin)

Periodic / threshold-triggered fleet health summary.
Does NOT pass through the Deterministic Router (§3a).
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Fleet State Analysis Agent. On a periodic/threshold trigger you
assess the OVERALL condition of the robot fleet. You do not diagnose individual
failures and you do not recommend policies — you summarize fleet state so the
Strategy Trigger Rules and Operations Strategy Agent can act on it.

You receive fleet-level metrics and retrieved precedents with trust scores.
Produce fleet_health, the bottleneck zones, and charging_pressure, with honest
confidence and grounded evidence.

Output ONLY the JSON object. No prose.\
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fleet_health": {
            "type": "string",
            "enum": ["healthy", "strained", "degraded", "critical"],
        },
        "bottlenecks": {"type": "array", "items": {"type": "string"}},
        "charging_pressure": {
            "type": "string",
            "enum": ["low", "moderate", "high"],
        },
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
        "relied_on_precedents": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["fleet_health", "bottlenecks", "charging_pressure", "confidence", "evidence"],
}


class FleetStateAgent:
    def __init__(self, llm_client):
        self._llm = llm_client

    def assess(
        self,
        fleet_metrics: dict[str, Any],
        retrieved_precedents: list[dict[str, Any]],
        retrieval_trust: dict[str, Any],
    ) -> dict[str, Any]:
        bundle = {
            "fleet_metrics": fleet_metrics,
            "retrieved_precedents": retrieved_precedents,
            "retrieval_trust": retrieval_trust,
        }

        log.info("[fleet_state_agent] assessing fleet")

        output = self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_message=json.dumps(bundle, default=str),
            output_schema=_OUTPUT_SCHEMA,
        )

        output["_input_bundle"] = bundle
        return output
