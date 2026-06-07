"""
Operations Strategy Agent — §2 of mars_agent_contracts.md
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mars.config import POLICY_WHITELIST

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Operations Strategy Agent in a warehouse robot supervisory system.
You recommend how operations should ADAPT to a situation. You are advisory only:
you NEVER assign robots, schedule missions, or issue robot commands. You ONLY
recommend policies, and ONLY from the provided available_policy_types — never
invent a policy type.

You receive: the incident analysis (if any), the current fleet analysis (if it
is current — it may be null), live operational metrics, the list of currently
active policies, and retrieved precedents with trust scores.

Recommend zero or more policy_updates. Doing NOTHING is a valid, often correct
answer — if the situation is transient, already covered by an active policy, or
the evidence is too thin, return an empty policy_updates list with a reason.
Do not duplicate or contradict an active policy.

For each policy:
  - type        from available_policy_types only
  - params      the fields that type requires (e.g. avoid_zone needs "zone")
  - duration_sec  how long it should hold (the system stamps absolute expiry)
  - rationale   one line tying it to specific evidence

Also produce confidence (0-1, trust-bounded as above), evidence (grounded with
refs), and relied_on_precedents.

Every evidence ref MUST be a dotted path into the bundle you were given.
incident_analysis has ONLY these sub-fields — use exactly these, nothing else:
  incident_analysis.cause
  incident_analysis.scope
  incident_analysis.persistence
  incident_analysis.affected_zone
  incident_analysis.confidence
  incident_analysis.evidence[0]            (and [1], [2], ...)
Other valid top-level keys: fleet_analysis, operational_metrics,
active_policies[0], retrieved_precedents[0], retrieval_trust.
Do NOT invent paths. There is NO incident_analysis.mission_failures,
no .zone_state, no .robot_history, no .health — the diagnosis does NOT
contain raw failure rows, only the fields listed above. Cite
incident_analysis.scope and incident_analysis.affected_zone for a zone issue.

If retrieved_precedents is empty or retrieval_trust.set_level is "LOW",
set relied_on_precedents to [] and do NOT claim precedent support.

When you recommend ANY policy you MUST include at least one evidence item with
a valid ref (e.g. incident_analysis.scope, incident_analysis.affected_zone) —
never return an empty evidence list alongside a non-empty policy_updates.
Base confidence on the incident diagnosis, not only on precedents: a zone_wide
incident with high incident_analysis.confidence warrants confidence >= 0.6 even
when retrieved_precedents is empty.

Output ONLY the JSON object. No prose.\
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "policy_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "params": {"type": "object"},
                    "duration_sec": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["type", "params", "duration_sec", "rationale"],
            },
        },
        "no_action_reason": {"type": ["string", "null"]},
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
    "required": ["policy_updates", "confidence", "evidence"],
}


class OperationsStrategyAgent:
    def __init__(self, llm_client):
        self._llm = llm_client

    def recommend(
        self,
        incident_analysis: dict[str, Any] | None,
        fleet_analysis: dict[str, Any] | None,
        operational_metrics: dict[str, Any],
        active_policies: list[dict[str, Any]],
        retrieved_precedents: list[dict[str, Any]],
        retrieval_trust: dict[str, Any],
    ) -> dict[str, Any]:
        bundle = {
            "incident_analysis": incident_analysis,
            "fleet_analysis": fleet_analysis,
            "operational_metrics": operational_metrics,
            "available_policy_types": POLICY_WHITELIST,
            "active_policies": active_policies,
            "retrieved_precedents": retrieved_precedents,
            "retrieval_trust": retrieval_trust,
        }

        log.info(
            "[ops_strategy_agent] recommending  incident_scope=%s fleet_health=%s",
            (incident_analysis or {}).get("scope"),
            (fleet_analysis or {}).get("fleet_health"),
        )

        output = self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_message=json.dumps(bundle, default=str),
            output_schema=_OUTPUT_SCHEMA,
        )

        output["_input_bundle"] = bundle
        return output
