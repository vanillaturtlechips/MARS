"""
Intent Agent — translate an operator's natural-language intent into candidate
fleet policies (강 claim). Advisory only: it proposes policies from the whitelist;
the deterministic Policy Guardrail (mars/guardrail) validates/rejects them before
anything reaches the fleet. The agent NEVER invents a policy type and NEVER acts.

Mirrors OperationsStrategyAgent: a single complete_structured() call with a
constrained output schema. Ambiguous intents -> needs_clarification (no policy).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mars.config import POLICY_WHITELIST

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Intent Agent in a warehouse robot supervisory system. A human
operator gives an instruction in natural language. Your ONLY job is to translate
it into zero or more fleet POLICIES drawn STRICTLY from available_policy_types.
You are advisory: a deterministic guardrail validates your output before it can
affect robots. You NEVER invent a policy type and NEVER command robots directly.

You receive: the operator utterance, available_policy_types, current
active_policies, and fleet context (valid zones, robot/charger counts).

For each policy emit: type (from available_policy_types ONLY), params (the fields
that type needs, e.g. avoid_zone needs "zone"), duration_sec (from the utterance's
time phrase; default 1800 if unspecified; "permanently"/"indefinitely" -> 7200),
and a short rationale quoting the operator intent.

Policy mapping (use ONLY these; pick by the operator's intent):
  - avoid_zone(zone): keep robots out of a zone / block / clear a zone.
  - delay_low_priority_missions: prioritize urgent work, hold/defer low-priority
    missions, "X 먼저 돌려 / 급한 것만 / 저우선 미션 멈춰 / deprioritize".
  - reserve_chargers_for_critical(reserve_count): keep chargers for critical/low
    -battery robots only.
  - lower_target_charge_level: charge less per cycle so robots return to work
    faster, "80%까지만 / cycle faster".
  - pre_charge_for_demand_spike: top up BEFORE an expected surge, "이따 물량 /
    미리 충전 / ahead of demand".

Rules:
  - CRITICAL: if the instruction asks for something NONE of the policies above
    can express (robot speed, firmware, lighting, payload, audio, anything not
    in the list), you MUST set out_of_scope=true and return an EMPTY
    policy_updates list. NEVER substitute an unrelated policy (e.g. do NOT emit
    avoid_zone or reserve_chargers for a "drive faster" request). Forcing an
    unrelated valid policy is a SAFETY VIOLATION — the guardrail cannot catch it.
  - If the instruction is too vague to act on (no clear target/action), set
    needs_clarification=true, empty policy_updates, and a clarification question.
  - One utterance may map to MULTIPLE policies (compositional): emit each.
  - Use ONLY zones that appear in the fleet context's valid_zones.
  - Do NOT worry about safety/feasibility/duplicates — the guardrail handles
    those. Just translate the intent faithfully into the policies above.

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
        "needs_clarification": {"type": "boolean"},
        "clarification": {"type": ["string", "null"]},
        "out_of_scope": {"type": "boolean"},
        "out_of_scope_reason": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["policy_updates", "needs_clarification", "out_of_scope", "confidence"],
}


class IntentAgent:
    def __init__(self, llm_client):
        self._llm = llm_client

    def translate(
        self,
        utterance: str,
        active_policies: list[dict[str, Any]] | None = None,
        fleet_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bundle = {
            "operator_utterance": utterance,
            "available_policy_types": POLICY_WHITELIST,
            "active_policies": active_policies or [],
            "fleet_context": fleet_context or {},
        }
        log.info("[intent_agent] translating: %r", utterance)
        out = self._llm.complete_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_message=json.dumps(bundle, default=str),
            output_schema=_OUTPUT_SCHEMA,
        )
        out["_input_bundle"] = bundle
        return out
