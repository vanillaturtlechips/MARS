"""
Decision Validator — §5

Deterministic gate on agent output.  Applies to Failure Analysis Agent,
Operations Strategy Agent, and Fleet State Analysis Agent.

Four checks:
  1. CONFIDENCE THRESHOLD  — confidence >= tau(action_class)
  2. EVIDENCE GROUNDING    — every evidence.refs resolves against the input bundle
  3. EVIDENCE CONSISTENCY  — evidence supports the stated scope/cause
  4. RETRIEVAL COHERENCE   — HIGH confidence but LOW retrieval_trust + used precedent

Outcomes: PASS | DEGRADE | REJECT
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from mars.config import (
    DV_TAU_DIAGNOSIS,
    DV_TAU_POLICY_HIGH,
    DV_TAU_POLICY_MEDIUM,
)

log = logging.getLogger(__name__)

# Impact tiers for Operations Strategy Agent
_HIGH_IMPACT_POLICIES = {"fleet_wide_throttle"}   # not in current whitelist but future-proof
_MEDIUM_IMPACT_POLICIES = {
    "avoid_zone",
    "delay_low_priority_missions",
    "reserve_chargers_for_critical",
    "lower_target_charge_level",
    "pre_charge_for_demand_spike",
}


class DVResult(str, Enum):
    PASS    = "PASS"
    DEGRADE = "DEGRADE"
    REJECT  = "REJECT"


_MISSING = object()  # sentinel: field does not exist


def _resolve_ref(ref: str, bundle: dict[str, Any]) -> bool:
    """
    Walk a JSON-path-style ref string against the bundle dict.

    Supports:
      "field_name"
      "field_name.subfield"
      "list_field[0]"
      "list_field[0].subfield"

    Returns True when the path EXISTS in the bundle, regardless of whether the
    value is None — a null field IS verifiable evidence (e.g. fault_flag=null
    means no fault was detected).  Returns False only when the path cannot be
    walked (field does not exist, list index out of range, etc.).
    """
    parts = []
    for segment in ref.split("."):
        if "[" in segment:
            name, rest = segment.split("[", 1)
            idx = int(rest.rstrip("]"))
            parts.append((name, idx))
        else:
            parts.append((segment, None))

    current: Any = bundle
    for name, idx in parts:
        if not isinstance(current, dict):
            return False
        current = current.get(name, _MISSING)
        if current is _MISSING:
            return False          # field does not exist → unresolvable
        if idx is not None:
            if not isinstance(current, list) or idx >= len(current):
                return False
            current = current[idx]
    return True                   # path exists; value may be None


def _tau_for_diagnosis() -> float:
    return DV_TAU_DIAGNOSIS


def _tau_for_policy(policy_type: str) -> float:
    if policy_type in _HIGH_IMPACT_POLICIES:
        return DV_TAU_POLICY_HIGH
    return DV_TAU_POLICY_MEDIUM


def validate_diagnosis(
    agent_output: dict[str, Any],
    input_bundle: dict[str, Any],
    retrieval_trust: dict[str, Any] | None = None,
) -> tuple[DVResult, str]:
    """
    Validate Failure Analysis Agent output.

    Returns (DVResult, notes_string).
    """
    notes = []
    result = DVResult.PASS

    confidence = float(agent_output.get("confidence", 0.0))
    tau = _tau_for_diagnosis()

    # 1. Confidence threshold
    if confidence < tau:
        notes.append(f"confidence {confidence:.2f} < tau {tau:.2f}")
        result = DVResult.DEGRADE

    # 2. Evidence grounding — every ref must resolve
    evidence = agent_output.get("evidence", [])
    if not evidence:
        notes.append("evidence is empty")
        result = DVResult.DEGRADE
    else:
        for item in evidence:
            for ref in item.get("refs", []):
                if not _resolve_ref(ref, input_bundle):
                    notes.append(f"unresolvable ref: {ref!r}")
                    result = DVResult.REJECT

    # 3. Consistency: zone_wide/fleet_wide scope should reference multiple robots
    scope = agent_output.get("scope", "")
    if scope in ("zone_wide", "fleet_wide"):
        refs_all = [r for item in evidence for r in item.get("refs", [])]
        mission_refs = [r for r in refs_all if "mission_failures" in r]
        if len(mission_refs) < 2:
            notes.append(
                f"scope={scope} but evidence references <2 mission_failures entries"
            )
            # Downgrade to DEGRADE only (not REJECT) — maybe agent cited aggregate
            if result == DVResult.PASS:
                result = DVResult.DEGRADE

    # 4. Retrieval coherence
    if retrieval_trust:
        relied = agent_output.get("relied_on_precedents", [])
        trust_level = retrieval_trust.get("set_level", "LOW")
        if relied and trust_level == "LOW" and confidence > 0.7:
            notes.append(
                "HIGH confidence despite LOW retrieval_trust and relied_on_precedents non-empty"
            )
            result = DVResult.DEGRADE

    notes_str = "; ".join(notes) if notes else "ok"
    log.info("[decision_validator] diagnosis → %s  notes=%s", result, notes_str)
    return result, notes_str


def validate_strategy(
    agent_output: dict[str, Any],
    input_bundle: dict[str, Any],
    retrieval_trust: dict[str, Any] | None = None,
) -> tuple[DVResult, str]:
    """
    Validate Operations Strategy Agent output.
    """
    notes = []
    result = DVResult.PASS

    confidence = float(agent_output.get("confidence", 0.0))
    policy_updates = agent_output.get("policy_updates", [])

    # Determine highest-impact policy in the output
    max_tau = DV_TAU_POLICY_MEDIUM
    for p in policy_updates:
        max_tau = max(max_tau, _tau_for_policy(p.get("type", "")))

    # 1. Confidence threshold
    if confidence < max_tau:
        notes.append(f"confidence {confidence:.2f} < tau {max_tau:.2f}")
        result = DVResult.DEGRADE

    # 2. Evidence grounding
    evidence = agent_output.get("evidence", [])
    for item in evidence:
        for ref in item.get("refs", []):
            if not _resolve_ref(ref, input_bundle):
                notes.append(f"unresolvable ref: {ref!r}")
                result = DVResult.REJECT

    # 3. Consistency: policy recommendation must tie to evidence
    # (light check — full consistency is subjective, so we only reject if
    #  there is NO evidence at all for a non-empty recommendation)
    if policy_updates and not evidence:
        notes.append("policy_updates non-empty but evidence is empty")
        result = DVResult.DEGRADE

    # 4. Retrieval coherence
    if retrieval_trust:
        relied = agent_output.get("relied_on_precedents", [])
        trust_level = retrieval_trust.get("set_level", "LOW")
        if relied and trust_level == "LOW" and confidence > 0.7:
            notes.append("HIGH confidence despite LOW retrieval_trust")
            result = DVResult.DEGRADE

    notes_str = "; ".join(notes) if notes else "ok"
    log.info("[decision_validator] strategy → %s  notes=%s", result, notes_str)
    return result, notes_str


def validate_fleet_state(
    agent_output: dict[str, Any],
    input_bundle: dict[str, Any],
    retrieval_trust: dict[str, Any] | None = None,
) -> tuple[DVResult, str]:
    """
    Validate Fleet State Analysis Agent output (same checks, diagnosis tau).
    """
    # Reuse diagnosis validation logic (same threshold)
    return validate_diagnosis(agent_output, input_bundle, retrieval_trust)
