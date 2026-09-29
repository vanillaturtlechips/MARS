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
import re
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
_MISSION_ENTRY_RE = re.compile(r"^mission_failures\[(\d+)\]")
_TRAILING_INDEX = re.compile(r"\[\d+\]$")


def _walk(ref: str, bundle: dict[str, Any]) -> bool:
    """True when `ref` walks exactly against the bundle.

    A path whose value is None still resolves — a null field IS verifiable
    evidence (fault_flag=null means no fault was detected).
    """
    parts = []
    for segment in ref.split("."):
        if "[" in segment:
            name, rest = segment.split("[", 1)
            idx_str = rest.rstrip("]")
            if not idx_str.isdigit():
                # Malformed index ("field[]", "field[x]", "field[-1]"). The
                # grammar (CONTRACT_C.md C4.2) admits key[i], i >= 0 only;
                # Python-style negative indices would otherwise resolve silently.
                return False
            parts.append((name, int(idx_str)))
        else:
            parts.append((segment, None))

    current: Any = bundle
    for name, idx in parts:
        if not isinstance(current, dict):
            return False
        current = current.get(name, _MISSING)
        if current is _MISSING:
            return False
        if idx is not None:
            if not isinstance(current, list) or idx >= len(current):
                return False
            current = current[idx]
    return True


def _enumerate(obj: Any, prefix: str = "") -> list[str]:
    """Every path in the bundle, so a near-miss citation can be looked up."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.append(p)
            out += _enumerate(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.append(p)
            out += _enumerate(v, p)
    return out


def _leaf(ref: str) -> str:
    return ref.split(".")[-1].split("[")[0]


def resolve_ref(ref: str, bundle: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve a citation. Returns (status, intended_path).

    status is one of:
      "exact"      the path walks as written
      "imprecise"  it does not, but the field it names occurs EXACTLY ONCE in the
                   bundle, so the citation identifies a real datum by the wrong
                   path — a provenance defect, not an invention
      "unresolved" no such datum, or the name is ambiguous so the citation does
                   not identify anything

    The three-way split exists because the two-way one was wrong in practice. Of
    13 rejected citations in a 100-case run, 10 were a real field named at the
    wrong depth ("trigger_event.fault_codes" for
    "trigger_event.health_at_failure.fault_codes"), and the diagnoses carrying
    them were mostly CORRECT. Treating a mis-typed path as a fabricated one spent
    the validator's harshest verdict on right answers while catching no wrong
    ones. An invented field ("mission_failures[0].fault_code", which exists
    nowhere) is still unresolved and still rejected.

    Ambiguity is not repaired: if the field name occurs in several places the
    citation does not pick out an observation, which is the thing being checked.
    """
    if _walk(ref, bundle):
        return "exact", ref
    leaf = _leaf(ref)
    # A list and its elements share a leaf name ("fault_codes" and
    # "fault_codes[0]"), which counted as two candidates and made every such
    # citation look ambiguous. Normalise the trailing index away first.
    hits = sorted({_TRAILING_INDEX.sub("", p)
                   for p in _enumerate(bundle) if _leaf(p) == leaf})
    if len(hits) == 1:
        return "imprecise", hits[0]
    return "unresolved", None


def _resolve_ref(ref: str, bundle: dict[str, Any]) -> bool:
    """Back-compat: True when the citation identifies a real datum at all."""
    return resolve_ref(ref, bundle)[0] != "unresolved"


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
    tau: float | None = None,
) -> tuple[DVResult, str]:
    """
    Validate Failure Analysis Agent output.

    `tau` overrides the configured confidence threshold. Runtime callers leave it
    None and get the configured value; the evaluation passes it so this validator
    can be swept across operating points and compared with other validators at a
    matched hold rate (comparing hold-rate-mismatched validators just rewards
    whichever one blocks more).

    Returns (DVResult, notes_string).
    """
    notes = []
    result = DVResult.PASS

    confidence = float(agent_output.get("confidence", 0.0))
    tau = _tau_for_diagnosis() if tau is None else tau

    # 1. Confidence threshold
    if confidence < tau:
        notes.append(f"confidence {confidence:.2f} < tau {tau:.2f}")
        result = DVResult.DEGRADE

    # 2. Evidence grounding — every ref must identify a real datum.
    #    A citation that names a real field at the wrong depth is a provenance
    #    defect (DEGRADE: the decision is held, and the note says what was meant),
    #    not a fabrication (REJECT). Collapsing the two spent REJECT on correct
    #    diagnoses with a mis-typed path — see resolve_ref().
    evidence = agent_output.get("evidence", [])
    if not evidence:
        notes.append("evidence is empty")
        result = DVResult.DEGRADE
    else:
        for item in evidence:
            for ref in item.get("refs", []):
                status, intended = resolve_ref(ref, input_bundle)
                if status == "unresolved":
                    notes.append(f"unresolvable ref: {ref!r}")
                    result = DVResult.REJECT
                elif status == "imprecise":
                    notes.append(f"imprecise ref: {ref!r} -> {intended!r}")
                    if result is not DVResult.REJECT:
                        result = DVResult.DEGRADE

    # 3. Consistency: zone_wide/fleet_wide scope should reference multiple robots
    scope = agent_output.get("scope", "")
    if scope in ("zone_wide", "fleet_wide"):
        refs_all = [r for item in evidence for r in item.get("refs", [])]
        # Count DISTINCT mission_failures entries (C4.3): two refs into the same
        # entry, or a ref to the whole list, do not evidence multiple robots.
        mission_refs = {m.group(1) for r in refs_all
                        if (m := _MISSION_ENTRY_RE.match(r))}
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
