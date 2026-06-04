"""
Unit tests for Decision Validator — §5

Critical tests per coding prompt:
  - ref resolver (grounding check must actually resolve refs)
  - threshold / confidence check
  - consistency check (scope vs evidence)
"""
from __future__ import annotations

import pytest

from mars.validators.decision_validator import (
    DVResult,
    _resolve_ref,
    validate_diagnosis,
    validate_strategy,
)


# ---------------------------------------------------------------------------
# ref resolver tests
# ---------------------------------------------------------------------------

class TestResolveRef:
    def test_simple_field(self):
        bundle = {"trigger_event": {"robot_id": "R1"}}
        assert _resolve_ref("trigger_event", bundle) is True

    def test_nested_field(self):
        bundle = {"trigger_event": {"robot_id": "R1"}}
        assert _resolve_ref("trigger_event.robot_id", bundle) is True

    def test_list_index(self):
        bundle = {"mission_failures": [{"robot_id": "R1"}, {"robot_id": "R2"}]}
        assert _resolve_ref("mission_failures[0]", bundle) is True
        assert _resolve_ref("mission_failures[1]", bundle) is True

    def test_list_index_subfield(self):
        bundle = {"mission_failures": [{"robot_id": "R1"}]}
        assert _resolve_ref("mission_failures[0].robot_id", bundle) is True

    def test_out_of_bounds_index(self):
        bundle = {"mission_failures": [{"robot_id": "R1"}]}
        assert _resolve_ref("mission_failures[5]", bundle) is False

    def test_missing_field(self):
        bundle = {"trigger_event": {"robot_id": "R1"}}
        assert _resolve_ref("trigger_event.missing_key", bundle) is False

    def test_missing_top_level(self):
        bundle = {}
        assert _resolve_ref("nonexistent", bundle) is False

    def test_null_value_returns_true(self):
        # A null field IS verifiable evidence (e.g. fault_flag=null means no fault).
        # Only a non-existent path is unresolvable.
        bundle = {"trigger_event": {"fault_flag": None}}
        assert _resolve_ref("trigger_event.fault_flag", bundle) is True

    def test_nonexistent_field_returns_false(self):
        bundle = {"trigger_event": {"fault_flag": None}}
        assert _resolve_ref("trigger_event.nonexistent_key", bundle) is False


# ---------------------------------------------------------------------------
# Confidence threshold tests
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    def _make_output(self, confidence, scope="isolated"):
        return {
            "cause": "zone_congestion",
            "scope": scope,
            "persistence": "persistent",
            "confidence": confidence,
            "evidence": [
                {"observation": "test", "refs": ["trigger_event.robot_id"]},
            ],
            "relied_on_precedents": [],
        }

    def _make_bundle(self):
        return {
            "trigger_event": {"robot_id": "R1", "fault_flag": None},
            "mission_failures": [{"robot_id": "R1"}],
        }

    def test_pass_above_threshold(self):
        result, _ = validate_diagnosis(self._make_output(0.8), self._make_bundle())
        assert result == DVResult.PASS

    def test_degrade_below_threshold(self):
        result, _ = validate_diagnosis(self._make_output(0.2), self._make_bundle())
        assert result == DVResult.DEGRADE

    def test_exactly_at_threshold_passes(self):
        # DV_TAU_DIAGNOSIS = 0.5 by default
        result, _ = validate_diagnosis(self._make_output(0.5), self._make_bundle())
        assert result == DVResult.PASS


# ---------------------------------------------------------------------------
# Grounding / ref resolution tests
# ---------------------------------------------------------------------------

class TestGrounding:
    def test_unresolvable_ref_causes_reject(self):
        output = {
            "cause": "zone_congestion",
            "scope": "zone_wide",
            "persistence": "persistent",
            "confidence": 0.9,
            "evidence": [
                {"observation": "test", "refs": ["mission_failures[0].robot_id"]},
                {"observation": "hallucinated ref", "refs": ["nonexistent_field.count"]},
            ],
            "relied_on_precedents": [],
        }
        bundle = {
            "trigger_event": {"robot_id": "R1", "fault_flag": None},
            "mission_failures": [{"robot_id": "R1"}],
        }
        result, notes = validate_diagnosis(output, bundle)
        assert result == DVResult.REJECT
        assert "unresolvable ref" in notes

    def test_all_refs_valid_passes_grounding(self):
        output = {
            "cause": "zone_congestion",
            "scope": "zone_wide",
            "persistence": "persistent",
            "confidence": 0.9,
            "evidence": [
                {"observation": "multiple robots", "refs": ["mission_failures[0]", "mission_failures[1]"]},
            ],
            "relied_on_precedents": [],
        }
        bundle = {
            "trigger_event": {"robot_id": "R1", "fault_flag": None},
            "mission_failures": [{"robot_id": "R1"}, {"robot_id": "R2"}],
        }
        result, _ = validate_diagnosis(output, bundle)
        assert result == DVResult.PASS


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_zone_wide_scope_needs_multiple_mission_failures_refs(self):
        output = {
            "cause": "zone_congestion",
            "scope": "zone_wide",
            "persistence": "persistent",
            "confidence": 0.8,
            "evidence": [
                # Only ONE mission_failures ref — inconsistent with zone_wide
                {"observation": "one robot", "refs": ["mission_failures[0]"]},
            ],
            "relied_on_precedents": [],
        }
        bundle = {
            "trigger_event": {"robot_id": "R1", "fault_flag": None},
            "mission_failures": [{"robot_id": "R1"}],
        }
        result, notes = validate_diagnosis(output, bundle)
        assert result == DVResult.DEGRADE


# ---------------------------------------------------------------------------
# Retrieval coherence tests
# ---------------------------------------------------------------------------

class TestRetrievalCoherence:
    def test_high_confidence_low_trust_precedent_degrades(self):
        output = {
            "cause": "zone_congestion",
            "scope": "zone_wide",
            "persistence": "persistent",
            "confidence": 0.9,
            "evidence": [
                {"observation": "ref", "refs": ["mission_failures[0]", "mission_failures[1]"]},
            ],
            "relied_on_precedents": ["STRAT#19"],
        }
        bundle = {
            "trigger_event": {"robot_id": "R1", "fault_flag": None},
            "mission_failures": [{"robot_id": "R1"}, {"robot_id": "R2"}],
        }
        retrieval_trust = {"set_level": "LOW", "support_count": 0}
        result, notes = validate_diagnosis(output, bundle, retrieval_trust)
        assert result == DVResult.DEGRADE
        assert "LOW retrieval_trust" in notes


# ---------------------------------------------------------------------------
# Strategy validation
# ---------------------------------------------------------------------------

class TestStrategyValidation:
    def test_strategy_with_grounded_evidence_passes(self):
        output = {
            "policy_updates": [
                {"type": "avoid_zone", "params": {"zone": "Dock"}, "duration_sec": 900, "rationale": "x"}
            ],
            "no_action_reason": None,
            "confidence": 0.8,
            "evidence": [
                {"observation": "zone congestion", "refs": ["incident_analysis.scope"]},
            ],
            "relied_on_precedents": [],
        }
        bundle = {"incident_analysis": {"scope": "zone_wide", "affected_zone": "Dock"}}
        result, _ = validate_strategy(output, bundle)
        assert result == DVResult.PASS

    def test_empty_policy_updates_with_reason_passes(self):
        output = {
            "policy_updates": [],
            "no_action_reason": "situation is transient",
            "confidence": 0.7,
            "evidence": [],
            "relied_on_precedents": [],
        }
        bundle = {}
        result, _ = validate_strategy(output, bundle)
        # No evidence but no policy_updates either — this is valid restraint
        assert result in (DVResult.PASS, DVResult.DEGRADE)
