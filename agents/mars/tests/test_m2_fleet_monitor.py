"""
M2 tests — FleetMonitor and the charging pressure → policy loop

Tests:
  - FleetMonitor.run_once() produces a validated fleet analysis
  - Fleet State Agent output shape (fleet_health, bottlenecks, charging_pressure)
  - FleetMonitor.update_fleet_analysis() arms the StrategyTrigger correlation window
  - evaluate_fleet() fires when charging_pressure == high
  - evaluate_fleet() → avoid action when fleet is healthy
  - Charging pressure loop: high pressure → policy activated
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pytest

from mars.agents.fleet_state import FleetStateAgent
from mars.orchestrator.fleet_monitor import FleetMonitor
from mars.orchestrator.strategy_trigger import StrategyTrigger, TRIGGER, NO_TRIGGER
from mars.validators.decision_validator import validate_fleet_state
from mars.validators.retrieval_validator import validate_retrieval_set
from mars.policy.policy_manager import PolicyManager

from tests.conftest import ZONE_WIDE_DIAGNOSIS, AVOID_ZONE_STRATEGY


# ---------------------------------------------------------------------------
# Canned fleet agent outputs
# ---------------------------------------------------------------------------

HEALTHY_FLEET = {
    "fleet_health":      "healthy",
    "bottlenecks":       [],
    "charging_pressure": "low",
    "confidence":        0.90,
    "evidence": [
        {"observation": "low backlog and utilization",
         "refs": ["fleet_metrics.mission_backlog", "fleet_metrics.robot_utilization"]},
    ],
    "relied_on_precedents": [],
}

DEGRADED_FLEET_HIGH_PRESSURE = {
    "fleet_health":      "degraded",
    "bottlenecks":       ["Receiving Dock"],
    "charging_pressure": "high",
    "confidence":        0.82,
    "evidence": [
        {"observation": "charging queue forming, 3 robots below LOW",
         "refs": ["fleet_metrics.charging.queue_len", "fleet_metrics.charging.below_low_count"]},
    ],
    "relied_on_precedents": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeConn:
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


class _InMemBB:
    """Minimal in-memory BB for fleet-monitor tests."""

    def __init__(self, fleet_metrics=None):
        self._fleet_metrics = fleet_metrics or {
            "robot_utilization": 0.4,
            "mission_backlog": 3,
            "charging": {"queue_len": 0, "mean_wait_sec": 0, "occupied_pct": 0.0,
                         "below_low_count": 0, "below_critical_count": 0},
            "zone_health": {},
            "recent_failure_clusters": [],
        }
        self.strategy_runs: list[dict] = []
        self.policies: dict[str, dict] = {}

    def get_fleet_metrics(self, conn):
        return self._fleet_metrics

    def search_similar(self, conn, embedding, source_types=None, zone=None, limit=10):
        return []

    def get_operational_metrics(self, conn):
        return {"mission_backlog": self._fleet_metrics["mission_backlog"],
                "robot_utilization": self._fleet_metrics["robot_utilization"],
                "charging": self._fleet_metrics["charging"]}

    def get_world_state(self, conn):
        return {
            "zones": {"Charging Bay": {"is_charger_zone": True, "is_mandatory": False}},
            "charger_zones": ["Charging Bay"],
        }

    def write_policy(self, conn, policy):
        pid = policy.get("policy_id") or f"POL{len(self.policies)+1:04d}"
        self.policies[pid] = {**policy, "policy_id": pid}
        return pid

    def get_active_policies(self, conn):
        return list(self.policies.values())

    def deactivate_policy(self, conn, pid, reason): pass


def _make_fleet_llm(canned=None):
    canned = canned or DEGRADED_FLEET_HIGH_PRESSURE

    class _Mock:
        def complete_structured(self, system_prompt, user_message, output_schema, *, temperature=0.0):
            return dict(canned)

    return _Mock()


def _make_strategy_llm():
    class _Mock:
        def complete_structured(self, system_prompt, user_message, output_schema, *, temperature=0.0):
            # Return a charging policy recommendation
            return {
                "policy_updates": [
                    {"type": "lower_target_charge_level",
                     "params": {"target_pct": 70.0},
                     "duration_sec": 1800,
                     "rationale": "charging pressure is high"},
                ],
                "no_action_reason": None,
                "confidence": 0.78,
                "evidence": [
                    {"observation": "charging queue forming",
                     "refs": ["fleet_analysis.charging_pressure"]},
                ],
                "relied_on_precedents": [],
            }

    return _Mock()


def _make_strategy_trigger(bb, embedder, conn_factory):
    from mars.llm.client import MockLLMClient
    from mars.agents.operations_strategy import OperationsStrategyAgent

    pm = PolicyManager(conn_factory)
    # Patch activate to use in-memory BB
    def _patched_activate(policy):
        pid = bb.write_policy(None, policy)
        policy["policy_id"] = pid
        pm._active[pid] = policy
        pm._last_applied[policy.get("type", "")] = time.time()
        for ft, cb in pm._consumers:
            if ft is None or ft == policy.get("type"):
                try: cb("activated", policy)
                except Exception: pass
        return pid
    pm.activate = _patched_activate

    trigger = StrategyTrigger(
        ops_strategy_agent=OperationsStrategyAgent(_make_strategy_llm()),
        retrieval_validator_fn=validate_retrieval_set,
        decision_validator_fn=__import__("mars.validators.decision_validator", fromlist=["validate_strategy"]).validate_strategy,
        policy_manager=pm,
        blackboard_queries=bb,
        embedder=embedder,
    )
    return trigger, pm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFleetStateAgentOutputShape:
    def test_healthy_fleet_output(self):
        agent = FleetStateAgent(_make_fleet_llm(HEALTHY_FLEET))
        out   = agent.assess(
            fleet_metrics={"robot_utilization": 0.4, "mission_backlog": 2,
                           "charging": {"queue_len": 0, "occupied_pct": 0.0,
                                        "below_low_count": 0}},
            retrieved_precedents=[],
            retrieval_trust={"set_level": "LOW", "support_count": 0},
        )
        assert out["fleet_health"] in ("healthy", "strained", "degraded", "critical")
        assert isinstance(out["bottlenecks"], list)
        assert out["charging_pressure"] in ("low", "moderate", "high")
        assert 0.0 <= out["confidence"] <= 1.0

    def test_degraded_high_pressure_output(self):
        agent = FleetStateAgent(_make_fleet_llm(DEGRADED_FLEET_HIGH_PRESSURE))
        out   = agent.assess(
            fleet_metrics={"robot_utilization": 0.9, "mission_backlog": 40,
                           "charging": {"queue_len": 3, "occupied_pct": 0.9,
                                        "below_low_count": 3}},
            retrieved_precedents=[],
            retrieval_trust={"set_level": "LOW", "support_count": 0},
        )
        assert out["fleet_health"] == "degraded"
        assert out["charging_pressure"] == "high"


class TestFleetStateDecisionValidator:
    def test_valid_output_passes(self):
        bundle = {
            "fleet_metrics": {
                "mission_backlog": 3,
                "robot_utilization": 0.4,
                "charging": {"queue_len": 0, "occupied_pct": 0.0, "below_low_count": 0},
            }
        }
        dv_result, notes = validate_fleet_state(
            {**HEALTHY_FLEET, "_input_bundle": bundle}, bundle
        )
        assert dv_result.value == "PASS", notes

    def test_unresolvable_ref_degrades(self):
        from mars.validators.decision_validator import DVResult
        output = {
            **HEALTHY_FLEET,
            "evidence": [
                {"observation": "hallucinated", "refs": ["fleet_metrics.nonexistent_key"]},
            ],
        }
        bundle = {"fleet_metrics": {"mission_backlog": 3}}
        dv_result, notes = validate_fleet_state(output, bundle)
        assert dv_result in (DVResult.DEGRADE, DVResult.REJECT)


class TestStrategyTriggerFleetOnly:
    def test_high_pressure_triggers(self):
        bb      = _InMemBB()
        embedder = __import__("mars.llm.client", fromlist=["MockEmbedder"]).MockEmbedder(dim=1024)
        trigger, _ = _make_strategy_trigger(bb, embedder, lambda: _FakeConn())

        result = trigger.evaluate_fleet(DEGRADED_FLEET_HIGH_PRESSURE, _FakeConn())
        assert result == TRIGGER

    def test_healthy_fleet_no_trigger(self):
        bb      = _InMemBB()
        embedder = __import__("mars.llm.client", fromlist=["MockEmbedder"]).MockEmbedder(dim=1024)
        trigger, _ = _make_strategy_trigger(bb, embedder, lambda: _FakeConn())

        result = trigger.evaluate_fleet(HEALTHY_FLEET, _FakeConn())
        assert result == NO_TRIGGER

    def test_fleet_trigger_activates_charging_policy(self):
        bb      = _InMemBB()
        embedder = __import__("mars.llm.client", fromlist=["MockEmbedder"]).MockEmbedder(dim=1024)
        trigger, pm = _make_strategy_trigger(bb, embedder, lambda: _FakeConn())

        trigger.evaluate_fleet(DEGRADED_FLEET_HIGH_PRESSURE, _FakeConn())

        active = pm.get_active()
        charging_policies = [p for p in active
                             if p.get("type") in ("lower_target_charge_level",
                                                   "reserve_chargers_for_critical")]
        assert charging_policies, (
            f"Expected a charging policy; active={[p.get('type') for p in active]}"
        )

    def test_correlation_window_expires(self):
        """Fleet analysis older than correlation window should not be used."""
        bb      = _InMemBB()
        embedder = __import__("mars.llm.client", fromlist=["MockEmbedder"]).MockEmbedder(dim=1024)
        trigger, _ = _make_strategy_trigger(bb, embedder, lambda: _FakeConn())

        trigger._last_fleet_out = DEGRADED_FLEET_HIGH_PRESSURE
        # Simulate stale timestamp
        trigger._last_fleet_ts = time.time() - 9999
        assert trigger._current_fleet_analysis() is None


class TestFleetMonitorRunOnce:
    def test_run_once_returns_validated_output(self):
        bb       = _InMemBB()
        embedder = __import__("mars.llm.client", fromlist=["MockEmbedder"]).MockEmbedder(dim=1024)
        trigger, _ = _make_strategy_trigger(bb, embedder, lambda: _FakeConn())

        monitor = FleetMonitor(
            fleet_state_agent=FleetStateAgent(_make_fleet_llm(HEALTHY_FLEET)),
            strategy_trigger=trigger,
            blackboard_queries=bb,
            embedder=embedder,
            conn_factory=lambda: _FakeConn(),
        )

        result = monitor.run_once(_FakeConn())
        assert result is not None
        assert "fleet_health" in result
        assert monitor._last_output is not None
        assert monitor._last_run_ts > 0

    def test_run_once_arms_trigger_correlation_window(self):
        bb       = _InMemBB()
        embedder = __import__("mars.llm.client", fromlist=["MockEmbedder"]).MockEmbedder(dim=1024)
        trigger, _ = _make_strategy_trigger(bb, embedder, lambda: _FakeConn())

        monitor = FleetMonitor(
            fleet_state_agent=FleetStateAgent(_make_fleet_llm(HEALTHY_FLEET)),
            strategy_trigger=trigger,
            blackboard_queries=bb,
            embedder=embedder,
            conn_factory=lambda: _FakeConn(),
        )

        monitor.run_once(_FakeConn())

        # Trigger should now have a fleet analysis and a recent timestamp
        assert trigger._last_fleet_out is not None
        assert trigger._last_fleet_ts > 0
