"""
M1 integration test — runs the full vertical slice with mock LLM and
an in-memory dict blackboard.  No Postgres or network calls required.

Tests the graded concepts:
  - Failure → Aggregator enrichment → Router (SLOW path)
  - Failure Analysis Agent → Decision Validator (PASS)
  - Strategy Trigger fired → Operations Strategy Agent
  - Decision Validator (PASS) → Policy Guardrail (ACCEPT)
  - Policy Manager activates avoid_zone
  - Scheduler defers dock missions
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from mars.aggregator.aggregator import Aggregator
from mars.blackboard.hot_state import HotState
from mars.orchestrator.orchestrator import Orchestrator, fast_disposition, slow_disposition
from mars.orchestrator.strategy_trigger import StrategyTrigger
from mars.agents.failure_analysis import FailureAnalysisAgent
from mars.agents.operations_strategy import OperationsStrategyAgent
from mars.validators.retrieval_validator import validate_retrieval_set
from mars.validators.decision_validator import validate_diagnosis, validate_strategy, DVResult
from mars.guardrail.guardrail import check as guardrail_check
from mars.policy.policy_manager import PolicyManager
from mars.router.router import route, Path

from tests.conftest import ZONE_WIDE_DIAGNOSIS, AVOID_ZONE_STRATEGY


# ---------------------------------------------------------------------------
# Minimal in-memory blackboard (no psycopg / Postgres)
# ---------------------------------------------------------------------------

class _InMemBB:
    """Thin dict-backed blackboard used in integration tests."""

    def __init__(self):
        self.failures:   dict[str, dict] = {}
        self.diagnoses:  dict[str, dict] = {}
        self.strategy_runs: dict[str, dict] = {}
        self.policies:   dict[str, dict] = {}
        self.missions:   dict[str, dict] = {}
        self.robots:     dict[str, dict] = {}
        self.zones:      dict[str, dict] = {}
        self._next_id    = 0

    def _new_id(self, prefix=""):
        self._next_id += 1
        return f"{prefix}{self._next_id:04d}"

    # failures
    def write_failure(self, conn, event):
        fid = event.get("failure_id") or self._new_id("F")
        self.failures[fid] = {**event, "failure_id": fid}
        return fid

    def get_recent_failures(self, conn, window_seconds=900, zone=None):
        rows = list(self.failures.values())
        if zone:
            rows = [r for r in rows if r.get("zone") == zone]
        return rows

    # diagnoses
    def write_diagnosis(self, conn, diag):
        did = diag.get("diagnosis_id") or self._new_id("DX")
        self.diagnoses[did] = {**diag, "diagnosis_id": did}
        return did

    # strategy runs
    def write_strategy_run(self, conn, run):
        rid = run.get("strategy_run_id") or self._new_id("SR")
        self.strategy_runs[rid] = {**run, "strategy_run_id": rid}
        return rid

    # policies
    def write_policy(self, conn, policy):
        pid = policy.get("policy_id") or self._new_id("POL")
        policy = dict(policy)
        if not policy.get("expires_at"):
            policy["expires_at"] = datetime.now(timezone.utc) + timedelta(seconds=policy.get("duration_sec", 900))
        self.policies[pid] = {**policy, "policy_id": pid}
        return pid

    def get_active_policies(self, conn):
        now = datetime.now(timezone.utc)
        return [
            p for p in self.policies.values()
            if p.get("is_active", True)
            and p.get("expires_at", now + timedelta(1)) > now
        ]

    def deactivate_policy(self, conn, policy_id, reason):
        if policy_id in self.policies:
            self.policies[policy_id]["is_active"] = False

    # robots
    def upsert_robot(self, conn, robot):
        self.robots[robot["robot_id"]] = robot

    def get_robot(self, conn, robot_id):
        return self.robots.get(robot_id)

    def get_idle_robots(self, conn):
        return [r for r in self.robots.values() if r.get("allocation_state") == "IDLE"]

    def cas_allocation_state(self, conn, robot_id, expected, new_state):
        robot = self.robots.get(robot_id)
        if robot and robot.get("allocation_state") == expected:
            robot["allocation_state"] = new_state
            return True
        return False

    # missions
    def upsert_mission(self, conn, mission):
        self.missions[mission["mission_id"]] = mission

    def get_pending_missions(self, conn):
        return [m for m in self.missions.values() if m.get("state") == "PENDING"]

    # zones
    def upsert_zone(self, conn, zone):
        self.zones[zone["zone_id"]] = zone

    # search (always returns empty — no pgvector in tests)
    def search_similar(self, conn, embedding, source_types=None, zone=None, limit=10):
        return []

    # operational metrics
    def get_operational_metrics(self, conn):
        backlog = sum(1 for m in self.missions.values() if m.get("state") == "PENDING")
        busy    = sum(1 for r in self.robots.values()
                      if r.get("allocation_state") in ("BUSY", "RESERVED"))
        total   = len(self.robots) or 1
        return {
            "mission_backlog": backlog,
            "robot_utilization": busy / total,
            "charging": {"queue_len": 0, "occupied_pct": 0.0, "below_low_count": 0},
        }

    # world state
    def get_world_state(self, conn):
        zones = {
            zid: {"is_mandatory": z.get("is_mandatory", False),
                  "is_charger_zone": z.get("is_charger_zone", False)}
            for zid, z in self.zones.items()
        }
        charger_zones = [zid for zid, z in zones.items() if z["is_charger_zone"]]
        return {"zones": zones, "charger_zones": charger_zones}


class _FakeConn:
    """No-op connection stub for the in-memory BB."""
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bb():
    db = _InMemBB()
    # Zone IDs match the zone strings used in failure events and canned agent
    # output (ZONE_WIDE_DIAGNOSIS.affected_zone = "Receiving Dock").
    db.upsert_zone(None, {"zone_id": "Receiving Dock", "display_name": "Receiving Dock",
                           "is_charger_zone": False, "is_mandatory": False})
    db.upsert_zone(None, {"zone_id": "Charging Bay", "display_name": "Charging Bay",
                           "is_charger_zone": True, "is_mandatory": False})
    for i, rid in enumerate(["R1", "R2", "R3", "R4", "R5"]):
        db.upsert_robot(None, {"robot_id": rid, "battery_pct": 75.0,
                                "allocation_state": "IDLE", "mode": "IDLE",
                                "current_zone": "Receiving Dock"})
        db.upsert_mission(None, {
            "mission_id": f"M{i+1:03d}", "robot_id": None,
            "state": "PENDING", "zone": "Receiving Dock",
            "priority": 5, "scheduling_priority": 5,
        })
    return db


@pytest.fixture
def conn():
    return _FakeConn()


@pytest.fixture
def hot_state():
    return HotState(redis_client=None)


@pytest.fixture
def mock_llm_multi():
    """Alternates between diagnosis and strategy canned outputs."""

    class _Multi:
        def complete_structured(
            self,
            system_prompt: str,
            user_message: str,
            output_schema: dict,
            *,
            temperature: float = 0.0,
        ) -> dict:
            # Route by system prompt content so diagnosis and strategy never
            # swap regardless of call order.
            if "DIAGNOSE" in system_prompt:
                return dict(ZONE_WIDE_DIAGNOSIS)
            return dict(AVOID_ZONE_STRATEGY)

    return _Multi()


@pytest.fixture
def mock_embedder():
    from mars.llm.client import MockEmbedder
    return MockEmbedder(dim=1024)


def _make_components(bb, hot_state, llm, embedder):
    conn_factory = lambda: _FakeConn()

    policy_manager = PolicyManager(conn_factory)
    # Patch policy_manager to use in-memory BB
    original_activate = policy_manager.activate
    def _patched_activate(policy):
        pid = bb.write_policy(None, policy)
        policy["policy_id"] = pid
        policy_manager._active[pid] = policy
        policy_manager._last_applied[policy.get("type", "")] = time.time()
        for filter_type, cb in policy_manager._consumers:
            if filter_type is None or filter_type == policy.get("type"):
                try:
                    cb("activated", policy)
                except Exception:
                    pass
        return pid
    policy_manager.activate = _patched_activate

    strategy_trigger = StrategyTrigger(
        ops_strategy_agent=OperationsStrategyAgent(llm),
        retrieval_validator_fn=validate_retrieval_set,
        decision_validator_fn=validate_strategy,
        policy_manager=policy_manager,
        blackboard_queries=bb,
        embedder=embedder,
    )

    orchestrator = Orchestrator(
        blackboard_queries=bb,
        hot_state=hot_state,
        failure_analysis_agent=FailureAnalysisAgent(llm),
        retrieval_validator_fn=validate_retrieval_set,
        decision_validator_fn=validate_diagnosis,
        strategy_trigger_fn=strategy_trigger.evaluate,
        policy_manager=policy_manager,
        embedder=embedder,
    )

    return orchestrator, policy_manager, strategy_trigger


# ---------------------------------------------------------------------------
# The four enriched failure events (zone-wide congestion scenario)
# ---------------------------------------------------------------------------

def _make_failure(robot_id: str, failures_for_mission: int, zone_spread: int) -> dict:
    return {
        "event_type": "navigation.aborted",
        "robot_id": robot_id,
        "mission_id": f"M00{ord(robot_id[-1]) - ord('0')}",
        "goal_id": f"goal-{robot_id}",
        # Zone string must match zone_id seeded in the bb fixture AND the
        # affected_zone in ZONE_WIDE_DIAGNOSIS ("Receiving Dock").
        "zone": "Receiving Dock",
        "nav_outcome": "aborted",
        "goal_status": 6,
        "health_at_failure": {"battery_pct": 75.0, "estop_active": False, "fault_codes": []},
        "fault_flag": None,
        "distribution": {
            "per_robot_zone_spread": 1,
            "per_zone_robot_spread": zone_spread,
        },
        "failures_for_this_mission": failures_for_mission,
        "occurred_at": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestM1VerticalSlice:
    def test_router_selects_slow_for_zone_pattern(self):
        """4+ distinct robots failing in one zone → SLOW path."""
        event = _make_failure("R1", failures_for_mission=0, zone_spread=4)
        assert route(event) == Path.SLOW

    def test_fast_path_for_isolated_first_failure(self):
        event = _make_failure("R1", failures_for_mission=0, zone_spread=1)
        assert route(event) == Path.FAST

    def test_aggregator_sets_fault_flag_on_low_battery(self, hot_state):
        """Aggregator sets fault_flag='battery_critical' when battery < CRITICAL_PCT."""
        from mars.ros.interfaces import BatteryState, NavGoalStatus

        events = []
        agg = Aggregator(hot_state, on_failure_event=events.append)
        agg._zone_map["R1"] = "receiving_dock"

        agg.on_battery_update(BatteryState(robot_id="R1", percentage=0.10, power_supply_status=2))
        agg.on_nav_status(
            NavGoalStatus(goal_id="g1", robot_id="R1", status=6),
            mission_id="M001",
        )

        assert len(events) == 1
        assert events[0]["fault_flag"] == "battery_critical"

    def test_aggregator_no_fault_flag_for_healthy_robot(self, hot_state):
        from mars.ros.interfaces import BatteryState, NavGoalStatus, RobotHealth

        events = []
        agg = Aggregator(hot_state, on_failure_event=events.append)
        agg._zone_map["R1"] = "receiving_dock"

        agg.on_battery_update(BatteryState(robot_id="R1", percentage=0.75, power_supply_status=2))
        agg.on_health_update(RobotHealth(robot_id="R1", level=0))
        agg.on_nav_status(
            NavGoalStatus(goal_id="g1", robot_id="R1", status=6),
            mission_id="M001",
        )

        assert events[0]["fault_flag"] is None

    def test_failure_analysis_agent_output_shape(self, mock_llm_multi):
        """Agent returns a dict with required keys."""
        from mars.agents.failure_analysis import FailureAnalysisAgent

        agent = FailureAnalysisAgent(mock_llm_multi)
        event = _make_failure("R1", 0, 4)
        out = agent.analyze(
            trigger_event=event,
            mission_failures=[event],
            robot_state={"robot_id": "R1", "battery_pct": 75},
            zone_state={"zone": "receiving_dock", "occupancy": 4},
            retrieved_precedents=[],
            retrieval_trust={"set_level": "LOW", "support_count": 0},
        )
        assert out["scope"] == "zone_wide"
        assert out["cause"] == "zone_congestion"
        assert 0.0 <= out["confidence"] <= 1.0

    def test_decision_validator_passes_zone_wide_diagnosis(self):
        """Well-formed diagnosis with 4 distinct mission_failures refs → PASS."""
        bundle = {
            "trigger_event": {
                "robot_id": "R1",
                "health_at_failure": {"battery_pct": 41},
                "fault_flag": None,
            },
            "mission_failures": [
                {"robot_id": "R1"}, {"robot_id": "R2"},
                {"robot_id": "R3"}, {"robot_id": "R4"},
            ],
        }
        result, notes = validate_diagnosis(
            {**ZONE_WIDE_DIAGNOSIS, "_input_bundle": bundle},
            bundle,
        )
        assert result == DVResult.PASS, notes

    def test_full_pipeline_activates_avoid_zone(self, bb, conn, hot_state,
                                                 mock_llm_multi, mock_embedder):
        """End-to-end: 4-robot zone failure → avoid_zone policy active.

        Pre-seed 3 failures into the BB so that when handle_failure is called
        for R1 it reads back 4 items in mission_failures — matching the four
        refs in ZONE_WIDE_DIAGNOSIS and satisfying the Decision Validator's
        scope-consistency check.
        """
        orchestrator, policy_manager, _ = _make_components(
            bb, hot_state, mock_llm_multi, mock_embedder
        )

        # Pre-seed failures from R2/R3/R4 directly (don't process through pipeline)
        for rid in ["R2", "R3", "R4"]:
            bb.write_failure(None, _make_failure(rid, 2, 4))

        # Now process R1 — get_recent_failures returns all 4 events → PASS
        event = _make_failure("R1", failures_for_mission=2, zone_spread=4)
        orchestrator.handle_failure(event, conn)

        active = policy_manager.get_active()
        avoid_policies = [p for p in active if p.get("type") == "avoid_zone"]
        assert avoid_policies, (
            f"Expected an avoid_zone policy to be active; got: {active}"
        )
        assert avoid_policies[0]["params"]["zone"] == "Receiving Dock"

    def test_scheduler_has_avoid_zone_after_pipeline(
        self, bb, conn, hot_state, mock_llm_multi, mock_embedder
    ):
        """
        After the pipeline runs, the SchedulingService's avoid_zones set
        contains 'receiving_dock', so it would skip dock missions on any sweep.
        (The sweep itself requires a real Postgres cursor; we verify the state
        machine outcome instead.)
        """
        from mars.services.scheduling import SchedulingService

        orchestrator, policy_manager, _ = _make_components(
            bb, hot_state, mock_llm_multi, mock_embedder
        )
        sched = SchedulingService(lambda: conn)
        policy_manager.register_consumer(sched.on_policy_change)

        # Pre-seed 3 failures so the 4th handle_failure call sees all 4 in bundle
        for rid in ["R2", "R3", "R4"]:
            bb.write_failure(None, _make_failure(rid, 2, 4))

        event = _make_failure("R1", failures_for_mission=2, zone_spread=4)
        orchestrator.handle_failure(event, conn)

        # The scheduler must have received the policy activation callback
        assert "Receiving Dock" in sched._avoid_zones, (
            f"Expected 'Receiving Dock' in scheduler avoid_zones; got {sched._avoid_zones}"
        )

    def test_avoid_zone_guardrail_rejects_only_charger_zone(self):
        """Guardrail rejects avoid_zone when it would strand robots from chargers."""
        policy = {
            "type": "avoid_zone",
            "params": {"zone": "charging_bay"},
            "duration_sec": 900,
        }
        world_state = {
            "zones": {
                "charging_bay": {"is_charger_zone": True, "is_mandatory": False},
            },
            "charger_zones": ["charging_bay"],
        }
        from mars.guardrail.guardrail import GuardrailResult
        result, _, notes = guardrail_check(policy, [], world_state)
        assert result == GuardrailResult.REJECT
        assert "charger" in notes.lower()

    def test_fast_disposition_handoff_on_fault_flag(self):
        event = {**_make_failure("R1", 0, 1), "fault_flag": "diagnostics_error"}
        assert fast_disposition(event) == "handoff"

    def test_fast_disposition_retry_on_first_failure(self):
        event = _make_failure("R1", failures_for_mission=0, zone_spread=1)
        event["fault_flag"] = None
        assert fast_disposition(event) == "retry"

    def test_slow_disposition_reschedule_for_zone_wide(self):
        event = _make_failure("R1", 2, 4)
        diagnosis = {"scope": "zone_wide", "persistence": "persistent"}
        assert slow_disposition(event, diagnosis, []) == "reschedule"

    def test_slow_disposition_handoff_for_robot_specific(self):
        event = _make_failure("R1", 3, 1)
        diagnosis = {"scope": "robot_specific", "persistence": "persistent"}
        assert slow_disposition(event, diagnosis, []) == "handoff"
