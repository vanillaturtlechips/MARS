"""
Test fixtures shared across the test suite.

Key patterns:
  - LLM is always mocked — tests are deterministic and need no network
  - Blackboard uses a real Postgres if DB_DSN is set, else an in-memory dict stub
  - MockSim is used as the ROS boundary
"""
from __future__ import annotations

import pytest

from mars.llm.client import MockLLMClient, MockEmbedder
from mars.blackboard.hot_state import HotState
from mars.sim.mock_sim import MockSim
from mars.sim.fault_injector import FaultInjector


# ---------------------------------------------------------------------------
# Canned agent outputs for deterministic testing
# ---------------------------------------------------------------------------

ZONE_WIDE_DIAGNOSIS = {
    "cause": "zone_congestion",
    "scope": "zone_wide",
    "persistence": "persistent",
    "affected_zone": "Receiving Dock",
    "confidence": 0.84,
    "evidence": [
        {
            "observation": "4 distinct robots failed in Receiving Dock in the window",
            "refs": [
                "mission_failures[0]",
                "mission_failures[1]",
                "mission_failures[2]",
                "mission_failures[3]",
            ],
        },
        {
            "observation": "focal robot battery healthy (41%), no fault_flag",
            "refs": ["trigger_event.health_at_failure", "trigger_event.fault_flag"],
        },
    ],
    "relied_on_precedents": [],
}

AVOID_ZONE_STRATEGY = {
    "policy_updates": [
        {
            "type": "avoid_zone",
            "params": {"zone": "Receiving Dock"},
            "duration_sec": 900,
            "rationale": "zone_wide congestion confirmed by incident_analysis",
        }
    ],
    "no_action_reason": None,
    "confidence": 0.8,
    "evidence": [
        {
            "observation": "incident diagnosed zone_wide congestion at the dock",
            "refs": ["incident_analysis.scope", "incident_analysis.affected_zone"],
        }
    ],
    "relied_on_precedents": [],
}

HEALTHY_FLEET = {
    "fleet_health": "healthy",
    "bottlenecks": [],
    "charging_pressure": "low",
    "confidence": 0.9,
    "evidence": [
        {
            "observation": "low backlog, normal utilization",
            "refs": ["fleet_metrics.mission_backlog", "fleet_metrics.robot_utilization"],
        }
    ],
    "relied_on_precedents": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_diagnosis():
    return MockLLMClient(canned_output=ZONE_WIDE_DIAGNOSIS)


@pytest.fixture
def mock_llm_strategy():
    return MockLLMClient(canned_output=AVOID_ZONE_STRATEGY)


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dim=1536)


@pytest.fixture
def hot_state():
    return HotState(redis_client=None)  # in-process fallback


@pytest.fixture
def mock_sim():
    sim = MockSim(robot_ids=["R1", "R2", "R3", "R4", "R5"])
    yield sim
    sim.stop()


@pytest.fixture
def fault_injector(mock_sim):
    return FaultInjector(mock_sim)


@pytest.fixture
def enriched_failure_event():
    """A representative zone-wide failure event (4 robots, same zone)."""
    return {
        "event_type": "navigation.aborted",
        "robot_id": "R5",
        "mission_id": "M123",
        "goal_id": "goal-abc123",
        "zone": "Receiving Dock",
        "nav_outcome": "aborted",
        "goal_status": 6,
        "health_at_failure": {
            "battery_pct": 41.0,
            "estop_active": False,
            "fault_codes": [],
        },
        "fault_flag": None,
        "distribution": {
            "per_robot_zone_spread": 1,
            "per_zone_robot_spread": 4,
        },
        "failures_for_this_mission": 2,
    }


@pytest.fixture
def mission_failures():
    """Window of recent failures matching enriched_failure_event scenario."""
    return [
        {"robot_id": "R5", "mission_id": "M123", "zone": "Receiving Dock",
         "nav_outcome": "aborted", "health": {"battery_pct": 41}},
        {"robot_id": "R2", "mission_id": "M118", "zone": "Receiving Dock",
         "nav_outcome": "aborted", "health": {"battery_pct": 72}},
        {"robot_id": "R7", "mission_id": "M131", "zone": "Receiving Dock",
         "nav_outcome": "aborted", "health": {"battery_pct": 55}},
        {"robot_id": "R9", "mission_id": "M140", "zone": "Receiving Dock",
         "nav_outcome": "aborted", "health": {"battery_pct": 88}},
    ]
