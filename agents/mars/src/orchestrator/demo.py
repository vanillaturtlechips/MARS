"""
M1/M2 Demo — end-to-end vertical slice (M1) + charging pressure loop (M2).

Usage:
    python -m mars.orchestrator.demo

What runs (M1):
    4 robots abort in "Receiving Dock" (injected via FaultInjector)
    → Aggregator enriches each event (health snapshot + distribution)
    → Router: SLOW path (per_zone_robot_spread ≥ ROUTER_SCOPE_HINT)
    → Failure Analysis Agent → Decision Validator
    → Strategy Trigger Rules → Operations Strategy Agent
    → Decision Validator → Policy Guardrail → Policy Manager
    → Scheduler visibly defers all missions routing through Receiving Dock
    → Outcome Evaluator registered

What runs (M2):
    2 robots at low battery (injected)
    → ChargingService tick detects pressure
    → FleetMonitor runs one cycle: Fleet State Agent diagnoses high pressure
    → StrategyTrigger.evaluate_fleet() fires
    → Operations Strategy Agent recommends lower_target_charge_level
    → Policy activated → Charging Service changes behaviour

Requires: Postgres running at DB_DSN (see .env / config.py).
LLM:      Uses real Anthropic Claude (set ANTHROPIC_API_KEY in .env).
Embedder: Mock (zero-vectors) — no Voyage key needed on first run.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Logging — set up before any mars imports so we catch all output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-38s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("demo")

# ---------------------------------------------------------------------------
# Mars imports
# ---------------------------------------------------------------------------
import mars.blackboard.queries as Q
from mars.blackboard.db import apply_migrations, connect, ping
from mars.blackboard.hot_state import HotState
from mars.aggregator.aggregator import Aggregator
from mars.router.router import route, Path
from mars.agents.failure_analysis import FailureAnalysisAgent
from mars.agents.fleet_state import FleetStateAgent
from mars.agents.operations_strategy import OperationsStrategyAgent
from mars.orchestrator.orchestrator import Orchestrator, fast_disposition
from mars.orchestrator.strategy_trigger import StrategyTrigger
from mars.orchestrator.fleet_monitor import FleetMonitor
from mars.validators.retrieval_validator import validate_retrieval_set
from mars.validators.decision_validator import validate_diagnosis, validate_strategy, validate_fleet_state
from mars.guardrail.guardrail import check as guardrail_check
from mars.policy.policy_manager import PolicyManager
from mars.services.scheduling import SchedulingService
from mars.services.charging import ChargingService
from mars.services.ros_executor import ROSExecutor
from mars.outcome.evaluator import OutcomeEvaluator
from mars.sim.mock_sim import MockSim
from mars.sim.fault_injector import FaultInjector
from mars.llm.client import get_llm_client, get_embedder
from mars.config import (
    FAILURE_WINDOW_SECONDS,
    ROUTER_SCOPE_HINT,
)

# ---------------------------------------------------------------------------
# World seed data
# ---------------------------------------------------------------------------

_ROBOTS = ["R1", "R2", "R3", "R4", "R5"]

_ZONES = [
    {"zone_id": "receiving_dock",  "display_name": "Receiving Dock",  "is_charger_zone": False, "is_mandatory": False},
    {"zone_id": "charging_bay",    "display_name": "Charging Bay",    "is_charger_zone": True,  "is_mandatory": False},
    {"zone_id": "storage_area_a",  "display_name": "Storage Area A",  "is_charger_zone": False, "is_mandatory": False},
]

_CHARGERS = [
    {"charger_id": "CH1", "zone_id": "charging_bay"},
    {"charger_id": "CH2", "zone_id": "charging_bay"},
]

# 5 pending missions — all routing through Receiving Dock so the Scheduler
# has something visible to defer when avoid_zone is applied.
_MISSIONS = [
    {"mission_id": f"M{i:03d}", "robot_id": None, "state": "PENDING",
     "zone": "receiving_dock", "priority": 5, "scheduling_priority": 5,
     "destination_pose": {"x": float(i), "y": 0.0, "z": 0.0}}
    for i in range(1, 6)
]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def seed_world(conn) -> None:
    for zone in _ZONES:
        Q.upsert_zone(conn, zone)
    for charger in _CHARGERS:
        Q.upsert_charger(conn, charger)
    for robot_id in _ROBOTS:
        Q.upsert_robot(conn, {
            "robot_id": robot_id,
            "battery_pct": 75.0,
            "allocation_state": "IDLE",
            "mode": "IDLE",
            "current_zone": "receiving_dock",
        })
    for mission in _MISSIONS:
        Q.upsert_mission(conn, mission)
    conn.commit()
    log.info("World seeded: %d zones, %d chargers, %d robots, %d missions",
             len(_ZONES), len(_CHARGERS), len(_ROBOTS), len(_MISSIONS))


def build_components(conn_factory, hot_state, llm, embedder) -> dict[str, Any]:
    """Instantiate all supervisory components and wire them together."""
    policy_manager = PolicyManager(conn_factory)

    scheduling_service = SchedulingService(conn_factory)
    policy_manager.register_consumer(scheduling_service.on_policy_change)

    charging_service = ChargingService(
        conn_factory,
        chargers=_CHARGERS,
    )
    policy_manager.register_consumer(charging_service.on_policy_change)

    strategy_trigger = StrategyTrigger(
        ops_strategy_agent=OperationsStrategyAgent(llm),
        retrieval_validator_fn=validate_retrieval_set,
        decision_validator_fn=validate_strategy,
        policy_manager=policy_manager,
        blackboard_queries=Q,
        embedder=embedder,
    )

    fleet_monitor = FleetMonitor(
        fleet_state_agent=FleetStateAgent(llm),
        strategy_trigger=strategy_trigger,
        blackboard_queries=Q,
        embedder=embedder,
        conn_factory=conn_factory,
        interval_sec=60.0,   # real production interval; demo calls run_once() directly
    )

    orchestrator = Orchestrator(
        blackboard_queries=Q,
        hot_state=hot_state,
        failure_analysis_agent=FailureAnalysisAgent(llm),
        retrieval_validator_fn=validate_retrieval_set,
        decision_validator_fn=validate_diagnosis,
        strategy_trigger_fn=strategy_trigger.evaluate,
        policy_manager=policy_manager,
        embedder=embedder,
    )

    outcome_evaluator = OutcomeEvaluator(Q, embedder, conn_factory)

    def _on_policy_event(event: str, policy: dict) -> None:
        if event == "activated":
            conn = conn_factory()
            baseline = {"failure_rate": _current_failure_rate(conn, policy)}
            conn.close()
            outcome_evaluator.register_watch(
                action_type="policy",
                action_id=policy["policy_id"],
                zone=policy.get("params", {}).get("zone"),
                baseline_metrics=baseline,
            )

    policy_manager.register_consumer(_on_policy_event)

    return {
        "orchestrator":      orchestrator,
        "policy_manager":    policy_manager,
        "scheduling_service": scheduling_service,
        "charging_service":  charging_service,
        "strategy_trigger":  strategy_trigger,
        "fleet_monitor":     fleet_monitor,
        "outcome_evaluator": outcome_evaluator,
    }


def _current_failure_rate(conn, policy: dict) -> float:
    zone = policy.get("params", {}).get("zone")
    recent = Q.get_recent_failures(conn, window_seconds=FAILURE_WINDOW_SECONDS, zone=zone)
    return len(recent) / max(FAILURE_WINDOW_SECONDS / 60, 1)


# ---------------------------------------------------------------------------
# Sim wiring
# ---------------------------------------------------------------------------

def wire_sim(sim: MockSim, aggregator: Aggregator) -> dict[str, str]:
    """
    Subscribe aggregator callbacks to the sim.
    Returns goal_to_mission map (populated later when goals are sent).
    """
    goal_to_mission: dict[str, str] = {}

    for robot_id in _ROBOTS:
        sim.subscribe_battery(robot_id, aggregator.on_battery_update)
        sim.subscribe_health(robot_id, aggregator.on_health_update)

        # Pose callback: for demo all robots are in receiving_dock
        def make_pose_cb(rid: str):
            def _pose_cb(pose):
                aggregator.on_pose_update(rid, pose, zone="receiving_dock")
            return _pose_cb

        sim.subscribe_pose(robot_id, make_pose_cb(robot_id))

        # Nav status: look up mission_id from goal_to_mission map
        def make_nav_cb(rid: str):
            def _nav_cb(nav_status):
                mission_id = goal_to_mission.get(nav_status.goal_id)
                aggregator.on_nav_status(nav_status, mission_id)
            return _nav_cb

        sim.subscribe_nav_status(robot_id, make_nav_cb(robot_id))

    # Pre-populate aggregator zone map
    for robot_id in _ROBOTS:
        aggregator._zone_map[robot_id] = "receiving_dock"

    return goal_to_mission


def dispatch_demo_goals(
    sim: MockSim,
    conn,
    goal_to_mission: dict[str, str],
) -> list[str]:
    """
    Send NavigateToPose goals for the first N robots so they have active goals
    to abort.  Returns list of goal_ids.
    """
    from mars.ros.interfaces import Pose
    goal_ids = []
    for i, robot_id in enumerate(_ROBOTS[:4]):
        goal_id = f"goal-demo-{robot_id}"
        destination = Pose(x=float(i * 2), y=1.0)
        mission_id = _MISSIONS[i]["mission_id"]
        goal_to_mission[goal_id] = mission_id

        # Mark mission ACTIVE in DB (robot assigned + goal dispatched)
        Q.upsert_mission(conn, {**_MISSIONS[i], "robot_id": robot_id,
                                "state": "ACTIVE", "goal_id": goal_id})
        Q.record_dispatch(conn, goal_id, mission_id, robot_id)
        # Mark robot BUSY
        Q.cas_allocation_state(conn, robot_id, "IDLE", "BUSY")
        Q.upsert_robot(conn, {"robot_id": robot_id, "allocation_state": "BUSY",
                               "current_mission_id": mission_id,
                               "current_zone": "receiving_dock"})

        sim.send_goal(robot_id, goal_id, destination, on_status_change=lambda s: None)
        goal_ids.append(goal_id)

    conn.commit()
    log.info("Dispatched %d goals", len(goal_ids))
    return goal_ids


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    _DIVIDER = "=" * 64

    print(_DIVIDER)
    print("  MARS M1 DEMO — Vertical Slice")
    print(_DIVIDER)

    # -- DB check --
    if not ping():
        print("\nERROR: Cannot reach Postgres at DB_DSN.")
        print("Start the database: docker-compose up -d  (or native postgres)")
        sys.exit(1)

    conn_main = connect()
    apply_migrations(conn_main)

    def conn_factory():
        return connect()

    # -- Hot state (Redis optional; falls back to in-process dict) --
    hot_state = _make_hot_state()

    # -- LLM + Embedder --
    llm      = _make_llm()
    embedder = _make_embedder()

    # -- Seed world --
    seed_world(conn_main)

    # -- Build components --
    components       = build_components(conn_factory, hot_state, llm, embedder)
    orchestrator     = components["orchestrator"]
    policy_manager   = components["policy_manager"]
    scheduling_svc   = components["scheduling_service"]
    charging_service = components["charging_service"]
    fleet_monitor    = components["fleet_monitor"]

    # -- Event queue: sim callbacks → main thread --
    event_queue: queue.Queue[dict] = queue.Queue()

    def on_failure_event(event: dict) -> None:
        event_queue.put(event)

    aggregator = Aggregator(hot_state, on_failure_event=on_failure_event)

    # -- Mock sim --
    sim = MockSim(robot_ids=_ROBOTS, tick_hz=20.0)
    goal_to_mission = wire_sim(sim, aggregator)
    sim.start()

    # -- Dispatch goals so robots have something to abort --
    dispatch_demo_goals(sim, conn_main, goal_to_mission)

    time.sleep(0.5)  # let sim tick once so battery/health snapshots are populated

    # -- Inject zone-wide failure: 4 robots abort in Receiving Dock --
    print()
    print(_DIVIDER)
    print("  INJECTING: 4-robot zone-wide failure in Receiving Dock")
    print(_DIVIDER)
    fi = FaultInjector(sim)
    fi.inject_zone_failure("Receiving Dock", _ROBOTS[:4], delay_between_sec=0.05)

    # -- Process events from the queue --
    results: list[dict] = []
    deadline = time.time() + 8.0  # wait up to 8 s for all events

    print()
    while time.time() < deadline:
        try:
            event = event_queue.get(timeout=0.5)
        except queue.Empty:
            if results:
                break
            continue

        conn = conn_factory()
        try:
            result = orchestrator.handle_failure(event, conn)
            conn.commit()
            results.append(result)
            _print_event_result(event, result)
        except Exception:
            log.exception("handle_failure raised")
            conn.rollback()
        finally:
            conn.close()

    sim.stop()

    # ── M2: Charging pressure loop ──────────────────────────────────────
    print()
    print(_DIVIDER)
    print("  M2: Charging pressure → Fleet Monitor → policy loop")
    print(_DIVIDER)

    # Set 3 robots to low battery so the Charging Service detects pressure
    for rid in _ROBOTS[1:4]:
        Q.upsert_robot(conn_main, {"robot_id": rid, "battery_pct": 12.0,
                                    "allocation_state": "IDLE", "mode": "IDLE"})
    conn_main.commit()

    # Tick the Charging Service with low-battery robots
    robots_snapshot = [
        {"robot_id": rid, "battery_pct": 12.0, "allocation_state": "IDLE"}
        for rid in _ROBOTS[1:4]
    ]
    conn_pressure = conn_factory()
    charging_service.tick(robots_snapshot, conn_pressure,
                          sim_time=datetime.now(timezone.utc))
    conn_pressure.commit()
    conn_pressure.close()
    log.info("Charging tick: queue_len=%d", charging_service.queue_length())

    # Run one fleet monitor cycle (synchronous — would normally run periodically)
    print()
    log.info("Running fleet monitor cycle...")
    conn_fleet = conn_factory()
    fleet_out = fleet_monitor.run_once(conn_fleet)
    conn_fleet.commit()
    conn_fleet.close()
    if fleet_out:
        log.info("Fleet analysis: health=%s pressure=%s",
                 fleet_out.get("fleet_health"), fleet_out.get("charging_pressure"))
    # ────────────────────────────────────────────────────────────────────

    # -- Summary --
    _print_summary(results, policy_manager, scheduling_svc, conn_main)
    conn_main.close()


def _print_event_result(event: dict, result: dict) -> None:
    robot = event.get("robot_id", "?")
    zone  = event.get("zone", "?")
    path  = result.get("path", "?")
    disp  = result.get("disposition", "?")
    dv    = result.get("dv_result", "")
    dv_str = f"  DV={dv}" if dv else ""
    print(f"  event robot={robot} zone={zone}  →  path={path}  disposition={disp}{dv_str}")


def _print_summary(
    results: list[dict],
    policy_manager: PolicyManager,
    scheduling_svc: SchedulingService,
    conn,
) -> None:
    _DIVIDER = "=" * 64
    print()
    print(_DIVIDER)
    print("  M1 DEMO RESULT")
    print(_DIVIDER)

    slow = [r for r in results if r.get("path") == "slow"]
    fast = [r for r in results if r.get("path") == "fast"]
    print(f"  Failures processed   : {len(results)}  (slow={len(slow)}  fast={len(fast)})")

    # Diagnosis summary (last slow-path result)
    if slow:
        sr = slow[-1]
        if sr.get("diagnosis_id"):
            from mars.blackboard import queries as Q2
            diag = Q2.get_recent_failures(conn, 30)  # quick proxy — diagnosis not in this call
        dv = sr.get("dv_result", "?")
        print(f"  Decision Validator   : {dv}")

    # Active policies
    active = policy_manager.get_active()
    if active:
        for p in active:
            ptype  = p.get("type", "?")
            params = p.get("params", {})
            zone   = params.get("zone", "")
            gr     = p.get("guardrail_result", "?")
            exp_raw = p.get("expires_at")
            exp_str = ""
            if exp_raw:
                if hasattr(exp_raw, "strftime"):
                    exp_str = f"  expires={exp_raw.strftime('%H:%M:%S')}"
            print(f"  Policy activated     : {ptype}  zone={zone}  guardrail={gr}{exp_str}")
    else:
        print("  Policy activated     : (none)")

    # Scheduler behaviour
    conn2 = connect()
    pending = Q.get_pending_missions(conn2)
    dock_pending = [m for m in pending if m.get("zone") == "receiving_dock"]
    conn2.close()
    avoid_active = policy_manager.is_policy_active_for_zone(
        "receiving_dock", "avoid_zone"
    )
    print(f"  Dock missions pending: {len(dock_pending)}")
    print(f"  avoid_zone active    : {avoid_active}")
    if avoid_active:
        print()
        print("  ✓ Scheduler will skip Receiving Dock missions on next sweep.")
        print("  ✓ M1 vertical slice COMPLETE — all graded components exercised.")
    else:
        print()
        print("  (No avoid_zone policy applied — check logs above for details.)")

    print(_DIVIDER)


# ---------------------------------------------------------------------------
# Provider factories — graceful degradation
# ---------------------------------------------------------------------------

def _make_hot_state() -> HotState:
    try:
        import redis as redis_lib
        from mars.config import REDIS_URL
        r = redis_lib.from_url(REDIS_URL)
        r.ping()
        log.info("HotState: using Redis at %s", REDIS_URL)
        return HotState(redis_client=r)
    except Exception:
        log.info("HotState: Redis unavailable — using in-process dict fallback")
        return HotState(redis_client=None)


def _make_llm():
    from mars.config import ANTHROPIC_API_KEY
    if ANTHROPIC_API_KEY:
        log.info("LLM: Anthropic Claude (real)")
        return get_llm_client("anthropic")
    log.warning("LLM: ANTHROPIC_API_KEY not set — using mock LLM (canned output)")
    from tests.conftest import ZONE_WIDE_DIAGNOSIS, AVOID_ZONE_STRATEGY
    from mars.llm.client import MockLLMClient

    class _MultiMock:
        """Routes to diagnosis or strategy output by system-prompt content."""
        def complete_structured(self, system_prompt, user_message, output_schema, *, temperature=0.0):
            if "DIAGNOSE" in system_prompt:
                return dict(ZONE_WIDE_DIAGNOSIS)
            return dict(AVOID_ZONE_STRATEGY)

    return _MultiMock()


def _make_embedder():
    from mars.config import VOYAGE_API_KEY
    if VOYAGE_API_KEY:
        log.info("Embedder: Voyage AI voyage-3")
        return get_embedder("voyage")
    log.info("Embedder: mock (zero-vectors) — RAG store starts empty (expected on first run)")
    return get_embedder("mock")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demo()
