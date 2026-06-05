"""
ROS2 Node Runner — wires the Isaac Sim adapter to the Aggregator.

This is the entry point when running against real Isaac Sim + Nav2.
Equivalent role to demo.py but for production use.

Usage:
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash   # for mars_msgs
    python -m mars.ros.ros2_node

The node:
  1. Reads configuration (robot IDs, DB DSN) from env / config.py
  2. Loads zones from the blackboard to build the ZoneResolver
  3. Creates ROS2SimAdapter (replaces MockSim)
  4. Creates Aggregator with an event queue for the main thread
  5. Builds all supervisory components (same as demo.py)
  6. Spins the ROS2 node

Swap guide (mock → real):
  Before (demo.py):
      sim = MockSim(robot_ids=..., tick_hz=20)
      sim.start()
  After (this file):
      rclpy.init()
      node = rclpy.create_node("mars_supervisor", ...)
      sim = ROS2SimAdapter(node, robot_ids, zone_resolver)
  Everything downstream is unchanged.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any

log = logging.getLogger(__name__)

_ROBOT_IDS = ["R1", "R2", "R3", "R4", "R5"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-38s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # 1. ROS2 init
    # ------------------------------------------------------------------
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import MultiThreadedExecutor
    except ImportError:
        log.error("rclpy not available — cannot run ROS2 node.  "
                  "Use demo.py with MockSim for testing without ROS2.")
        sys.exit(1)

    rclpy.init()
    node = rclpy.create_node(
        "mars_supervisor",
        parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True),
        ],
    )
    log.info("[ros2_node] node created, use_sim_time=True")

    # ------------------------------------------------------------------
    # 2. Blackboard connection + world load
    # ------------------------------------------------------------------
    import mars.blackboard.queries as Q
    from mars.blackboard.db import apply_migrations, connect
    from mars.blackboard.hot_state import HotState

    conn_main = connect()
    apply_migrations(conn_main)
    conn_factory = connect

    # Load zones for ZoneResolver
    from mars.ros.zone_resolver import ZoneResolver
    with conn_main.cursor() as cur:
        cur.execute("SELECT zone_id, polygon FROM zones")
        zones_raw = [{"zone_id": r[0], "polygon": r[1] or []} for r in cur.fetchall()]
    zone_resolver = ZoneResolver(zones_raw)

    # ------------------------------------------------------------------
    # 3. Hot state
    # ------------------------------------------------------------------
    try:
        import redis as redis_lib
        from mars.config import REDIS_URL
        r = redis_lib.from_url(REDIS_URL)
        r.ping()
        hot_state = HotState(redis_client=r)
    except Exception:
        log.info("[ros2_node] Redis unavailable — hot state uses in-process dict")
        hot_state = HotState(redis_client=None)

    # ------------------------------------------------------------------
    # 4. LLM + Embedder
    # ------------------------------------------------------------------
    from mars.llm.client import get_llm_client, get_embedder
    from mars.config import ANTHROPIC_API_KEY, VOYAGE_API_KEY
    llm      = get_llm_client("anthropic") if ANTHROPIC_API_KEY else get_llm_client("mock")
    embedder = get_embedder("voyage") if VOYAGE_API_KEY else get_embedder("mock")

    # ------------------------------------------------------------------
    # 5. Supervisory components
    # ------------------------------------------------------------------
    from mars.agents.failure_analysis import FailureAnalysisAgent
    from mars.agents.fleet_state import FleetStateAgent
    from mars.agents.operations_strategy import OperationsStrategyAgent
    from mars.orchestrator.orchestrator import Orchestrator
    from mars.orchestrator.strategy_trigger import StrategyTrigger
    from mars.orchestrator.fleet_monitor import FleetMonitor
    from mars.validators.retrieval_validator import validate_retrieval_set
    from mars.validators.decision_validator import validate_diagnosis, validate_strategy
    from mars.policy.policy_manager import PolicyManager
    from mars.services.scheduling import SchedulingService
    from mars.services.charging import ChargingService
    from mars.outcome.evaluator import OutcomeEvaluator

    chargers_raw = []
    with conn_main.cursor() as cur:
        cur.execute("SELECT charger_id, zone_id FROM chargers WHERE is_online=TRUE")
        for row in cur.fetchall():
            chargers_raw.append({"charger_id": row[0], "zone_id": row[1]})

    policy_manager = PolicyManager(conn_factory)
    scheduling_svc = SchedulingService(conn_factory)
    charging_svc   = ChargingService(conn_factory, chargers_raw)
    policy_manager.register_consumer(scheduling_svc.on_policy_change)
    policy_manager.register_consumer(charging_svc.on_policy_change)

    strategy_trigger = StrategyTrigger(
        ops_strategy_agent=OperationsStrategyAgent(llm),
        retrieval_validator_fn=validate_retrieval_set,
        decision_validator_fn=validate_strategy,
        policy_manager=policy_manager,
        blackboard_queries=Q,
        embedder=embedder,
    )
    outcome_evaluator = OutcomeEvaluator(Q, embedder, conn_factory)
    fleet_monitor = FleetMonitor(
        fleet_state_agent=FleetStateAgent(llm),
        strategy_trigger=strategy_trigger,
        blackboard_queries=Q,
        embedder=embedder,
        conn_factory=conn_factory,
        outcome_evaluator=outcome_evaluator,
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

    # ------------------------------------------------------------------
    # 6. Adapter + Aggregator
    # ------------------------------------------------------------------
    from mars.ros.isaac_sim_adapter import ROS2SimAdapter
    from mars.aggregator.aggregator import Aggregator

    event_queue: queue.Queue[dict] = queue.Queue()

    def on_failure_event(event: dict) -> None:
        event_queue.put(event)

    aggregator = Aggregator(hot_state, on_failure_event=on_failure_event)

    # Pre-populate zone map from blackboard (best-effort; TF updates live)
    with conn_main.cursor() as cur:
        cur.execute("SELECT robot_id, current_zone FROM robots")
        for row in cur.fetchall():
            if row[1]:
                aggregator._zone_map[row[0]] = row[1]

    adapter = ROS2SimAdapter(node, _ROBOT_IDS, zone_resolver)

    for robot_id in _ROBOT_IDS:
        # Aggregator pose callback resolves zone via ZoneResolver
        def _make_pose_cb(rid: str):
            def _cb(pose):
                zone = zone_resolver.resolve(pose.x, pose.y)
                if zone:
                    aggregator.on_pose_update(rid, pose, zone)
            return _cb

        def _make_nav_cb(rid: str):
            def _cb(nav_status):
                # Look up mission_id via dispatch ledger
                conn = conn_factory()
                row  = Q.get_dispatch_by_goal(conn, nav_status.goal_id)
                conn.close()
                mission_id = row["mission_id"] if row else None
                aggregator.on_nav_status(nav_status, mission_id)
            return _cb

        adapter.subscribe_battery(robot_id, aggregator.on_battery_update)
        adapter.subscribe_health(robot_id, aggregator.on_health_update)
        adapter.subscribe_pose(robot_id, _make_pose_cb(robot_id))
        adapter.subscribe_nav_status(robot_id, _make_nav_cb(robot_id))

    # ------------------------------------------------------------------
    # 7. Main processing loop (event queue → orchestrator) + fleet monitor
    # ------------------------------------------------------------------
    fleet_monitor.start()
    log.info("[ros2_node] fleet monitor started")

    def _process_events() -> None:
        """Main-thread loop: drain event queue and run orchestrator."""
        while rclpy.ok():
            try:
                event = event_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            conn = conn_factory()
            try:
                result = orchestrator.handle_failure(event, conn)
                conn.commit()
                log.info("[ros2_node] handled failure: %s", result)
            except Exception:
                log.exception("[ros2_node] handle_failure raised")
                conn.rollback()
            finally:
                conn.close()

    event_thread = threading.Thread(target=_process_events, daemon=True)
    event_thread.start()

    # ------------------------------------------------------------------
    # 8. Spin ROS2 (blocks until Ctrl+C)
    # ------------------------------------------------------------------
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        log.info("[ros2_node] spinning (Ctrl+C to stop)")
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        fleet_monitor.stop()
        node.destroy_node()
        rclpy.shutdown()
        conn_main.close()
        log.info("[ros2_node] shutdown complete")


if __name__ == "__main__":
    main()
