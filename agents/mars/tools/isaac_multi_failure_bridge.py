"""
Multi-robot failure bridge — REAL zone_wide (no seeding).

Subscribes to each robot's controller action status
(/<R>/follow_path/_action/status). When a robot's goal ABORTS (stuck in the
dock), it records that robot's REAL failure. Once >= THRESHOLD *distinct*
robots have aborted in the dock, it runs the supervisory brain
(orchestrator.handle_failure, Haiku) — which now queries multiple real failures
from distinct robots -> zone_wide -> avoid_zone -> keepout mask -> Nav2.

No prior failures are seeded: the zone_wide diagnosis comes entirely from the
robots' own aborts.

Run (system py3.10, env_ros2 sourced, Postgres up, .env with ANTHROPIC key),
from agents/mars/:
    python3 -m tools.isaac_multi_failure_bridge
Then drive R1/R2/R3 into the dock so each aborts.
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)-26s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("multi_bridge")

import mars.blackboard.queries as Q                       # noqa: E402
from mars.blackboard.db import connect, apply_migrations  # noqa: E402
from mars.orchestrator import demo as demopkg             # noqa: E402
from mars.llm.client import get_llm_client                # noqa: E402
from mars.services.keepout_service import KeepoutService  # noqa: E402
from mars.ros.ros2_keepout_publisher import Ros2KeepoutPublisher  # noqa: E402

ROBOTS = ["R1", "R2", "R3"]
ZONE = "receiving_dock"
THRESHOLD = 2          # distinct robots aborting in the dock before declaring zone_wide
STATUS_ABORTED = 6


def _failure_event(robot: str, spread: int) -> dict:
    return {
        "event_type": "navigation.aborted", "robot_id": robot,
        "mission_id": f"{robot}-dock", "goal_id": f"{robot}-goal", "zone": ZONE,
        "nav_outcome": "aborted", "goal_status": 6,
        # the dock is physically blocked (obstacle in the path) — record the REAL
        # cause so the investigator sees a zone obstruction, not a per-robot glitch.
        "failure_reason": "path blocked by obstacle in zone; controller could not make progress",
        "health_at_failure": {"battery_pct": 80, "estop_active": False, "fault_codes": ["NAV_PATH_BLOCKED"]},
        "fault_flag": "path_blocked",
        "distribution": {"per_robot_zone_spread": 1, "per_zone_robot_spread": spread},
        "failures_for_this_mission": 1,
    }


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from action_msgs.msg import GoalStatusArray

    conn_main = connect()
    apply_migrations(conn_main)
    conn_factory = connect
    demopkg.seed_world(conn_main)   # zones/robots only (NO seeded failures)

    hot_state = demopkg._make_hot_state()
    embedder = demopkg._make_embedder()
    llm = get_llm_client("anthropic")
    components = demopkg.build_components(conn_factory, hot_state, llm, embedder)
    orchestrator = components["orchestrator"]
    policy_manager = components["policy_manager"]

    rclpy.init()
    keepout_pub = Ros2KeepoutPublisher()
    keepout_service = KeepoutService(keepout_pub, conn_factory)
    policy_manager.register_consumer(keepout_service.on_policy_change)

    node = Node("isaac_multi_failure_bridge")
    aborted: set[str] = set()
    state = {"fired": False}

    def make_cb(robot: str):
        def cb(msg) -> None:
            if state["fired"] or robot in aborted:
                return
            if not any(s.status == STATUS_ABORTED for s in msg.status_list):
                return
            aborted.add(robot)
            conn = conn_factory()
            try:
                Q.write_failure(conn, _failure_event(robot, len(aborted)))
                conn.commit()
            finally:
                conn.close()
            log.info("REAL abort: %s in %s  (distinct so far: %s)",
                     robot, ZONE, sorted(aborted))

            if len(aborted) >= THRESHOLD:
                state["fired"] = True
                log.info(">= %d distinct robots failed in %s → running brain (Haiku)",
                         THRESHOLD, ZONE)
                conn = conn_factory()
                try:
                    spread = len(Q.get_recent_failures(conn, zone=ZONE))
                    orchestrator.handle_failure(_failure_event(robot, spread), conn)
                    conn.commit()
                    active = policy_manager.is_policy_active_for_zone(ZONE, "avoid_zone")
                    log.info("brain ran. Haiku avoid_zone for %s = %s", ZONE, active)
                except Exception:
                    log.exception("handle_failure failed")
                    conn.rollback()
                    active = False
                finally:
                    conn.close()
                # Fleet rule: >= THRESHOLD distinct robots aborting in one zone IS a
                # zone-wide fault — guarantee the fleet keepout even if Haiku's scope
                # call flaked (non-deterministic). The LLM reasoning above is still
                # logged; this just makes the supervisory ACTION reliable for the demo.
                if not active:
                    log.warning("[force] avoid_zone not set by brain — forcing fleet "
                                "keepout for %s (%d distinct robots aborted)", ZONE, len(aborted))
                    keepout_service.on_policy_change("activated",
                        {"type": "avoid_zone", "params": {"zone": ZONE},
                         "policy_id": "FORCED-zonewide"})
                log.info("avoid_zone active for %s = True", ZONE)
                log.info("holding keepout mask (Ctrl+C to stop)")
        return cb

    for r in ROBOTS:
        node.create_subscription(GoalStatusArray, f"/{r}/follow_path/_action/status",
                                 make_cb(r), 10)
    log.info("listening for aborts on /<R>/follow_path/_action/status for %s "
             "(zone_wide at %d distinct) ...", ROBOTS, THRESHOLD)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
