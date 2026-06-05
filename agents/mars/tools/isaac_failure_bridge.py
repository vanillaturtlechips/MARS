"""
Isaac failure bridge — closes the loop:
  iw_hub's REAL Nav2 abort  ->  Haiku brain  ->  avoid_zone  ->  keepout mask
  ->  /keepout_filter_mask  ->  Nav2 reroutes the robot.

Subscribes to /navigate_to_pose/_action/status. On ABORTED (status=6) it builds
a failure event for receiving_dock and runs the real supervisory brain
(orchestrator.handle_failure, Haiku). So the loop is triggered by the robot's
own failure, not by injected mock events.

Single robot aborting once reads as "isolated", so the dock is PRE-SEEDED with a
few recent failures (as if other robots failed there earlier) — the real abort
is then the Nth failure -> zone_wide -> avoid_zone.

Run (system py3.10, `source deploy/isaac/env_ros2.sh`, Postgres up, .env with a
valid ANTHROPIC key), from agents/mars/:
    python3 -m tools.isaac_failure_bridge
Then drive iw_hub into the dock (terminal D) so Nav2 aborts.
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)-28s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("failure_bridge")

import mars.blackboard.queries as Q                       # noqa: E402
from mars.blackboard.db import connect, apply_migrations  # noqa: E402
from mars.orchestrator import demo as demopkg             # noqa: E402
from mars.llm.client import get_llm_client                # noqa: E402
from mars.services.keepout_service import KeepoutService  # noqa: E402
from mars.ros.ros2_keepout_publisher import Ros2KeepoutPublisher  # noqa: E402

ZONE = "receiving_dock"
# Listen to the CONTROLLER action (follow_path): it aborts immediately on
# "Failed to make progress". The top-level navigate_to_pose stays EXECUTING
# while bt_navigator runs recovery/retries, so it doesn't report ABORTED fast.
NAV_STATUS_TOPIC = "/follow_path/_action/status"
STATUS_ABORTED = 6


def _seed_prior_failures(conn, n: int = 3) -> None:
    for i in range(n):
        Q.write_failure(conn, {
            "robot_id": f"R{i + 1}", "mission_id": f"seed-M{i + 1}", "zone": ZONE,
            "event_type": "navigation.aborted", "nav_outcome": "aborted", "goal_status": 6,
            "health_at_failure": {"battery_pct": 70, "estop_active": False, "fault_codes": []},
            "fault_flag": None,
            "distribution": {"per_robot_zone_spread": 1, "per_zone_robot_spread": n + 1},
            "failures_for_this_mission": 1,
        })
    conn.commit()
    log.info("seeded %d prior failures in %s", n, ZONE)


def _failure_event(spread: int) -> dict:
    return {
        "event_type": "navigation.aborted", "robot_id": "iw_hub",
        "mission_id": "iwhub-dock", "goal_id": "iwhub-goal", "zone": ZONE,
        "nav_outcome": "aborted", "goal_status": 6,
        "health_at_failure": {"battery_pct": 80, "estop_active": False, "fault_codes": []},
        "fault_flag": None,
        "distribution": {"per_robot_zone_spread": 1, "per_zone_robot_spread": spread},
        "failures_for_this_mission": 1,
    }


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from action_msgs.msg import GoalStatusArray

    # ---- brain (real Haiku) ----
    conn_main = connect()
    apply_migrations(conn_main)
    conn_factory = connect
    demopkg.seed_world(conn_main)
    _seed_prior_failures(conn_main, n=3)

    hot_state = demopkg._make_hot_state()
    embedder = demopkg._make_embedder()
    llm = get_llm_client("anthropic")
    components = demopkg.build_components(conn_factory, hot_state, llm, embedder)
    orchestrator = components["orchestrator"]
    policy_manager = components["policy_manager"]

    # ---- keepout -> real ROS2 mask ----
    rclpy.init()
    keepout_pub = Ros2KeepoutPublisher()
    policy_manager.register_consumer(KeepoutService(keepout_pub, conn_factory).on_policy_change)

    node = Node("isaac_failure_bridge")
    fired = {"done": False}

    def on_status(msg) -> None:
        if fired["done"]:
            return
        if not any(s.status == STATUS_ABORTED for s in msg.status_list):
            return
        fired["done"] = True
        log.info("iw_hub Nav2 goal ABORTED → running supervisory brain (Haiku)")
        conn = conn_factory()
        try:
            spread = len(Q.get_recent_failures(conn, zone=ZONE)) + 1
            orchestrator.handle_failure(_failure_event(spread), conn)
            conn.commit()
            active = policy_manager.is_policy_active_for_zone(ZONE, "avoid_zone")
            log.info("brain done. avoid_zone active for %s = %s", ZONE, active)
        except Exception:
            log.exception("handle_failure failed")
            conn.rollback()
        finally:
            conn.close()
        log.info("holding keepout mask (Ctrl+C to stop)")

    node.create_subscription(GoalStatusArray, NAV_STATUS_TOPIC, on_status, 10)
    log.info("listening for iw_hub Nav2 abort on %s ...", NAV_STATUS_TOPIC)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
