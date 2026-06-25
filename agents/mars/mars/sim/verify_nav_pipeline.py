"""
Step A — adapter → Aggregator verification harness (NO Isaac Sim).

Stands up a mock NavigateToPose action server that ABORTS every goal, then drives
one goal through the REAL adapter and checks that an enriched failure event
reaches the Aggregator.  This isolates our code: if this passes, any later
Isaac Sim failure is in the ROS2 bridge, not in the adapter/aggregator.

    ROS2SimAdapter.send_goal(R1) --NavigateToPose--> MockNavServer (abort)
        adapter._on_result(status=ABORTED) -> nav_status callback
            -> Aggregator.on_nav_status -> enriched failure event   ← assert

PASS = enriched event with nav_outcome == "aborted" reaches the Aggregator.

Run (venv OFF, ROS + mars_msgs sourced):
    cd agents/mars
    source /opt/ros/jazzy/setup.bash
    source ros2_ws/install/setup.bash
    python3 -m mars.sim.verify_nav_pipeline
"""
from __future__ import annotations

import logging
import sys
import threading
import time

log = logging.getLogger("verify_nav")

ROBOT = "R1"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import rclpy
        from rclpy.action import ActionServer
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from nav2_msgs.action import NavigateToPose
    except ImportError as exc:
        print(f"FAIL: ROS2 not available ({exc}). Source /opt/ros/jazzy/setup.bash first.")
        sys.exit(1)

    from mars.aggregator.aggregator import Aggregator
    from mars.blackboard.hot_state import HotState
    from mars.ros.interfaces import Pose
    from mars.ros.isaac_sim_adapter import ROS2SimAdapter
    from mars.ros.zone_resolver import ZoneResolver

    rclpy.init()

    # ── Mock NavigateToPose server: aborts every goal ──────────────────
    server_node = rclpy.create_node("mock_nav_server")
    server_cb = ReentrantCallbackGroup()

    def execute_cb(goal_handle):
        log.info("[mock_nav] goal received for %s -> ABORT", ROBOT)
        time.sleep(0.3)
        goal_handle.abort()
        return NavigateToPose.Result()

    ActionServer(
        server_node,
        NavigateToPose,
        f"/{ROBOT}/navigate_to_pose",
        execute_callback=execute_cb,
        callback_group=server_cb,
    )

    # ── Real adapter + Aggregator ──────────────────────────────────────
    adapter_node = rclpy.create_node("mars_verify")
    adapter = ROS2SimAdapter(adapter_node, [ROBOT], ZoneResolver([]))

    if ROBOT not in getattr(adapter, "_action_clients", {}):
        print("FAIL: adapter._setup did not create an action client "
              "(likely an ImportError no-op). Check isaac_sim_adapter._setup imports.")
        rclpy.shutdown()
        sys.exit(1)

    hot = HotState(redis_client=None)
    got = threading.Event()
    events: list[dict] = []

    def on_failure(ev: dict) -> None:
        events.append(ev)
        got.set()

    agg = Aggregator(hot, on_failure_event=on_failure)
    agg._zone_map[ROBOT] = "TestZone"  # so the enriched event carries a zone
    adapter.subscribe_nav_status(ROBOT, lambda ns: agg.on_nav_status(ns, "M_TEST"))

    # ── Spin both nodes ────────────────────────────────────────────────
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(server_node)
    executor.add_node(adapter_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    ac = adapter._action_clients[ROBOT]
    if not ac.wait_for_server(timeout_sec=10.0):
        print("FAIL: mock action server not discoverable within 10s")
        rclpy.shutdown()
        sys.exit(1)

    log.info("sending goal to %s ...", ROBOT)
    adapter.send_goal(ROBOT, "goal_test_1", Pose(x=5.0, y=5.0), on_status_change=lambda s: None)

    ok = got.wait(timeout=15.0)

    print("\n================ RESULT ================")
    if ok and events and events[0].get("nav_outcome") == "aborted":
        ev = events[0]
        print("PASS  adapter -> Aggregator path works")
        print(f"  nav_outcome = {ev['nav_outcome']}")
        print(f"  robot_id    = {ev['robot_id']}")
        print(f"  mission_id  = {ev['mission_id']}")
        print(f"  zone        = {ev['zone']}")
        print(f"  goal_status = {ev['goal_status']}  (6 = ABORTED)")
        print(f"  fault_flag  = {ev['fault_flag']}")
        rc = 0
    else:
        print("FAIL  no aborted enriched event reached the Aggregator")
        print(f"  got_event={ok}  events={events}")
        rc = 1
    print("========================================")

    rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
