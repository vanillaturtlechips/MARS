"""
Charging arbitration bridge — drives the Isaac robots from a REAL supervisory
decision (NOT hardcoded order).

Two scenarios share one physical charger; the ORDER robots take it comes from
the real ``ChargingService`` priority queue + compare-and-set serialization —
exactly the fleet-level arbitration a per-robot RL policy cannot do:

  --scenario charging  (demo2): R1 critical (8%) + R2 low (25%) both want the one
      charger -> the queue grants it to the CRITICAL robot first; the low one is
      delayed until the charger frees.
  --scenario priority  (demo3): R2 + R3 both critical (12%) contend for the one
      charger -> the queue + CAS serialize them (one charges, the other waits its
      turn, then charges) so they never deadlock over the same spot.

The LLM fleet loop (FleetMonitor -> FleetStateAgent -> OperationsStrategyAgent)
is also exercised best-effort so the brain is genuinely in the loop (logged).

Run (system py3.10, env_ros2 sourced, Nav2 up for R1/R2/R3, Postgres up, .env),
from agents/mars/:
    python3 -m tools.isaac_charging_bridge --scenario charging
    python3 -m tools.isaac_charging_bridge --scenario priority
"""
from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)-22s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("charge_bridge")

import mars.blackboard.queries as Q                       # noqa: E402
from mars.blackboard.db import connect, apply_migrations  # noqa: E402
from mars.orchestrator import demo as demopkg             # noqa: E402
from mars.services.charging import ChargingService        # noqa: E402

# Physical poses live in the KNOWN-OPEN lane the robots already drive (x[-3,3]
# y[-8,5]). The DB charger zone (x[4,7]) may sit inside a warehouse wall, so we
# use a reachable lane pose as the physical "charging bay". The ORDER robots use
# it is the real supervisory decision; these coordinates are just where it is.
CHARGER_POSE = (0.0, -7.0)
PARK_POSES = {"R1": (-3.0, -8.0), "R2": (-3.0, -8.0), "R3": (3.0, -8.0)}


def decide_order(scenario: str, conn) -> tuple[list[tuple[str, str]], list[dict]]:
    """Run the REAL ChargingService once; return [(robot, role), ...] in the
    order the supervisory queue grants the single charger."""
    chargers = [{"charger_id": "CH1", "zone_id": "charging_bay", "is_online": True}]
    cs = ChargingService(connect, chargers=chargers)

    if scenario == "charging":
        snap = [{"robot_id": "R1", "battery_pct": 8.0, "allocation_state": "IDLE"},
                {"robot_id": "R2", "battery_pct": 25.0, "allocation_state": "IDLE"}]
        roles = {"R1": "critical", "R2": "low"}
    else:  # priority — two same-tier robots contend for one charger
        snap = [{"robot_id": "R2", "battery_pct": 12.0, "allocation_state": "IDLE"},
                {"robot_id": "R3", "battery_pct": 12.0, "allocation_state": "IDLE"}]
        roles = {"R2": "first", "R3": "waits"}

    for r in snap:
        Q.upsert_robot(conn, {"robot_id": r["robot_id"], "battery_pct": r["battery_pct"],
                              "allocation_state": "IDLE", "mode": "IDLE"})
    conn.commit()

    cs.tick(snap, conn)
    conn.commit()

    on_charger = list(cs._charging.keys())
    queued = [e.robot_id for e in sorted(cs._queue)]
    order = on_charger + queued
    log.info("[charging-svc] granted=%s queued=%s -> serve order=%s",
             on_charger, queued, order)
    return [(r, roles.get(r, "?")) for r in order], snap


def run_fleet_llm(snap: list[dict]) -> None:
    """Best-effort: exercise the real LLM fleet loop so the brain is in the loop.
    The serve order above is already the real supervisory decision; this only
    adds (and logs) the fleet-state reasoning on top."""
    from datetime import datetime, timezone
    from mars.llm.client import get_llm_client
    hot = demopkg._make_hot_state()
    emb = demopkg._make_embedder()
    llm = get_llm_client("anthropic")
    comp = demopkg.build_components(connect, hot, llm, emb)
    c = connect()
    comp["charging_service"].tick(snap, c, sim_time=datetime.now(timezone.utc))
    c.commit(); c.close()
    c = connect()
    out = comp["fleet_monitor"].run_once(c)
    c.commit(); c.close()
    if out:
        log.info("[fleet-llm] health=%s charging_pressure=%s",
                 out.get("fleet_health"), out.get("charging_pressure"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["charging", "priority"], required=True)
    ap.add_argument("--dwell", type=float, default=8.0,
                    help="seconds the robot 'charges' at the bay before leaving")
    a = ap.parse_args()

    conn = connect()
    apply_migrations(conn)
    demopkg.seed_world(conn)
    order, snap = decide_order(a.scenario, conn)

    try:
        run_fleet_llm(snap)
    except Exception:
        log.exception("fleet LLM loop failed (non-fatal — serve order already decided)")

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from nav2_msgs.action import NavigateToPose
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = Node("isaac_charging_bridge")
    clients = {r: ActionClient(node, NavigateToPose, f"/{r}/navigate_to_pose")
               for r in ("R1", "R2", "R3")}

    def goal_msg(x: float, y: float):
        p = PoseStamped()
        p.header.frame_id = "map"
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.orientation.w = 1.0
        g = NavigateToPose.Goal()
        g.pose = p
        return g

    def drive(robot: str, x: float, y: float, timeout: float = 70.0) -> bool:
        cl = clients[robot]
        if not cl.wait_for_server(timeout_sec=20.0):
            log.warning("  %s: nav action server not available", robot)
        log.info("  -> %s navigating to (%.1f, %.1f)", robot, x, y)
        fut = cl.send_goal_async(goal_msg(x, y))
        rclpy.spin_until_future_complete(node, fut, timeout_sec=20.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            log.warning("  %s: goal not accepted", robot)
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(node, rf, timeout_sec=timeout)
        log.info("  %s reached / finished", robot)
        return True

    # serialize the single charger: each robot drives in, dwells (charging), then
    # parks away so the next robot in the real serve order can take it.
    for robot, role in order:
        log.info("charger granted to %s (%s)", robot, role)
        drive(robot, *CHARGER_POSE, timeout=70.0)
        log.info("  %s charging at the bay (%.0fs)...", robot, a.dwell)
        time.sleep(a.dwell)
        px, py = PARK_POSES.get(robot, (3.0, -8.0))
        drive(robot, px, py, timeout=70.0)   # leave so the next robot can charge

    log.info("charging arbitration demo done (scenario=%s)", a.scenario)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
