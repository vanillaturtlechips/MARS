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

# The charging station (real packing_table) sits at (0,3) in the open south area
# (inside the building, south of the shelves); robots dock in FRONT at (0,5).
# Park spots are distinct free cells in that open band. The ORDER robots take the
# one charger is the real supervisory decision; these are just where it is.
CHARGER_POSE = (0.0, 5.0)
# Park spots after charging — kept near the pad and in the camera's view (charging
# spawns are R1(0,2.5) R2(3,2) R3(-4,2)); a charged robot pulls aside so the next can dock.
PARK_POSES = {"R1": (-2.0, 2.0), "R2": (4.0, 3.0), "R3": (-4.0, 2.0)}


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

    # Drive via direct cmd_vel closed on /<r>/odom — NO Nav2. Three Nav2 stacks
    # deadlock this env's FastDDS discovery and randomly freeze a robot (e.g. R2);
    # the charging area is open floor so straight-line driving is enough. Same control
    # law as follow_waypoints.py. The serve ORDER above is still the real agent decision.
    import math
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist

    SPAWN = {"R1": (0.0, 2.5), "R2": (3.0, 2.0), "R3": (-4.0, 2.0)}   # = scene --charge spawns
    rclpy.init()
    node = Node("isaac_charging_bridge")
    st = {r: {"x": 0.0, "y": 0.0, "yaw": 0.0, "have": False} for r in ("R1", "R2", "R3")}
    pubs = {r: node.create_publisher(Twist, f"/{r}/cmd_vel", 10) for r in st}

    def _mk(r):
        def cb(m):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            st[r]["x"], st[r]["y"] = p.x, p.y
            st[r]["yaw"] = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                      1 - 2 * (q.y * q.y + q.z * q.z))
            st[r]["have"] = True
        return cb
    for r in st:
        node.create_subscription(Odometry, f"/{r}/odom", _mk(r), 10)

    def drive(robot: str, X: float, Y: float, timeout: float = 60.0) -> bool:
        sx, sy = SPAWN[robot]
        gx, gy = X - sx, Y - sy          # world target -> odom frame (zeroed at spawn, map-aligned)
        log.info("  -> %s driving to (%.1f, %.1f)", robot, X, Y)
        t0 = time.monotonic()
        pub = pubs[robot]
        while rclpy.ok() and time.monotonic() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            s = st[robot]
            if not s["have"]:
                continue
            dx, dy = gx - s["x"], gy - s["y"]
            if math.hypot(dx, dy) < 0.3:
                pub.publish(Twist())     # stop (diff drive latches last velocity)
                log.info("  %s reached (%.1f, %.1f)", robot, X, Y)
                return True
            err = math.atan2(math.sin(math.atan2(dy, dx) - s["yaw"]),
                             math.cos(math.atan2(dy, dx) - s["yaw"]))
            tw = Twist()
            tw.angular.z = max(-0.8, min(0.8, 1.5 * err))
            tw.linear.x = 0.4 * max(0.0, math.cos(err))
            pub.publish(tw)
        pub.publish(Twist())
        log.warning("  %s drive timed out", robot)
        return False

    # serialize the single charger: each robot drives in, dwells (charging), then
    # parks away so the next robot in the real serve order can take it.
    for robot, role in order:
        log.info("charger granted to %s (%s)", robot, role)
        drive(robot, *CHARGER_POSE)
        log.info("  %s charging at the bay (%.0fs)...", robot, a.dwell)
        time.sleep(a.dwell)
        px, py = PARK_POSES.get(robot, (4.0, 3.0))
        drive(robot, px, py)             # leave so the next robot can charge

    log.info("charging arbitration demo done (scenario=%s)", a.scenario)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
