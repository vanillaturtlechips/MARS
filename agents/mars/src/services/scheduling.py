"""
Scheduling Service — §6a stub

Heuristic: priority → nearest idle robot → flat battery threshold.
No energy math, no ETA optimization.

Invariant: compare-and-set on allocation_state (IDLE→RESERVED) prevents
double-claim with the Charging Service.
"""
from __future__ import annotations

import logging

from mars.config import BATTERY_MIN_DISPATCH_PCT
from mars.blackboard.queries import (
    cas_allocation_state,
    get_idle_robots,
    get_pending_missions,
)

log = logging.getLogger(__name__)


class SchedulingService:
    def __init__(self, conn_factory, avoid_zones: set[str] | None = None):
        self._conn = conn_factory
        self._avoid_zones: set[str] = avoid_zones or set()

    def on_policy_change(self, event: str, policy: dict) -> None:
        """Called by PolicyManager when a scheduling-relevant policy changes."""
        if policy.get("type") == "avoid_zone":
            zone = policy.get("params", {}).get("zone")
            if zone:
                if event == "activated":
                    self._avoid_zones.add(zone)
                    log.info("[scheduler] added avoid_zone: %s", zone)
                else:
                    self._avoid_zones.discard(zone)
                    log.info("[scheduler] removed avoid_zone: %s", zone)
        self.run_sweep()

    def run_sweep(self) -> int:
        """Assign pending missions to idle robots. Returns number of assignments."""
        conn = self._conn()
        assigned = 0

        pending = get_pending_missions(conn)
        idle = get_idle_robots(conn)

        available = [
            r for r in idle
            if r["battery_pct"] >= BATTERY_MIN_DISPATCH_PCT
        ]

        for mission in pending:
            if not available:
                break

            # Policy filter: skip missions routing through avoided zones
            if mission.get("zone") in self._avoid_zones:
                log.debug("[scheduler] skipping mission %s — zone %s avoided",
                          mission["mission_id"], mission["zone"])
                continue

            # Pick first available robot (nearest-idle heuristic placeholder)
            robot = available.pop(0)

            # Compare-and-set allocation_state IDLE → RESERVED
            if cas_allocation_state(conn, robot["robot_id"], "IDLE", "RESERVED"):
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE missions SET state='ASSIGNED', robot_id=%s
                        WHERE mission_id=%s AND state='PENDING'
                        """,
                        (robot["robot_id"], mission["mission_id"]),
                    )
                conn.commit()
                assigned += 1
                log.info(
                    "[scheduler] assigned mission=%s → robot=%s",
                    mission["mission_id"], robot["robot_id"],
                )
            else:
                log.debug("[scheduler] CAS failed for robot %s — already claimed", robot["robot_id"])

        return assigned
