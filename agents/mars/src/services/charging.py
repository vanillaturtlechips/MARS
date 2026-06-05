"""
Charging Service — §6b

Deterministic executor.  The charging pressure → policy LOOP is what matters
for this project; everything below is the machinery that loop needs.

Battery tiers
─────────────
  OK        eligible for any mission
  LOW       finish current mission, then charge; ineligible for new LONG missions
  CRITICAL  go to charger now (interrupt if necessary)

Hysteresis  (prevents ping-pong)
──────────────────────────────────
  START_CHARGING   = BATTERY_LOW_PCT          (30%) — summon when below this
  OK_TO_DISPATCH   = BATTERY_LOW_PCT + HYSTERESIS_GAP_PCT (35%) — eligible again

Priority queue
──────────────
  CRITICAL (priority 0) → LOW (priority 1) → OPPORTUNISTIC (priority 2)
  reserve_chargers_for_critical keeps N chargers free for priority-0 robots.

Target charge level
───────────────────
  Default: BATTERY_TARGET_CHARGE_PCT (80%).  Under pressure a charging policy
  can lower it (lower_target_charge_level).  Idle robots near free chargers
  top off to 100% only if nobody is queued (opportunistic).

Shared allocation state
───────────────────────
  compare-and-set IDLE→CHARGING; Scheduler cannot claim a CHARGING robot.
"""
from __future__ import annotations

import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mars.config import (
    BATTERY_CRITICAL_PCT,
    BATTERY_LOW_PCT,
    BATTERY_MIN_DISPATCH_PCT,
    BATTERY_TARGET_CHARGE_PCT,
    BATTERY_TOP_UP_PCT,
)

log = logging.getLogger(__name__)

HYSTERESIS_GAP_PCT: float = 5.0   # gap between start-charging and ok-to-dispatch
OK_TO_DISPATCH_PCT: float = BATTERY_LOW_PCT + HYSTERESIS_GAP_PCT

# Priority constants (lower = higher urgency)
_PRI_CRITICAL     = 0
_PRI_LOW          = 1
_PRI_OPPORTUNISTIC = 2

# How many chargers to keep free for critical robots when
# reserve_chargers_for_critical policy is active.
_DEFAULT_RESERVE = 1


@dataclass(order=True)
class _QueueEntry:
    priority: int
    robot_id: str = field(compare=False)
    requested_at: float = field(compare=False, default_factory=time.time)


class ChargingService:
    def __init__(self, conn_factory, chargers: list[dict[str, Any]]):
        """
        chargers: list of {charger_id, zone_id, is_online}
        """
        self._conn = conn_factory
        self._chargers: dict[str, dict] = {c["charger_id"]: dict(c) for c in chargers}
        self._target_pct: float = BATTERY_TARGET_CHARGE_PCT
        self._reserved_for_critical: int = 0   # set by policy

        # Priority queue of _QueueEntry
        self._queue: list[_QueueEntry] = []
        # Robots currently charging: robot_id → {charger_id, session_id, started_at, target_pct}
        self._charging: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Policy consumer
    # ------------------------------------------------------------------

    def on_policy_change(self, event: str, policy: dict) -> None:
        ptype = policy.get("type")
        if ptype == "lower_target_charge_level":
            if event == "activated":
                self._target_pct = float(policy.get("params", {}).get("target_pct", 70.0))
                log.info("[charging] target_charge_level → %.0f%%", self._target_pct)
            else:
                self._target_pct = BATTERY_TARGET_CHARGE_PCT
                log.info("[charging] target_charge_level restored → %.0f%%", self._target_pct)

        elif ptype == "reserve_chargers_for_critical":
            if event == "activated":
                self._reserved_for_critical = int(policy.get("params", {}).get("reserve_count", _DEFAULT_RESERVE))
                log.info("[charging] reserving %d charger(s) for critical robots", self._reserved_for_critical)
            else:
                self._reserved_for_critical = 0

    # ------------------------------------------------------------------
    # Per-robot tier
    # ------------------------------------------------------------------

    @staticmethod
    def battery_tier(battery_pct: float) -> str:
        if battery_pct <= BATTERY_CRITICAL_PCT:
            return "critical"
        if battery_pct <= BATTERY_LOW_PCT:
            return "low"
        return "ok"

    def is_eligible_for_dispatch(self, battery_pct: float) -> bool:
        """Hysteresis gate: must be above dispatch level before scheduling can claim."""
        return battery_pct >= OK_TO_DISPATCH_PCT

    # ------------------------------------------------------------------
    # Tick — called periodically
    # ------------------------------------------------------------------

    def tick(
        self,
        robots: list[dict[str, Any]],
        conn,
        sim_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Main periodic driver.

        robots: list of dicts with robot_id, battery_pct, allocation_state.
        Returns pressure metrics dict (also written to DB).
        """
        t = sim_time or datetime.now(timezone.utc)

        # --- Simulate charging progress (increment battery for charging robots) ---
        self._advance_charging(robots, conn)

        # --- Triage: robots that need to charge ---
        below_low      = 0
        below_critical = 0
        for robot in robots:
            pct   = float(robot.get("battery_pct", 100.0))
            rid   = robot["robot_id"]
            alloc = robot.get("allocation_state", "IDLE")
            tier  = self.battery_tier(pct)

            if tier == "critical":
                below_critical += 1
                if alloc != "CHARGING" and rid not in self._charging:
                    self._enqueue(rid, _PRI_CRITICAL)

            elif tier == "low":
                below_low += 1
                if alloc == "IDLE" and rid not in self._charging:
                    self._enqueue(rid, _PRI_LOW)

            elif alloc == "IDLE" and pct < BATTERY_TOP_UP_PCT and rid not in self._charging:
                # Opportunistic: idle + below top-up + charger free
                if self._free_charger_count() > self._reserved_for_critical:
                    self._enqueue(rid, _PRI_OPPORTUNISTIC)

        # --- Drain the queue into free chargers ---
        self._dispatch_queue(conn, t)

        # --- Emit pressure metrics ---
        occupied = len(self._charging)
        total    = len(self._chargers)
        occ_pct  = occupied / max(total, 1)
        metrics  = {
            "queue_length":        len(self._queue),
            "mean_wait_sec":       self._mean_wait_sec(),
            "p95_wait_sec":        None,
            "occupied_pct":        occ_pct,
            "below_low_count":     below_low,
            "below_critical_count": below_critical,
            "recorded_at":         t,
        }
        self._write_metrics(conn, metrics, t)
        return metrics

    # ------------------------------------------------------------------
    # Public: check if a robot can be summoned immediately
    # ------------------------------------------------------------------

    def summon_if_needed(self, robot_id: str, conn) -> bool:
        """
        Imperatively summon a robot to a charger if its battery is low.
        Returns True if the robot was claimed for charging.
        """
        from mars.blackboard.queries import cas_allocation_state, get_robot
        robot = get_robot(conn, robot_id)
        if not robot:
            return False
        pct = float(robot.get("battery_pct", 100.0))
        tier = self.battery_tier(pct)
        if tier == "ok":
            return False
        priority = _PRI_CRITICAL if tier == "critical" else _PRI_LOW
        charger_id = self._find_free_charger(priority)
        if not charger_id:
            self._enqueue(robot_id, priority)
            return False
        if cas_allocation_state(conn, robot_id, "IDLE", "CHARGING"):
            self._start_charging(robot_id, charger_id, pct, conn, datetime.now(timezone.utc))
            conn.commit()
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, robot_id: str, priority: int) -> None:
        # Avoid duplicates
        if any(e.robot_id == robot_id for e in self._queue):
            return
        heapq.heappush(self._queue, _QueueEntry(priority=priority, robot_id=robot_id))
        log.debug("[charging] queued robot=%s priority=%d", robot_id, priority)

    def _dispatch_queue(self, conn, sim_time: datetime) -> None:
        """Send queued robots to free chargers."""
        from mars.blackboard.queries import cas_allocation_state

        while self._queue:
            entry = self._queue[0]   # peek
            priority = entry.priority

            charger_id = self._find_free_charger(priority)
            if not charger_id:
                break   # no charger available at this priority level

            heapq.heappop(self._queue)   # consume
            robot_id = entry.robot_id

            if cas_allocation_state(conn, robot_id, "IDLE", "CHARGING"):
                self._start_charging(robot_id, charger_id, None, conn, sim_time)
                conn.commit()
            else:
                log.debug("[charging] CAS failed for robot=%s — skipping", robot_id)

    def _start_charging(
        self,
        robot_id: str,
        charger_id: str,
        start_battery_pct: float | None,
        conn,
        sim_time: datetime,
    ) -> str:
        session_id = f"CS{uuid.uuid4().hex[:10]}"
        self._chargers[charger_id]["is_occupied"]    = True
        self._chargers[charger_id]["current_robot_id"] = robot_id
        self._charging[robot_id] = {
            "charger_id": charger_id,
            "session_id": session_id,
            "started_at": sim_time,
            "start_battery_pct": start_battery_pct,
            "target_pct": self._target_pct,
        }
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO charging_sessions
                  (session_id, robot_id, charger_id, started_at, start_battery_pct, target_pct)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (session_id, robot_id, charger_id, sim_time, start_battery_pct, self._target_pct),
            )
        log.info("[charging] started  robot=%s charger=%s target=%.0f%%",
                 robot_id, charger_id, self._target_pct)
        return session_id

    def _advance_charging(self, robots: list[dict], conn) -> None:
        """
        Simulate charging progress.  In a real system the robot reports its
        battery level; here we increment by a fixed rate per tick.
        Robots that reach target_pct are released.
        """
        from mars.blackboard.queries import cas_allocation_state, upsert_robot
        CHARGE_RATE_PER_TICK = 5.0  # % per tick (tunable)

        for robot in robots:
            rid = robot["robot_id"]
            if rid not in self._charging:
                continue
            session = self._charging[rid]
            target  = session["target_pct"]
            pct     = float(robot.get("battery_pct", 0.0))
            new_pct = min(pct + CHARGE_RATE_PER_TICK, 100.0)
            robot["battery_pct"] = new_pct  # update in-place so emit_pressure sees it

            if new_pct >= target:
                # Done charging — release
                charger_id = session["charger_id"]
                self._chargers[charger_id]["is_occupied"]    = False
                self._chargers[charger_id]["current_robot_id"] = None
                del self._charging[rid]
                cas_allocation_state(conn, rid, "CHARGING", "IDLE")
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE charging_sessions
                        SET ended_at = NOW(), end_battery_pct = %s
                        WHERE session_id = %s
                        """,
                        (new_pct, session["session_id"]),
                    )
                log.info("[charging] complete robot=%s  %.0f%% → %.0f%%", rid, pct, new_pct)

    def _find_free_charger(self, priority: int) -> str | None:
        """
        Find a free charger, respecting the critical reserve.
        CRITICAL robots (priority=0) may use reserved chargers.
        """
        free = [
            cid for cid, c in self._chargers.items()
            if c.get("is_online") and not c.get("is_occupied")
        ]
        if not free:
            return None
        # For non-critical robots, leave _reserved_for_critical chargers free
        if priority > _PRI_CRITICAL:
            available_count = len(free) - self._reserved_for_critical
            if available_count <= 0:
                return None
        return free[0]

    def _free_charger_count(self) -> int:
        return sum(
            1 for c in self._chargers.values()
            if c.get("is_online") and not c.get("is_occupied")
        )

    def _mean_wait_sec(self) -> float | None:
        if not self._queue:
            return None
        now = time.time()
        waits = [now - e.requested_at for e in self._queue]
        return sum(waits) / len(waits)

    def _write_metrics(self, conn, metrics: dict, sim_time: datetime) -> None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO charging_pressure_metrics
                      (recorded_at, queue_length, mean_wait_sec, p95_wait_sec,
                       occupied_pct, below_low_count, below_critical_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sim_time,
                        metrics["queue_length"],
                        metrics["mean_wait_sec"],
                        metrics["p95_wait_sec"],
                        metrics["occupied_pct"],
                        metrics["below_low_count"],
                        metrics["below_critical_count"],
                    ),
                )
            conn.commit()
        except Exception:
            log.exception("[charging] failed to write pressure metrics")

    # ------------------------------------------------------------------
    # State accessors (for tests + fleet monitor)
    # ------------------------------------------------------------------

    def queue_length(self) -> int:
        return len(self._queue)

    def charging_count(self) -> int:
        return len(self._charging)

    def get_pressure_summary(self) -> dict[str, Any]:
        return {
            "queue_length": self.queue_length(),
            "occupied_pct": len(self._charging) / max(len(self._chargers), 1),
        }
