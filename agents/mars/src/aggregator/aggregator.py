"""
Aggregator — §1a

Subscribes to sensor topics, attaches a health snapshot + distribution counts
to every failure event, and emits enriched failure events to the Orchestrator.

Two axes determined here (never by the agents):
  - distribution  → per_robot_zone_spread / per_zone_robot_spread
  - fault_flag    → battery_critical | estop | diagnostics_error | None

Invariant: the agent receives raw mission_failures; it NEVER gets the
precomputed distribution scalars that go to the Router.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from mars.config import BATTERY_CRITICAL_PCT, FAILURE_WINDOW_SECONDS
from mars.ros.interfaces import BatteryState, NavGoalStatus, Pose, RobotHealth

log = logging.getLogger(__name__)


class Aggregator:
    """
    Stateful enrichment layer.

    Call on_battery_update / on_health_update / on_nav_status as callbacks
    from the SensorInterface, then call on_nav_status with an aborted status
    to produce an enriched failure event.

    on_failure_event is called with the enriched dict for every resolved abort.
    """

    def __init__(
        self,
        hot_state,                    # HotState instance
        on_failure_event: Callable[[dict], None],
    ):
        self._hot = hot_state
        self._on_failure = on_failure_event
        # last known health per robot
        self._battery: dict[str, BatteryState] = {}
        self._health:  dict[str, RobotHealth]  = {}
        self._zone_map: dict[str, str] = {}   # robot_id → current zone

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def on_battery_update(self, msg: BatteryState) -> None:
        self._battery[msg.robot_id] = msg
        self._hot.update_robot(msg.robot_id, {"battery_pct": msg.percentage * 100})

    def on_health_update(self, msg: RobotHealth) -> None:
        self._health[msg.robot_id] = msg
        self._hot.update_robot(
            msg.robot_id,
            {
                "health_level": msg.level,
                "estop_active": msg.estop_active,
                "fault_codes": msg.fault_codes,
            },
        )

    def on_pose_update(self, robot_id: str, pose: Pose, zone: str) -> None:
        self._zone_map[robot_id] = zone
        self._hot.update_robot(robot_id, {"current_zone": zone})

    def on_nav_status(self, msg: NavGoalStatus, mission_id: str | None) -> None:
        if msg.status not in (5, 6):  # only CANCELED or ABORTED
            return
        nav_outcome = "canceled" if msg.status == 5 else "aborted"
        self._emit_enriched_event(msg, mission_id, nav_outcome)

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _emit_enriched_event(
        self, msg: NavGoalStatus, mission_id: str | None, nav_outcome: str
    ) -> None:
        robot_id = msg.robot_id
        zone = self._zone_map.get(robot_id, "unknown")

        # --- Health snapshot ---
        battery = self._battery.get(robot_id)
        health  = self._health.get(robot_id)
        battery_pct = (battery.percentage * 100) if battery else 100.0
        estop   = health.estop_active if health else False
        fault_codes = health.fault_codes if health else []
        health_level = health.level if health else 0

        # --- fault_flag (deterministic) ---
        fault_flag = None
        if battery_pct < BATTERY_CRITICAL_PCT:
            fault_flag = "battery_critical"
        elif estop:
            fault_flag = "estop"
        elif health_level >= 2:
            fault_flag = "diagnostics_error"

        # --- Distribution counts from hot state ---
        self._hot.record_failure_for_distribution(
            robot_id, zone, FAILURE_WINDOW_SECONDS
        )
        per_robot_zone_spread = self._hot.get_robot_zone_spread(robot_id)
        per_zone_robot_spread = self._hot.get_zone_robot_spread(zone)

        # --- Mission failure count ---
        failures_for_this_mission = 0
        if mission_id:
            failures_for_this_mission = self._hot.increment_mission_failures(
                mission_id, FAILURE_WINDOW_SECONDS
            )

        event = {
            "event_type": "navigation.aborted",
            "robot_id": robot_id,
            "mission_id": mission_id,
            "goal_id": msg.goal_id,
            "zone": zone,
            "nav_outcome": nav_outcome,
            "goal_status": msg.status,
            "health_at_failure": {
                "battery_pct": battery_pct,
                "estop_active": estop,
                "fault_codes": fault_codes,
            },
            "fault_flag": fault_flag,
            "distribution": {
                "per_robot_zone_spread": per_robot_zone_spread,
                "per_zone_robot_spread": per_zone_robot_spread,
            },
            "failures_for_this_mission": failures_for_this_mission,
            "occurred_at": time.time(),
        }

        log.info(
            "[aggregator] enriched event robot=%s zone=%s fault_flag=%s spread=%d",
            robot_id, zone, fault_flag, per_zone_robot_spread,
        )
        self._on_failure(event)
