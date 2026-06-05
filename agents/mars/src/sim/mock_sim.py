"""
Mock robot sim — implements NavigationInterface and SensorInterface without
Isaac Sim or a GPU.

Each simulated robot:
  - Runs a NavigateToPose action server (or its mock equivalent).
  - Publishes BatteryState, RobotHealth, and Pose at configurable rates.
  - Can have faults injected via FaultInjector (see fault_injector.py).

This module deliberately avoids importing rclpy at module load.  When ROS2 is
present, the ROS2MockSimNode subclass wires these to real topics.  Without
ROS2 the MockSim runs as a standalone thread-based simulator — the Aggregator
or tests drive it directly.

Sim time advances at wall-clock speed (1:1) or faster via MOCK_SIM_SPEED_FACTOR.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from mars.ros.interfaces import (
    BatteryState,
    NavGoalStatus,
    NavigationInterface,
    Pose,
    RobotHealth,
    SensorInterface,
)

log = logging.getLogger(__name__)

# Nav2 GoalStatus constants
STATUS_EXECUTING  = 2
STATUS_SUCCEEDED  = 4
STATUS_CANCELED   = 5
STATUS_ABORTED    = 6


# ---------------------------------------------------------------------------
# Per-robot simulated state
# ---------------------------------------------------------------------------

@dataclass
class RobotSimState:
    robot_id: str
    pose: Pose = field(default_factory=Pose)
    battery_pct: float = 100.0        # 0–100
    health_level: int = 0             # 0=OK 1=WARN 2=ERROR
    estop_active: bool = False
    fault_codes: list[str] = field(default_factory=list)
    active_goal_id: str | None = None
    destination: Pose | None = None
    mission_duration_sec: float = 5.0   # simulated travel time per goal
    _goal_start: float = 0.0

    # Injected failure — set by FaultInjector
    inject_abort: bool = False
    inject_estop: bool = False
    inject_motor_fault: bool = False


# ---------------------------------------------------------------------------
# Core mock sim (no ROS dependency)
# ---------------------------------------------------------------------------

class MockSim(NavigationInterface, SensorInterface):
    """
    Thread-safe mock sim.  Call start() to begin the background tick loop.
    Call stop() to shut down.

    Callbacks are called from the background thread; downstream code must
    handle thread safety.
    """

    def __init__(self, robot_ids: list[str], tick_hz: float = 10.0):
        self._robots: dict[str, RobotSimState] = {
            rid: RobotSimState(robot_id=rid) for rid in robot_ids
        }
        self._tick_hz = tick_hz
        self._running = False
        self._thread: threading.Thread | None = None

        # callbacks keyed by robot_id
        self._battery_cbs:    dict[str, list[Callable]] = {}
        self._health_cbs:     dict[str, list[Callable]] = {}
        self._pose_cbs:       dict[str, list[Callable]] = {}
        self._nav_status_cbs: dict[str, list[Callable]] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # NavigationInterface
    # ------------------------------------------------------------------

    def send_goal(
        self,
        robot_id: str,
        goal_id: str,
        destination: Pose,
        on_status_change: Callable[[NavGoalStatus], None],
    ) -> None:
        with self._lock:
            robot = self._robots.get(robot_id)
            if robot is None:
                log.warning("send_goal: unknown robot %s", robot_id)
                return
            robot.active_goal_id = goal_id
            robot.destination = destination
            robot._goal_start = time.monotonic()

        # Register caller's callback under nav_status
        self._nav_status_cbs.setdefault(robot_id, []).append(on_status_change)
        log.info("[sim] %s → goal %s dispatched to %s", robot_id, goal_id, destination)

    def cancel_goal(self, robot_id: str, goal_id: str) -> None:
        with self._lock:
            robot = self._robots.get(robot_id)
            if robot and robot.active_goal_id == goal_id:
                robot.active_goal_id = None
                robot.destination = None
        status = NavGoalStatus(
            goal_id=goal_id, robot_id=robot_id,
            status=STATUS_CANCELED,
        )
        self._fire(self._nav_status_cbs.get(robot_id, []), status)
        log.info("[sim] %s goal %s canceled", robot_id, goal_id)

    # ------------------------------------------------------------------
    # SensorInterface
    # ------------------------------------------------------------------

    def subscribe_battery(self, robot_id, callback):
        self._battery_cbs.setdefault(robot_id, []).append(callback)

    def subscribe_health(self, robot_id, callback):
        self._health_cbs.setdefault(robot_id, []).append(callback)

    def subscribe_pose(self, robot_id, callback):
        self._pose_cbs.setdefault(robot_id, []).append(callback)

    def subscribe_nav_status(self, robot_id, callback):
        self._nav_status_cbs.setdefault(robot_id, []).append(callback)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("[sim] started with robots: %s", list(self._robots))

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("[sim] stopped")

    # ------------------------------------------------------------------
    # Internal tick loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        interval = 1.0 / self._tick_hz
        counter = 0
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                robots = list(self._robots.values())
            for robot in robots:
                self._tick_robot(robot, counter)
            counter += 1
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _tick_robot(self, robot: RobotSimState, counter: int) -> None:
        # Battery drain: ~0.05% per tick at 10 Hz  (18 min to empty)
        robot.battery_pct = max(0.0, robot.battery_pct - 0.05 / self._tick_hz)

        # Publish battery every tick
        bs = BatteryState(
            robot_id=robot.robot_id,
            percentage=robot.battery_pct / 100.0,
            power_supply_status=2,  # DISCHARGING
        )
        self._fire(self._battery_cbs.get(robot.robot_id, []), bs)

        # Publish health every 10 ticks (~1 Hz)
        if counter % 10 == 0:
            health_level = robot.health_level
            estop = robot.estop_active or robot.inject_estop
            codes = list(robot.fault_codes)
            if robot.inject_motor_fault:
                health_level = 2
                codes.append("MOTOR_FAULT")
            rh = RobotHealth(
                robot_id=robot.robot_id,
                level=health_level,
                estop_active=estop,
                fault_codes=codes,
            )
            self._fire(self._health_cbs.get(robot.robot_id, []), rh)

        # Publish pose every 10 ticks (~1 Hz)
        if counter % 10 == 0 and robot.destination:
            # Simple linear interpolation toward destination
            t_elapsed = time.monotonic() - robot._goal_start
            frac = min(1.0, t_elapsed / max(robot.mission_duration_sec, 0.1))
            robot.pose = Pose(
                x=robot.pose.x + (robot.destination.x - robot.pose.x) * 0.1,
                y=robot.pose.y + (robot.destination.y - robot.pose.y) * 0.1,
            )
            self._fire(self._pose_cbs.get(robot.robot_id, []), robot.pose)

        # Navigate: resolve goal if time elapsed or abort injected
        if robot.active_goal_id:
            t_elapsed = time.monotonic() - robot._goal_start
            goal_id = robot.active_goal_id

            if robot.inject_abort:
                robot.inject_abort = False
                robot.active_goal_id = None
                robot.destination = None
                status = NavGoalStatus(
                    goal_id=goal_id, robot_id=robot.robot_id,
                    status=STATUS_ABORTED,
                )
                self._fire(self._nav_status_cbs.get(robot.robot_id, []), status)
                log.info("[sim] %s goal %s ABORTED (injected)", robot.robot_id, goal_id)

            elif t_elapsed >= robot.mission_duration_sec:
                robot.active_goal_id = None
                robot.destination = None
                status = NavGoalStatus(
                    goal_id=goal_id, robot_id=robot.robot_id,
                    status=STATUS_SUCCEEDED,
                )
                self._fire(self._nav_status_cbs.get(robot.robot_id, []), status)
                log.info("[sim] %s goal %s SUCCEEDED", robot.robot_id, goal_id)

    @staticmethod
    def _fire(callbacks: list[Callable], msg) -> None:
        for cb in callbacks:
            try:
                cb(msg)
            except Exception:
                log.exception("Callback error")

    # ------------------------------------------------------------------
    # Direct state access (for tests)
    # ------------------------------------------------------------------

    def get_robot_state(self, robot_id: str) -> RobotSimState | None:
        return self._robots.get(robot_id)

    def set_robot_battery(self, robot_id: str, pct: float) -> None:
        with self._lock:
            if robot_id in self._robots:
                self._robots[robot_id].battery_pct = pct
