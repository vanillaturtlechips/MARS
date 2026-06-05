"""
Fault injector — test driver for M0/M1 scenarios.

Provides a simple API to inject faults into the MockSim so the Aggregator and
the supervisory pipeline can be tested end-to-end without Isaac Sim.

Usage:
    sim = MockSim(["R1", "R2"])
    sim.start()
    fi = FaultInjector(sim)

    # Scenario: zone-wide congestion (4 robots abort in the same zone)
    fi.inject_zone_failure("Receiving Dock", robots=["R1","R2","R3","R4"])

    # Scenario: robot-internal fault
    fi.inject_hardware_fault("R1", fault_codes=["MOTOR_FAULT"])
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mars.sim.mock_sim import MockSim

log = logging.getLogger(__name__)


class FaultInjector:
    def __init__(self, sim: "MockSim"):
        self._sim = sim

    # ------------------------------------------------------------------
    # Individual fault injection
    # ------------------------------------------------------------------

    def abort_goal(self, robot_id: str) -> None:
        """Cause the robot's current active goal to abort on the next tick."""
        state = self._sim.get_robot_state(robot_id)
        if state is None:
            log.warning("abort_goal: unknown robot %s", robot_id)
            return
        state.inject_abort = True
        log.info("[inject] abort_goal → %s", robot_id)

    def inject_estop(self, robot_id: str, active: bool = True) -> None:
        state = self._sim.get_robot_state(robot_id)
        if state:
            state.inject_estop = active
            log.info("[inject] estop %s → %s", robot_id, active)

    def inject_hardware_fault(
        self, robot_id: str, fault_codes: list[str] | None = None
    ) -> None:
        """Set health level to ERROR and add fault codes."""
        state = self._sim.get_robot_state(robot_id)
        if state:
            state.health_level = 2  # ERROR
            state.inject_motor_fault = True
            if fault_codes:
                state.fault_codes = fault_codes
            log.info("[inject] hardware_fault → %s  codes=%s", robot_id, fault_codes)

    def clear_fault(self, robot_id: str) -> None:
        state = self._sim.get_robot_state(robot_id)
        if state:
            state.health_level = 0
            state.estop_active = False
            state.inject_estop = False
            state.inject_motor_fault = False
            state.inject_abort = False
            state.fault_codes = []
            log.info("[inject] cleared faults → %s", robot_id)

    def set_battery(self, robot_id: str, pct: float) -> None:
        self._sim.set_robot_battery(robot_id, pct)
        log.info("[inject] battery %s → %.1f%%", robot_id, pct)

    # ------------------------------------------------------------------
    # Canned scenarios
    # ------------------------------------------------------------------

    def inject_zone_failure(
        self,
        zone: str,
        robots: list[str],
        delay_between_sec: float = 0.5,
    ) -> None:
        """
        Abort the active goal for each listed robot in sequence, simulating
        a zone-wide congestion event that fires the slow path.
        """
        log.info("[inject] zone_failure scenario → zone=%s robots=%s", zone, robots)
        for rid in robots:
            self.abort_goal(rid)
            time.sleep(delay_between_sec)

    def inject_robot_internal_failure(
        self, robot_id: str, fault_codes: list[str] | None = None
    ) -> None:
        """
        Inject a hardware fault AND abort the current goal, simulating a
        robot-internal fault event that sets fault_flag.
        """
        self.inject_hardware_fault(robot_id, fault_codes or ["MOTOR_FAULT"])
        self.abort_goal(robot_id)
        log.info("[inject] robot_internal_failure → %s", robot_id)

    def inject_low_battery(self, robot_id: str, pct: float = 14.0) -> None:
        """Set battery below CRITICAL_THRESHOLD and abort the goal."""
        self.set_battery(robot_id, pct)
        self.abort_goal(robot_id)
        log.info("[inject] low_battery → %s  pct=%.1f", robot_id, pct)
