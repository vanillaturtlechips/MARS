"""
M2 tests — ChargingService

Tests:
  - battery_tier() for all three tiers
  - Hysteresis: robot below LOW is summoned; below dispatch threshold blocks dispatch
  - Priority queue: CRITICAL enqueued before LOW
  - reserve_chargers_for_critical: non-critical robots blocked when all chargers reserved
  - on_policy_change: lower_target_charge_level and reserve_chargers_for_critical
  - _advance_charging: robot reaches target → released to IDLE
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from mars.services.charging import ChargingService, OK_TO_DISPATCH_PCT, _PRI_CRITICAL, _PRI_LOW
from mars.config import BATTERY_CRITICAL_PCT, BATTERY_LOW_PCT, BATTERY_TARGET_CHARGE_PCT


# ---------------------------------------------------------------------------
# Minimal fake connection (in-memory, no SQL)
# ---------------------------------------------------------------------------

class _FakeConn:
    def __init__(self, robots: dict):
        self._robots = robots
        self._allocation = {rid: r["allocation_state"] for rid, r in robots.items()}

    def cursor(self):
        return _FakeCursor(self._robots, self._allocation)

    def commit(self): pass
    def rollback(self): pass

    def cas(self, robot_id, expected, new_state) -> bool:
        if self._allocation.get(robot_id) == expected:
            self._allocation[robot_id] = new_state
            if robot_id in self._robots:
                self._robots[robot_id]["allocation_state"] = new_state
            return True
        return False


class _FakeCursor:
    def __init__(self, robots, allocation):
        self._robots = robots
        self._allocation = allocation
        self.rowcount = 0

    def __enter__(self): return self
    def __exit__(self, *_): pass

    def execute(self, sql, params=None):
        # Handle allocation CAS
        if "UPDATE robots" in sql and "allocation_state" in sql:
            new_state, robot_id, expected = params
            if self._allocation.get(robot_id) == expected:
                self._allocation[robot_id] = new_state
                if robot_id in self._robots:
                    self._robots[robot_id]["allocation_state"] = new_state
                self.rowcount = 1
            else:
                self.rowcount = 0
        # Ignore all other SQL (INSERT INTO charging_sessions, UPDATE, etc.)


def _make_chargers(n: int = 2) -> list[dict]:
    return [{"charger_id": f"CH{i}", "zone_id": "charging_bay", "is_online": True}
            for i in range(n)]


def _make_robots(specs: list[tuple]) -> dict:
    """specs: [(robot_id, battery_pct, allocation_state)]"""
    return {
        rid: {"robot_id": rid, "battery_pct": pct, "allocation_state": alloc}
        for rid, pct, alloc in specs
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatteryTier:
    def test_critical(self):
        assert ChargingService.battery_tier(BATTERY_CRITICAL_PCT - 1) == "critical"

    def test_critical_boundary(self):
        assert ChargingService.battery_tier(BATTERY_CRITICAL_PCT) == "critical"

    def test_low(self):
        assert ChargingService.battery_tier(BATTERY_CRITICAL_PCT + 1) == "low"

    def test_low_boundary(self):
        assert ChargingService.battery_tier(BATTERY_LOW_PCT) == "low"

    def test_ok_above_low(self):
        assert ChargingService.battery_tier(BATTERY_LOW_PCT + 1) == "ok"

    def test_ok_full(self):
        assert ChargingService.battery_tier(100.0) == "ok"


class TestHysteresis:
    def test_robot_below_low_is_not_eligible_for_dispatch(self):
        svc = ChargingService(lambda: None, _make_chargers())
        assert not svc.is_eligible_for_dispatch(BATTERY_LOW_PCT)

    def test_robot_above_dispatch_level_is_eligible(self):
        svc = ChargingService(lambda: None, _make_chargers())
        assert svc.is_eligible_for_dispatch(OK_TO_DISPATCH_PCT)

    def test_hysteresis_gap_exists(self):
        # There is a gap between start-charging (LOW) and ok-to-dispatch
        assert OK_TO_DISPATCH_PCT > BATTERY_LOW_PCT


class TestPriorityQueue:
    def test_critical_queued_before_low(self):
        svc = ChargingService(lambda: None, [])  # no chargers → everything queues
        svc._enqueue("R_low",      _PRI_LOW)
        svc._enqueue("R_critical", _PRI_CRITICAL)
        # Heap invariant: lowest priority value (most urgent) at top
        assert svc._queue[0].robot_id == "R_critical"

    def test_no_duplicate_in_queue(self):
        svc = ChargingService(lambda: None, [])
        svc._enqueue("R1", _PRI_LOW)
        svc._enqueue("R1", _PRI_LOW)  # duplicate
        assert svc.queue_length() == 1

    def test_queue_drains_to_charger(self):
        robots = _make_robots([("R1", 20.0, "IDLE")])
        conn   = _FakeConn(robots)

        svc = ChargingService(lambda: conn, _make_chargers(1))
        svc._enqueue("R1", _PRI_LOW)
        svc._dispatch_queue(conn, datetime.now(timezone.utc))

        # Robot should now be CHARGING
        assert robots["R1"]["allocation_state"] == "CHARGING"
        assert svc.charging_count() == 1
        assert svc.queue_length() == 0


class TestReserveForCritical:
    def test_reserve_blocks_low_tier_robots(self):
        svc = ChargingService(lambda: None, _make_chargers(1))
        svc._reserved_for_critical = 1  # all chargers reserved

        # Low-priority robot cannot get a charger
        charger = svc._find_free_charger(_PRI_LOW)
        assert charger is None

    def test_critical_robot_can_use_reserved_charger(self):
        svc = ChargingService(lambda: None, _make_chargers(1))
        svc._reserved_for_critical = 1

        charger = svc._find_free_charger(_PRI_CRITICAL)
        assert charger == "CH0"

    def test_policy_sets_reservation(self):
        svc = ChargingService(lambda: None, _make_chargers(2))
        svc.on_policy_change("activated", {
            "type": "reserve_chargers_for_critical",
            "params": {"reserve_count": 1},
        })
        assert svc._reserved_for_critical == 1

    def test_policy_deactivation_clears_reservation(self):
        svc = ChargingService(lambda: None, _make_chargers(2))
        svc.on_policy_change("activated", {
            "type": "reserve_chargers_for_critical",
            "params": {"reserve_count": 1},
        })
        svc.on_policy_change("deactivated", {
            "type": "reserve_chargers_for_critical",
        })
        assert svc._reserved_for_critical == 0


class TestTargetChargeLevel:
    def test_policy_lowers_target(self):
        svc = ChargingService(lambda: None, _make_chargers())
        svc.on_policy_change("activated", {
            "type": "lower_target_charge_level",
            "params": {"target_pct": 70.0},
        })
        assert svc._target_pct == 70.0

    def test_policy_deactivation_restores_target(self):
        svc = ChargingService(lambda: None, _make_chargers())
        svc.on_policy_change("activated", {
            "type": "lower_target_charge_level",
            "params": {"target_pct": 70.0},
        })
        svc.on_policy_change("deactivated", {"type": "lower_target_charge_level"})
        assert svc._target_pct == BATTERY_TARGET_CHARGE_PCT


class TestChargeAdvancement:
    def test_robot_reaches_target_and_released(self):
        robots = _make_robots([("R1", 75.0, "CHARGING")])
        conn   = _FakeConn(robots)

        svc = ChargingService(lambda: conn, _make_chargers(1))
        # Manually put R1 into the charging map
        svc._chargers["CH0"]["is_occupied"]     = True
        svc._chargers["CH0"]["current_robot_id"] = "R1"
        svc._charging["R1"] = {
            "charger_id": "CH0",
            "session_id": "SES001",
            "started_at": datetime.now(timezone.utc),
            "start_battery_pct": 75.0,
            "target_pct": 80.0,
        }
        robots["R1"]["allocation_state"] = "CHARGING"

        svc._advance_charging(list(robots.values()), conn)

        # 75 + 5 = 80 → target reached → should be released
        assert robots["R1"]["allocation_state"] == "IDLE"
        assert "R1" not in svc._charging
        assert not svc._chargers["CH0"]["is_occupied"]

    def test_robot_below_target_stays_charging(self):
        robots = _make_robots([("R1", 60.0, "CHARGING")])
        conn   = _FakeConn(robots)

        svc = ChargingService(lambda: conn, _make_chargers(1))
        svc._chargers["CH0"]["is_occupied"]      = True
        svc._chargers["CH0"]["current_robot_id"] = "R1"
        svc._charging["R1"] = {
            "charger_id":    "CH0",
            "session_id":    "SES002",
            "started_at":    datetime.now(timezone.utc),
            "start_battery_pct": 60.0,
            "target_pct":    80.0,
        }

        svc._advance_charging(list(robots.values()), conn)

        # 60 + 5 = 65 < 80 → still charging
        assert robots["R1"]["allocation_state"] == "CHARGING"
        assert "R1" in svc._charging
