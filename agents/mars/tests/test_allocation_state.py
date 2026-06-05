"""
Unit tests for allocation state machine (§6a)

Ensures that compare-and-set prevents double-claim under concurrent
Scheduler and Charging Service access.

Uses an in-memory dict instead of a real DB so no Postgres is needed.
"""
from __future__ import annotations

import threading
import pytest


class _FakeConn:
    """Minimal connection stub for testing the CAS function."""

    def __init__(self, robots: dict):
        self._robots = robots

    def cursor(self):
        return _FakeCursor(self._robots)

    def commit(self):
        pass


class _FakeCursor:
    def __init__(self, robots):
        self._robots = robots
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params=None):
        if "UPDATE robots" in sql and "allocation_state" in sql:
            # params: (new_state, robot_id, expected)
            new_state, robot_id, expected = params
            robot = self._robots.get(robot_id)
            if robot and robot["allocation_state"] == expected:
                robot["allocation_state"] = new_state
                self.rowcount = 1
            else:
                self.rowcount = 0


def _make_fake_conn(robots):
    return _FakeConn(robots)


class TestAllocationStateCAS:
    def test_scheduler_claims_idle_robot(self):
        from mars.blackboard.queries import cas_allocation_state

        robots = {"R1": {"allocation_state": "IDLE"}}
        conn = _FakeConn(robots)
        result = cas_allocation_state(conn, "R1", "IDLE", "RESERVED")
        assert result is True
        assert robots["R1"]["allocation_state"] == "RESERVED"

    def test_charging_cannot_claim_reserved_robot(self):
        from mars.blackboard.queries import cas_allocation_state

        robots = {"R1": {"allocation_state": "RESERVED"}}
        conn = _FakeConn(robots)
        result = cas_allocation_state(conn, "R1", "IDLE", "CHARGING")
        assert result is False
        assert robots["R1"]["allocation_state"] == "RESERVED"

    def test_only_one_thread_wins_concurrent_cas(self):
        """Simulate two concurrent CAS attempts — only one should succeed."""
        from mars.blackboard.queries import cas_allocation_state

        robots = {"R1": {"allocation_state": "IDLE"}}
        results = []
        lock = threading.Lock()

        def try_claim(label):
            conn = _FakeConn(robots)
            with lock:  # serialize for in-memory dict safety
                ok = cas_allocation_state(conn, "R1", "IDLE", "RESERVED")
            results.append((label, ok))

        t1 = threading.Thread(target=try_claim, args=("scheduler",))
        t2 = threading.Thread(target=try_claim, args=("charging",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        successes = [r for _, r in results if r]
        assert len(successes) == 1, f"Expected exactly 1 success, got: {results}"

    def test_idle_transition_sequence(self):
        from mars.blackboard.queries import cas_allocation_state

        robots = {"R1": {"allocation_state": "IDLE"}}
        conn = _FakeConn(robots)

        assert cas_allocation_state(conn, "R1", "IDLE", "RESERVED")
        assert cas_allocation_state(conn, "R1", "RESERVED", "BUSY")
        assert cas_allocation_state(conn, "R1", "BUSY", "IDLE")
        assert robots["R1"]["allocation_state"] == "IDLE"
