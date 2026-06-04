"""
Unit tests for Deterministic Router — §2
"""
from __future__ import annotations

import pytest

from mars.router.router import Path, route
from mars.config import FAST_PATH_BUDGET, ROUTER_SCOPE_HINT


def _event(failures_for_mission=0, zone_robot_spread=1):
    return {
        "robot_id": "R1",
        "zone": "Zone A",
        "fault_flag": None,
        "failures_for_this_mission": failures_for_mission,
        "distribution": {
            "per_robot_zone_spread": 1,
            "per_zone_robot_spread": zone_robot_spread,
        },
    }


class TestRouter:
    def test_first_failure_isolated_fast(self):
        assert route(_event(0, 1)) == Path.FAST

    def test_exceeds_fast_path_budget_slow(self):
        assert route(_event(FAST_PATH_BUDGET + 1, 1)) == Path.SLOW

    def test_zone_spread_at_hint_slow(self):
        assert route(_event(0, ROUTER_SCOPE_HINT)) == Path.SLOW

    def test_active_policy_on_zone_slow(self):
        assert route(_event(0, 1), active_policy_on_zone=True) == Path.SLOW

    def test_degraded_zone_slow(self):
        assert route(_event(0, 1), zone_in_degraded_set=True) == Path.SLOW

    def test_second_failure_still_fast_if_no_pattern(self):
        # failures_for_this_mission == 1 < FAST_PATH_BUDGET
        assert route(_event(1, 1)) == Path.FAST

    def test_spread_below_hint_fast(self):
        assert route(_event(0, ROUTER_SCOPE_HINT - 1)) == Path.FAST
