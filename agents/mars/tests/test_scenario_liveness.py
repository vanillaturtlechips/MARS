"""
Offline tests for the physical scenario-2 driver (deploy/nav2/scenario_liveness.py).

drive_to_charger had never run: it imported a function that does not exist,
called send_goal with the wrong signature, polled a status method the adapter
does not have, and aimed at a charger both arms put under keepout. These tests
exercise everything in it that is not ROS — which charger is targeted, what mask
is published, which Pose is sent, how outcomes are recorded — against a fake
adapter that behaves like Nav2 + KeepoutFilter: a goal whose cell is lethal in
the last published mask is ABORTED, any other goal SUCCEEDS.

Also pins the ROS2SimAdapter fix that makes it honour send_goal's
on_status_change (it used to store the callback and never call it).
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from mars.ros.interfaces import NavGoalStatus

_SCRIPT = Path(__file__).resolve().parents[3] / "deploy" / "nav2" / "scenario_liveness.py"


@pytest.fixture(scope="module")
def S():
    spec = importlib.util.spec_from_file_location("scenario_liveness", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MASK_SETTLE_S = 0.0          # no costmap to wait for
    return mod


def avoid(zone):
    return {"type": "avoid_zone", "params": {"zone": zone}, "duration_sec": 3600}


def cell_value(grid, x, y):
    info = grid["info"]
    res = info["resolution"]
    ox, oy = info["origin"]["position"]["x"], info["origin"]["position"]["y"]
    col, row = int((x - ox) // res), int((y - oy) // res)
    if not (0 <= col < info["width"] and 0 <= row < info["height"]):
        return 0
    return grid["data"][row * info["width"] + col]


class FakeNode:
    def __init__(self, subscribers=1):
        self.subscribers = subscribers

    def count_subscribers(self, topic):
        assert topic == "/keepout_filter_mask"
        return self.subscribers


class FakeNav2:
    """send_goal -> status via on_status_change, decided by the current mask."""

    def __init__(self, server=True, respond=True):
        self.server, self.respond = server, respond
        self.masks, self.goals, self.cancelled = [], [], []

    def wait_for_server(self, robot_id, timeout_sec):
        return self.server

    def publish_keepout_mask(self, grid):
        self.masks.append(grid)

    def send_goal(self, robot_id, goal_id, destination, on_status_change):
        self.goals.append((robot_id, goal_id, destination))
        if not self.respond:
            return
        lethal = self.masks and cell_value(self.masks[-1], destination.x, destination.y) >= 100
        on_status_change(NavGoalStatus(goal_id=goal_id, robot_id=robot_id,
                                       status=6 if lethal else 4))

    def cancel_goal(self, robot_id, goal_id):
        self.cancelled.append(goal_id)


def run(S, active, **kw):
    nav = kw.pop("nav", None) or FakeNav2()
    node = kw.pop("node", None) or FakeNode()
    res = S.drive_to_charger(active, node=node, adapter=nav,
                             spin=lambda sec: time.sleep(min(sec, 0.01)), **kw)
    return res, nav


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def test_target_is_first_open_charger(S):
    assert S.pick_charger([]) == "charge_A"
    assert S.pick_charger(["charge_A"]) == "charge_B"


def test_target_falls_back_to_charge_b_when_all_avoided(S):
    assert S.pick_charger(["charge_A", "charge_B"]) == "charge_B"


def test_goal_poses_lie_inside_their_charger(S):
    from mars.ros.keepout import _points_in_polygon
    import numpy as np
    for z, (x, y, _) in S.CHARGE_GOALS.items():
        assert _points_in_polygon(np.array([x]), np.array([y]), S.ZONES[z]["polygon"])[0], z


def test_only_avoid_zone_policies_count(S):
    active = [avoid("charge_A"), {"type": "prefer_alternate_route",
                                  "params": {"zone": "charge_B"}}]
    assert S.avoided_zones(active) == ["charge_A"]


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def test_mask_marks_exactly_the_avoided_zones(S):
    grid = S.keepout_grid(["charge_A"])
    assert grid["header"]["frame_id"] == "map"
    assert grid["info"]["resolution"] == 0.05
    assert cell_value(grid, *S.CHARGE_GOALS["charge_A"][:2]) == 100
    assert cell_value(grid, *S.CHARGE_GOALS["charge_B"][:2]) == 0
    assert cell_value(grid, *S.START_POSE[:2]) == 0


def test_mask_extent_is_fixed_regardless_of_zones(S):
    # Every mask must fully replace the previous one in the KeepoutFilter, so the
    # extent cannot depend on which zones happen to be avoided.
    a, b, empty = S.keepout_grid(["charge_A"]), S.keepout_grid(["charge_A", "charge_B"]), S.keepout_grid([])
    assert a["info"] == b["info"] == empty["info"]
    assert not any(v for v in empty["data"])
    # ...and it covers every zone, the dock included.
    for d in S.ZONES.values():
        for x, y in d["polygon"]:
            info = a["info"]
            ox, oy = info["origin"]["position"]["x"], info["origin"]["position"]["y"]
            assert ox <= x <= ox + info["width"] * info["resolution"]
            assert oy <= y <= oy + info["height"] * info["resolution"]


# ---------------------------------------------------------------------------
# drive_to_charger end to end (fake Nav2)
# ---------------------------------------------------------------------------

def test_validated_arm_reaches_charge_b(S):
    res, nav = run(S, [avoid("charge_A")])
    assert res["target_charger"] == "charge_B"
    assert res["outcome"] == "succeeded" and res["reached"] is True
    assert res["reset_outcome"] == "succeeded"
    # reset first (cleared mask, dock), then the charge goal under the real mask
    assert [g[1] for g in nav.goals] == ["reset", "charge-charge_B"]
    assert not any(nav.masks[0]["data"])
    assert cell_value(nav.masks[1], *S.CHARGE_GOALS["charge_A"][:2]) == 100
    robot, _, pose = nav.goals[1]
    assert robot == "jongky"
    assert (pose.x, pose.y, pose.qz, pose.qw, pose.frame_id) == (1.7, -1.1, 0.0, 1.0, "map")
    reset_pose = nav.goals[0][2]
    assert (reset_pose.x, reset_pose.y) == S.START_POSE[:2]


def test_unvalidated_arm_attempts_charge_b_and_is_aborted(S):
    res, nav = run(S, [avoid("charge_A"), avoid("charge_B")])
    assert res["target_charger"] == "charge_B"
    assert res["outcome"] == "aborted" and res["reached"] is False
    assert cell_value(nav.masks[-1], 1.7, -1.1) == 100


def test_timeout_is_recorded_distinctly_and_cancels(S):
    nav = FakeNav2(respond=False)
    res, nav = run(S, [avoid("charge_A")], nav=nav, timeout_s=0.2, reset=False)
    assert res["outcome"] == "timeout" and res["reached"] is False
    assert res["elapsed_s"] >= 0.2
    assert nav.cancelled == ["charge-charge_B"]


def test_no_server_sends_nothing(S):
    res, nav = run(S, [avoid("charge_A")], nav=FakeNav2(server=False))
    assert res["outcome"] == "no_server" and nav.goals == [] and nav.masks == []


def test_failed_reset_skips_the_charge_goal(S):
    class ResetBlocked(FakeNav2):
        def send_goal(self, robot_id, goal_id, destination, on_status_change):
            self.goals.append((robot_id, goal_id, destination))
            on_status_change(NavGoalStatus(goal_id, robot_id, 6))
    res, nav = run(S, [avoid("charge_A")], nav=ResetBlocked())
    assert res["outcome"] == "reset_failed" and len(nav.goals) == 1


def test_no_keepout_listener_is_recorded(S):
    res, _ = run(S, [avoid("charge_A")], node=FakeNode(subscribers=0), reset=False)
    assert res["mask_subscribers"] == 0


# ---------------------------------------------------------------------------
# ROS2SimAdapter: send_goal's on_status_change is actually called, once
# ---------------------------------------------------------------------------

def test_adapter_fires_on_status_change_once():
    from mars.ros.isaac_sim_adapter import ROS2SimAdapter

    adapter = ROS2SimAdapter(None, ["jongky"], None)   # rclpy absent -> no _setup
    got = []
    adapter._active_goals["g1"] = ("jongky", object(), got.append)

    class Fut:
        def result(self):
            class R:
                status = 6
            return R()

    adapter._on_result("jongky", "g1", Fut())
    adapter._on_result("jongky", "g1", Fut())          # second path: no re-fire
    assert [(s.goal_id, s.status) for s in got] == [("g1", 6)]
