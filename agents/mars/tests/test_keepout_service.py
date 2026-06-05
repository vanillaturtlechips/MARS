"""
Unit tests for KeepoutService (policy → Nav2 keepout mask).

DB-free: get_zone_polygons is monkeypatched, and the mask lands in MockSim
via publish_keepout_mask.  Verifies the full policy→mask path locally so the
only unverified piece is the rclpy OccupancyGrid publish (RunPod).
"""
import mars.services.keepout_service as ks
from mars.services.keepout_service import KeepoutService
from mars.ros.keepout import KEEPOUT_VALUE
from mars.sim.mock_sim import MockSim


# Canned zone polygons (meters, map frame) — receiving_dock + storage_area_a.
ZONE_POLYS = {
    "receiving_dock": [{"x": 0, "y": 0}, {"x": 4, "y": 0},
                       {"x": 4, "y": 3}, {"x": 0, "y": 3}],
    "storage_area_a": [{"x": 6, "y": 6}, {"x": 9, "y": 6},
                       {"x": 9, "y": 9}, {"x": 6, "y": 9}],
}


class _DummyConn:
    def close(self):
        pass


def _service(monkeypatch, sim):
    monkeypatch.setattr(
        ks, "get_zone_polygons",
        lambda conn, zone_ids: {z: ZONE_POLYS[z] for z in zone_ids if z in ZONE_POLYS},
    )
    return KeepoutService(sim, conn_factory=lambda: _DummyConn(), resolution=0.1)


def _keepout_cells(grid):
    return sum(1 for v in grid["data"] if v >= KEEPOUT_VALUE)


def test_activate_publishes_keepout_mask(monkeypatch):
    sim = MockSim(robot_ids=["R1"])
    svc = _service(monkeypatch, sim)

    svc.on_policy_change("activated", {"type": "avoid_zone",
                                       "params": {"zone": "receiving_dock"}})

    assert sim.last_keepout_grid is not None
    assert _keepout_cells(sim.last_keepout_grid) > 0
    assert sim.last_keepout_grid["header"]["frame_id"] == "map"


def test_non_avoid_zone_policy_ignored(monkeypatch):
    sim = MockSim(robot_ids=["R1"])
    svc = _service(monkeypatch, sim)

    svc.on_policy_change("activated", {"type": "reserve_chargers_for_critical",
                                       "params": {}})

    assert sim.last_keepout_grid is None
    assert sim.keepout_grids == []


def test_deactivate_clears_mask(monkeypatch):
    sim = MockSim(robot_ids=["R1"])
    svc = _service(monkeypatch, sim)

    svc.on_policy_change("activated", {"type": "avoid_zone",
                                       "params": {"zone": "receiving_dock"}})
    assert _keepout_cells(sim.last_keepout_grid) > 0

    svc.on_policy_change("deactivated", {"type": "avoid_zone",
                                         "params": {"zone": "receiving_dock"}})
    # Cleared: tiny all-free mask, zero keepout cells.
    assert _keepout_cells(sim.last_keepout_grid) == 0


def test_two_zones_union_has_more_keepout(monkeypatch):
    sim = MockSim(robot_ids=["R1"])
    svc = _service(monkeypatch, sim)

    svc.on_policy_change("activated", {"type": "avoid_zone",
                                       "params": {"zone": "receiving_dock"}})
    one = _keepout_cells(sim.last_keepout_grid)

    svc.on_policy_change("activated", {"type": "avoid_zone",
                                       "params": {"zone": "storage_area_a"}})
    two = _keepout_cells(sim.last_keepout_grid)

    assert two > one  # second zone added more keepout cells


def test_missing_polygon_does_not_crash(monkeypatch):
    sim = MockSim(robot_ids=["R1"])
    svc = _service(monkeypatch, sim)

    # Zone not in ZONE_POLYS → no usable polygon → no publish, no exception.
    svc.on_policy_change("activated", {"type": "avoid_zone",
                                       "params": {"zone": "unknown_zone"}})
    assert sim.last_keepout_grid is None
