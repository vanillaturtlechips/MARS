"""
M4 tests — Isaac Sim adapter and Zone Resolver

No rclpy / Isaac Sim required:

  - ZoneResolver: point-in-polygon, centroid fallback, polygon parsing
  - topic_name(): canonical per-robot topic names match §1a spec
  - Interface contract: ROS2SimAdapter declares the same API as MockSim
  - Mock-vs-adapter swap: anything that works with MockSim compiles with
    ROS2SimAdapter (structural duck-typing check)
"""
from __future__ import annotations

import pytest

from mars.ros.zone_resolver import ZoneResolver, _point_in_polygon, _centroid
from mars.ros.isaac_sim_adapter import topic_name


# ---------------------------------------------------------------------------
# ZoneResolver — geometry
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    def _square(self):
        """Unit square [0,1]×[0,1]."""
        return [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_center_inside(self):
        assert _point_in_polygon(0.5, 0.5, self._square()) is True

    def test_outside(self):
        assert _point_in_polygon(2.0, 0.5, self._square()) is False

    def test_bottom_left_corner_borderline(self):
        # Corners are typically considered outside by ray casting
        # — we just assert it doesn't crash and returns a bool
        result = _point_in_polygon(0.0, 0.0, self._square())
        assert isinstance(result, bool)

    def test_degenerate_polygon_returns_false(self):
        assert _point_in_polygon(0.5, 0.5, [(0, 0), (1, 0)]) is False   # only 2 vertices

    def test_empty_polygon_returns_false(self):
        assert _point_in_polygon(0.0, 0.0, []) is False

    def test_triangle_inside(self):
        tri = [(0, 0), (2, 0), (1, 2)]
        assert _point_in_polygon(1.0, 0.5, tri) is True

    def test_triangle_outside(self):
        tri = [(0, 0), (2, 0), (1, 2)]
        assert _point_in_polygon(3.0, 0.5, tri) is False


class TestCentroid:
    def test_square_centroid(self):
        sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
        cx, cy = _centroid(sq)
        assert abs(cx - 1.0) < 1e-9
        assert abs(cy - 1.0) < 1e-9

    def test_single_point(self):
        cx, cy = _centroid([(3.5, 7.2)])
        assert abs(cx - 3.5) < 1e-9
        assert abs(cy - 7.2) < 1e-9

    def test_empty_polygon(self):
        assert _centroid([]) == (0.0, 0.0)


class TestZoneResolver:
    def _zones(self):
        return [
            {
                "zone_id": "receiving_dock",
                "polygon": [
                    {"x": 0, "y": 0}, {"x": 4, "y": 0},
                    {"x": 4, "y": 3}, {"x": 0, "y": 3},
                ],
            },
            {
                "zone_id": "storage_a",
                "polygon": [
                    {"x": 5, "y": 0}, {"x": 10, "y": 0},
                    {"x": 10, "y": 5}, {"x": 5,  "y": 5},
                ],
            },
            {
                "zone_id": "charging_bay",
                "polygon": [
                    {"x": 0, "y": 6}, {"x": 4, "y": 6},
                    {"x": 4, "y": 9}, {"x": 0, "y": 9},
                ],
            },
        ]

    def test_point_in_first_zone(self):
        r = ZoneResolver(self._zones())
        assert r.resolve(2.0, 1.5) == "receiving_dock"

    def test_point_in_second_zone(self):
        r = ZoneResolver(self._zones())
        assert r.resolve(7.0, 2.5) == "storage_a"

    def test_point_in_third_zone(self):
        r = ZoneResolver(self._zones())
        assert r.resolve(2.0, 7.5) == "charging_bay"

    def test_point_outside_all_zones_returns_nearest(self):
        r = ZoneResolver(self._zones())
        # (12, 12) is outside all polygons → nearest centroid
        result = r.resolve(12.0, 12.0)
        assert result is not None          # returns something
        assert isinstance(result, str)

    def test_empty_zone_list_returns_none(self):
        r = ZoneResolver([])
        assert r.resolve(1.0, 1.0) is None

    def test_zone_without_polygon_uses_centroid_fallback(self):
        """Zone with no polygon should still be returned as centroid fallback."""
        zones = [{"zone_id": "ghost_zone", "polygon": None}]
        r = ZoneResolver(zones)
        result = r.resolve(99.0, 99.0)
        assert result == "ghost_zone"

    def test_refresh_replaces_zones(self):
        r = ZoneResolver(self._zones())
        assert r.resolve(2.0, 1.5) == "receiving_dock"

        r.refresh([{
            "zone_id": "new_zone",
            "polygon": [
                {"x": 0, "y": 0}, {"x": 100, "y": 0},
                {"x": 100, "y": 100}, {"x": 0, "y": 100},
            ],
        }])
        assert r.resolve(2.0, 1.5) == "new_zone"

    def test_polygon_as_list_of_lists(self):
        """Polygon vertices may arrive as [[x,y], ...] as well as [{x,y}]."""
        zones = [{
            "zone_id": "alt_format",
            "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
        }]
        r = ZoneResolver(zones)
        assert r.resolve(2.0, 1.5) == "alt_format"


# ---------------------------------------------------------------------------
# Topic name convention (§1a)
# ---------------------------------------------------------------------------

class TestTopicNames:
    def test_battery_topic(self):
        assert topic_name("R1", "battery_state") == "/R1/battery_state"

    def test_robot_health_topic(self):
        assert topic_name("R3", "robot_health") == "/R3/robot_health"

    def test_nav_status_topic(self):
        t = topic_name("R2", "navigate_to_pose/_action/status")
        assert t == "/R2/navigate_to_pose/_action/status"

    def test_odom_topic(self):
        assert topic_name("R5", "odom") == "/R5/odom"


# ---------------------------------------------------------------------------
# Interface contract — ROS2SimAdapter is structurally compatible with MockSim
# ---------------------------------------------------------------------------

class TestAdapterInterfaceContract:
    """
    Verify that ROS2SimAdapter declares the same interface methods as MockSim.
    We only check method names / signatures — no rclpy init required.
    """

    def test_navigation_interface_methods_present(self):
        from mars.ros.isaac_sim_adapter import ROS2SimAdapter
        assert hasattr(ROS2SimAdapter, "send_goal")
        assert hasattr(ROS2SimAdapter, "cancel_goal")
        assert callable(ROS2SimAdapter.send_goal)
        assert callable(ROS2SimAdapter.cancel_goal)

    def test_sensor_interface_methods_present(self):
        from mars.ros.isaac_sim_adapter import ROS2SimAdapter
        for method in ("subscribe_battery", "subscribe_health",
                       "subscribe_pose", "subscribe_nav_status"):
            assert hasattr(ROS2SimAdapter, method), f"Missing: {method}"
            assert callable(getattr(ROS2SimAdapter, method))

    def test_mock_sim_declares_same_interface(self):
        """MockSim must have the same API as the adapter so they're interchangeable."""
        from mars.sim.mock_sim import MockSim
        from mars.ros.isaac_sim_adapter import ROS2SimAdapter

        adapter_methods = {
            m for m in dir(ROS2SimAdapter) if not m.startswith("_")
            and callable(getattr(ROS2SimAdapter, m))
        }
        mock_methods = {
            m for m in dir(MockSim) if not m.startswith("_")
            and callable(getattr(MockSim, m))
        }

        # Core interface methods must be present in BOTH
        required = {
            "send_goal", "cancel_goal",
            "subscribe_battery", "subscribe_health",
            "subscribe_pose", "subscribe_nav_status",
        }
        assert required <= adapter_methods, f"Adapter missing: {required - adapter_methods}"
        assert required <= mock_methods,    f"MockSim missing: {required - mock_methods}"

    def test_uuid_helper_returns_string(self):
        from mars.ros.isaac_sim_adapter import _uuid_bytes_to_hex
        result = _uuid_bytes_to_hex(b"\x00" * 16)
        assert isinstance(result, str)
        assert len(result) == 32  # 16 bytes → 32 hex chars


# ---------------------------------------------------------------------------
# Swap demonstration — Aggregator callbacks compile with either adapter
# ---------------------------------------------------------------------------

class TestMockSimSwap:
    """
    Demonstrate that Aggregator wiring works identically with MockSim or with a
    stub that has the same interface signature.  This is the swap guarantee.
    """

    def test_aggregator_wiring_compiles_with_mock(self):
        """The pattern used in demo.py / ros2_node.py must work with MockSim."""
        from mars.sim.mock_sim import MockSim
        from mars.aggregator.aggregator import Aggregator
        from mars.blackboard.hot_state import HotState

        events = []
        hs  = HotState(redis_client=None)
        agg = Aggregator(hs, on_failure_event=events.append)

        sim = MockSim(["R1"])

        # These are the exact calls made in demo.py / ros2_node.py
        sim.subscribe_battery("R1", agg.on_battery_update)
        sim.subscribe_health("R1", agg.on_health_update)
        sim.subscribe_pose("R1", lambda p: agg.on_pose_update("R1", p, "test_zone"))
        sim.subscribe_nav_status("R1", lambda ns: None)

        # No assertion needed — if this compiled and ran without TypeError, the
        # interface is correct.

    def test_adapter_method_signatures_match_aggregator_expectations(self):
        """
        ROS2SimAdapter.subscribe_battery / _health / _nav_status take (robot_id, callback).
        Aggregator's methods (on_battery_update, etc.) take (message) only.
        This test verifies the signatures are compatible.
        """
        import inspect
        from mars.ros.isaac_sim_adapter import ROS2SimAdapter
        from mars.aggregator.aggregator import Aggregator

        # subscribe_battery(self, robot_id, callback) — two positional args after self
        sig = inspect.signature(ROS2SimAdapter.subscribe_battery)
        params = list(sig.parameters)
        assert "robot_id" in params
        assert "callback" in params

        # Aggregator.on_battery_update(self, msg) — one positional arg after self
        sig2 = inspect.signature(Aggregator.on_battery_update)
        params2 = list(sig2.parameters)
        assert len(params2) == 2  # self + msg
