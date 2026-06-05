"""
Standalone keepout-mask publisher — isolates the "mask → Nav2 KeepoutFilter"
plumbing WITHOUT Isaac Sim or the full agent brain.

It publishes one keepout mask (receiving_dock by default) on the same latched
topic the real supervisor uses (/keepout_filter_mask), reusing the exact same
mask-building code (mars.ros.keepout).  Pair it with:

    ros2 launch deploy/nav2/keepout_filter.launch.py     # costmap_filter_info_server
    python3 -m tools.keepout_publish_test                 # this script
    ros2 topic echo /keepout_filter_mask --once           # see the mask
    ros2 topic echo /costmap_filter_info --once           # see the filter info

Then a Nav2 global_costmap configured with the keepout_filter plugin will mark
the zone lethal and the planner will route around it.

Run from agents/mars/ (so `tools` and `mars` are importable):
    source /opt/ros/humble/setup.bash
    python3 -m tools.keepout_publish_test --zone receiving_dock
"""
from __future__ import annotations

import argparse

from mars.ros.keepout import MapMeta, build_occupancy_grid_dict

# Same demo polygon as mars/orchestrator/demo.py _ZONES (meters, map frame).
_DEMO_POLYGONS = {
    "receiving_dock": [(0.0, -1.0), (4.0, -1.0), (4.0, 2.0), (0.0, 2.0)],
    "charging_bay":   [(4.0, -1.0), (7.0, -1.0), (7.0, 2.0), (4.0, 2.0)],
    "storage_area_a": [(7.0, -1.0), (10.0, -1.0), (10.0, 2.0), (7.0, 2.0)],
}

KEEPOUT_MASK_TOPIC = "/keepout_filter_mask"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="receiving_dock", choices=list(_DEMO_POLYGONS))
    ap.add_argument("--resolution", type=float, default=0.05)
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import Pose, Point, Quaternion

    poly = _DEMO_POLYGONS[args.zone]
    xs = [x for x, _ in poly]
    ys = [y for _, y in poly]
    meta = MapMeta.covering(min(xs), min(ys), max(xs), max(ys),
                            resolution=args.resolution, margin=0.5)
    grid = build_occupancy_grid_dict([poly], meta)

    rclpy.init()
    node = Node("keepout_test_publisher")
    latched = QoSProfile(
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )
    pub = node.create_publisher(OccupancyGrid, KEEPOUT_MASK_TOPIC, latched)

    info = grid["info"]
    origin = info["origin"]
    msg = OccupancyGrid()
    msg.header.frame_id = grid["header"]["frame_id"]
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.info.resolution = float(info["resolution"])
    msg.info.width = int(info["width"])
    msg.info.height = int(info["height"])
    msg.info.origin = Pose(
        position=Point(x=float(origin["position"]["x"]),
                       y=float(origin["position"]["y"]), z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    msg.data = [int(v) for v in grid["data"]]
    pub.publish(msg)

    keepout_cells = sum(1 for v in grid["data"] if v >= 100)
    node.get_logger().info(
        f"published keepout mask for '{args.zone}' on {KEEPOUT_MASK_TOPIC}: "
        f"{msg.info.width}x{msg.info.height}, {keepout_cells} keepout cells "
        f"(res={msg.info.resolution} origin=({origin['position']['x']},"
        f"{origin['position']['y']})). Latched — Ctrl+C to stop."
    )
    try:
        rclpy.spin(node)   # keep the latched publisher alive for subscribers
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
