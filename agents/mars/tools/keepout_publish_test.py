"""
Standalone keepout-mask publisher — drives the "agent decided avoid_zone"
side of the Nav2 keepout demo WITHOUT the full brain, so we can verify
mask -> KeepoutFilter -> planner reroute in isolation.

Publishes one rectangular keepout zone (a "wall" across the corridor) on the
same latched topic the real supervisor uses (/keepout_filter_mask), reusing
mars.ros.keepout. Default places a wall centered at (4,0), 1m thick x 5m wide,
so a straight path from (0,0) to (8,0) is blocked and the planner must detour.

Run (RunPod, `source deploy/isaac/env_ros2.sh`, from agents/mars/):
    python3 -m tools.keepout_publish_test                  # wall at (4,0) 1x5
    python3 -m tools.keepout_publish_test --cx 4 --cy 0 --w 1 --h 5
Leave it running (latched). Ctrl+C to clear.
"""
from __future__ import annotations

import argparse

from mars.ros.keepout import MapMeta, build_occupancy_grid_dict

KEEPOUT_MASK_TOPIC = "/keepout_filter_mask"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cx", type=float, default=4.0, help="zone center x (m)")
    ap.add_argument("--cy", type=float, default=0.0, help="zone center y (m)")
    ap.add_argument("--w", type=float, default=1.0, help="zone width along x (m)")
    ap.add_argument("--h", type=float, default=5.0, help="zone height along y (m)")
    ap.add_argument("--resolution", type=float, default=0.05)
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import Pose, Point, Quaternion

    hx, hy = args.w / 2.0, args.h / 2.0
    poly = [
        (args.cx - hx, args.cy - hy),
        (args.cx + hx, args.cy - hy),
        (args.cx + hx, args.cy + hy),
        (args.cx - hx, args.cy + hy),
    ]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
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
        f"published keepout wall center=({args.cx},{args.cy}) "
        f"size=({args.w}x{args.h}) on {KEEPOUT_MASK_TOPIC}: "
        f"{msg.info.width}x{msg.info.height} cells, {keepout_cells} keepout. "
        f"Latched — Ctrl+C to clear."
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
