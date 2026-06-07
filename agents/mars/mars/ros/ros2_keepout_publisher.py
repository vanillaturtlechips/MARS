"""
Ros2KeepoutPublisher — a NavigationInterface-shaped object whose
publish_keepout_mask() emits a real nav_msgs/OccupancyGrid on
/keepout_filter_mask (latched), for wiring the live supervisory brain
(demo.py M1) to a running Nav2 KeepoutFilter.

Only publish_keepout_mask is implemented (the only method KeepoutService
calls). rclpy is imported lazily so this module stays importable without ROS2.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

KEEPOUT_MASK_TOPIC = "/keepout_filter_mask"


class Ros2KeepoutPublisher:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
        from nav_msgs.msg import OccupancyGrid

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("mars_keepout_publisher")
        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub = self._node.create_publisher(OccupancyGrid, KEEPOUT_MASK_TOPIC, latched)
        self._OccupancyGrid = OccupancyGrid
        log.info("[ros2_keepout] publisher ready on %s", KEEPOUT_MASK_TOPIC)

    def publish_keepout_mask(self, grid: dict) -> None:
        from geometry_msgs.msg import Pose, Point, Quaternion
        info = grid["info"]
        origin = info["origin"]
        msg = self._OccupancyGrid()
        msg.header.frame_id = grid["header"]["frame_id"]
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.info.resolution = float(info["resolution"])
        msg.info.width = int(info["width"])
        msg.info.height = int(info["height"])
        msg.info.origin = Pose(
            position=Point(x=float(origin["position"]["x"]),
                           y=float(origin["position"]["y"]), z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        msg.data = [int(v) for v in grid["data"]]
        self._pub.publish(msg)
        keepout_cells = sum(1 for v in grid["data"] if v >= 100)
        log.info("[ros2_keepout] published mask %dx%d, %d keepout cells",
                 msg.info.width, msg.info.height, keepout_cells)

    def spin(self) -> None:
        """Keep the node (latched publisher) alive until Ctrl+C."""
        import rclpy
        try:
            rclpy.spin(self._node)
        except KeyboardInterrupt:
            pass
