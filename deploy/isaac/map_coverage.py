"""Print SLAM /map coverage so you can tell (headless) when it's filled in.

Run on the ROS2 side:
    source deploy/isaac/env_ros2.sh
    python3 deploy/isaac/map_coverage.py

Each /map update prints: size, known cells (free+occupied), unknown, and the
known fraction of the bounding box. When 'known' and the W x H size stop growing
as you drive, the reachable area is mapped — stop and save.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class Cov(Node):
    def __init__(self):
        super().__init__("map_coverage")
        # /map is latched (transient_local); match it.
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, "/map", self.cb, qos)
        self.prev_known = 0

    def cb(self, msg):
        w, h, res = msg.info.width, msg.info.height, msg.info.resolution
        data = msg.data
        free = sum(1 for v in data if 0 <= v < 50)
        occ = sum(1 for v in data if v >= 50)
        unknown = sum(1 for v in data if v < 0)
        known = free + occ
        total = w * h
        frac = (known / total * 100) if total else 0
        grew = known - self.prev_known
        self.prev_known = known
        area = known * res * res
        self.get_logger().info(
            f"size {w}x{h} ({w*res:.1f}x{h*res:.1f} m)  "
            f"known {known} (+{grew})  occ {occ}  free {free}  unknown {unknown}  "
            f"known {frac:.1f}%  mapped ~{area:.1f} m^2"
        )


def main():
    rclpy.init()
    rclpy.spin(Cov())


if __name__ == "__main__":
    main()
