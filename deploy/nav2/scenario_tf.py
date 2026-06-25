"""Publish static map -> <R>/odom TFs for an S1~S6 scenario.

Isaac publishes only <R>/odom -> <R>/base_link (IsaacComputeOdometry zeroes at
spawn). Nav2's costmaps need the `map` frame, so we supply the spawn offset:
each robot's odom origin == its spawn world position, identity rotation. Without
this every Nav2 costmap dies with 'frame "map" does not exist'.

Run BEFORE the per-robot Nav2 stacks:
    source deploy/isaac/env_ros2.sh
    python3 deploy/nav2/scenario_tf.py S1
"""
from __future__ import annotations

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # deploy/
import scenarios as SC  # noqa: E402


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "S1"
    if scenario not in SC.SCENARIOS:
        print(f"unknown scenario {scenario}; choose {list(SC.SCENARIOS)}")
        sys.exit(1)

    rclpy.init()
    node = Node("scenario_tf")
    bc = StaticTransformBroadcaster(node)

    tfs = []
    for name, x, y in SC.robots_for(scenario):
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = f"{name}/odom"
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.rotation.w = 1.0
        tfs.append(t)
        node.get_logger().info(f"map -> {name}/odom @ ({x:.2f},{y:.2f})")
    bc.sendTransform(tfs)

    node.get_logger().info(f"[{scenario}] static map->odom TFs published; spinning.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
