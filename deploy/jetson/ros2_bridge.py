"""ROS2 Humble 브릿지 — RL policy → /cmd_vel 퍼블리시.

실행:
  python3 deploy/jetson/ros2_bridge.py

구독:
  /goal_pose  (geometry_msgs/PoseStamped)  — 목표 위치 (world frame)
  /odom       (nav_msgs/Odometry)          — 현재 로봇 pose + velocity

퍼블리시:
  /cmd_vel    (geometry_msgs/Twist)        — 속도 명령 (body frame)
  /policy_hz  (std_msgs/Float32)           — 추론 주기 모니터링

좌표계: ROS2 표준 (REP-105)
  - world frame: ENU (x=East, y=North, z=Up)
  - body frame:  Forward-Left-Up
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from inference import WarehousePolicy

# 선반 AABB (warehouse_obstacle_env.py 와 동일한 좌표)
SHELF_CENTERS = [(-2.0, 2.5), (2.0, 2.5), (-2.0, -2.5), (2.0, -2.5)]
SHELF_HALF = (1.5, 0.25)   # (x, y) half-size


def _min_shelf_dist(rx: float, ry: float) -> float:
    """로봇 위치 → 가장 가까운 선반까지 AABB 거리 [m], 최대 5.0 클램프."""
    min_d = 5.0
    for cx, cy in SHELF_CENTERS:
        dx = max(abs(rx - cx) - SHELF_HALF[0], 0.0)
        dy = max(abs(ry - cy) - SHELF_HALF[1], 0.0)
        d  = math.hypot(dx, dy)
        if d < min_d:
            min_d = d
    return min_d


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """쿼터니언 → yaw (z-axis rotation) [rad]."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PolicyBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__("mars_policy_bridge")

        model_path = Path(__file__).parent / "actor_phase15.pt"
        self._policy = WarehousePolicy(model_path)

        # 상태 버퍼 — odom 콜백에서 갱신
        self._robot_x  = 0.0
        self._robot_y  = 0.0
        self._robot_yaw = 0.0
        self._vx_world = 0.0
        self._vy_world = 0.0
        self._omega_z  = 0.0

        # 목표 버퍼 — goal 콜백에서 갱신
        self._goal_x = 0.0
        self._goal_y = 0.0
        self._has_goal = False

        # 구독
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_cb, 10)
        self.create_subscription(Odometry,    "/odom",      self._odom_cb, 10)

        # 퍼블리셔
        self._cmd_pub  = self.create_publisher(Twist,   "/cmd_vel",   10)
        self._hz_pub   = self.create_publisher(Float32, "/policy_hz", 10)

        # 15 Hz 타이머 (warehouse_env.py 와 동일한 policy rate)
        self._timer = self.create_timer(1.0 / 15.0, self._step)
        self._last_step_t = time.perf_counter()

        self.get_logger().info("PolicyBridgeNode 시작 — /goal_pose 대기 중")

    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal_x  = msg.pose.position.x
        self._goal_y  = msg.pose.position.y
        self._has_goal = True
        self.get_logger().info(f"새 목표: ({self._goal_x:.2f}, {self._goal_y:.2f})")

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        t = msg.twist.twist

        self._robot_x   = p.x
        self._robot_y   = p.y
        self._robot_yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self._vx_world  = t.linear.x
        self._vy_world  = t.linear.y
        self._omega_z   = t.angular.z

    def _step(self) -> None:
        if not self._has_goal:
            return

        # goal 벡터 → body frame
        gx_w = self._goal_x - self._robot_x
        gy_w = self._goal_y - self._robot_y
        yaw  = self._robot_yaw
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        goal_x_body =  cos_y * gx_w + sin_y * gy_w
        goal_y_body = -sin_y * gx_w + cos_y * gy_w
        goal_dist   = math.hypot(gx_w, gy_w)

        # velocity → body frame
        vx_body =  cos_y * self._vx_world + sin_y * self._vy_world
        vy_body = -sin_y * self._vx_world + cos_y * self._vy_world

        # 장애물 거리 (world → local 오프셋 없음, 절대 좌표 그대로)
        min_obs = _min_shelf_dist(self._robot_x, self._robot_y)

        # 목표 도달 판정 (0.35 m — warehouse_env.py 와 동일)
        if goal_dist < 0.35:
            self._publish_zero()
            self.get_logger().info("목표 도달! cmd_vel = 0")
            self._has_goal = False
            return

        cmd_vx, cmd_vy, cmd_omega = self._policy.act(
            goal_x_body, goal_y_body, goal_dist,
            vx_body, vy_body, self._omega_z,
            min_obs,
        )

        twist = Twist()
        twist.linear.x  = cmd_vx
        twist.linear.y  = cmd_vy
        twist.angular.z = cmd_omega
        self._cmd_pub.publish(twist)

        # Hz 모니터링
        now = time.perf_counter()
        hz = 1.0 / max(now - self._last_step_t, 1e-9)
        self._last_step_t = now
        hz_msg = Float32()
        hz_msg.data = float(hz)
        self._hz_pub.publish(hz_msg)

    def _publish_zero(self) -> None:
        self._cmd_pub.publish(Twist())


def main() -> None:
    rclpy.init()
    node = PolicyBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
