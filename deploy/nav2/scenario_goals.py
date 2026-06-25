"""Send S1~S6 scenario goals to per-robot Nav2 stacks (NavigateToPose).

Improved (sensor + Nav2) version of the S1~S6 multi-robot scenarios: each
iw_hub robot navigates its goal with lidar-costmap avoidance instead of the
pure-RL policy. Coordinates come from deploy/scenarios.py (single source).

Run (ROS2 side, after Isaac scene + per-robot Nav2 are up):
    source deploy/isaac/env_ros2.sh
    python3 deploy/nav2/scenario_goals.py S1

S4 (동일 목표): two robots share one goal point. The driver lets whichever
reaches first WIN; on the other's abort (goal blocked) it reassigns the loser to
a small yield offset — the deadlock is resolved at the task layer, not by more
driving. (This reassignment hook is where the LLM orchestrator can plug in; the
default here is a deterministic rule so S4 works standalone.)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # deploy/
import scenarios as SC  # noqa: E402


def _pose(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "map"
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.orientation.z = math.sin(yaw / 2.0)
    p.pose.orientation.w = math.cos(yaw / 2.0)
    return p


class ScenarioRunner(Node):
    def __init__(self, scenario: str):
        super().__init__("scenario_goals")
        self.scenario = scenario
        self.spec = SC.SCENARIOS[scenario]
        self.goals = SC.goals_for(scenario)            # [(name,x,y)...]
        self.shared = set(self.spec.get("shared_goal", []))
        self.clients: dict[str, ActionClient] = {}
        self.done: dict[str, str] = {}                 # name -> "reached"/"aborted"
        self.reassigned: set[str] = set()

        for name, _, _ in self.goals:
            self.clients[name] = ActionClient(self, NavigateToPose, f"/{name}/navigate_to_pose")

        self.get_logger().info(f"[{scenario}] {self.spec['desc']}")
        for name, x, y in self.goals:
            self.get_logger().info(f"  {name} -> ({x:.2f}, {y:.2f})")
        self._send_all()

    def _send_all(self):
        for i, (name, x, y) in enumerate(self.goals):
            self._send(name, x, y, idx=i)

    def _send(self, name: str, x: float, y: float, idx: int):
        cli = self.clients[name]
        if not cli.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"{name}: navigate_to_pose server not available")
            self.done[name] = "no_server"
            return
        goal = NavigateToPose.Goal()
        goal.pose = _pose(x, y)
        self.get_logger().info(f"{name}: sending goal ({x:.2f},{y:.2f})")
        fut = cli.send_goal_async(goal)
        fut.add_done_callback(lambda f, n=name, ix=idx: self._on_accepted(f, n, ix))

    def _on_accepted(self, future, name: str, idx: int):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn(f"{name}: goal rejected")
            self.done[name] = "rejected"
            return
        handle.get_result_async().add_done_callback(
            lambda f, n=name, ix=idx: self._on_result(f, n, ix))

    def _on_result(self, future, name: str, idx: int):
        status = future.result().status   # 4=SUCCEEDED,5=CANCELED,6=ABORTED
        if status == 4:
            self.done[name] = "reached"
            self.get_logger().info(f"{name}: REACHED")
        else:
            self.get_logger().warn(f"{name}: did not reach (status={status})")
            # S4 deadlock resolution: a shared-goal loser that aborts gets a
            # yield offset so the scenario completes (one wins, one yields).
            if idx in self.shared and name not in self.reassigned:
                self.reassigned.add(name)
                gx, gy = self.goals[idx][1], self.goals[idx][2]
                yx, yy = gx, gy + 1.0          # yield 1 m aside of the contested point
                self.get_logger().warn(f"{name}: shared-goal yield -> ({yx:.2f},{yy:.2f})")
                self._send(name, yx, yy, idx=idx)
            else:
                self.done[name] = "aborted"

    def all_done(self) -> bool:
        return len(self.done) >= len(self.goals)


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "S1"
    if scenario not in SC.SCENARIOS:
        print(f"unknown scenario {scenario}; choose from {list(SC.SCENARIOS)}")
        sys.exit(1)
    rclpy.init()
    node = ScenarioRunner(scenario)
    try:
        while rclpy.ok() and not node.all_done():
            rclpy.spin_once(node, timeout_sec=0.5)
        node.get_logger().info(f"[{scenario}] outcomes: {node.done}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
