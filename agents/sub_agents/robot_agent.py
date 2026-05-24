"""로봇 제어 에이전트 — goal → RL 정책 트리거.

실배포: ROS2 /goal_pose 토픽 퍼블리시
데모:   목표 출력 + 시뮬레이션 env에 직접 goal 전달
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RobotStatus:
    robot_id: str
    x: float
    y: float
    battery_pct: float
    status: str          # idle / moving / blocked / charging / error
    current_task: str | None = None
    goal_x: float | None = None
    goal_y: float | None = None


class RobotAgent:
    """단일 로봇 담당 에이전트."""

    def __init__(self, robot_id: str, use_ros2: bool = False):
        self.robot_id = robot_id
        self.use_ros2 = use_ros2
        self._status = RobotStatus(
            robot_id=robot_id, x=0.0, y=0.0,
            battery_pct=100.0, status="idle"
        )

    def assign_goal(self, goal_x: float, goal_y: float, task: str = "") -> bool:
        """목표 배정 — Propose-then-Commit 두 번째 단계."""
        print(f"  [{self.robot_id}] 목표 배정: ({goal_x:+.1f}, {goal_y:+.1f})  {task}")

        if self.use_ros2:
            return self._publish_ros2_goal(goal_x, goal_y)
        else:
            return self._sim_goal(goal_x, goal_y, task)

    def _publish_ros2_goal(self, goal_x: float, goal_y: float) -> bool:
        """ROS2 /goal_pose 퍼블리시 (실배포용 — rclpy 필요)."""
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from rclpy.node import Node

            class GoalPublisher(Node):
                def __init__(self, robot_id: str):
                    super().__init__(f"mars_{robot_id.lower()}_agent")
                    self.pub = self.create_publisher(
                        PoseStamped, f"/{robot_id.lower()}/goal_pose", 10
                    )

            rclpy.init()
            node = GoalPublisher(self.robot_id)
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.pose.position.x = goal_x
            msg.pose.position.y = goal_y
            node.pub.publish(msg)
            time.sleep(0.1)
            rclpy.shutdown()
            return True
        except ImportError:
            print(f"  [{self.robot_id}] rclpy 없음 — 시뮬 모드로 전환")
            return self._sim_goal(goal_x, goal_y, "")

    def _sim_goal(self, goal_x: float, goal_y: float, task: str) -> bool:
        """데모/시뮬용: 목표 상태 기록."""
        self._status.goal_x = goal_x
        self._status.goal_y = goal_y
        self._status.current_task = task
        self._status.status = "moving"
        return True

    def get_status(self) -> RobotStatus:
        return self._status

    def update_position(self, x: float, y: float) -> None:
        self._status.x = x
        self._status.y = y

    def mark_arrived(self) -> None:
        self._status.status = "idle"
        self._status.current_task = None
        self._status.goal_x = None
        self._status.goal_y = None


class RobotFleet:
    """로봇 3대 통합 관리."""

    def __init__(self, robot_ids: list[str] | None = None, use_ros2: bool = False):
        ids = robot_ids or ["R1", "R2", "R3"]
        self.agents = {rid: RobotAgent(rid, use_ros2=use_ros2) for rid in ids}

    def dispatch(self, assignments: list[dict]) -> dict[str, bool]:
        """
        assignments: [{"robot_id": "R1", "goal_x": 3.5, "goal_y": 2.0, "task": "..."}]
        반환: {"R1": True, "R2": True, ...}
        """
        results = {}
        for a in sorted(assignments, key=lambda x: x.get("priority", 1)):
            rid = a["robot_id"]
            if rid not in self.agents:
                print(f"  [Fleet] 알 수 없는 로봇: {rid}")
                results[rid] = False
                continue
            ok = self.agents[rid].assign_goal(
                a["goal_x"], a["goal_y"], a.get("task", "")
            )
            results[rid] = ok
        return results

    def status_summary(self) -> str:
        lines = ["[ 로봇 상태 ]"]
        for rid, agent in self.agents.items():
            s = agent.get_status()
            goal_str = (
                f"→ ({s.goal_x:+.1f}, {s.goal_y:+.1f})"
                if s.goal_x is not None else "대기"
            )
            lines.append(f"  {rid}: {s.status:<8} {goal_str}  {s.current_task or ''}")
        return "\n".join(lines)


# ── 데모 실행 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fleet = RobotFleet()

    assignments = [
        {"robot_id": "R1", "goal_x":  3.5, "goal_y":  2.0, "task": "C1→G1", "priority": 1},
        {"robot_id": "R2", "goal_x":  3.5, "goal_y":  0.0, "task": "C2→G1", "priority": 2},
        {"robot_id": "R3", "goal_x":  4.5, "goal_y":  0.0, "task": "G1 대기", "priority": 3},
    ]

    print("\n[RobotFleet] 목표 배정 시작")
    results = fleet.dispatch(assignments)
    print(f"\n배정 결과: {results}")
    print(f"\n{fleet.status_summary()}")
