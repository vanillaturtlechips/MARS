"""
ROS boundary interfaces — abstract contracts that both the mock sim and a real
Isaac Sim adapter must satisfy.

These are plain Python ABCs with no rclpy import; the concrete adapter imports
rclpy.  This keeps the supervisory logic importable without a ROS installation.

Data types are plain dicts rather than ROS message objects so callers don't
need rclpy either.

GoalStatus constants (Nav2 Humble):
    STATUS_UNKNOWN   = 0
    STATUS_ACCEPTED  = 1
    STATUS_EXECUTING = 2
    STATUS_CANCELING = 3
    STATUS_SUCCEEDED = 4
    STATUS_CANCELED  = 5
    STATUS_ABORTED   = 6
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Shared data types (plain Python — no rclpy dependency)
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    frame_id: str = "map"

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "qx": self.qx, "qy": self.qy, "qz": self.qz, "qw": self.qw,
            "frame_id": self.frame_id,
        }


@dataclass
class BatteryState:
    robot_id: str
    percentage: float        # 0.0–1.0
    power_supply_status: int  # 1=CHARGING 2=DISCHARGING 4=FULL


@dataclass
class RobotHealth:
    robot_id: str
    level: int               # 0=OK 1=WARN 2=ERROR
    estop_active: bool = False
    fault_codes: list[str] = field(default_factory=list)


@dataclass
class NavGoalStatus:
    goal_id: str
    robot_id: str
    status: int              # 4=SUCCEEDED 5=CANCELED 6=ABORTED
    number_of_recoveries: int = 0
    distance_remaining: float = 0.0


# ---------------------------------------------------------------------------
# Navigation action interface
# ---------------------------------------------------------------------------

class NavigationInterface(ABC):
    """
    Abstraction over Nav2's NavigateToPose action.
    The mock sim and the Isaac Sim adapter both implement this.
    """

    @abstractmethod
    def send_goal(
        self,
        robot_id: str,
        goal_id: str,
        destination: Pose,
        on_status_change: Callable[[NavGoalStatus], None],
    ) -> None:
        """
        Dispatch a NavigateToPose goal.  on_status_change is called whenever
        the action status changes (executing, succeeded, aborted, canceled).
        """
        ...

    @abstractmethod
    def cancel_goal(self, robot_id: str, goal_id: str) -> None:
        """Cancel an active goal."""
        ...


# ---------------------------------------------------------------------------
# Sensor publishing interface
# ---------------------------------------------------------------------------

class SensorInterface(ABC):
    """
    Abstraction over the topics the Aggregator subscribes to.
    In the mock sim these are generated programmatically; in Isaac Sim they
    come from actual ROS topics.
    """

    @abstractmethod
    def subscribe_battery(
        self, robot_id: str, callback: Callable[[BatteryState], None]
    ) -> None:
        ...

    @abstractmethod
    def subscribe_health(
        self, robot_id: str, callback: Callable[[RobotHealth], None]
    ) -> None:
        ...

    @abstractmethod
    def subscribe_pose(
        self, robot_id: str, callback: Callable[[Pose], None]
    ) -> None:
        ...

    @abstractmethod
    def subscribe_nav_status(
        self, robot_id: str, callback: Callable[[NavGoalStatus], None]
    ) -> None:
        ...
