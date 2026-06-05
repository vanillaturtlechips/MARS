"""
Python shim for mars_msgs when running without a compiled ROS2 workspace.
Provides plain dataclasses that mirror the .msg definitions so the rest of
the code can `from mars.mars_msgs import RobotHealth` without rclpy.
"""
from dataclasses import dataclass, field


@dataclass
class Header:
    stamp_sec: int = 0
    stamp_nanosec: int = 0
    frame_id: str = ""


@dataclass
class RobotHealth:
    LEVEL_OK: int = 0
    LEVEL_WARN: int = 1
    LEVEL_ERROR: int = 2

    header: Header = field(default_factory=Header)
    robot_id: str = ""
    level: int = 0          # 0=OK 1=WARN 2=ERROR
    estop_active: bool = False
    fault_codes: list[str] = field(default_factory=list)
