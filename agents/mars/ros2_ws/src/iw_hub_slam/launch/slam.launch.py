from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CONFIG = str(Path(__file__).parent.parent / "config" / "slam_params.yaml")


def generate_launch_description():
    ns_arg = DeclareLaunchArgument(
        "namespace", default_value="",
        description="로봇 네임스페이스 (예: R1). 비워두면 단일 로봇 모드."
    )
    ns = LaunchConfiguration("namespace")

    slam_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            CONFIG,
            # namespace가 있으면 scan_topic을 /{ns}/scan 으로 remap
            {"scan_topic": "/scan"},
        ],
        remappings=[
            # namespace 지정 시: ros2 launch ... namespace:=R1
            # → 이 remapping은 launch argument로 동적 처리 필요 시 확장
        ],
    )

    return LaunchDescription([ns_arg, slam_node])
