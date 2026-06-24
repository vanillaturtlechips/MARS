from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command

URDF = str(Path(__file__).parent.parent / "urdf" / "iw_hub.urdf")


def generate_launch_description():
    robot_description = ParameterValue(
        Command(["cat ", URDF]),
        value_type=str,
    )

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(Path(__file__).parent.parent / "rviz" / "iw_hub.rviz")],
        ),
    ])
