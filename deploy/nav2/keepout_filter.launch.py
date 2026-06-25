"""
Launch the Nav2 keepout costmap-filter info server (ROS2 Jazzy).

This brings up ONLY the costmap_filter_info_server (a lifecycle node) plus a
lifecycle manager to activate it.  The keepout MASK is published dynamically by
the MARS supervisor (ROS2SimAdapter.publish_keepout_mask) on /keepout_filter_mask,
so no static map_server for the mask is launched here.

Run (on RunPod, after sourcing ROS2 Jazzy + Nav2):
    source deploy/isaac/env_ros2.sh
    ros2 launch deploy/nav2/keepout_filter.launch.py

Then start the supervisor (mars.ros.ros2_node) and your Nav2 bringup whose
global_costmap includes the keepout_filter plugin (see README_keepout.md).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare  # noqa: F401  (optional)
import os


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    default_params = os.path.join(
        os.path.dirname(__file__), "keepout_costmap_filter_info.yaml"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file", default_value=default_params,
            description="costmap_filter_info_server params",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use Isaac Sim /clock",
        ),
        Node(
            package="nav2_map_server",
            executable="costmap_filter_info_server",
            name="costmap_filter_info_server",
            output="screen",
            emulate_tty=True,
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_costmap_filters",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["costmap_filter_info_server"],
            }],
        ),
    ])
