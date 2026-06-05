"""
Global (shared) Nav2 pieces for the multi-robot keepout demo:
  map_server + costmap_filter_info_server (+ lifecycle manager).
One map, one keepout filter-info, one /keepout_filter_mask — shared by every
robot's namespaced costmap. Run ONCE.

    ros2 launch deploy/nav2/bringup_global.launch.py
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "nav2_keepout_demo.params.yaml")
FILTER_PARAMS = os.path.join(HERE, "keepout_costmap_filter_info.yaml")
USE_SIM_TIME = {"use_sim_time": True}


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(package="nav2_map_server", executable="map_server", name="map_server",
             output="screen", parameters=[PARAMS, USE_SIM_TIME]),
        Node(package="nav2_map_server", executable="costmap_filter_info_server",
             name="costmap_filter_info_server",
             output="screen", parameters=[FILTER_PARAMS, USE_SIM_TIME]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_global", output="screen",
             parameters=[{"use_sim_time": True, "autostart": True,
                          "node_names": ["map_server", "costmap_filter_info_server"]}]),
    ])
