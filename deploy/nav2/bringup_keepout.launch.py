"""
Nav2 bring-up for the Isaac iw_hub keepout demo (ROS2 Humble).

Launches exactly the servers configured in nav2_keepout_demo.params.yaml:
  map_server, costmap_filter_info_server, planner_server, controller_server,
  behavior_server, bt_navigator  (+ one lifecycle manager).

No AMCL (Isaac publishes tf map->base_link directly). The KeepoutFilter on the
global costmap reads /costmap_filter_info and /keepout_filter_mask; with no mask
published it is inert (2a: free-space nav), and turns on when the MARS supervisor
publishes a mask (2b).

Run (RunPod, after `source deploy/isaac/env_ros2.sh`):
    cd /workspace/MARS/deploy/nav2
    python make_empty_map.py            # once
    ros2 launch /workspace/MARS/deploy/nav2/bringup_keepout.launch.py
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "nav2_keepout_demo.params.yaml")
FILTER_PARAMS = os.path.join(HERE, "keepout_costmap_filter_info.yaml")

USE_SIM_TIME = {"use_sim_time": True}

LIFECYCLE_NODES = [
    "map_server",
    "costmap_filter_info_server",
    "planner_server",
    "controller_server",
    "behavior_server",
    "bt_navigator",
]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package="nav2_map_server", executable="map_server", name="map_server",
            output="screen", parameters=[PARAMS, USE_SIM_TIME],
        ),
        Node(
            package="nav2_map_server", executable="costmap_filter_info_server",
            name="costmap_filter_info_server",
            output="screen", parameters=[FILTER_PARAMS, USE_SIM_TIME],
        ),
        Node(
            package="nav2_planner", executable="planner_server", name="planner_server",
            output="screen", parameters=[PARAMS, USE_SIM_TIME],
        ),
        Node(
            package="nav2_controller", executable="controller_server", name="controller_server",
            output="screen", parameters=[PARAMS, USE_SIM_TIME],
        ),
        Node(
            package="nav2_behaviors", executable="behavior_server", name="behavior_server",
            output="screen", parameters=[PARAMS, USE_SIM_TIME],
        ),
        Node(
            package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator",
            output="screen", parameters=[PARAMS, USE_SIM_TIME],
        ),
        Node(
            package="nav2_lifecycle_manager", executable="lifecycle_manager",
            name="lifecycle_manager_nav",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "autostart": True,
                "node_names": LIFECYCLE_NODES,
            }],
        ),
    ])
