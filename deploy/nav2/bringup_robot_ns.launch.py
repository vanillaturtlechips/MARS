"""
Per-robot namespaced Nav2 stack for the multi-robot keepout demo.

Launches planner/controller/behavior/bt_navigator (+ lifecycle) under a robot
namespace (e.g. /R2), with robot_base_frame = <ns>/base_link. Topics get the
namespace (/<ns>/cmd_vel, /<ns>/navigate_to_pose); tf is global; the static map
(/map) and keepout filter info (/costmap_filter_info) are shared globals.

Run the global pieces first (bringup_global.launch.py), then one per robot:
    ros2 launch deploy/nav2/bringup_robot_ns.launch.py namespace:=R2
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "nav2_keepout_demo.params.yaml")

# tf must stay global for all namespaced nodes.
TF_REMAPS = [("/tf", "/tf"), ("/tf_static", "/tf_static")]
# static_layer subscribes to "map"; point it at the global /map.
MAP_REMAP = [("map", "/map")]

LIFECYCLE_NODES = ["planner_server", "controller_server", "behavior_server", "bt_navigator"]


def generate_launch_description() -> LaunchDescription:
    ns = LaunchConfiguration("namespace")

    configured = RewrittenYaml(
        source_file=PARAMS,
        root_key=ns,
        param_rewrites={"robot_base_frame": [ns, "/base_link"]},
        convert_types=True,
    )
    p = [configured, {"use_sim_time": True}]

    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="R2"),
        Node(package="nav2_planner", executable="planner_server", name="planner_server",
             namespace=ns, output="screen", parameters=p, remappings=TF_REMAPS + MAP_REMAP),
        Node(package="nav2_controller", executable="controller_server", name="controller_server",
             namespace=ns, output="screen", parameters=p, remappings=TF_REMAPS),
        Node(package="nav2_behaviors", executable="behavior_server", name="behavior_server",
             namespace=ns, output="screen", parameters=p, remappings=TF_REMAPS),
        Node(package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator",
             namespace=ns, output="screen", parameters=p, remappings=TF_REMAPS),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager", namespace=ns, output="screen",
             parameters=[{"use_sim_time": True, "autostart": True,
                          "node_names": LIFECYCLE_NODES}]),
    ])
