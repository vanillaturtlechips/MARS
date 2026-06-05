"""
Step 1b — iw_hub in (optional) Simple_Warehouse, publishing ROS2 /clock,
/odom and /tf (map -> base_link).  No driving yet; we verify pose/odom
telemetry and discover the robot's joint names before adding cmd_vel.

Run (RunPod, after `source deploy/isaac/env_isaac.sh`, stale sims killed):
    python deploy/isaac/isaac_warehouse_ros2.py            # ground plane only (fast)
    python deploy/isaac/isaac_warehouse_ros2.py --warehouse  # + full_warehouse.usd (S3 download)

Verify (other shell, `source deploy/isaac/env_ros2.sh`):
    ros2 topic list                       # /clock /odom /tf
    ros2 topic echo /odom --once          # pose ~ (0,0)
    ros2 run tf2_ros tf2_echo map base_link
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse  # noqa: E402
import carb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--warehouse", action="store_true",
                help="load full_warehouse.usd (downloads from Isaac cloud)")
args, _ = ap.parse_known_args()

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import omni.usd  # noqa: E402
import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
IW_HUB_USD   = f"{_ISAAC_CLOUD}/Isaac/Robots/Idealworks/iwhub/iw_hub.usd"
WAREHOUSE_USD = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

ROBOT_PRIM = "/World/iw_hub"

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

if args.warehouse:
    carb.log_warn("[1b] loading full_warehouse.usd (S3) ...")
    add_reference_to_stage(WAREHOUSE_USD, "/World/Warehouse")

carb.log_warn(f"[1b] loading iw_hub: {IW_HUB_USD}")
add_reference_to_stage(IW_HUB_USD, ROBOT_PRIM)

# ------------------------------------------------------------------
# OmniGraph: clock + odometry(map->base_link) publishing
# ------------------------------------------------------------------
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("PublishRawTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("PublishOdom.inputs:odomFrameId", "map"),
            ("PublishOdom.inputs:chassisFrameId", "base_link"),
            ("PublishRawTF.inputs:parentFrameId", "map"),
            ("PublishRawTF.inputs:childFrameId", "base_link"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("OnTick.outputs:tick", "ComputeOdom.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishOdom.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishRawTF.inputs:execIn"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            ("Context.outputs:context", "PublishOdom.inputs:context"),
            ("Context.outputs:context", "PublishRawTF.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishRawTF.inputs:timeStamp"),
            ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
            ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
            ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
            ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
            ("ComputeOdom.outputs:position", "PublishRawTF.inputs:translation"),
            ("ComputeOdom.outputs:orientation", "PublishRawTF.inputs:rotation"),
        ],
    },
)
# ComputeOdometry needs to know which prim is the robot chassis.
og.Controller.edit(
    "/ActionGraph",
    {og.Controller.Keys.SET_VALUES: [
        ("ComputeOdom.inputs:chassisPrim", [og.Controller.attribute(f"{ROBOT_PRIM}")]),
    ]},
)

world.reset()

# Discover joints (for the upcoming cmd_vel / differential-drive step).
stage = omni.usd.get_context().get_stage()
carb.log_warn("[1b] ===== joints under the robot =====")
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.startswith(ROBOT_PRIM) and ("Joint" in prim.GetTypeName() or "joint" in prim.GetName().lower()):
        carb.log_warn(f"[1b] joint: {p}  ({prim.GetTypeName()})")
carb.log_warn("[1b] ===================================")

import omni.timeline  # noqa: E402
timeline = omni.timeline.get_timeline_interface()
timeline.play()
carb.log_warn("[1b] timeline playing; publishing /clock /odom /tf. Ctrl+C to stop.")

try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    pass
finally:
    timeline.stop()
    simulation_app.close()
