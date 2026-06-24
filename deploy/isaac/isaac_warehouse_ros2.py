"""
Step 1b — iw_hub in (optional) Simple_Warehouse, publishing ROS2 /clock,
/odom and /tf (map -> base_link).  No driving yet; we verify pose/odom
telemetry and discover the robot's joint names before adding cmd_vel.

Run (RunPod, after `source deploy/isaac/env_isaac.sh`, stale sims killed):
    python deploy/isaac/isaac_warehouse_ros2.py            # ground plane only (fast)
    python deploy/isaac/isaac_warehouse_ros2.py --warehouse  # + full_warehouse.usd (S3 download)
    python deploy/isaac/isaac_warehouse_ros2.py --warehouse --slam  # SLAM scan run

--slam: odom is published as odom->base_link (relative) instead of ground-truth
map->base_link, so slam_toolbox owns map->odom. The RTX lidar /scan is published
in both modes. Pair with: ros2 launch iw_hub_slam slam.launch.py

Verify (other shell, `source deploy/isaac/env_ros2.sh`):
    ros2 topic list                       # /clock /odom /tf /scan
    ros2 topic echo /scan --once          # LaserScan ranges
    ros2 run tf2_ros tf2_echo odom base_link
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse  # noqa: E402
import carb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--warehouse", action="store_true",
                help="load full_warehouse.usd (downloads from Isaac cloud)")
ap.add_argument("--obstacle", action="store_true",
                help="spawn a blocking box in receiving_dock (x~4) so a goal "
                     "there aborts — used to trigger a real failure")
ap.add_argument("--slam", action="store_true",
                help="publish odometry as odom->base_link (NOT ground-truth "
                     "map->base_link) so slam_toolbox can own map->odom. Also "
                     "needed for the lidar /scan to feed SLAM.")
args, _ = ap.parse_known_args()

import numpy as np  # noqa: E402  (lidar position, obstacle scale)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import omni.usd  # noqa: E402
import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
IW_HUB_USD   = f"{_ISAAC_CLOUD}/Isaac/Robots/Idealworks/iwhub/iw_hub.usd"
WAREHOUSE_USD = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

ROBOT_PRIM = "/World/iw_hub"

# SLAM mode: slam_toolbox owns map->odom, so Isaac must publish odom->base_link
# (relative odometry). Default (no --slam) keeps the ground-truth map->base_link
# shortcut used by the known-map / AMCL demos.
ODOM_PARENT = "odom" if args.slam else "map"

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

if args.warehouse:
    carb.log_warn("[1b] loading full_warehouse.usd (S3) ...")
    add_reference_to_stage(WAREHOUSE_USD, "/World/Warehouse")

carb.log_warn(f"[1b] loading iw_hub: {IW_HUB_USD}")
add_reference_to_stage(IW_HUB_USD, ROBOT_PRIM)

# Blocking box in receiving_dock (x~4) — robot can't reach a goal there, Nav2
# aborts (progress checker) -> a real failure for the supervisor to react to.
if args.obstacle:
    from isaacsim.core.api.objects import FixedCuboid
    world.scene.add(FixedCuboid(
        prim_path="/World/dock_block", name="dock_block",
        position=np.array([4.0, 0.0, 0.5]),
        scale=np.array([1.0, 3.0, 1.0]),
    ))
    carb.log_warn("[obstacle] spawned blocking box at (4,0) size (1,3,1)")

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
            ("PublishOdom.inputs:odomFrameId", ODOM_PARENT),
            ("PublishOdom.inputs:chassisFrameId", "base_link"),
            ("PublishRawTF.inputs:parentFrameId", ODOM_PARENT),
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
# ComputeOdometry needs to know which prim is the robot chassis. chassisPrim
# is a target/relationship input — set it with Isaac's set_target_prims helper
# (NOT og.Controller.attribute, which is for graph attributes).
try:
    from isaacsim.core.nodes.scripts.utils import set_target_prims
except ImportError:
    from omni.isaac.core_nodes.scripts.utils import set_target_prims
set_target_prims(
    primPath="/ActionGraph/ComputeOdom",
    inputName="inputs:chassisPrim",
    targetPrimPaths=[ROBOT_PRIM],
)

# ------------------------------------------------------------------
# Drive graph: /cmd_vel -> DifferentialController -> wheel joints.
# Separate graph + try/except so a node/port typo can't break the verified
# odom/tf graph above. iw_hub drive wheels: left_wheel_joint, right_wheel_joint.
# ------------------------------------------------------------------
WHEEL_RADIUS = 0.08    # m — tuned so cmd≈actual (0.125 gave 0.3cmd→0.19actual)
WHEEL_BASE   = 0.54    # m distance between drive wheels (approx)
try:
    og.Controller.edit(
        {"graph_path": "/DriveGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLin", "omni.graph.nodes.BreakVector3"),
                ("BreakAng", "omni.graph.nodes.BreakVector3"),
                ("DiffCtrl", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArtCtrl", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("SubTwist.inputs:topicName", "cmd_vel"),
                ("DiffCtrl.inputs:wheelRadius", WHEEL_RADIUS),
                ("DiffCtrl.inputs:wheelDistance", WHEEL_BASE),
                ("ArtCtrl.inputs:jointNames", ["left_wheel_joint", "right_wheel_joint"]),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "SubTwist.inputs:execIn"),
                ("OnTick.outputs:tick", "DiffCtrl.inputs:execIn"),
                ("OnTick.outputs:tick", "ArtCtrl.inputs:execIn"),
                ("Context.outputs:context", "SubTwist.inputs:context"),
                ("SubTwist.outputs:linearVelocity", "BreakLin.inputs:tuple"),
                ("SubTwist.outputs:angularVelocity", "BreakAng.inputs:tuple"),
                ("BreakLin.outputs:x", "DiffCtrl.inputs:linearVelocity"),
                ("BreakAng.outputs:z", "DiffCtrl.inputs:angularVelocity"),
                ("DiffCtrl.outputs:velocityCommand", "ArtCtrl.inputs:velocityCommand"),
            ],
        },
    )
    set_target_prims(
        primPath="/DriveGraph/ArtCtrl",
        inputName="inputs:targetPrim",
        targetPrimPaths=[ROBOT_PRIM],
    )
    carb.log_warn("[1c] drive graph ready: /cmd_vel -> diff drive (left/right_wheel_joint)")
except Exception as exc:  # noqa: BLE001
    carb.log_error(f"[1c] drive graph FAILED (odom/tf still OK): {exc}")

world.reset()

# ------------------------------------------------------------------
# RTX Lidar -> /scan (sensor_msgs/LaserScan).  frame_id=lidar_link; the
# slam.launch.py static_transform_publisher provides base_link->lidar_link.
# URDF lidar_joint origin: x=+0.35, y=0, z=+0.25 (chassis front, top).
#
# Created AFTER world.reset() and kept in a MODULE-LEVEL ref (LIDAR) on purpose:
#  - reset tears down render products, so building the lidar before it publishes
#    nothing ("hydratexture already released").
#  - LidarRtx.__init__ creates its OWN render product (128x128); we must use
#    lidar.get_render_product_path() and NOT make a second one.
#  - LidarRtx.__del__ destroys that render product, so if the object is GC'd
#    (e.g. not assigned to a variable) the sensor silently stops producing data.
# ------------------------------------------------------------------
LIDAR = None
_lidar_writer = None
try:
    from isaacsim.sensors.rtx import LidarRtx
    import omni.replicator.core as rep
    LIDAR = LidarRtx(
        prim_path=f"{ROBOT_PRIM}/lidar",
        name="iw_hub_lidar",
        position=np.array([0.35, 0.0, 0.25]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        config_file_name="RPLIDAR_S2E",   # 2D, 360°, 30 m
    )
    RP_PATH = LIDAR.get_render_product_path()
    carb.log_warn(f"[1d] lidar render product: {RP_PATH}")
    # Attach the ROS2 LaserScan writer to the lidar's OWN render product path
    # (string), exactly as OgnROS2RtxLidarHelper does internally.
    _lidar_writer = rep.writers.get("RtxLidarROS2PublishLaserScan")
    _lidar_writer.initialize(topicName="scan", frameId="lidar_link")
    _lidar_writer.attach([RP_PATH])
    carb.log_warn("[1d] lidar writer attached: /scan (frame=lidar_link)")
except Exception as exc:  # noqa: BLE001
    carb.log_error(f"[1d] lidar FAILED (odom/tf still OK): {exc}")

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
carb.log_warn(
    f"[1b] timeline playing; publishing /clock /odom /tf /scan "
    f"(odom parent={ODOM_PARENT}). Ctrl+C to stop.")

try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    pass
finally:
    timeline.stop()
    simulation_app.close()
