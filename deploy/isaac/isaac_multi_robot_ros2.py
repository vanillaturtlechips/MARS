"""
Multi-robot Isaac scene (R1, R2, R3) for the Nav2 keepout demo.

Each robot is an iw_hub, namespaced:
  /<R>/cmd_vel   (subscribe) -> differential drive -> wheels
  /<R>/odom      (publish, frame map -> <R>/base_link)
  /tf            (global) map -> <R>/base_link
  /clock         (shared)
A blocking box sits in receiving_dock (x~4) so goals there abort -> real
zone-wide failures (no seeding needed).

Run (RunPod, `source deploy/isaac/env_isaac.sh`, stale sims killed):
    python deploy/isaac/isaac_multi_robot_ros2.py [--warehouse]

Verify (other shell, env_ros2.sh):
    ros2 topic list                 # /R1/odom /R2/odom /R3/odom /clock /tf
    ros2 topic echo /R2/odom --once
    ros2 run tf2_ros tf2_echo map R2/base_link
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse  # noqa: E402
import carb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--warehouse", action="store_true")
ap.add_argument("--no-obstacle", action="store_true",
                help="skip the dock blocking box (default: spawn it)")
args, _ = ap.parse_known_args()

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402

try:
    from isaacsim.core.nodes.scripts.utils import set_target_prims
except ImportError:
    from omni.isaac.core_nodes.scripts.utils import set_target_prims

_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
IW_HUB_USD   = f"{_ISAAC_CLOUD}/Isaac/Robots/Idealworks/iwhub/iw_hub.usd"
WAREHOUSE_USD = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# (name, spawn x, spawn y) — a ROW across x at the north end; all three drive
# south (-y) down separate 3 m-apart lanes so they never touch (no obstacle
# layer = robots are invisible to each other). The keepout wall is dropped
# across the middle of the lanes; robots whose lane it blocks detour around it.
ROBOTS = [("R1", -3.0, 5.0), ("R2", 0.0, 5.0), ("R3", 3.0, 5.0)]
WHEEL_RADIUS = 0.08
WHEEL_BASE = 0.54

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

if args.warehouse:
    carb.log_warn("[multi] loading full_warehouse.usd ...")
    add_reference_to_stage(WAREHOUSE_USD, "/World/Warehouse")

if not args.no_obstacle:
    from isaacsim.core.api.objects import FixedCuboid
    world.scene.add(FixedCuboid(
        prim_path="/World/dock_block", name="dock_block",
        position=np.array([4.0, 0.0, 0.5]),
        scale=np.array([1.0, 5.0, 1.0]),   # spans the dock width (y -2.5..2.5)
    ))
    carb.log_warn("[multi] spawned dock blocking box (4,0) size (1,5,1)")


def spawn_robot(name: str, x: float, y: float) -> str:
    prim_path = f"/World/{name}"
    add_reference_to_stage(IW_HUB_USD, prim_path)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.0))
    carb.log_warn(f"[multi] spawned {name} at ({x},{y})")
    return prim_path


def build_robot_graph(name: str, prim_path: str, sx: float, sy: float) -> None:
    ns = f"/{name}"
    g = f"/{name}_Graph"
    og.Controller.edit(
        {"graph_path": g, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("PublishRawTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLin", "omni.graph.nodes.BreakVector3"),
                ("BreakAng", "omni.graph.nodes.BreakVector3"),
                ("DiffCtrl", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArtCtrl", "isaacsim.core.nodes.IsaacArticulationController"),
                # IsaacComputeOdometry zeroes at the robot's spawn, so the raw
                # tf would say every robot is at (0,0). Add the spawn offset so
                # map->base_link is the TRUE global pose (Nav2 has no odom frame).
                ("OdomOffset", "omni.graph.nodes.Add"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("OdomOffset.inputs:b", [float(sx), float(sy), 0.0]),
                ("PublishOdom.inputs:nodeNamespace", ns),
                ("PublishOdom.inputs:odomFrameId", "map"),
                ("PublishOdom.inputs:chassisFrameId", f"{name}/base_link"),
                ("PublishRawTF.inputs:parentFrameId", "map"),
                ("PublishRawTF.inputs:childFrameId", f"{name}/base_link"),
                ("SubTwist.inputs:nodeNamespace", ns),
                ("SubTwist.inputs:topicName", "cmd_vel"),
                ("DiffCtrl.inputs:wheelRadius", WHEEL_RADIUS),
                ("DiffCtrl.inputs:wheelDistance", WHEEL_BASE),
                ("ArtCtrl.inputs:jointNames", ["left_wheel_joint", "right_wheel_joint"]),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishOdom.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishRawTF.inputs:execIn"),
                ("OnTick.outputs:tick", "SubTwist.inputs:execIn"),
                ("OnTick.outputs:tick", "DiffCtrl.inputs:execIn"),
                ("OnTick.outputs:tick", "ArtCtrl.inputs:execIn"),
                ("Context.outputs:context", "PublishOdom.inputs:context"),
                ("Context.outputs:context", "PublishRawTF.inputs:context"),
                ("Context.outputs:context", "SubTwist.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishRawTF.inputs:timeStamp"),
                ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
                ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
                ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
                ("ComputeOdom.outputs:position", "OdomOffset.inputs:a"),
                ("OdomOffset.outputs:sum", "PublishRawTF.inputs:translation"),
                ("ComputeOdom.outputs:orientation", "PublishRawTF.inputs:rotation"),
                ("SubTwist.outputs:linearVelocity", "BreakLin.inputs:tuple"),
                ("SubTwist.outputs:angularVelocity", "BreakAng.inputs:tuple"),
                ("BreakLin.outputs:x", "DiffCtrl.inputs:linearVelocity"),
                ("BreakAng.outputs:z", "DiffCtrl.inputs:angularVelocity"),
                ("DiffCtrl.outputs:velocityCommand", "ArtCtrl.inputs:velocityCommand"),
            ],
        },
    )
    set_target_prims(primPath=f"{g}/ComputeOdom", inputName="inputs:chassisPrim",
                     targetPrimPaths=[prim_path])
    set_target_prims(primPath=f"{g}/ArtCtrl", inputName="inputs:targetPrim",
                     targetPrimPaths=[prim_path])
    carb.log_warn(f"[multi] graph ready for {name} ({ns}/cmd_vel, {ns}/odom)")


# Shared /clock
og.Controller.edit(
    {"graph_path": "/ClockGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ],
    },
)

for _name, _x, _y in ROBOTS:
    _prim = spawn_robot(_name, _x, _y)
    build_robot_graph(_name, _prim, _x, _y)

world.reset()

import omni.timeline  # noqa: E402
timeline = omni.timeline.get_timeline_interface()
timeline.play()
carb.log_warn(f"[multi] timeline playing; {len(ROBOTS)} robots publishing. Ctrl+C to stop.")

try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    pass
finally:
    timeline.stop()
    simulation_app.close()
