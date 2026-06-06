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
import sys as _sys
from isaacsim import SimulationApp

# Camera/RTX recording (--record) needs the camera render pipeline enabled at
# app launch; with only {"headless": True} the offscreen camera hangs at NGX/RTX
# init. enable_cameras (what IsaacLab's render scripts set) fixes it. Off for
# non-record runs so the plain path stays light.
simulation_app = SimulationApp({"headless": True, "enable_cameras": "--record" in _sys.argv})

import argparse  # noqa: E402
import carb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--warehouse", action="store_true")
ap.add_argument("--no-obstacle", action="store_true",
                help="skip the dock blocking box (default: spawn it)")
ap.add_argument("--record", type=str, default="",
                help="if set, capture an offscreen camera to this .mp4 (headless, like the RL render scripts)")
# default cam sits INSIDE the verified-open lane volume (robots run x[-3,3] y[-8,5])
# so it never lands in a warehouse wall; elevated, looking north up the lane.
ap.add_argument("--cam-eye", type=str, default="0,-11,7", help="record camera position x,y,z")
ap.add_argument("--cam-target", type=str, default="0,1,0.8", help="record camera look-at x,y,z")
ap.add_argument("--fps", type=int, default=20)
args, _ = ap.parse_known_args()

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
if args.record:
    enable_extension("isaacsim.sensors.camera")
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
        position=np.array([0.0, 0.0, 0.5]),
        scale=np.array([3.0, 1.5, 1.0]),   # receiving_dock at origin: x[-1.5,1.5] y[-0.75,0.75]
    ))
    carb.log_warn("[multi] spawned dock blocking box (0,0) size (3,1.5,1)")


def spawn_robot(name: str, x: float, y: float) -> str:
    prim_path = f"/World/{name}"
    add_reference_to_stage(IW_HUB_USD, prim_path)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.0))
    carb.log_warn(f"[multi] spawned {name} at ({x},{y})")
    return prim_path


def build_robot_graph(name: str, prim_path: str) -> None:
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
            ],
            og.Controller.Keys.SET_VALUES: [
                # IsaacComputeOdometry zeroes at spawn, so publish odom->base_link
                # (per-robot odom frame). A static_transform_publisher supplies the
                # map->{name}/odom spawn offset; tf2 chains map->base_link for Nav2.
                ("PublishOdom.inputs:nodeNamespace", ns),
                ("PublishOdom.inputs:odomFrameId", f"{name}/odom"),
                ("PublishOdom.inputs:chassisFrameId", f"{name}/base_link"),
                ("PublishRawTF.inputs:parentFrameId", f"{name}/odom"),
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
                ("ComputeOdom.outputs:position", "PublishRawTF.inputs:translation"),
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
    build_robot_graph(_name, _prim)

world.reset()

import omni.timeline  # noqa: E402
timeline = omni.timeline.get_timeline_interface()
timeline.play()
carb.log_warn(f"[multi] timeline playing; {len(ROBOTS)} robots publishing. Ctrl+C to stop.")

# ---- offscreen recording: headless camera -> PNG frames -> ffmpeg mp4 (RL-style) ----
_rec = None
if args.record:
    import os, math, tempfile, subprocess  # noqa: E402
    import numpy as _np  # noqa: E402
    from isaacsim.sensors.camera import Camera  # noqa: E402

    def _xyz(s):
        return [float(v) for v in s.split(",")]

    def _lookat_quat(eye, tgt):
        e = _np.array(eye, float); t = _np.array(tgt, float)
        f = t - e; f /= (_np.linalg.norm(f) + 1e-9)
        up = _np.array([0.0, 0.0, 1.0])
        r = _np.cross(f, up)
        if _np.linalg.norm(r) < 1e-6:
            up = _np.array([0.0, 1.0, 0.0]); r = _np.cross(f, up)
        r /= (_np.linalg.norm(r) + 1e-9)
        u = _np.cross(r, f)
        m = _np.array([[r[0], u[0], -f[0]], [r[1], u[1], -f[1]], [r[2], u[2], -f[2]]])
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2; w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
        return [w, x, y, z]

    from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
    eye = _xyz(args.cam_eye); tgt = _xyz(args.cam_target)
    _cam = Camera(prim_path="/World/RecordCam", resolution=(1280, 720))
    _cam.initialize()
    _cam.set_focal_length(24.0)          # default aperture is ~2.1 (≈5° telephoto);
    _cam.set_horizontal_aperture(20.955) # standard ~20.955 -> ~47° FOV, whole scene fits
    set_camera_view(eye=eye, target=tgt, camera_prim_path="/World/RecordCam")  # correct aim
    for _ in range(20):
        world.step(render=True)
    _frame_dir = "/tmp/keepout_frames"          # fixed path so the runner can encode it
    os.system(f"rm -rf {_frame_dir} && mkdir -p {_frame_dir}")
    _rec = {"cam": _cam, "dir": _frame_dir, "i": 0}
    carb.log_warn(f"[multi] recording -> {args.record}  (frames in {_frame_dir})")

_step = 0
try:
    while simulation_app.is_running():
        world.step(render=True)
        if _rec is not None:
            _step += 1
            if _step % 3 == 0:                         # ~throttle to a third of sim rate
                rgba = _rec["cam"].get_rgba()
                if rgba is not None and rgba.size > 0:
                    from PIL import Image as _Img
                    _Img.fromarray(rgba[:, :, :3]).save(
                        os.path.join(_rec["dir"], f"frame_{_rec['i']:06d}.png"))
                    _rec["i"] += 1
except KeyboardInterrupt:
    pass
finally:
    if _rec is not None and _rec["i"] > 0:
        carb.log_warn(f"[multi] encoding {_rec['i']} frames -> {args.record}")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                        "-i", os.path.join(_rec["dir"], "frame_%06d.png"),
                        "-pix_fmt", "yuv420p", args.record])
    timeline.stop()
    simulation_app.close()
