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
# cam_5 from cam_sweep (south overview): frames the dock + 3 robots + detour well.
# south view looking NORTH up the aisle (x=8) between the shelves (x=3.5/12.5);
# frames the box at (8,13), the robots, and the left detour at x=6.
ap.add_argument("--cam-eye", type=str, default="8,-10,10", help="record camera position x,y,z")
ap.add_argument("--cam-target", type=str, default="8,12,1", help="record camera look-at x,y,z")
ap.add_argument("--fps", type=int, default=20)
args, _ = ap.parse_known_args()

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
if args.record:
    enable_extension("isaacsim.sensors.camera")
simulation_app.update()

import os  # noqa: E402
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
# real NVIDIA props (no standalone shelf asset exists — the shelves come from
# full_warehouse itself; these are the obstacle box + the charging station).
CARDBOX_USD  = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"
CHARGER_USD  = f"{_ISAAC_CLOUD}/Isaac/Props/PackingTable/packing_table.usd"
RACK_USD     = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_06.usd"  # the stocked shelf full_warehouse itself uses

# Coordinates read off the occupancy map. The only two CLEAR straight aisles
# between the real shelves are x≈8 (flanked by shelves x≈5.5 & 10.4) and x≈13
# (flanked by x≈10.4 & 15.4); both run y≈9..18. Robots start in the open band
# just south of the shelf fronts and drive UP into the aisles.
#   demo1: R1 goes up aisle-8, rams the box (8,13) BETWEEN the shelves and is
#   stuck; the agent bans aisle-8; R2,R3 are rerouted up the clear aisle-13.
ROBOTS = [("R1", 8.0, 5.0), ("R2", 11.0, 5.0), ("R3", 5.0, 5.0)]
DOCK_XY = (8.0, 13.0)       # obstacle box / keepout center — inside aisle-8, between the shelves
CHARGER_XY = (12.0, 3.0)    # charging station in the open band (south of the shelves)
WHEEL_RADIUS = 0.08
WHEEL_BASE = 0.54

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

if args.warehouse:
    carb.log_warn("[multi] loading full_warehouse.usd ...")
    add_reference_to_stage(WAREHOUSE_USD, "/World/Warehouse")

# ---- demo props: SAME scene for all three demos (shelves come from the
# warehouse; here we add the obstacle box + the charging station) ----
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid

def _place(prim_path: str, usd: str, x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> None:
    add_reference_to_stage(usd, prim_path)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
    if yaw:
        xf.AddRotateZOp().Set(float(yaw))

_dx, _dy = DOCK_XY
# Obstacle: an INVISIBLE collider (guarantees the robot physically stalls -> a
# real follow_path abort, like the original FixedCuboid) with the REAL NVIDIA
# cardboard box as the visible obstacle on top. Nav2 has no obstacle layer, so
# the box is invisible to planning until the agent publishes the keepout mask.
world.scene.add(FixedCuboid(
    prim_path="/World/dock_collider", name="dock_collider",
    position=np.array([_dx, _dy, 0.25]), scale=np.array([1.2, 0.8, 0.5]),
))
UsdGeom.Imageable(stage.GetPrimAtPath("/World/dock_collider")).MakeInvisible()
_place("/World/dock_box", CARDBOX_USD, _dx, _dy, 0.0)
carb.log_warn(f"[multi] obstacle: real SM_CardBoxA at {DOCK_XY} (collider 1.2x0.8x0.5)")

# Stocked shelves (SM_RackPile_06 — the asset full_warehouse itself uses) flanking
# the aisle so the obstacle box sits BETWEEN real shelving. We place our own at
# known coords (the occupancy map's shelf positions didn't match the render).
# Rows at x=3.5 and x=12.5 (a wide, safe aisle); the robot lane is x=8 and the
# left detour is x=6, both well clear of the shelves. yaw=90 turns each rack so
# its long side runs ALONG the aisle (forming a wall).
for _ry in (10.0, 13.0, 16.0):
    _place(f"/World/shelf_L_{int(_ry)}", RACK_USD, 3.0, _ry, 0.0, yaw=90.0)
    _place(f"/World/shelf_R_{int(_ry)}", RACK_USD, 13.0, _ry, 0.0, yaw=90.0)
carb.log_warn("[multi] placed SM_RackPile shelves flanking aisle x=8 (box between them)")

# Charging station: real NVIDIA packing table at the R3-lane mouth.
_place("/World/charger", CHARGER_USD, *CHARGER_XY, 0.0)
carb.log_warn(f"[multi] charging station: real packing_table at {CHARGER_XY}")

# Small red keepout slab (1.6x1.6) over the receiving_dock polygon center (3,3).
# Hidden until the agent declares avoid_zone (runner touches /tmp/keepout_zone_go
# — only demo1 does this, so it stays hidden in the charging demos).
world.scene.add(VisualCuboid(
    prim_path="/World/keepout_zone", name="keepout_zone",
    position=np.array([_dx, _dy, 0.06]), scale=np.array([2.0, 2.0, 0.12]),
    color=np.array([0.9, 0.05, 0.05]),
))
UsdGeom.Imageable(stage.GetPrimAtPath("/World/keepout_zone")).MakeInvisible()
carb.log_warn("[multi] keepout slab 2x2 created in aisle-8 (hidden until avoid_zone)")


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
    # one-time TOP-DOWN calibration still of the whole scene (warehouse + robots at
    # their known spawns + shelves + box) so the world<->render layout can be read
    # directly instead of guessed. Saved once, then the demo camera is restored.
    from PIL import Image as _ImgTD
    set_camera_view(eye=[0.0, 0.0, 60.0], target=[0.0, 0.0, 0.0], camera_prim_path="/World/RecordCam")
    for _ in range(15):
        world.step(render=True)
    _td = _cam.get_rgba()
    if _td is not None and _td.size > 0:
        _ImgTD.fromarray(_td[:, :, :3]).save("/tmp/keepout_topdown.png")
        carb.log_warn("[multi] saved top-down calibration -> /tmp/keepout_topdown.png")
    set_camera_view(eye=eye, target=tgt, camera_prim_path="/World/RecordCam")
    for _ in range(15):
        world.step(render=True)
    _frame_dir = "/tmp/keepout_frames"          # fixed path so the runner can encode it
    os.system(f"rm -rf {_frame_dir} && mkdir -p {_frame_dir}")
    _rec = {"cam": _cam, "dir": _frame_dir, "i": 0}
    carb.log_warn(f"[multi] recording -> {args.record}  (frames in {_frame_dir})")

# readiness marker for the runner (file, not buffered log). written once Isaac is
# up + playing — BEFORE the runner brings up Nav2, so Nav2 starts without camera load.
open("/tmp/keepout_isaac_ready", "w").close()

_step = 0
# reveal the red keepout slab once the agent declares avoid_zone; the runner
# touches this marker right after the bridge logs avoid_zone=True (demo1 only,
# so the slab stays hidden in the charging demos).
_keepout_pending = True
try:
    while simulation_app.is_running():
        world.step(render=True)
        if _keepout_pending and os.path.exists("/tmp/keepout_zone_go"):
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/keepout_zone")).MakeVisible()
            _keepout_pending = False
            carb.log_warn("[multi] keepout zone revealed (agent avoid_zone active)")
        if _rec is not None:
            _step += 1
            # only capture once the runner touches the GO marker (right before goals),
            # so the heavy camera render doesn't starve Nav2's lifecycle handshakes.
            import os as _os
            if _step % 3 == 0 and _os.path.exists("/tmp/keepout_record_go"):
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
