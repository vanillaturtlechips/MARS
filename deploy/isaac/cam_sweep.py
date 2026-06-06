"""
Camera contact sheet for the REAL-aisle demo scene. Loads full_warehouse + the
3 robots + the obstacle box (in the real aisle x=-8, between the real shelves at
x=-10.5/-5.5) at a mid-scenario pose (R1 stuck at the box, R2/R3 rerouted up the
next aisle x=-3) and captures 10 camera angles to /tmp/cam_sweep/cam_1..10.png.

Look at the PNGs, pick the best cam_N, and tell me — that eye/target becomes the
demo camera (--cam-eye / --cam-target in isaac_multi_robot_ros2.py).

  source deploy/isaac/env_isaac.sh
  python deploy/isaac/cam_sweep.py
  # then view /tmp/cam_sweep/cam_*.png
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "enable_cameras": True})

import os  # noqa: E402
import numpy as np  # noqa: E402
import carb  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.sensors.camera")
simulation_app.update()

import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402

_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
IW_HUB = f"{_CLOUD}/Isaac/Robots/Idealworks/iwhub/iw_hub.usd"
WAREHOUSE = f"{_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
CARDBOX = f"{_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"
DOCK = (-8.0, 15.0)
# mid-scenario pose: R1 stuck at the box; R2,R3 rerouted up the next aisle x=-3
ROBOTS = [("R1", -8.0, 13.6), ("R2", -3.0, 16.0), ("R3", -3.0, 10.0)]

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(WAREHOUSE, "/World/Warehouse")

# obstacle: invisible collider + real cardbox visual + red keepout slab
world.scene.add(FixedCuboid(prim_path="/World/dock_collider", name="dock_collider",
                            position=np.array([DOCK[0], DOCK[1], 0.25]), scale=np.array([1.2, 0.8, 0.5])))
UsdGeom.Imageable(stage.GetPrimAtPath("/World/dock_collider")).MakeInvisible()
add_reference_to_stage(CARDBOX, "/World/dock_box")
UsdGeom.Xformable(stage.GetPrimAtPath("/World/dock_box")).AddTranslateOp().Set(Gf.Vec3d(DOCK[0], DOCK[1], 0.0))
world.scene.add(VisualCuboid(prim_path="/World/keepout_zone", name="keepout_zone",
                             position=np.array([DOCK[0], DOCK[1], 0.06]), scale=np.array([2.0, 2.0, 0.12]),
                             color=np.array([0.9, 0.05, 0.05])))
for name, x, y in ROBOTS:
    p = f"/World/{name}"
    add_reference_to_stage(IW_HUB, p)
    UsdGeom.Xformable(stage.GetPrimAtPath(p)).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

world.reset()
cam = Camera(prim_path="/World/SweepCam", resolution=(1280, 720))
cam.initialize()
cam.set_focal_length(24.0)
cam.set_horizontal_aperture(20.955)   # ~47 deg FOV
for _ in range(40):
    world.step(render=True)

# (eye, target) presets around the aisle x=-8 action (box at -8,15; shelves
# x=-10.5/-5.5; reroute aisle x=-3; robots came from the open south y<8.5).
PRESETS = [
    ((-8, 2, 3),    (-8, 18, 1.5)),   # 1: straight down aisle x=-8, low
    ((-8, 4, 7),    (-8, 20, 1.0)),   # 2: down aisle x=-8, mid height
    ((-5.5, -6, 14),(-5.5, 15, 1.0)), # 3: elevated overview of both aisles
    ((-1, -2, 7),   (-8, 15, 1.0)),   # 4: 3/4 from south-east
    ((-15, -2, 7),  (-6, 15, 1.0)),   # 5: 3/4 from south-west
    ((-3, 2, 4),    (-3, 18, 1.5)),   # 6: down the reroute aisle x=-3
    ((-8, 7, 9),    (-8, 22, 1.0)),   # 7: high, looking down aisle x=-8
    ((-18, 14, 6),  (-5, 15, 1.0)),   # 8: side view across the aisles
    ((-8, 10, 3),   (-8, 16, 1.0)),   # 9: close on the box / stuck robot
    ((-7, -10, 12), (-7, 14, 1.0)),   # 10: wide south overview
]

from PIL import Image  # noqa: E402
out = "/tmp/cam_sweep"
os.system(f"rm -rf {out} && mkdir -p {out}")
for i, (eye, tgt) in enumerate(PRESETS, 1):
    set_camera_view(eye=list(eye), target=list(tgt), camera_prim_path="/World/SweepCam")
    for _ in range(10):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is not None and rgba.size > 0:
        Image.fromarray(rgba[:, :, :3]).save(f"{out}/cam_{i}.png")
        carb.log_warn(f"[sweep] cam_{i}: eye={eye} target={tgt} -> {out}/cam_{i}.png")
    else:
        carb.log_warn(f"[sweep] cam_{i}: no image")

carb.log_warn(f"[sweep] done -> {out}/cam_1..10.png")
simulation_app.close()
