"""
Camera contact sheet: load the warehouse + dock box + 3 robots at their spawns,
then capture ONE still from each of several camera presets to /tmp/cam_sweep/.
Look at the PNGs, pick the best preset #, and use its eye/target for the demo.

  source deploy/isaac/env_isaac.sh
  python deploy/isaac/cam_sweep.py
  # then view /tmp/cam_sweep/cam_*.png  (scp or open)
"""
import sys as _sys  # noqa
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "enable_cameras": True})

import os, math  # noqa: E402
import numpy as np  # noqa: E402
import carb  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.sensors.camera")
simulation_app.update()

import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402

_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
IW_HUB = f"{_CLOUD}/Isaac/Robots/Idealworks/iwhub/iw_hub.usd"
WAREHOUSE = f"{_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
ROBOTS = [("R1", -3.0, 5.0), ("R2", 0.0, 5.0), ("R3", 3.0, 5.0)]

# (eye, target) presets to try
PRESETS = [
    ((0, -11, 7),   (0, 1, 0.8)),    # 0: south, look north up the lane
    ((0, -12, 11),  (0, -1, 0.5)),   # 1: south, higher, look at dock
    ((0, -1.5, 14), (0, -1.5, 0)),   # 2: top-down (mid height)
    ((0, -1.5, 22), (0, -1.5, 0)),   # 3: top-down (high)
    ((4, -10, 8),   (0, -1, 0.8)),   # 4: 3/4 from +x/south corner
    ((-4, -10, 8),  (0, -1, 0.8)),   # 5: 3/4 from -x/south corner
]

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(WAREHOUSE, "/World/Warehouse")
world.scene.add(FixedCuboid(prim_path="/World/dock_block", name="dock_block",
                            position=np.array([0.0, 0.0, 0.5]),
                            scale=np.array([3.0, 1.5, 1.0])))
for name, x, y in ROBOTS:
    p = f"/World/{name}"
    add_reference_to_stage(IW_HUB, p)
    UsdGeom.Xformable(stage.GetPrimAtPath(p)).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

world.reset()
cam = Camera(prim_path="/World/SweepCam", resolution=(1280, 720))
cam.initialize()
for _ in range(30):
    world.step(render=True)

from PIL import Image  # noqa: E402
out = "/tmp/cam_sweep"
os.system(f"rm -rf {out} && mkdir -p {out}")
for i, (eye, tgt) in enumerate(PRESETS):
    set_camera_view(eye=list(eye), target=list(tgt), camera_prim_path="/World/SweepCam")
    for _ in range(8):
        world.step(render=True)
    rgba = cam.get_rgba()
    if rgba is not None and rgba.size > 0:
        Image.fromarray(rgba[:, :, :3]).save(f"{out}/cam_{i}.png")
        carb.log_warn(f"[sweep] cam_{i}: eye={eye} target={tgt} -> {out}/cam_{i}.png")
    else:
        carb.log_warn(f"[sweep] cam_{i}: no image")

carb.log_warn(f"[sweep] done. view {out}/cam_*.png")
simulation_app.close()
