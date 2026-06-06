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
# place robots where the ACTION is (clustered at the dock), not at north spawn,
# so the stills show the real moment: R2/R3 stuck at the box, R1 mid-detour.
ROBOTS = [("R1", -1.8, 0.8), ("R2", 0.0, 0.6), ("R3", 1.3, -0.2)]

# (eye, target) presets — closer + lower so the small dock + robots fill the frame
PRESETS = [
    ((0, -6, 3.5),  (0, -0.3, 0.4)),  # 0: south, low, close
    ((0, -8, 5),    (0, -0.3, 0.4)),  # 1: south, mid
    ((3.5, -6, 4),  (0, -0.3, 0.4)),  # 2: slight SE corner, low
    ((-3.5, -6, 4), (0, -0.3, 0.4)),  # 3: slight SW corner, low
    ((0, -1, 9),    (0, -0.5, 0)),    # 4: top-down (lower, z=9)
    ((0, -10, 7),   (0, -0.3, 0.5)),  # 5: south overview
]

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(WAREHOUSE, "/World/Warehouse")
world.scene.add(FixedCuboid(prim_path="/World/dock_block", name="dock_block",
                            position=np.array([0.0, 0.0, 0.25]),
                            scale=np.array([1.2, 0.8, 0.5])))   # match the demo's small dock box
for name, x, y in ROBOTS:
    p = f"/World/{name}"
    add_reference_to_stage(IW_HUB, p)
    UsdGeom.Xformable(stage.GetPrimAtPath(p)).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

world.reset()
cam = Camera(prim_path="/World/SweepCam", resolution=(1280, 720))
cam.initialize()
# default aperture came out ~2.1 (≈5° telephoto = everything looks zoomed in).
# set the standard ~20.955 aperture -> ~47° FOV so the whole scene fits.
cam.set_focal_length(24.0)
cam.set_horizontal_aperture(20.955)
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
