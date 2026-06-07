"""
Camera contact sheet for the CHARGING demo scene (demo2/3). Loads full_warehouse +
the packing-table charger at (0,3) + the green charge pad at the dock (0,5) + the 3
robots in a mid-scenario charging pose (R1 on the pad charging, R2 waiting at its
park, R3 approaching from the aisle), and captures camera angles to
/tmp/cam_sweep/charge_1..N.png.

Look at the PNGs, pick the best charge_N, tell me — that eye/target becomes the
charge-demo camera (--cam-eye=/--cam-target= in run_all_demos.sh charge_demo).

  source deploy/isaac/env_isaac.sh
  python deploy/isaac/cam_sweep_charge.py
  # then view /tmp/cam_sweep/charge_*.png
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
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402

_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
IW_HUB = f"{_CLOUD}/Isaac/Robots/Idealworks/iwhub/iw_hub.usd"
WAREHOUSE = f"{_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
CHARGER = f"{_CLOUD}/Isaac/Props/PackingTable/packing_table.usd"

CHARGER_XY = (0.0, 3.0)      # packing table
PAD = (0.0, 5.0)             # green charge pad / dock (robots pull in here)
# mid-scenario charging pose: R1 on the pad (charging), R2 waiting at its park,
# R3 approaching the pad from the aisle side.
ROBOTS = [("R1", 0.0, 5.0), ("R2", 3.0, 3.0), ("R3", -4.0, 6.0)]

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(WAREHOUSE, "/World/Warehouse")

# charging station = green charge pad only (the packing-table stand-in read as a
# work table, not a charger, so it's dropped).
world.scene.add(VisualCuboid(prim_path="/World/charge_pad", name="charge_pad",
                             position=np.array([PAD[0], PAD[1], 0.02]),
                             scale=np.array([1.8, 1.8, 0.04]),
                             color=np.array([0.05, 0.85, 0.25])))
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

# (eye, target) presets around the charging area (pad/charger at x=0, y=3-5, open
# floor south of the shelves at y>=8.5; robots approach from the aisle x=-8 north).
PRESETS = [
    ((0, -6, 4),    (0, 4, 0.8)),     # 1: south, low, straight at the pad
    ((0, -9, 8),    (0, 4, 0.6)),     # 2: south, mid height
    ((-3, -10, 11), (-3, 4, 0.6)),    # 3: elevated SW overview (pad + approach aisle)
    ((6, -7, 7),    (0, 4, 0.8)),     # 4: 3/4 from south-east
    ((-6, -7, 7),   (0, 4, 0.8)),     # 5: 3/4 from south-west
    ((0, -4, 3),    (0, 4, 0.9)),     # 6: close on the pad/charger
    ((-2, -11, 13), (-2, 4, 0.5)),    # 7: high wide south overview
    ((8, 0, 6),     (0, 4, 0.8)),     # 8: from the east side
    ((-8, -8, 9),   (-3, 5, 0.6)),    # 9: catches the aisle-mouth approach + pad
    ((0, -12, 9),   (0, 3, 0.6)),     # 10: far south, charger centered, wide
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
        Image.fromarray(rgba[:, :, :3]).save(f"{out}/charge_{i}.png")
        carb.log_warn(f"[sweep] charge_{i}: eye={eye} target={tgt} -> {out}/charge_{i}.png")
    else:
        carb.log_warn(f"[sweep] charge_{i}: no image")

carb.log_warn(f"[sweep] done -> {out}/charge_1..10.png")
simulation_app.close()
