"""
Camera contact sheet for the CHARGING demo scene (demo2/3). Loads full_warehouse +
the charging dock (green pad + dock unit at (0,5)) + the 3 robots in a charging pose
(R1 docked on the pad, R2 waiting, R3 idle), and captures 10 angles to
/tmp/cam_sweep/charge_1..10.png.

ALL presets are constrained to what actually renders here:
  * eye INSIDE the building: x in [-5, 4] (floor ends at x=5), y >= -2 (south wall;
    y=-7/-10 were OUTSIDE -> grey)
  * eye BELOW the ceiling: z <= 7 (z=9-11 went ABOVE the ~8m ceiling -> grey)
Wide lens (focal 16 -> ~66 deg) so each frame is wide.

Look at the PNGs, pick the best charge_N, tell me — that eye/target becomes the
charge-demo camera in run_all_demos.sh.

  source deploy/isaac/env_isaac.sh
  python deploy/isaac/cam_sweep_charge.py
  # then view /tmp/cam_sweep/charge_*.png   (cp /tmp/cam_sweep/charge_*.png /workspace/)
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

CX, CY = 0.0, 5.0            # charger pad / dock
# charging pose: R1 docked on the pad, R2 waiting, R3 idle (= scene --charge spawns)
ROBOTS = [("R1", 0.0, 5.0), ("R2", 3.0, 2.0), ("R3", -4.0, 2.0)]

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(WAREHOUSE, "/World/Warehouse")

# charging station: green pad + compact dock unit + green indicator (matches the live scene)
world.scene.add(VisualCuboid(prim_path="/World/charge_pad", name="charge_pad",
                             position=np.array([CX, CY, 0.02]), scale=np.array([1.8, 1.8, 0.04]),
                             color=np.array([0.05, 0.85, 0.25])))
world.scene.add(VisualCuboid(prim_path="/World/charge_dock", name="charge_dock",
                             position=np.array([CX, CY + 0.95, 0.45]), scale=np.array([1.3, 0.45, 0.9]),
                             color=np.array([0.30, 0.32, 0.36])))
world.scene.add(VisualCuboid(prim_path="/World/charge_led", name="charge_led",
                             position=np.array([CX, CY + 0.72, 0.62]), scale=np.array([0.9, 0.06, 0.28]),
                             color=np.array([0.10, 0.90, 0.35])))
for name, x, y in ROBOTS:
    p = f"/World/{name}"
    add_reference_to_stage(IW_HUB, p)
    UsdGeom.Xformable(stage.GetPrimAtPath(p)).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

world.reset()
cam = Camera(prim_path="/World/SweepCam", resolution=(1280, 720))
cam.initialize()
cam.set_focal_length(16.0)            # wide (~66 deg)
cam.set_horizontal_aperture(20.955)
for _ in range(40):
    world.step(render=True)

# (eye, target) presets — ALL inside (x[-5,4], y>=-2) and below the ceiling (z<=7),
# aimed at the charging band (charger (0,5), spawns at y=2).
PRESETS = [
    ((0, -2, 6),    (0, 4.5, 0.4)),   # 1: south, centered, mid
    ((0, -2, 3.5),  (0, 4.5, 0.6)),   # 2: south, low
    ((0, -2, 7),    (0, 4.0, 0.3)),   # 3: south, high (just under ceiling)
    ((4, -1, 5),    (0, 4.5, 0.5)),   # 4: south-east 3/4
    ((-4, -1, 5),   (0, 4.5, 0.5)),   # 5: south-west 3/4
    ((0, 1, 3),     (0, 5.0, 0.6)),   # 6: close, low, on the dock
    ((4, 1, 5),     (-1, 4.5, 0.5)),  # 7: east 3/4
    ((-5, 1, 5),    (1, 4.5, 0.5)),   # 8: west 3/4
    ((0, -2, 5),    (0, 3.5, 0.3)),   # 9: south, target lower (catches the waiting robots at y=2)
    ((2, -2, 4),    (-1, 4.5, 0.5)),  # 10: south-east, low
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
