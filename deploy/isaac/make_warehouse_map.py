"""
Generate a Nav2 occupancy map (.pgm + .yaml) from full_warehouse.usd so Nav2
knows the warehouse walls/racks (the empty map made by deploy/nav2/make_empty_map.py
is all-free and makes robots ram invisible warehouse geometry).

Runs INSIDE Isaac (py3.11):  source deploy/isaac/env_isaac.sh
    python deploy/isaac/make_warehouse_map.py

Output (written next to the Nav2 params so map_server can load it):
    deploy/nav2/warehouse_map.pgm
    deploy/nav2/warehouse_map.yaml

SAME origin/resolution as warehouse_empty.* so the keepout coordinates and the
demo geometry (dock at x~4) stay aligned — only the occupied cells change.

After running, point map_server at it:
    deploy/nav2/nav2_keepout_demo.params.yaml -> yaml_filename: .../warehouse_map.yaml
(already switched by the accompanying commit; revert to warehouse_empty.yaml to undo.)
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os  # noqa: E402
import carb  # noqa: E402

# --- map geometry: MUST match deploy/nav2/make_empty_map.py so frames line up
RES = 0.05                       # m per cell
SIZE_M = 40.0                    # 40 x 40 m area
ORIGIN_XY = (-20.0, -20.0)       # lower-left corner in map frame (yaml origin)
SLICE_Z = 0.30                   # horizontal raycast height: catches walls/racks at robot-body height
# pixel values for the PGM (Nav2 trinary: 0=occupied/black, 255=free/white, 205=unknown/grey)
PGM_OCC, PGM_FREE, PGM_UNK = 0, 255, 205
# omap writes these tag values into the buffer per cell type; we remap them to PGM above
TAG_OCC, TAG_FREE, TAG_UNK = 1, 0, 2
# Nav2 image row 0 = TOP = highest y; flip the buffer's y so north is up.
FLIP_Y = True

_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
WAREHOUSE_USD = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.asset.gen.omap")
simulation_app.update()

import omni  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

# omap binding moved namespaces across versions; try new (5.x) then old.
try:
    from isaacsim.asset.gen.omap.bindings import _omap
except ImportError:  # pragma: no cover - older Isaac
    from omni.isaac.occupancy_map.bindings import _omap

world = World(stage_units_in_meters=1.0)
carb.log_warn("[omap] loading full_warehouse.usd ...")
add_reference_to_stage(WAREHOUSE_USD, "/World/Warehouse")

# colliders must exist before raycasting -> reset + step a few frames
world.reset()
for _ in range(20):
    world.step(render=False)

physx = omni.physx.acquire_physx_interface()
stage_id = omni.usd.get_context().get_stage_id()

gen = _omap.Generator(physx, stage_id)
# update_settings(cell_size, occupied_value, unoccupied_value, unknown_value)
gen.update_settings(RES, TAG_OCC, TAG_FREE, TAG_UNK)
# set_transform(origin, min_bound, max_bound): region = [origin+min, origin+max], slice at origin.z
half = SIZE_M / 2.0
center = (ORIGIN_XY[0] + half, ORIGIN_XY[1] + half, SLICE_Z)   # (0,0,z) for a -20..20 map
gen.set_transform(center, (-half, -half, 0.0), (half, half, 0.0))
carb.log_warn(f"[omap] generating 2D occupancy: center={center} extent=+/-{half}m z={SLICE_Z} res={RES}")
gen.generate2d()

buf = gen.get_buffer()
dims = gen.get_dimensions()      # (width, height[, depth])
w, h = int(dims[0]), int(dims[1])
carb.log_warn(f"[omap] dims={dims} buffer_len={len(buf)} (expect ~{int(SIZE_M/RES)}^2)")

# remap omap tags -> PGM pixel values
def to_pgm(v):
    if v == TAG_OCC:
        return PGM_OCC
    if v == TAG_FREE:
        return PGM_FREE
    return PGM_UNK

# buffer is row-major; reshape to rows of width w
rows = [buf[i * w:(i + 1) * w] for i in range(h)]
if FLIP_Y:
    rows = rows[::-1]

n_occ = sum(1 for v in buf if v == TAG_OCC)
carb.log_warn(f"[omap] occupied cells = {n_occ} / {len(buf)}")

HERE = os.path.dirname(os.path.abspath(__file__))
NAV2 = os.path.normpath(os.path.join(HERE, "..", "nav2"))
pgm_path = os.path.join(NAV2, "warehouse_map.pgm")
yaml_path = os.path.join(NAV2, "warehouse_map.yaml")

with open(pgm_path, "wb") as f:
    f.write(f"P5\n{w} {h}\n255\n".encode())
    for r in rows:
        f.write(bytes(to_pgm(v) for v in r))

with open(yaml_path, "w") as f:
    f.write(
        f"image: warehouse_map.pgm\n"
        f"resolution: {RES}\n"
        f"origin: [{ORIGIN_XY[0]}, {ORIGIN_XY[1]}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.25\n"
        f"mode: trinary\n"
    )

carb.log_warn(f"[omap] wrote {pgm_path} ({w}x{h}) and {yaml_path}")
simulation_app.close()
