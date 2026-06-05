"""
Generate an empty (all-free) occupancy map for the Nav2 keepout demo.

We localize perfectly via Isaac ground-truth (tf map->base_link), and there is
no lidar, so the static map just needs to give Nav2 a free area to plan in.
The agent's avoid_zone is added on top via the KeepoutFilter (not this map).

Run (anywhere with numpy):
    python deploy/nav2/make_empty_map.py        # writes warehouse_empty.pgm + .yaml here
"""
import os
import numpy as np

RES = 0.05          # m per cell (matches costmap)
SIZE_M = 40.0       # 40 x 40 m free area
ORIGIN = (-20.0, -20.0, 0.0)   # lower-left corner in map frame (robot starts ~0,0)

n = int(SIZE_M / RES)           # 800 x 800
HERE = os.path.dirname(os.path.abspath(__file__))
pgm = os.path.join(HERE, "warehouse_empty.pgm")
yaml = os.path.join(HERE, "warehouse_empty.yaml")

# PGM P5: 255 = free (white). Nav2 with negate=0 treats high values as free.
grid = np.full((n, n), 255, dtype=np.uint8)
with open(pgm, "wb") as f:
    f.write(f"P5\n{n} {n}\n255\n".encode())
    f.write(grid.tobytes())

with open(yaml, "w") as f:
    f.write(
        f"image: warehouse_empty.pgm\n"
        f"resolution: {RES}\n"
        f"origin: [{ORIGIN[0]}, {ORIGIN[1]}, {ORIGIN[2]}]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.25\n"
        f"mode: trinary\n"
    )

print(f"wrote {pgm} ({n}x{n}, {RES} m/cell) and {yaml}")
