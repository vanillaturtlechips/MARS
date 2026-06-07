"""
Generate an empty (all-free) occupancy map for the Nav2 keepout demo.

Perfect localization via Isaac ground-truth (tf map->base_link), no lidar, so
the static map just gives Nav2 a free area to plan in. avoid_zone is added on
top via the KeepoutFilter, not this map.

Pure stdlib (no numpy) so it runs in any python.
    python deploy/nav2/make_empty_map.py
"""
import os

RES = 0.05            # m per cell
SIZE_M = 40.0         # 40 x 40 m free area
ORIGIN = (-20.0, -20.0, 0.0)

n = int(SIZE_M / RES)            # 800 x 800
HERE = os.path.dirname(os.path.abspath(__file__))
pgm = os.path.join(HERE, "warehouse_empty.pgm")
yaml = os.path.join(HERE, "warehouse_empty.yaml")

# PGM P5, 255 = free. One row = n bytes of 0xFF; n rows.
row = b"\xff" * n
with open(pgm, "wb") as f:
    f.write(f"P5\n{n} {n}\n255\n".encode())
    for _ in range(n):
        f.write(row)

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
