"""
Build a CORRECT Nav2 occupancy map from the REAL shelf/wall positions read out of
full_warehouse.usd (via usd-core; no Isaac/GPU/render needed).

The previously generated warehouse_map.pgm was offset/wrong — its occupied cells
fell on open floor in the render, so demos placed outside the building looked like
an empty field. These coordinates were read directly from the USD:
  shelf rows (run along Y, Y≈8.5..25):  x ≈ -20.5, -15.5, -10.5, -5.5, -0.5, 4.5
  clear aisles between them:            x ≈ -18, -13, -8, -3, +2
  floor extent:                         x[-26.3, 5.0]  y[-20.4, 27.6]

  python3 deploy/nav2/make_real_map.py   # writes warehouse_real.pgm/.yaml

Output is committed so map_server can load it without re-running.
"""
import os

OX, OY, RES, W, H = -30.0, -25.0, 0.1, 400, 600   # covers x[-30,10] y[-25,35]
SHELF_ROWS_X = (-20.5, -15.5, -10.5, -5.5, -0.5, 4.5)
SHELF_Y = (8.5, 25.2)
SHELF_HALF_W = 0.7
HERE = os.path.dirname(os.path.abspath(__file__))


def build() -> bytearray:
    grid = bytearray([255]) * (W * H)   # 255 = free

    def occ(x0, x1, y0, y1):
        for yy in range(int((y0 - OY) / RES), int((y1 - OY) / RES) + 1):
            if 0 <= yy < H:
                row = (H - 1 - yy) * W      # pgm row 0 = top = +y
                for xx in range(int((x0 - OX) / RES), int((x1 - OX) / RES) + 1):
                    if 0 <= xx < W:
                        grid[row + xx] = 0

    for rx in SHELF_ROWS_X:
        occ(rx - SHELF_HALF_W, rx + SHELF_HALF_W, *SHELF_Y)
    # perimeter walls (approx from floor extent)
    occ(-26.5, -26.0, -24, 31)
    occ(5.0, 5.5, -24, 31)
    occ(-26.5, 5.5, -24, -23.5)
    occ(-26.5, 5.5, 30.6, 31.1)
    return grid


def main() -> None:
    grid = build()
    with open(os.path.join(HERE, "warehouse_real.pgm"), "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (W, H))
        f.write(bytes(grid))
    with open(os.path.join(HERE, "warehouse_real.yaml"), "w") as f:
        f.write("image: warehouse_real.pgm\nresolution: %g\norigin: [%g, %g, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n"
                % (RES, OX, OY))
    print("wrote warehouse_real.pgm/.yaml (%dx%d)" % (W, H))


if __name__ == "__main__":
    main()
