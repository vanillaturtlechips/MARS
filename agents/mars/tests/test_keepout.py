"""
Unit tests for the keepout mask generator (mars/ros/keepout.py).

Pure-Python — no rclpy/Nav2 needed, so these run locally and gate the
"policy → keepout mask" logic before any RunPod integration.
"""
import numpy as np

from mars.ros.keepout import (
    KEEPOUT_VALUE,
    FREE_VALUE,
    MapMeta,
    rasterize_polygons,
    build_occupancy_grid_dict,
)


# A 1 m-resolution 10×10 grid with origin at (0, 0): covers [0,10]×[0,10].
META = MapMeta(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)

# A 4×4 square keepout from (2,2) to (6,6) in world coords.
SQUARE = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]


def test_empty_polygons_is_all_free():
    grid = rasterize_polygons([], META)
    assert grid.shape == (10, 10)
    assert (grid == FREE_VALUE).all()


def test_square_interior_is_keepout():
    grid = rasterize_polygons([SQUARE], META)
    # Cell center (col=3,row=3) -> world (3.5,3.5), inside the square.
    assert grid[3, 3] == KEEPOUT_VALUE
    # Cell center (col=4,row=4) -> world (4.5,4.5), inside.
    assert grid[4, 4] == KEEPOUT_VALUE


def test_outside_square_is_free():
    grid = rasterize_polygons([SQUARE], META)
    # world (0.5,0.5) far outside
    assert grid[0, 0] == FREE_VALUE
    # world (8.5,8.5) outside
    assert grid[8, 8] == FREE_VALUE


def test_keepout_count_matches_area():
    # Square spans world x,y in (2,6): cell centers at 2.5,3.5,4.5,5.5 -> 4 per axis.
    grid = rasterize_polygons([SQUARE], META)
    assert int((grid == KEEPOUT_VALUE).sum()) == 16


def test_row_major_indexing_matches_world_y():
    # A keepout only in the lower band y in (0,3) should mark low rows, not high.
    band = [(0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (0.0, 3.0)]
    grid = rasterize_polygons([band], META)
    assert grid[0, 5] == KEEPOUT_VALUE   # row 0 -> world y=0.5, inside band
    assert grid[9, 5] == FREE_VALUE      # row 9 -> world y=9.5, outside band


def test_multiple_polygons_union():
    a = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
    b = [(7.0, 7.0), (9.0, 7.0), (9.0, 9.0), (7.0, 9.0)]
    grid = rasterize_polygons([a, b], META)
    assert grid[2, 2] == KEEPOUT_VALUE   # inside a
    assert grid[8, 8] == KEEPOUT_VALUE   # inside b
    assert grid[5, 5] == FREE_VALUE      # between, free


def test_degenerate_polygon_ignored():
    grid = rasterize_polygons([[(1.0, 1.0), (2.0, 2.0)]], META)  # <3 vertices
    assert (grid == FREE_VALUE).all()


def test_occupancy_grid_dict_shape_and_fields():
    msg = build_occupancy_grid_dict([SQUARE], META, stamp_sec=42)
    assert msg["info"]["width"] == 10
    assert msg["info"]["height"] == 10
    assert msg["info"]["resolution"] == 1.0
    assert msg["info"]["origin"]["position"]["x"] == 0.0
    assert msg["header"]["frame_id"] == "map"
    assert msg["header"]["stamp"]["sec"] == 42
    # data is flat row-major of length width*height
    assert len(msg["data"]) == 100
    assert max(msg["data"]) == KEEPOUT_VALUE
    assert min(msg["data"]) == FREE_VALUE


def test_covering_factory_encloses_bbox():
    meta = MapMeta.covering(min_x=2.0, min_y=2.0, max_x=6.0, max_y=6.0,
                            resolution=0.5, margin=1.0)
    # origin pushed out by margin
    assert meta.origin_x == 1.0
    assert meta.origin_y == 1.0
    # grid covers (2-1)..(6+1) = 6 m / 0.5 = 12 cells
    assert meta.width == 12
    assert meta.height == 12
    # the square should rasterize fully inside this grid
    grid = rasterize_polygons([SQUARE], meta)
    assert int((grid == KEEPOUT_VALUE).sum()) > 0
