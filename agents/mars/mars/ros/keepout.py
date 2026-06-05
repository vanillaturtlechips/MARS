"""
Keepout mask generation — avoid_zone polygons → Nav2 KeepoutFilter mask.

Nav2's KeepoutFilter consumes a filter mask published as a
``nav_msgs/OccupancyGrid``: a row-major grid whose cells hold occupancy
values in [0, 100].  The KeepoutFilter treats high-value cells as keepout
(no-go) regions in the global costmap, so the planner routes around them.

This module is **pure Python (+ numpy)** with NO rclpy dependency, so the
mask logic is unit-testable without a ROS installation.  The ROS2 adapter
imports build_occupancy_grid_dict() and copies the fields into a real
nav_msgs/OccupancyGrid message before publishing.

Conventions (match nav_msgs/OccupancyGrid):
  - data is row-major: data[row * width + col]
  - cell (col=0, row=0) sits at ``origin`` (the grid's lower-left corner)
  - world coord of a cell *center* (col i, row j):
        x = origin_x + (i + 0.5) * resolution
        y = origin_y + (j + 0.5) * resolution
  - keepout cells = KEEPOUT_VALUE (100), free cells = 0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Occupancy value the KeepoutFilter treats as a no-go cell.
KEEPOUT_VALUE: int = 100
FREE_VALUE: int = 0

# A polygon is a sequence of (x, y) vertices in the map frame (meters).
Polygon = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class MapMeta:
    """
    Geometry of the filter mask grid.  Must match the costmap it filters
    (same frame, and a resolution/extent that covers the navigable area).

    resolution : meters per cell
    origin_x/y : map-frame coords of the grid's lower-left corner (cell 0,0)
    width      : number of cells along x (columns)
    height     : number of cells along y (rows)
    frame_id   : TF frame the grid is expressed in (Nav2 global frame)
    """
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    frame_id: str = "map"

    @classmethod
    def covering(
        cls,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        resolution: float,
        margin: float = 1.0,
        frame_id: str = "map",
    ) -> "MapMeta":
        """
        Build a MapMeta whose grid covers the axis-aligned bounding box
        [min_x, max_x] × [min_y, max_y] plus a margin (meters) on every side.
        """
        ox = min_x - margin
        oy = min_y - margin
        width = max(1, int(np.ceil((max_x + margin - ox) / resolution)))
        height = max(1, int(np.ceil((max_y + margin - oy) / resolution)))
        return cls(resolution, ox, oy, width, height, frame_id)


def rasterize_polygons(polygons: Sequence[Polygon], meta: MapMeta) -> np.ndarray:
    """
    Rasterize avoid-zone polygons into an int8 occupancy grid.

    Returns a (height, width) numpy array: KEEPOUT_VALUE inside any polygon,
    FREE_VALUE elsewhere.  A cell is "inside" when its center is inside the
    polygon (even-odd ray-casting), vectorized over the whole grid.
    """
    grid = np.full((meta.height, meta.width), FREE_VALUE, dtype=np.int8)

    # World coords of every cell center.
    cols = np.arange(meta.width)
    rows = np.arange(meta.height)
    xs = meta.origin_x + (cols + 0.5) * meta.resolution        # (width,)
    ys = meta.origin_y + (rows + 0.5) * meta.resolution        # (height,)
    gx, gy = np.meshgrid(xs, ys)                               # (height, width)

    for poly in polygons:
        mask = _points_in_polygon(gx, gy, poly)
        if mask is not None:
            grid[mask] = KEEPOUT_VALUE
    return grid


def _points_in_polygon(
    gx: np.ndarray, gy: np.ndarray, polygon: Polygon
) -> np.ndarray | None:
    """
    Vectorized even-odd ray casting.  gx/gy are equal-shaped coordinate
    grids; returns a boolean mask (same shape) True where the point is inside.
    Returns None for a degenerate polygon (<3 vertices).
    """
    pts = [(float(x), float(y)) for x, y in polygon]
    n = len(pts)
    if n < 3:
        return None

    inside = np.zeros(gx.shape, dtype=bool)
    xi, yi = pts[0]
    for k in range(1, n + 1):
        xj, yj = pts[k % n]
        # Edge straddles the horizontal ray at each test point's y?
        cond = ((yi > gy) != (yj > gy))
        # x of the edge at the point's y; compare to point x.
        x_cross = (xj - xi) * (gy - yi) / (yj - yi + 1e-12) + xi
        inside ^= cond & (gx < x_cross)
        xi, yi = xj, yj
    return inside


def build_occupancy_grid_dict(
    polygons: Sequence[Polygon],
    meta: MapMeta,
    stamp_sec: int = 0,
    stamp_nanosec: int = 0,
) -> dict:
    """
    Produce a plain dict mirroring nav_msgs/OccupancyGrid.  The ROS2 adapter
    copies these fields into a real message (keeps this module rclpy-free).

    data is row-major (data[row*width + col]) as a flat list of ints.
    """
    grid = rasterize_polygons(polygons, meta)
    return {
        "header": {
            "frame_id": meta.frame_id,
            "stamp": {"sec": stamp_sec, "nanosec": stamp_nanosec},
        },
        "info": {
            "resolution": meta.resolution,
            "width": meta.width,
            "height": meta.height,
            "origin": {
                "position": {"x": meta.origin_x, "y": meta.origin_y, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        "data": grid.flatten().astype(int).tolist(),
    }
