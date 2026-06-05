"""
Zone Resolver — maps a robot's TF pose (map → base_link) to a zone ID.

Used by the Isaac Sim adapter to attach a zone to every NavGoalStatus event,
mirroring what the mock sim does via its hard-coded zone map.

Algorithm:
  1. For each zone that has a polygon, run point-in-polygon test.
  2. If no polygon match (or zone has no polygon), fall back to nearest
     zone centroid.
  3. If no zones are configured, return None.

Zone polygons are lists of {x, y} dicts stored in the blackboard.  The
resolver is initialised once with the zone list and can be refreshed
by calling refresh(zones).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Zone:
    zone_id: str
    polygon: list[tuple[float, float]] = field(default_factory=list)  # (x, y) pairs
    centroid: tuple[float, float] = field(default=(0.0, 0.0))


class ZoneResolver:
    """
    Thread-safe (reads only after construction / refresh).

    Usage:
        resolver = ZoneResolver(zones)
        zone_id = resolver.resolve(x=3.2, y=1.1)
    """

    def __init__(self, zones: list[dict[str, Any]]):
        self._zones: list[_Zone] = []
        self.refresh(zones)

    def refresh(self, zones: list[dict[str, Any]]) -> None:
        self._zones = [_parse_zone(z) for z in zones]

    def resolve(self, x: float, y: float) -> str | None:
        """Return the zone_id that contains (x, y), or the nearest zone."""
        if not self._zones:
            return None

        # Point-in-polygon pass (exact)
        for zone in self._zones:
            if zone.polygon and _point_in_polygon(x, y, zone.polygon):
                return zone.zone_id

        # Fallback: nearest centroid
        best_zone = min(
            self._zones,
            key=lambda z: (x - z.centroid[0]) ** 2 + (y - z.centroid[1]) ** 2,
        )
        return best_zone.zone_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_zone(zone_dict: dict[str, Any]) -> _Zone:
    polygon_raw = zone_dict.get("polygon") or []
    polygon: list[tuple[float, float]] = []
    for pt in polygon_raw:
        if isinstance(pt, dict):
            polygon.append((float(pt.get("x", 0)), float(pt.get("y", 0))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            polygon.append((float(pt[0]), float(pt[1])))

    centroid = _centroid(polygon) if polygon else (0.0, 0.0)
    return _Zone(
        zone_id=zone_dict["zone_id"],
        polygon=polygon,
        centroid=centroid,
    )


def _centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    if not polygon:
        return (0.0, 0.0)
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    return (cx, cy)


def _point_in_polygon(
    x: float, y: float, polygon: list[tuple[float, float]]
) -> bool:
    """
    Ray casting algorithm.  Returns True if (x, y) is inside the polygon.
    The polygon is a list of (x, y) vertices; edges are assumed closed
    (last vertex connects back to first).
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    xi, yi = polygon[0]
    for j in range(1, n + 1):
        xj, yj = polygon[j % n]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        xi, yi = xj, yj
    return inside
