"""
Keepout Service — turns active avoid_zone policies into a Nav2 KeepoutFilter
mask and pushes it to the navigation backend.

Lifecycle: registered as a PolicyManager consumer.  When an avoid_zone policy
activates/deactivates, this service recomputes the set of avoided zones,
rasterizes their polygons (mars.ros.keepout) into a single OccupancyGrid mask,
and calls nav.publish_keepout_mask(grid).

The whole "policy → mask grid" path is plain Python and unit-testable with the
mock sim; only nav.publish_keepout_mask's real implementation (the rclpy
OccupancyGrid publish) needs ROS2 — see ROS2SimAdapter.
"""
from __future__ import annotations

import logging

from mars.blackboard.queries import get_zone_polygons
from mars.ros.keepout import MapMeta, build_occupancy_grid_dict

log = logging.getLogger(__name__)


def _parse_polygon(raw) -> list[tuple[float, float]]:
    """Accept [{x,y}, ...] or [[x,y], ...]; return [(x, y), ...]."""
    out: list[tuple[float, float]] = []
    for pt in raw or []:
        if isinstance(pt, dict):
            out.append((float(pt.get("x", 0.0)), float(pt.get("y", 0.0))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append((float(pt[0]), float(pt[1])))
    return out


class KeepoutService:
    def __init__(
        self,
        nav,
        conn_factory,
        *,
        resolution: float = 0.05,
        margin: float = 0.5,
    ):
        self._nav = nav
        self._conn = conn_factory
        self._resolution = resolution
        self._margin = margin
        self._avoid_zones: set[str] = set()

    def on_policy_change(self, event: str, policy: dict) -> None:
        """PolicyManager consumer callback."""
        if policy.get("type") != "avoid_zone":
            return
        zone = policy.get("params", {}).get("zone")
        if not zone:
            return
        if event == "activated":
            self._avoid_zones.add(zone)
            log.info("[keepout] avoid_zone activated: %s", zone)
        else:
            self._avoid_zones.discard(zone)
            log.info("[keepout] avoid_zone cleared: %s", zone)
        self._rebuild_and_publish()

    def _rebuild_and_publish(self) -> None:
        """Rasterize all active avoid zones into one mask and publish it."""
        if not self._avoid_zones:
            # Nothing avoided → publish a tiny all-free mask to clear keepout.
            empty = MapMeta(resolution=self._resolution,
                            origin_x=0.0, origin_y=0.0, width=1, height=1)
            self._nav.publish_keepout_mask(build_occupancy_grid_dict([], empty))
            log.info("[keepout] no active zones — published cleared mask")
            return

        conn = self._conn()
        try:
            zone_polys = get_zone_polygons(conn, list(self._avoid_zones))
        finally:
            conn.close()

        polygons = [p for p in (_parse_polygon(v) for v in zone_polys.values())
                    if len(p) >= 3]
        if not polygons:
            log.warning("[keepout] active zones %s have no usable polygons — "
                        "Nav2 keepout skipped (scheduler still gates dispatch)",
                        sorted(self._avoid_zones))
            return

        # Size the mask to cover every active polygon plus a margin.
        xs = [x for poly in polygons for x, _ in poly]
        ys = [y for poly in polygons for _, y in poly]
        meta = MapMeta.covering(
            min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys),
            resolution=self._resolution, margin=self._margin,
        )
        grid = build_occupancy_grid_dict(polygons, meta)
        self._nav.publish_keepout_mask(grid)
        log.info("[keepout] published mask for zones=%s  grid=%dx%d res=%.3f",
                 sorted(self._avoid_zones), meta.width, meta.height, meta.resolution)
