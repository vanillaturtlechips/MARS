"""
Hot-state store — fast read/write for per-robot live fields (pose, battery,
current_zone) and zone occupancy.  Backed by Redis; falls back to an in-process
dict when Redis is unavailable (useful in tests).

Keys:
    robot:{robot_id}       → hash  {battery_pct, current_zone, pose_json,
                                     health_level, estop_active, allocation_state}
    zone:{zone_id}:occ     → int   (current occupancy count)
    zone:{zone_id}:health  → str   (ok | degraded | blocked)
    dist:robot:{robot_id}:zones      → HyperLogLog of distinct zones with failures
    dist:zone:{zone_id}:robots       → HyperLogLog of distinct robots with failures
    fail_count:mission:{mission_id}  → int
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_ROBOT_TTL = 30  # seconds before a robot is considered stale


class HotState:
    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local: dict[str, Any] = {}  # fallback store

    # ------------------------------------------------------------------
    # Robot hot state
    # ------------------------------------------------------------------

    def update_robot(self, robot_id: str, fields: dict[str, Any]) -> None:
        if self._redis:
            serialized = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in fields.items()
            }
            key = f"robot:{robot_id}"
            self._redis.hset(key, mapping=serialized)
            self._redis.expire(key, _ROBOT_TTL)
        else:
            self._local.setdefault(f"robot:{robot_id}", {}).update(fields)

    def get_robot(self, robot_id: str) -> dict[str, Any] | None:
        if self._redis:
            raw = self._redis.hgetall(f"robot:{robot_id}")
            if not raw:
                return None
            result = {}
            for k, v in raw.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = val
            return result
        return self._local.get(f"robot:{robot_id}")

    # ------------------------------------------------------------------
    # Zone occupancy / health
    # ------------------------------------------------------------------

    def set_zone_occupancy(self, zone_id: str, count: int) -> None:
        if self._redis:
            self._redis.set(f"zone:{zone_id}:occ", count)
        else:
            self._local[f"zone:{zone_id}:occ"] = count

    def get_zone_occupancy(self, zone_id: str) -> int:
        if self._redis:
            v = self._redis.get(f"zone:{zone_id}:occ")
            return int(v) if v else 0
        return self._local.get(f"zone:{zone_id}:occ", 0)

    def set_zone_health(self, zone_id: str, status: str) -> None:
        if self._redis:
            self._redis.set(f"zone:{zone_id}:health", status)
        else:
            self._local[f"zone:{zone_id}:health"] = status

    def get_zone_health(self, zone_id: str) -> str:
        if self._redis:
            v = self._redis.get(f"zone:{zone_id}:health")
            return (v.decode() if isinstance(v, bytes) else v) or "ok"
        return self._local.get(f"zone:{zone_id}:health", "ok")

    # ------------------------------------------------------------------
    # Failure distribution counters (rolling window)
    # Used by Aggregator to compute per_robot_zone_spread and
    # per_zone_robot_spread cheaply without hitting Postgres.
    # ------------------------------------------------------------------

    def record_failure_for_distribution(
        self, robot_id: str, zone_id: str, window_seconds: int = 900
    ) -> None:
        """Record a failure occurrence for both direction axes."""
        if not self._redis:
            # Fallback: accumulate in local dicts
            rkey = f"dist:robot:{robot_id}:zones"
            zkey = f"dist:zone:{zone_id}:robots"
            self._local.setdefault(rkey, set()).add(zone_id)
            self._local.setdefault(zkey, set()).add(robot_id)
            return

        # Per-robot: how many DISTINCT zones has this robot failed in?
        rkey = f"dist:robot:{robot_id}:zones:{zone_id}"
        self._redis.setex(rkey, window_seconds, 1)

        # Per-zone: how many DISTINCT robots failed in this zone?
        zkey = f"dist:zone:{zone_id}:robots:{robot_id}"
        self._redis.setex(zkey, window_seconds, 1)

    def get_robot_zone_spread(self, robot_id: str) -> int:
        """Distinct zones this robot has failed in within the window."""
        if not self._redis:
            return len(self._local.get(f"dist:robot:{robot_id}:zones", set()))
        pattern = f"dist:robot:{robot_id}:zones:*"
        keys = self._redis.keys(pattern)
        return len(keys)

    def get_zone_robot_spread(self, zone_id: str) -> int:
        """Distinct robots that failed in this zone within the window."""
        if not self._redis:
            return len(self._local.get(f"dist:zone:{zone_id}:robots", set()))
        pattern = f"dist:zone:{zone_id}:robots:*"
        keys = self._redis.keys(pattern)
        return len(keys)

    # ------------------------------------------------------------------
    # Mission failure counter
    # ------------------------------------------------------------------

    def increment_mission_failures(
        self, mission_id: str, window_seconds: int = 3600
    ) -> int:
        if not self._redis:
            key = f"fail_count:mission:{mission_id}"
            self._local[key] = self._local.get(key, 0) + 1
            return self._local[key]
        key = f"fail_count:mission:{mission_id}"
        count = self._redis.incr(key)
        if count == 1:
            self._redis.expire(key, window_seconds)
        return count

    def get_mission_failure_count(self, mission_id: str) -> int:
        if not self._redis:
            return self._local.get(f"fail_count:mission:{mission_id}", 0)
        v = self._redis.get(f"fail_count:mission:{mission_id}")
        return int(v) if v else 0
