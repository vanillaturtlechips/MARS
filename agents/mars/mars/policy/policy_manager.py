"""
Policy Manager — lifecycle, activation, expiry.

Activates policies that have passed the Decision Validator and Policy Guardrail.
Notifies registered consumers (Scheduler, Charging Service) on change.
Expires policies that have passed their expires_at.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

log = logging.getLogger(__name__)


class PolicyManager:
    def __init__(self, blackboard_conn_factory: Callable):
        self._conn_factory = blackboard_conn_factory
        # In-memory cache of active policies for fast reads by Scheduler etc.
        self._active: dict[str, dict] = {}
        # Consumers: (type_filter, callback)  — None = all types
        self._consumers: list[tuple[str | None, Callable]] = []
        # Track last-applied time per type (for guardrail cooldown)
        self._last_applied: dict[str, float] = {}

    def register_consumer(
        self, callback: Callable[[str, dict], None], policy_type: str | None = None
    ) -> None:
        """Register a callback to be notified when a policy is activated/deactivated."""
        self._consumers.append((policy_type, callback))

    def activate(self, policy: dict) -> str:
        """Write policy to blackboard and notify consumers."""
        from mars.blackboard.queries import write_policy

        conn = self._conn_factory()
        pid = write_policy(conn, policy)
        conn.commit()

        policy["policy_id"] = pid
        self._active[pid] = policy
        self._last_applied[policy.get("type", "")] = time.time()

        for filter_type, cb in self._consumers:
            if filter_type is None or filter_type == policy.get("type"):
                try:
                    cb("activated", policy)
                except Exception:
                    log.exception("Policy consumer callback error")

        log.info("[policy_manager] activated %s  id=%s", policy.get("type"), pid)
        return pid

    def deactivate(self, policy_id: str, reason: str = "expired") -> None:
        from mars.blackboard.queries import deactivate_policy

        conn = self._conn_factory()
        deactivate_policy(conn, policy_id, reason)
        conn.commit()

        policy = self._active.pop(policy_id, {})
        for filter_type, cb in self._consumers:
            if policy and (filter_type is None or filter_type == policy.get("type")):
                try:
                    cb("deactivated", policy)
                except Exception:
                    log.exception("Policy consumer callback error")

        log.info("[policy_manager] deactivated %s  reason=%s", policy_id, reason)

    def expire_stale(self) -> None:
        """Deactivate all policies whose expires_at has passed."""
        now = datetime.now(timezone.utc)
        expired = [
            pid
            for pid, p in list(self._active.items())
            if p.get("expires_at") and p["expires_at"] < now
        ]
        for pid in expired:
            self.deactivate(pid, reason="expired")

    def get_active(self) -> list[dict]:
        return list(self._active.values())

    def get_last_applied(self) -> dict[str, float]:
        return dict(self._last_applied)

    def is_policy_active_for_zone(self, zone: str, policy_type: str = "avoid_zone") -> bool:
        return any(
            p.get("type") == policy_type and p.get("params", {}).get("zone") == zone
            for p in self._active.values()
        )
