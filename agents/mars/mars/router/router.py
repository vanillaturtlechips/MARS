"""
Deterministic Router — §2

Reads cheap signals from the enriched failure event (produced by the Aggregator)
and selects FAST_PATH or SLOW_PATH.  No inference; no LLM.

Important: per_zone_robot_spread from the distribution is a ROUTING signal only.
It is NOT forwarded to the Failure Analysis Agent (which must derive scope from
raw mission_failures itself — §4 invariant).
"""
from __future__ import annotations

import logging
from enum import Enum

from mars.config import FAST_PATH_BUDGET, ROUTER_SCOPE_HINT

log = logging.getLogger(__name__)


class Path(str, Enum):
    FAST = "FAST"
    SLOW = "SLOW"


def route(failure_event: dict, active_policy_on_zone: bool = False, zone_in_degraded_set: bool = False) -> Path:
    """
    Deterministic path selection.

    Args:
        failure_event: enriched event from Aggregator (§1a shape).
        active_policy_on_zone: True if a policy is already active for this zone.
        zone_in_degraded_set: True if this zone is in a known-degraded set.

    Returns:
        Path.FAST or Path.SLOW
    """
    zone_spread = failure_event.get("distribution", {}).get("per_zone_robot_spread", 0)
    failures_for_mission = failure_event.get("failures_for_this_mission", 0)

    # Retry-loop guard: a mission that keeps failing is not "isolated"
    if failures_for_mission > FAST_PATH_BUDGET:
        log.info(
            "[router] SLOW — failures_for_mission=%d > budget=%d",
            failures_for_mission, FAST_PATH_BUDGET,
        )
        return Path.SLOW

    # Pattern signals → reason about it
    if zone_spread >= ROUTER_SCOPE_HINT:
        log.info("[router] SLOW — zone_robot_spread=%d >= hint=%d", zone_spread, ROUTER_SCOPE_HINT)
        return Path.SLOW

    if active_policy_on_zone:
        log.info("[router] SLOW — active_policy already on zone")
        return Path.SLOW

    if zone_in_degraded_set:
        log.info("[router] SLOW — zone in known degraded set")
        return Path.SLOW

    log.info("[router] FAST — isolated / first-time failure")
    return Path.FAST
