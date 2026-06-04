"""
Retrieval Validator — §4

Gates whether the agent is allowed to trust retrieved precedent.

Per-result scoring with gating (not a flat average):
  base = W_META*metadata_match + W_REC*recency + W_COV*coverage_match + W_SIM*similarity
  coverage mismatch below COV_FLOOR caps base at COV_MISMATCH_CAP
  stale recency below RECENCY_FLOOR caps base at STALE_CAP

Set-level: support_count + consistency → HIGH / MEDIUM / LOW trust level.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from mars.config import (
    RV_ACCEPT_THRESHOLD,
    RV_COV_FLOOR,
    RV_COV_MISMATCH_CAP,
    RV_RECENCY_FLOOR,
    RV_RECENCY_HALF_LIFE_DAYS,
    RV_STALE_CAP,
    RV_W_COV,
    RV_W_META,
    RV_W_REC,
    RV_W_SIM,
)

log = logging.getLogger(__name__)

_SCOPE_ORDER = ["isolated", "robot_specific", "zone_wide", "fleet_wide"]


def _scope_distance(s1: str | None, s2: str | None) -> float:
    """Distance between two scope values (0=identical, 1=max)."""
    if s1 is None or s2 is None:
        return 0.5
    try:
        return abs(_SCOPE_ORDER.index(s1) - _SCOPE_ORDER.index(s2)) / (len(_SCOPE_ORDER) - 1)
    except ValueError:
        return 0.5


def score_precedent(
    precedent: dict[str, Any],
    current_zone: str | None,
    current_failure_type: str | None,
    current_scope: str | None,
) -> float:
    """
    Compute per-result trust score with gating.

    Expects precedent to have fields from incident_embeddings:
      similarity, zone, failure_type, scope, recorded_at (ISO string or datetime)
    """
    # --- Metadata match ---
    meta_score = 0.0
    count = 0
    if current_zone and precedent.get("zone"):
        meta_score += 1.0 if precedent["zone"] == current_zone else 0.0
        count += 1
    if current_failure_type and precedent.get("failure_type"):
        meta_score += 1.0 if precedent["failure_type"] == current_failure_type else 0.3
        count += 1
    metadata_match = (meta_score / count) if count > 0 else 0.5

    # --- Recency ---
    recency = 1.0
    recorded_at = precedent.get("recorded_at")
    if recorded_at:
        if isinstance(recorded_at, str):
            try:
                recorded_at = datetime.fromisoformat(recorded_at)
            except ValueError:
                recorded_at = None
        if recorded_at:
            now = datetime.now(timezone.utc)
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=timezone.utc)
            age_days = (now - recorded_at).total_seconds() / 86400.0
            recency = math.exp(-math.log(2) * age_days / RV_RECENCY_HALF_LIFE_DAYS)

    # --- Coverage match ---
    p_scope = precedent.get("scope") or precedent.get("coverage")
    dist = _scope_distance(current_scope, p_scope)
    coverage_match = 1.0 - dist

    # --- Similarity (raw cosine from pgvector query) ---
    similarity = float(precedent.get("similarity", 0.5))

    # --- Base score ---
    base = (
        RV_W_META * metadata_match
        + RV_W_REC  * recency
        + RV_W_COV  * coverage_match
        + RV_W_SIM  * similarity
    )

    # --- Gating ---
    if coverage_match < RV_COV_FLOOR:
        base = min(base, RV_COV_MISMATCH_CAP)
    if recency < RV_RECENCY_FLOOR:
        base = min(base, RV_STALE_CAP)

    return base


def validate_retrieval_set(
    precedents: list[dict[str, Any]],
    current_zone: str | None,
    current_failure_type: str | None,
    current_scope: str | None,
) -> dict[str, Any]:
    """
    Score all precedents, filter by ACCEPT_THRESHOLD, and return a trust
    assessment that can be attached to the agent input bundle.

    Returns:
        {
          "filtered_precedents": [...],  # those that survived the threshold
          "set_level": "HIGH" | "MEDIUM" | "LOW",
          "support_count": int,
        }
    """
    scored = []
    for p in precedents:
        score = score_precedent(p, current_zone, current_failure_type, current_scope)
        scored.append({**p, "_trust_score": score})

    survivors = [p for p in scored if p["_trust_score"] >= RV_ACCEPT_THRESHOLD]
    support_count = len(survivors)

    # Set-level: check if survivors agree on cause/scope
    if support_count == 0:
        set_level = "LOW"
    elif support_count >= 3:
        # Check consistency: do survivors broadly agree on scope?
        scopes = [p.get("scope") or p.get("coverage") for p in survivors if p.get("scope") or p.get("coverage")]
        if scopes and len(set(scopes)) == 1:
            set_level = "HIGH"
        elif support_count >= 2:
            set_level = "MEDIUM"
        else:
            set_level = "LOW"
    elif support_count >= 2:
        set_level = "MEDIUM"
    else:
        set_level = "MEDIUM"  # single strong survivor

    log.info(
        "[retrieval_validator] %d/%d survived filter → set_level=%s",
        support_count, len(precedents), set_level,
    )

    return {
        "filtered_precedents": survivors,
        "set_level": set_level,
        "support_count": support_count,
    }
