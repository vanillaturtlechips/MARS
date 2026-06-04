"""
Unit tests for Retrieval Validator — §4

Key test: coverage mismatch must cap trust even at high similarity.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from mars.validators.retrieval_validator import score_precedent, validate_retrieval_set
from mars.config import RV_COV_MISMATCH_CAP, RV_ACCEPT_THRESHOLD


def _make_precedent(
    zone="Receiving Dock",
    failure_type="zone_congestion",
    scope="zone_wide",
    similarity=0.95,
    days_old=1,
):
    recorded_at = datetime.now(timezone.utc) - timedelta(days=days_old)
    return {
        "zone": zone,
        "failure_type": failure_type,
        "scope": scope,
        "similarity": similarity,
        "recorded_at": recorded_at.isoformat(),
    }


class TestScorePrecedent:
    def test_perfect_match_scores_high(self):
        p = _make_precedent()
        score = score_precedent(p, "Receiving Dock", "zone_congestion", "zone_wide")
        assert score >= 0.7

    def test_coverage_mismatch_caps_trust(self):
        # Precedent scope = isolated, current scope = fleet_wide → big mismatch
        p = _make_precedent(scope="isolated", similarity=0.99)
        score = score_precedent(p, "Receiving Dock", "zone_congestion", "fleet_wide")
        # Even with perfect similarity, coverage mismatch must cap the score
        assert score <= RV_COV_MISMATCH_CAP + 0.01, (
            f"Coverage mismatch at score={score:.3f} exceeds cap={RV_COV_MISMATCH_CAP}"
        )

    def test_stale_precedent_caps_trust(self):
        p = _make_precedent(days_old=500, similarity=0.99)
        score = score_precedent(p, "Receiving Dock", "zone_congestion", "zone_wide")
        # Old precedent should be capped by STALE_CAP
        from mars.config import RV_STALE_CAP
        assert score <= RV_STALE_CAP + 0.05

    def test_zone_mismatch_reduces_score(self):
        p_same = _make_precedent(zone="Receiving Dock")
        p_diff = _make_precedent(zone="Storage Area")
        s_same = score_precedent(p_same, "Receiving Dock", "zone_congestion", "zone_wide")
        s_diff = score_precedent(p_diff, "Receiving Dock", "zone_congestion", "zone_wide")
        assert s_same > s_diff

    def test_recent_beats_old_same_metadata(self):
        p_recent = _make_precedent(days_old=1)
        p_old    = _make_precedent(days_old=180)
        s_recent = score_precedent(p_recent, "Receiving Dock", "zone_congestion", "zone_wide")
        s_old    = score_precedent(p_old,    "Receiving Dock", "zone_congestion", "zone_wide")
        assert s_recent > s_old


class TestValidateRetrievalSet:
    def test_empty_precedents_gives_low_trust(self):
        result = validate_retrieval_set([], "Receiving Dock", "zone_congestion", "zone_wide")
        assert result["set_level"] == "LOW"
        assert result["support_count"] == 0

    def test_three_consistent_give_high_trust(self):
        precedents = [
            _make_precedent(similarity=0.9),
            _make_precedent(similarity=0.85),
            _make_precedent(similarity=0.88),
        ]
        result = validate_retrieval_set(precedents, "Receiving Dock", "zone_congestion", "zone_wide")
        assert result["set_level"] == "HIGH"
        assert result["support_count"] == 3

    def test_low_similarity_filtered_out(self):
        # All below accept threshold → LOW
        precedents = [_make_precedent(similarity=0.1, days_old=400)]
        result = validate_retrieval_set(precedents, "Receiving Dock", "zone_congestion", "zone_wide")
        assert result["set_level"] == "LOW"
        assert result["support_count"] == 0

    def test_coverage_mismatch_caps_trust_at_medium(self):
        # High similarity but totally wrong scope (isolated vs fleet_wide).
        # Per DECISIONS.md §4: COV_MISMATCH_CAP == ACCEPT_THRESHOLD (both 0.5),
        # so coverage mismatch ALONE still passes the filter (it is right at the
        # threshold), but caps the trust score — combined with any other weakness
        # the result would be filtered.  The important invariant is that the score
        # is <= CAP even at 0.99 similarity.
        precedents = [
            _make_precedent(scope="isolated", similarity=0.99),
        ]
        result = validate_retrieval_set(precedents, "Receiving Dock", "zone_congestion", "fleet_wide")
        # Score must be capped regardless of high similarity
        survivor_score = result["filtered_precedents"][0]["_trust_score"] if result["filtered_precedents"] else None
        if survivor_score is not None:
            assert survivor_score <= RV_COV_MISMATCH_CAP + 0.01
        # Set level must not be HIGH (single capped survivor is at best MEDIUM)
        assert result["set_level"] != "HIGH"
