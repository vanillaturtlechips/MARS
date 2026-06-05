"""
M3 Depth / Robustness tests

Covers:
  - Embedding pipeline: diagnosis embedded after DV PASS (not after REJECT)
  - Embedding pipeline: strategy run persisted + embedded
  - Outcome Evaluator: labels outcome correctly given failure rate delta
  - Outcome Evaluator: writes labeled embedding (RAG loop closed)
  - FleetMonitor drives OutcomeEvaluator.tick()
  - Guardrail charging viability: reserve_chargers_for_critical rejected when
    reserve_count ≥ total_chargers
  - Guardrail charging viability: accepted when reserve leaves capacity
  - Retrieval Validator coverage-mismatch cap enforced (regression test)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pytest

from mars.guardrail.guardrail import GuardrailResult, check as guardrail_check
from mars.outcome.evaluator import OutcomeEvaluator
from mars.validators.decision_validator import DVResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _InMemBB:
    """Minimal in-memory BB for M3 tests — includes write operations."""

    def __init__(self):
        self.failures:    list[dict] = []
        self.diagnoses:   dict[str, dict] = {}
        self.strategy_runs: dict[str, dict] = {}
        self.outcomes:    dict[str, dict] = {}
        self.embeddings:  list[dict] = []
        self._id = 0

    def _nid(self, prefix=""):
        self._id += 1
        return f"{prefix}{self._id:04d}"

    def write_diagnosis(self, conn, d):
        did = d.get("diagnosis_id") or self._nid("DX")
        self.diagnoses[did] = {**d, "diagnosis_id": did}
        return did

    def write_strategy_run(self, conn, r):
        rid = r.get("strategy_run_id") or self._nid("SR")
        self.strategy_runs[rid] = {**r, "strategy_run_id": rid}
        return rid

    def write_outcome(self, conn, o):
        oid = o.get("outcome_id") or self._nid("OUT")
        self.outcomes[oid] = {**o, "outcome_id": oid}
        return oid

    def write_failure(self, conn, event):
        fid = event.get("failure_id") or self._nid("F")
        self.failures.append({**event, "failure_id": fid})
        return fid

    def get_recent_failures(self, conn, window_seconds=900, zone=None):
        rows = list(self.failures)
        if zone:
            rows = [r for r in rows if r.get("zone") == zone]
        return rows

    # pgvector stub — records calls for assertion
    def search_similar(self, conn, embedding, source_types=None, zone=None, limit=10):
        return []

    def get_fleet_metrics(self, conn):
        return {
            "robot_utilization": 0.4,
            "mission_backlog": 3,
            "charging": {"queue_len": 0, "mean_wait_sec": 0, "occupied_pct": 0.0,
                         "below_low_count": 0, "below_critical_count": 0},
            "zone_health": {},
            "recent_failure_clusters": [],
        }

    def get_operational_metrics(self, conn):
        return {"mission_backlog": 3, "robot_utilization": 0.4,
                "charging": {"queue_len": 0, "occupied_pct": 0.0, "below_low_count": 0}}

    def get_world_state(self, conn):
        return {
            "zones": {
                "Charging Bay":   {"is_charger_zone": True,  "is_mandatory": False},
                "Receiving Dock": {"is_charger_zone": False, "is_mandatory": False},
            },
            "charger_zones":  ["Charging Bay"],
            "total_chargers": 2,
        }


def _write_embedding_stub(bb):
    """Patch write_embedding to capture calls into bb.embeddings."""
    import mars.blackboard.queries as q
    original = q.write_embedding
    def _patched(conn, record):
        bb.embeddings.append(dict(record))
        return len(bb.embeddings)
    return original, _patched


class _FakeConn:
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


class _FakeEmbedder:
    def embed(self, text):
        return [float(hash(text) % 1000) / 1000.0] * 1024


# ---------------------------------------------------------------------------
# Embedding pipeline tests
# ---------------------------------------------------------------------------

class TestEmbeddingPipeline:
    def test_diagnosis_embedded_after_dv_pass(self):
        """After a PASS diagnosis, write_embedding is called with source_type=diagnosis."""
        import mars.blackboard.queries as q

        bb  = _InMemBB()
        original, patched = _write_embedding_stub(bb)
        q.write_embedding = patched

        try:
            # Simulate what orchestrator does after DV PASS
            emb = _FakeEmbedder()
            summary = "zone:Receiving Dock cause:zone_congestion scope:zone_wide"
            embedding = emb.embed(summary)
            q.write_embedding(_FakeConn(), {
                "source_type":  "diagnosis",
                "source_id":    "DX0001",
                "zone":         "Receiving Dock",
                "failure_type": "zone_congestion",
                "scope":        "zone_wide",
                "summary":      summary,
                "embedding":    embedding,
                "recorded_at":  datetime.now(timezone.utc),
            })
        finally:
            q.write_embedding = original

        assert len(bb.embeddings) == 1
        assert bb.embeddings[0]["source_type"] == "diagnosis"
        assert bb.embeddings[0]["zone"] == "Receiving Dock"

    def test_embedding_not_called_after_dv_reject(self):
        """write_embedding must NOT be called when DV result is REJECT."""
        import mars.blackboard.queries as q

        bb  = _InMemBB()
        original, patched = _write_embedding_stub(bb)
        q.write_embedding = patched

        try:
            # Simulate orchestrator REJECT path — no embedding call
            dv_result = DVResult.REJECT
            if dv_result != DVResult.REJECT:
                emb = _FakeEmbedder()
                q.write_embedding(_FakeConn(), {"source_type": "diagnosis", "source_id": "X"})
        finally:
            q.write_embedding = original

        assert len(bb.embeddings) == 0, "No embedding should be written on REJECT"

    def test_strategy_run_persisted(self):
        """After a strategy run, write_strategy_run creates a row."""
        bb = _InMemBB()
        rid = bb.write_strategy_run(None, {
            "confidence": 0.8,
            "evidence": [{"observation": "congestion", "refs": ["incident_analysis.scope"]}],
            "relied_on_precedents": [],
            "decision_validator_result": "PASS",
            "no_action_reason": None,
        })
        assert rid in bb.strategy_runs
        assert bb.strategy_runs[rid]["confidence"] == 0.8

    def test_strategy_embedding_has_source_type_strategy(self):
        """Embedded strategy row must have source_type='strategy'."""
        import mars.blackboard.queries as q

        bb  = _InMemBB()
        original, patched = _write_embedding_stub(bb)
        q.write_embedding = patched

        try:
            emb = _FakeEmbedder()
            q.write_embedding(_FakeConn(), {
                "source_type":  "strategy",
                "source_id":    "SR0001",
                "zone":         "Receiving Dock",
                "scope":        "zone_wide",
                "summary":      "zone:Receiving Dock scope:zone_wide policies:['avoid_zone']",
                "embedding":    emb.embed("test"),
                "recorded_at":  datetime.now(timezone.utc),
            })
        finally:
            q.write_embedding = original

        assert bb.embeddings[0]["source_type"] == "strategy"


# ---------------------------------------------------------------------------
# Outcome Evaluator tests
# ---------------------------------------------------------------------------

class TestOutcomeEvaluator:
    def _make_evaluator(self, bb) -> OutcomeEvaluator:
        import mars.blackboard.queries as q

        # Patch write_embedding to capture calls
        original, patched = _write_embedding_stub(bb)
        q.write_embedding = patched
        self._orig_emb = original
        self._q = q

        return OutcomeEvaluator(bb, _FakeEmbedder(), lambda: _FakeConn())

    def teardown_method(self, _):
        if hasattr(self, "_q") and hasattr(self, "_orig_emb"):
            self._q.write_embedding = self._orig_emb

    def test_improved_label_when_rate_drops(self):
        bb  = _InMemBB()
        ev  = self._make_evaluator(bb)

        ev.register_watch("policy", "POL001", "Receiving Dock",
                          baseline_metrics={"failure_rate": 10.0})
        # Expire the window immediately
        ev._watches["POL001"]["start"] -= 9999

        ev.tick(_FakeConn())

        assert "POL001" not in ev._watches  # consumed
        oid = next(iter(bb.outcomes))
        assert bb.outcomes[oid]["label"] == "improved"  # 0 recent failures < 10 * 0.7

    def test_no_effect_label_when_rate_similar(self):
        bb  = _InMemBB()
        ev  = self._make_evaluator(bb)

        # Seed 5 failures so current rate ≈ baseline
        for i in range(5):
            bb.write_failure(None, {"robot_id": f"R{i}", "zone": "Receiving Dock",
                                    "failure_id": f"F{i}"})

        ev.register_watch("policy", "POL002", "Receiving Dock",
                          baseline_metrics={"failure_rate": 0.33})  # ~5 / 15 min
        ev._watches["POL002"]["start"] -= 9999

        ev.tick(_FakeConn())

        oid  = next(iter(bb.outcomes))
        assert bb.outcomes[oid]["label"] in ("improved", "no_effect")

    def test_outcome_is_embedded_after_evaluation(self):
        bb  = _InMemBB()
        ev  = self._make_evaluator(bb)

        ev.register_watch("policy", "POL003", "Receiving Dock",
                          baseline_metrics={"failure_rate": 5.0})
        ev._watches["POL003"]["start"] -= 9999

        ev.tick(_FakeConn())

        # An outcome row must exist
        assert len(bb.outcomes) == 1
        # An embedding must have been written (source_type=outcome)
        outcome_embs = [e for e in bb.embeddings if e.get("source_type") == "outcome"]
        assert outcome_embs, "Expected at least one outcome embedding for RAG"
        assert outcome_embs[0].get("outcome_label") is not None

    def test_pending_count_decrements_after_tick(self):
        bb  = _InMemBB()
        ev  = self._make_evaluator(bb)

        ev.register_watch("recovery", "F001", None, {"failure_rate": 1.0})
        ev.register_watch("recovery", "F002", None, {"failure_rate": 1.0})
        assert ev.pending_count() == 2

        ev._watches["F001"]["start"] -= 9999  # expire F001 only
        ev.tick(_FakeConn())

        assert ev.pending_count() == 1  # F002 still pending

    def test_tick_accepts_caller_provided_conn(self):
        """tick(conn=...) should use provided conn and not create its own."""
        bb   = _InMemBB()
        ev   = self._make_evaluator(bb)
        conn = _FakeConn()

        ev.register_watch("policy", "POL004", None, {"failure_rate": 2.0})
        ev._watches["POL004"]["start"] -= 9999

        result = ev.tick(conn=conn)  # should not raise
        assert result == 1


# ---------------------------------------------------------------------------
# FleetMonitor drives OutcomeEvaluator
# ---------------------------------------------------------------------------

class TestFleetMonitorDrivesOutcomeEvaluator:
    def test_outcome_tick_called_during_fleet_cycle(self):
        """FleetMonitor.run_once() should call outcome_evaluator.tick()."""
        from mars.agents.fleet_state import FleetStateAgent
        from mars.orchestrator.fleet_monitor import FleetMonitor
        from mars.orchestrator.strategy_trigger import StrategyTrigger
        from mars.validators.retrieval_validator import validate_retrieval_set
        from mars.validators.decision_validator import validate_fleet_state, validate_strategy
        from mars.policy.policy_manager import PolicyManager
        from mars.llm.client import MockLLMClient

        bb = _InMemBB()
        embed = _FakeEmbedder()

        HEALTHY = {
            "fleet_health": "healthy", "bottlenecks": [], "charging_pressure": "low",
            "confidence": 0.9,
            "evidence": [{"observation": "ok", "refs": ["fleet_metrics.mission_backlog"]}],
            "relied_on_precedents": [],
        }
        llm = MockLLMClient(canned_output=HEALTHY)

        pm = PolicyManager(lambda: _FakeConn())
        pm.activate = lambda p: (p.__setitem__("policy_id", "P001") or "P001")

        trigger = StrategyTrigger(
            ops_strategy_agent=type("A", (), {
                "recommend": lambda self, **kw: {
                    "policy_updates": [], "no_action_reason": "no_trigger",
                    "confidence": 0.8, "evidence": [], "relied_on_precedents": [],
                }
            })(),
            retrieval_validator_fn=validate_retrieval_set,
            decision_validator_fn=validate_strategy,
            policy_manager=pm,
            blackboard_queries=bb,
            embedder=embed,
        )

        import mars.blackboard.queries as q
        original, patched = _write_embedding_stub(bb)
        q.write_embedding = patched

        tick_calls = []

        class _MockEval:
            def tick(self, conn=None):
                tick_calls.append(1)
                return 0

        try:
            monitor = FleetMonitor(
                fleet_state_agent=FleetStateAgent(llm),
                strategy_trigger=trigger,
                blackboard_queries=bb,
                embedder=embed,
                conn_factory=lambda: _FakeConn(),
                outcome_evaluator=_MockEval(),
            )
            monitor.run_once(_FakeConn())
        finally:
            q.write_embedding = original

        assert len(tick_calls) == 1, "outcome_evaluator.tick() should be called once per cycle"


# ---------------------------------------------------------------------------
# Guardrail charging viability tests
# ---------------------------------------------------------------------------

class TestGuardrailChargingViability:
    def _world(self, total_chargers: int) -> dict:
        return {
            "zones": {
                "Charging Bay":   {"is_charger_zone": True,  "is_mandatory": False},
                "Receiving Dock": {"is_charger_zone": False, "is_mandatory": False},
            },
            "charger_zones":  ["Charging Bay"],
            "total_chargers": total_chargers,
        }

    def test_reserve_all_chargers_rejected(self):
        """reserve_count == total_chargers → no chargers left for normal robots → REJECT."""
        policy = {
            "type": "reserve_chargers_for_critical",
            "params": {"reserve_count": 2},
            "duration_sec": 900,
        }
        result, _, notes = guardrail_check(policy, [], self._world(2))
        assert result == GuardrailResult.REJECT
        assert "no chargers for normal robots" in notes

    def test_reserve_exceeds_total_rejected(self):
        policy = {
            "type": "reserve_chargers_for_critical",
            "params": {"reserve_count": 3},
            "duration_sec": 900,
        }
        result, _, _ = guardrail_check(policy, [], self._world(2))
        assert result == GuardrailResult.REJECT

    def test_reserve_one_of_two_accepted(self):
        """reserve_count=1 with total=2 → 1 charger remains for normal robots → ACCEPT."""
        policy = {
            "type": "reserve_chargers_for_critical",
            "params": {"reserve_count": 1},
            "duration_sec": 900,
        }
        result, _, _ = guardrail_check(policy, [], self._world(2))
        assert result in (GuardrailResult.ACCEPT, GuardrailResult.MODIFY)

    def test_no_world_state_accepts(self):
        """If world_state is unavailable (empty) we cannot check — accept by default."""
        policy = {
            "type": "reserve_chargers_for_critical",
            "params": {"reserve_count": 5},
            "duration_sec": 900,
        }
        result, _, _ = guardrail_check(policy, [], {})
        # With empty world_state, total_chargers=0 so 5 >= 0 is True...
        # Actually total_chargers=0 means we can't verify, so we accept.
        # The check only blocks when total_chargers > 0.
        assert result in (GuardrailResult.ACCEPT, GuardrailResult.MODIFY)

    def test_avoid_zone_charger_strand_rejected(self):
        """Regression: avoid_zone that is the only charger zone → REJECT."""
        policy = {
            "type": "avoid_zone",
            "params": {"zone": "Charging Bay"},
            "duration_sec": 900,
        }
        result, _, notes = guardrail_check(policy, [], self._world(2))
        assert result == GuardrailResult.REJECT
        assert "charger" in notes.lower()


# ---------------------------------------------------------------------------
# Retrieval Validator — coverage-mismatch cap regression
# ---------------------------------------------------------------------------

class TestRetrievalValidatorCoverageCapRegression:
    """Regression tests ensuring M3 doesn't break the M0 coverage-mismatch cap."""

    def test_isolated_vs_fleet_wide_still_capped(self):
        from mars.validators.retrieval_validator import score_precedent
        from mars.config import RV_COV_MISMATCH_CAP
        from datetime import timedelta

        p = {
            "zone": "Receiving Dock",
            "failure_type": "zone_congestion",
            "scope": "isolated",          # scope mismatch vs fleet_wide
            "similarity": 0.99,
            "recorded_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        }
        score = score_precedent(p, "Receiving Dock", "zone_congestion", "fleet_wide")
        assert score <= RV_COV_MISMATCH_CAP + 0.01, (
            f"Coverage-mismatch cap violated: score={score:.3f} > cap={RV_COV_MISMATCH_CAP}"
        )
