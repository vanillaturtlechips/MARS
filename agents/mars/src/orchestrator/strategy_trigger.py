"""
Strategy Trigger Rules — §3a

Time-windowed join of failure analysis (event-driven) and fleet analysis
(periodic).  Fires the Operations Strategy Agent when thresholds are crossed.

Two entry points:
  evaluate(failure_out, ...)  — called by Orchestrator after slow-path diagnosis
  evaluate_fleet(conn)        — called by FleetMonitor after periodic fleet analysis;
                                failure_out is None (fleet-only trigger path)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from mars.config import (
    STRATEGY_BACKLOG_THRESHOLD,
    STRATEGY_CORRELATION_WINDOW_SECONDS,
)

log = logging.getLogger(__name__)

TRIGGER    = "TRIGGER"
NO_TRIGGER = "NO_TRIGGER"


class StrategyTrigger:
    """
    Stateful — holds the most recent validated fleet analysis output so it can
    be correlated with incoming failure analyses within a time window (§3a).
    """

    def __init__(
        self,
        ops_strategy_agent,
        retrieval_validator_fn,
        decision_validator_fn,
        policy_manager,
        blackboard_queries,
        embedder,
    ):
        self._ops_agent = ops_strategy_agent
        self._rv        = retrieval_validator_fn
        self._dv        = decision_validator_fn
        self._pm        = policy_manager
        self._bb        = blackboard_queries
        self._embedder  = embedder
        self._last_fleet_out: dict[str, Any] | None = None
        self._last_fleet_ts: float = 0.0

    # ------------------------------------------------------------------
    # Called by FleetMonitor after a validated fleet analysis
    # ------------------------------------------------------------------

    def update_fleet_analysis(self, fleet_out: dict[str, Any]) -> None:
        self._last_fleet_out = fleet_out
        self._last_fleet_ts  = time.time()
        log.debug("[strategy_trigger] fleet analysis updated: health=%s pressure=%s",
                  fleet_out.get("fleet_health"), fleet_out.get("charging_pressure"))

    # ------------------------------------------------------------------
    # Called by Orchestrator after slow-path failure diagnosis
    # ------------------------------------------------------------------

    def evaluate(
        self,
        failure_out: dict[str, Any],
        failure_id: str,
        diagnosis_id: str,
        conn,
    ) -> str:
        """Failure-driven entry.  Merges with current fleet analysis if recent."""
        fleet_out = self._current_fleet_analysis()

        scope   = failure_out.get("scope", "isolated")
        backlog = (fleet_out or {}).get("mission_backlog", 0)

        triggered = scope in ("zone_wide", "fleet_wide") or backlog > STRATEGY_BACKLOG_THRESHOLD

        if not triggered:
            log.info("[strategy_trigger] NO_TRIGGER  scope=%s", scope)
            return NO_TRIGGER

        log.info("[strategy_trigger] TRIGGER  source=failure  scope=%s", scope)
        self._run_strategy(
            failure_out=failure_out,
            fleet_out=fleet_out,
            conn=conn,
            diagnosis_id=diagnosis_id,
        )
        return TRIGGER

    # ------------------------------------------------------------------
    # Called by FleetMonitor — fleet-only trigger (no current failure)
    # ------------------------------------------------------------------

    def evaluate_fleet(self, fleet_out: dict[str, Any], conn) -> str:
        """
        Fleet-only entry — fires when fleet metrics cross a threshold without a
        concurrent failure event.  Runs the same Operations Strategy path.
        """
        charging_pressure = fleet_out.get("charging_pressure", "low")
        backlog           = fleet_out.get("mission_backlog", 0)
        fleet_health      = fleet_out.get("fleet_health", "healthy")

        triggered = (
            charging_pressure == "high"
            or backlog > STRATEGY_BACKLOG_THRESHOLD
            or fleet_health in ("degraded", "critical")
        )

        if not triggered:
            log.info("[strategy_trigger] NO_TRIGGER  source=fleet  health=%s", fleet_health)
            return NO_TRIGGER

        log.info("[strategy_trigger] TRIGGER  source=fleet  health=%s pressure=%s",
                 fleet_health, charging_pressure)
        self._run_strategy(
            failure_out=None,
            fleet_out=fleet_out,
            conn=conn,
            diagnosis_id=None,
        )
        return TRIGGER

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_fleet_analysis(self) -> dict[str, Any] | None:
        if self._last_fleet_out is None:
            return None
        age = time.time() - self._last_fleet_ts
        if age > STRATEGY_CORRELATION_WINDOW_SECONDS:
            log.debug("[strategy_trigger] fleet analysis stale (%ds) — ignoring", int(age))
            return None
        return self._last_fleet_out

    def _run_strategy(
        self,
        failure_out: dict[str, Any] | None,
        fleet_out: dict[str, Any] | None,
        conn,
        diagnosis_id: str | None,
    ) -> None:
        from mars.validators.decision_validator import DVResult, validate_strategy
        from mars.guardrail.guardrail import check as guardrail_check, GuardrailResult

        # For retrieval: use zone and scope from failure_out when available
        zone         = (failure_out or {}).get("affected_zone")
        scope        = (failure_out or {}).get("scope")
        failure_type = (failure_out or {}).get("cause")

        active_policies = self._pm.get_active()
        ops_metrics     = (
            self._bb.get_operational_metrics(conn)
            if hasattr(self._bb, "get_operational_metrics")
            else {}
        )

        # Retrieve strategy precedents
        try:
            query     = f"zone:{zone} scope:{scope}"
            embedding = self._embedder.embed(query)
            raw       = self._bb.search_similar(
                conn, embedding,
                source_types=["strategy", "outcome"],
                zone=zone, limit=5,
            )
        except Exception:
            log.exception("[strategy_trigger] retrieval failed — proceeding without precedent")
            raw = []

        rv_result = self._rv(
            raw,
            current_zone=zone,
            current_failure_type=failure_type,
            current_scope=scope,
        )
        retrieval_trust = {
            "set_level":     rv_result["set_level"],
            "support_count": rv_result["support_count"],
        }

        strategy_out = self._ops_agent.recommend(
            incident_analysis=failure_out,   # may be None for fleet-only trigger
            fleet_analysis=fleet_out,
            operational_metrics=ops_metrics,
            active_policies=active_policies,
            retrieved_precedents=rv_result["filtered_precedents"],
            retrieval_trust=retrieval_trust,
        )

        dv_result, dv_notes = validate_strategy(
            strategy_out, strategy_out.get("_input_bundle", {}), retrieval_trust
        )

        if dv_result == DVResult.REJECT:
            log.warning("[strategy_trigger] strategy REJECTED: %s", dv_notes)
            return

        world_state  = self._bb.get_world_state(conn) if hasattr(self._bb, "get_world_state") else {}
        last_applied = self._pm.get_last_applied()

        # Persist the strategy run BEFORE activating policies so FKs resolve.
        strategy_run_id = None
        if hasattr(self._bb, "write_strategy_run"):
            try:
                strategy_run_id = self._bb.write_strategy_run(conn, {
                    "incident_diagnosis_id": diagnosis_id,
                    "confidence":            strategy_out.get("confidence", 0.0),
                    "evidence":              strategy_out.get("evidence", []),
                    "relied_on_precedents":  strategy_out.get("relied_on_precedents", []),
                    "decision_validator_result": dv_result.value,
                    "decision_validator_notes":  dv_notes,
                    "retrieval_trust_level":     retrieval_trust.get("set_level"),
                    "no_action_reason":      strategy_out.get("no_action_reason"),
                })
            except Exception:
                log.warning("[strategy_trigger] write_strategy_run failed — continuing without persist")

        # Embed the strategy recommendation for future RAG retrieval.
        if strategy_run_id is not None and hasattr(self._bb, "search_similar"):
            try:
                from mars.blackboard.queries import write_embedding
                from datetime import datetime, timezone as _tz
                _policy_types = [p.get("type") for p in strategy_out.get("policy_updates", [])]
                _summary = (
                    f"zone:{zone} scope:{scope} cause:{failure_type} "
                    f"policies:{_policy_types} "
                    f"confidence:{strategy_out.get('confidence', 0.0):.2f}"
                )
                _emb = self._embedder.embed(_summary)
                write_embedding(conn, {
                    "source_type":  "strategy",
                    "source_id":    strategy_run_id,
                    "zone":         zone,
                    "failure_type": failure_type,
                    "scope":        scope,
                    "summary":      _summary,
                    "embedding":    _emb,
                    "recorded_at":  datetime.now(_tz.utc),
                })
            except Exception:
                log.warning("[strategy_trigger] strategy embedding failed — skipping")

        for proposal in strategy_out.get("policy_updates", []):
            gr, modified, gr_notes = guardrail_check(
                proposal, active_policies, world_state, last_applied
            )
            if gr in (GuardrailResult.ACCEPT, GuardrailResult.MODIFY):
                modified["guardrail_result"] = gr.value
                modified["guardrail_notes"]  = gr_notes
                modified["diagnosis_id"]     = diagnosis_id
                modified["strategy_run_id"]  = strategy_run_id
                self._pm.activate(modified)
                log.info("[strategy_trigger] policy activated: %s", modified.get("type"))
            else:
                log.warning("[strategy_trigger] guardrail %s for %s: %s",
                            gr, proposal.get("type"), gr_notes)
