"""
Fleet Monitoring Workflow — §3a

Periodic (or threshold-triggered) fleet health assessment.  Runs independently
of the failure-driven Orchestrator and bypasses the Deterministic Router.

Loop:
  1. Read fleet_metrics from the blackboard
  2. Call Fleet State Analysis Agent
  3. Decision Validator (same gate as failure analysis)
  4. StrategyTrigger.update_fleet_analysis() — arms the time-windowed join
  5. StrategyTrigger.evaluate_fleet() — fires the Operations Strategy Agent
     when fleet metrics cross a threshold (charging pressure, backlog, health)

The charging pressure→policy loop closes here:
  ChargingService emits pressure metrics → blackboard
  → get_fleet_metrics() reads them
  → Fleet State Agent sees charging_pressure: high
  → evaluate_fleet() triggers Operations Strategy Agent
  → agent recommends lower_target_charge_level / reserve_chargers_for_critical
  → policy applied → ChargingService changes behaviour
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from mars.config import FLEET_MONITOR_INTERVAL_SEC
from mars.validators.decision_validator import validate_fleet_state
from mars.validators.retrieval_validator import validate_retrieval_set

log = logging.getLogger(__name__)


class FleetMonitor:
    """
    Runs the Fleet State Analysis Agent on a periodic interval.

    Start with:    monitor.start()
    Stop with:     monitor.stop()
    Single run:    monitor.run_once(conn)  (useful in tests)
    """

    def __init__(
        self,
        fleet_state_agent,
        strategy_trigger,
        blackboard_queries,
        embedder,
        conn_factory,
        interval_sec: float = FLEET_MONITOR_INTERVAL_SEC,
        outcome_evaluator=None,
    ):
        self._agent              = fleet_state_agent
        self._trigger            = strategy_trigger
        self._bb                 = blackboard_queries
        self._embedder           = embedder
        self._conn               = conn_factory
        self._interval_sec       = interval_sec
        self._outcome_evaluator  = outcome_evaluator   # optional; drives tick() each cycle
        self._thread: threading.Thread | None = None
        self._running            = False

        # Most recent validated output — readable by tests
        self._last_output: dict[str, Any] | None = None
        self._last_run_ts: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="fleet-monitor")
        self._thread.start()
        log.info("[fleet_monitor] started (interval=%ds)", int(self._interval_sec))

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._interval_sec + 2)
        log.info("[fleet_monitor] stopped")

    # ------------------------------------------------------------------
    # Single cycle — also used directly in tests
    # ------------------------------------------------------------------

    def run_once(self, conn) -> dict[str, Any] | None:
        """
        Execute one fleet assessment cycle.

        Returns the validated fleet analysis dict (or None if the run was
        rejected or failed).
        """
        try:
            return self._do_run(conn)
        except Exception:
            log.exception("[fleet_monitor] run_once failed")
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            t0   = time.monotonic()
            conn = self._conn()
            try:
                self._do_run(conn)
                conn.commit()
            except Exception:
                log.exception("[fleet_monitor] tick failed")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            elapsed = time.monotonic() - t0
            sleep   = max(0.0, self._interval_sec - elapsed)
            time.sleep(sleep)

    def _do_run(self, conn) -> dict[str, Any] | None:
        # 1. Assemble fleet metrics from blackboard
        fleet_metrics = (
            self._bb.get_fleet_metrics(conn)
            if hasattr(self._bb, "get_fleet_metrics")
            else {}
        )

        # 2. Retrieve fleet-level precedents (empty on first run — expected)
        try:
            embedding = self._embedder.embed(
                f"fleet_health:{fleet_metrics.get('robot_utilization', 0)} "
                f"charging_pressure:{fleet_metrics.get('charging', {}).get('occupied_pct', 0)}"
            )
            raw = self._bb.search_similar(
                conn, embedding,
                source_types=["diagnosis", "outcome"],
                limit=5,
            )
        except Exception:
            log.debug("[fleet_monitor] retrieval failed — using empty precedents")
            raw = []

        rv_result = validate_retrieval_set(
            raw,
            current_zone=None,
            current_failure_type=None,
            current_scope=None,
        )
        retrieval_trust = {
            "set_level":     rv_result["set_level"],
            "support_count": rv_result["support_count"],
        }

        # 3. Fleet State Analysis Agent
        fleet_out = self._agent.assess(
            fleet_metrics=fleet_metrics,
            retrieved_precedents=rv_result["filtered_precedents"],
            retrieval_trust=retrieval_trust,
        )

        # 4. Decision Validator (same gate as failure analysis)
        bundle = fleet_out.get("_input_bundle", {"fleet_metrics": fleet_metrics})
        dv_result, dv_notes = validate_fleet_state(fleet_out, bundle, retrieval_trust)

        log.info(
            "[fleet_monitor] health=%s pressure=%s  DV=%s",
            fleet_out.get("fleet_health"), fleet_out.get("charging_pressure"),
            dv_result.value,
        )

        if dv_result.value == "REJECT":
            log.warning("[fleet_monitor] fleet analysis REJECTED: %s", dv_notes)
            return None

        # 5. Update StrategyTrigger's fleet state snapshot
        self._trigger.update_fleet_analysis(fleet_out)
        self._last_output = fleet_out
        self._last_run_ts = time.time()

        # 6. Fleet-only strategy trigger (charging pressure or high backlog)
        self._trigger.evaluate_fleet(fleet_out, conn)

        # 7. Drive OutcomeEvaluator: label any expired policy/recovery windows.
        #    This is what closes the RAG loop — labeled outcomes become
        #    retrievable precedent for future agent calls (§7).
        if self._outcome_evaluator is not None:
            evaluated = self._outcome_evaluator.tick(conn)
            if evaluated:
                log.info("[fleet_monitor] outcome evaluator labeled %d outcome(s)", evaluated)

        return fleet_out
