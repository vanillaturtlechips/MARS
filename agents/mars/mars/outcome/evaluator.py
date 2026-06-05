"""
Outcome Evaluator — §7

Watches a metric window after a policy or recovery action, labels the outcome
(improved / no_effect / worsened), and writes the labeled record to the
blackboard with an embedding.

The embedding is the key part: RAG retrieval surfaces *outcome-labeled*
precedent — strategies that worked, strategies that didn't — so the agents
learn from prior decisions.

Driven by FleetMonitor.tick() via the optional outcome_evaluator parameter.
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Any

from mars.config import OUTCOME_WINDOW_SEC

log = logging.getLogger(__name__)


class OutcomeEvaluator:
    def __init__(self, blackboard_queries, embedder, conn_factory):
        self._bb   = blackboard_queries
        self._emb  = embedder
        self._conn = conn_factory
        # Pending watches: action_id → {action_type, zone, baseline, start_time}
        self._watches: dict[str, dict[str, Any]] = {}

    def register_watch(
        self,
        action_type: str,
        action_id: str,
        zone: str | None,
        baseline_metrics: dict[str, Any],
    ) -> None:
        """Start watching.  Called immediately after a policy is activated."""
        self._watches[action_id] = {
            "action_type": action_type,
            "zone":        zone,
            "baseline":    baseline_metrics,
            "start":       time.time(),
        }
        log.info(
            "[outcome_evaluator] watching %s=%s zone=%s for %ds",
            action_type, action_id, zone, OUTCOME_WINDOW_SEC,
        )

    def tick(self, conn=None) -> int:
        """
        Check if any watch windows have expired; label and write outcomes.
        Returns the number of outcomes evaluated this tick.

        conn: optional caller-provided connection.  If None, creates its own.
        """
        now           = time.time()
        own_conn      = conn is None
        if own_conn:
            conn = self._conn()

        evaluated = 0
        try:
            for action_id in list(self._watches):
                w = self._watches[action_id]
                if now - w["start"] >= OUTCOME_WINDOW_SEC:
                    try:
                        self._evaluate(action_id, w, conn)
                        evaluated += 1
                    except Exception:
                        log.exception(
                            "[outcome_evaluator] evaluation failed for %s=%s",
                            w["action_type"], action_id,
                        )
                    finally:
                        del self._watches[action_id]
            if evaluated:
                conn.commit()
        finally:
            if own_conn:
                try:
                    conn.close()
                except Exception:
                    pass

        return evaluated

    def pending_count(self) -> int:
        return len(self._watches)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate(self, action_id: str, watch: dict[str, Any], conn) -> None:
        zone = watch.get("zone")
        baseline_rate = watch["baseline"].get("failure_rate", 0.0)

        # Current failure rate in the observation window
        recent = self._bb.get_recent_failures(
            conn, window_seconds=OUTCOME_WINDOW_SEC, zone=zone
        )
        current_rate = len(recent) / max(OUTCOME_WINDOW_SEC / 60.0, 1.0)

        if current_rate < baseline_rate * 0.7:
            label     = "improved"
            magnitude = (baseline_rate - current_rate) / max(baseline_rate, 0.001)
        elif current_rate > baseline_rate * 1.3:
            label     = "worsened"
            magnitude = (current_rate - baseline_rate) / max(baseline_rate, 0.001)
        else:
            label     = "no_effect"
            magnitude = 0.0

        window_start = datetime.datetime.fromtimestamp(watch["start"],
                                                        tz=datetime.timezone.utc)
        window_end   = datetime.datetime.now(datetime.timezone.utc)

        outcome_id = self._bb.write_outcome(conn, {
            "action_type":      watch["action_type"],
            "action_id":        action_id,
            "baseline_metrics": watch["baseline"],
            "final_metrics":    {"failure_rate": current_rate},
            "label":            label,
            "magnitude":        magnitude,
            "window_start":     window_start,
            "window_end":       window_end,
        })

        # Embed the outcome.  outcome_id is attached so future retrievals return
        # outcome-labeled precedents (§7 — closes the RAG loop).
        summary = (
            f"action={watch['action_type']} zone={zone} outcome={label} "
            f"baseline_rate={baseline_rate:.2f} final_rate={current_rate:.2f}"
        )
        try:
            from mars.blackboard.queries import write_embedding
            embedding = self._emb.embed(summary)
            write_embedding(conn, {
                "source_type":  "outcome",
                "source_id":    outcome_id,
                "zone":         zone,
                "scope":        None,
                "outcome_label": label,
                "outcome_id":   outcome_id,
                "summary":      summary,
                "embedding":    embedding,
                "recorded_at":  window_end,
            })
        except Exception:
            log.warning(
                "[outcome_evaluator] embedding failed for outcome %s — skipping",
                outcome_id,
            )

        log.info(
            "[outcome_evaluator] %s=%s → %s (magnitude=%.2f)",
            watch["action_type"], action_id, label, magnitude,
        )
