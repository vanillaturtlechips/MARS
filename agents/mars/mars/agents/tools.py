"""
Investigator Tools — §1 (Failure Analysis Agent ReAct loop)

Five read-only tools the investigator may call to gather evidence:

  query_failures     — recent failures in a window (all robots or filtered)
  get_zone_state     — zone occupancy, health, recent failure count
  get_robot_history  — failure history for a specific robot
  search_incidents   — vector similarity search over incident_embeddings;
                       Retrieval Validator scores are applied per-call
                       and attached to each result (Retrieval Validator
                       is no longer a pre-stage in the orchestrator)
  get_active_policies — currently active policies

Enforcement:
  The connection passed to InvestigatorTools MUST use the mars_reader
  read-only role (see 0002_readonly_role.sql).  This is enforced at the
  DB level — no write is physically possible through this connection.
  Use mars.blackboard.db.connect_readonly() to get the right connection.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# OpenAI tool definition dicts — passed to chat_with_tools()
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_failures",
            "description": (
                "Query recent navigation failures from the blackboard. "
                "Returns the raw failure events for the fleet (or a subset) "
                "within the configured window. Use this to understand the "
                "distribution of failures across robots and zones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": ["string", "null"],
                        "description": "Filter to failures in this zone. null = all zones.",
                    },
                    "robot_id": {
                        "type": ["string", "null"],
                        "description": "Filter to failures by this robot. null = all robots.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 20).",
                        "default": 20,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_zone_state",
            "description": (
                "Get the current state of a zone: occupancy, health status, "
                "and recent failure count. Use to understand whether a zone "
                "is congested, degraded, or blocked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "description": "Zone ID (e.g. 'Receiving Dock').",
                    },
                },
                "required": ["zone"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_history",
            "description": (
                "Retrieve the recent failure history for a specific robot. "
                "Useful to distinguish a robot-specific pattern (same robot "
                "failing in many zones) from a zone-wide pattern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "robot_id": {
                        "type": "string",
                        "description": "Robot ID (e.g. 'R5').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["robot_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_incidents",
            "description": (
                "Search the incident embeddings store for historically similar "
                "incidents. Returns past diagnoses/strategies/outcomes, each "
                "annotated with a trust score from the Retrieval Validator "
                "(score 0-1; scores < 0.5 indicate low reliability). "
                "Use to find relevant precedents for your diagnosis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text description of the incident to search for.",
                    },
                    "zone": {
                        "type": ["string", "null"],
                        "description": "Restrict results to this zone.",
                        "default": None,
                    },
                    "failure_type": {
                        "type": ["string", "null"],
                        "description": "Filter by failure_type (failure_cause enum).",
                        "default": None,
                    },
                    "scope": {
                        "type": ["string", "null"],
                        "description": "Current scope hint for trust scoring.",
                        "default": None,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_policies",
            "description": (
                "Get the list of currently active operational policies "
                "(e.g. avoid_zone, delay_low_priority_missions). Use to "
                "understand whether the situation is already being handled."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


class InvestigatorTools:
    """
    Wraps the five read-only tool implementations.

    conn must be a read-only connection (mars_reader role).
    embedder is used by search_incidents.
    """

    def __init__(self, conn, embedder, window_seconds: int = 900):
        self._conn    = conn
        self._embedder = embedder
        self._window  = window_seconds

    def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        """Route a tool call by name and return its result."""
        if name == "query_failures":
            return self._query_failures(**args)
        if name == "get_zone_state":
            return self._get_zone_state(**args)
        if name == "get_robot_history":
            return self._get_robot_history(**args)
        if name == "search_incidents":
            return self._search_incidents(**args)
        if name == "get_active_policies":
            return self._get_active_policies()
        raise ValueError(f"Unknown tool: {name!r}")

    # ------------------------------------------------------------------
    # Implementations — READ ONLY
    # ------------------------------------------------------------------

    def _query_failures(
        self,
        zone: str | None = None,
        robot_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        from mars.blackboard.queries import get_recent_failures
        rows = get_recent_failures(self._conn, window_seconds=self._window, zone=zone)
        if robot_id:
            rows = [r for r in rows if r.get("robot_id") == robot_id]
        # Strip large/irrelevant fields for token efficiency
        return [_slim_failure(r) for r in rows[:limit]]

    def _get_zone_state(self, zone: str) -> dict:
        from mars.blackboard.queries import get_recent_failures
        rows = get_recent_failures(self._conn, window_seconds=self._window, zone=zone)

        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT current_occupancy, health_status FROM zones WHERE zone_id = %s",
                (zone,),
            )
            row = cur.fetchone()

        occupancy     = row[0] if row else 0
        health_status = row[1] if row else "unknown"
        return {
            "zone":                 zone,
            "occupancy":            occupancy,
            "health_status":        health_status,
            "recent_failure_count": len(rows),
        }

    def _get_robot_history(self, robot_id: str, limit: int = 10) -> list[dict]:
        from mars.blackboard.queries import get_recent_failures
        rows = get_recent_failures(self._conn, window_seconds=self._window)
        robot_rows = [r for r in rows if r.get("robot_id") == robot_id]
        return [_slim_failure(r) for r in robot_rows[:limit]]

    def _search_incidents(
        self,
        query: str,
        zone: str | None = None,
        failure_type: str | None = None,
        scope: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Vector search + Retrieval Validator scoring (moved here from pre-stage).
        Each result is annotated with _trust_score so the agent can weigh it.
        """
        from mars.blackboard.queries import search_similar
        from mars.validators.retrieval_validator import score_precedent

        try:
            embedding = self._embedder.embed(query)
            raw = search_similar(
                self._conn, embedding,
                source_types=["diagnosis", "strategy", "outcome"],
                zone=zone,
                limit=limit * 2,  # over-fetch so filtering doesn't leave us empty
            )
        except Exception:
            log.warning("[tools/search_incidents] retrieval failed — returning empty")
            return []

        scored = []
        for r in raw:
            trust = score_precedent(r, zone, failure_type, scope)
            scored.append({
                "id":           r.get("source_id", ""),
                "source_type":  r.get("source_type", ""),
                "summary":      r.get("summary", ""),
                "zone":         r.get("zone"),
                "failure_type": r.get("failure_type"),
                "scope":        r.get("scope"),
                "outcome_label": r.get("outcome_label"),
                "_trust_score": round(trust, 3),
            })

        # Sort by trust descending; return top N
        scored.sort(key=lambda x: x["_trust_score"], reverse=True)
        return scored[:limit]

    def _get_active_policies(self) -> list[dict]:
        from mars.blackboard.queries import get_active_policies
        return [
            {"policy_id": p.get("policy_id"), "type": p.get("type"),
             "params": p.get("params"), "expires_at": str(p.get("expires_at", ""))}
            for p in get_active_policies(self._conn)
        ]


# ---------------------------------------------------------------------------
# Mock tools for tests — no DB required
# ---------------------------------------------------------------------------

class MockInvestigatorTools:
    """
    In-memory tools for unit/integration tests.

    failures_getter — optional callable () → list[dict]; called on each
    query_failures dispatch so tests can use a live reference to the in-memory
    BB's failure store.  Pass failures= for a static list.
    """

    def __init__(
        self,
        failures: list[dict] | None = None,
        failures_getter: "Callable[[], list[dict]] | None" = None,
        zone_states: dict[str, dict] | None = None,
        incidents: list[dict] | None = None,
        policies: list[dict] | None = None,
    ):
        self._static_failures = failures or []
        self._failures_getter = failures_getter
        self._zone_states     = zone_states or {}
        self._incidents       = incidents or []
        self._policies        = policies or []

    def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        if name == "query_failures":
            zone     = args.get("zone")
            robot_id = args.get("robot_id")
            limit    = args.get("limit", 20)
            rows = self._failures_getter() if self._failures_getter else self._static_failures
            if zone:     rows = [r for r in rows if r.get("zone") == zone]
            if robot_id: rows = [r for r in rows if r.get("robot_id") == robot_id]
            return rows[:limit]

        if name == "get_zone_state":
            zone = args["zone"]
            return self._zone_states.get(zone, {
                "zone": zone, "occupancy": 0, "health_status": "ok",
                "recent_failure_count": 0,
            })

        if name == "get_robot_history":
            robot_id = args["robot_id"]
            limit    = args.get("limit", 10)
            return [r for r in self._failures if r.get("robot_id") == robot_id][:limit]

        if name == "search_incidents":
            return self._incidents[:args.get("limit", 5)]

        if name == "get_active_policies":
            return self._policies

        raise ValueError(f"Unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# Helper — trim failure rows for token efficiency
# ---------------------------------------------------------------------------

def _slim_failure(row: dict) -> dict:
    return {
        "failure_id":  row.get("failure_id"),
        "robot_id":    row.get("robot_id"),
        "zone":        row.get("zone"),
        "nav_outcome": row.get("nav_outcome"),
        "fault_flag":  row.get("fault_flag"),
        "occurred_at": str(row.get("occurred_at", "")),
        "health":      row.get("health_at_failure"),
    }
