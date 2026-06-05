"""
Blackboard query layer — structured reads and writes against PostgreSQL.

All components talk to the blackboard through this module.  Raw SQL is kept
here; callers receive plain dicts or lists of dicts.

psycopg3 (psycopg package) is required.  Call register_vector(conn) once per
connection before running vector queries.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from psycopg.rows import dict_row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Missions
# ---------------------------------------------------------------------------

def upsert_mission(conn, mission: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO missions (
                mission_id, robot_id, goal_id, state, priority, scheduling_priority,
                start_pose, destination_pose, zone, retry_count, handoff_count,
                failure_reason, created_at, assigned_at, started_at, completed_at
            ) VALUES (
                %(mission_id)s, %(robot_id)s, %(goal_id)s, %(state)s,
                %(priority)s, %(scheduling_priority)s,
                %(start_pose)s::jsonb, %(destination_pose)s::jsonb,
                %(zone)s, %(retry_count)s, %(handoff_count)s, %(failure_reason)s,
                %(created_at)s, %(assigned_at)s, %(started_at)s, %(completed_at)s
            )
            ON CONFLICT (mission_id) DO UPDATE SET
                robot_id            = EXCLUDED.robot_id,
                goal_id             = COALESCE(EXCLUDED.goal_id, missions.goal_id),
                state               = EXCLUDED.state,
                priority            = EXCLUDED.priority,
                scheduling_priority = EXCLUDED.scheduling_priority,
                start_pose          = COALESCE(EXCLUDED.start_pose, missions.start_pose),
                destination_pose    = COALESCE(EXCLUDED.destination_pose, missions.destination_pose),
                zone                = COALESCE(EXCLUDED.zone, missions.zone),
                retry_count         = EXCLUDED.retry_count,
                handoff_count       = EXCLUDED.handoff_count,
                failure_reason      = EXCLUDED.failure_reason,
                assigned_at         = COALESCE(EXCLUDED.assigned_at, missions.assigned_at),
                started_at          = COALESCE(EXCLUDED.started_at, missions.started_at),
                completed_at        = COALESCE(EXCLUDED.completed_at, missions.completed_at)
            """,
            {
                "mission_id": mission["mission_id"],
                "robot_id": mission.get("robot_id"),
                "goal_id": mission.get("goal_id"),
                "state": mission.get("state", "PENDING"),
                "priority": mission.get("priority", 5),
                "scheduling_priority": mission.get("scheduling_priority", 5),
                "start_pose": json.dumps(mission["start_pose"]) if mission.get("start_pose") else None,
                "destination_pose": json.dumps(mission["destination_pose"]) if mission.get("destination_pose") else None,
                "zone": mission.get("zone"),
                "retry_count": mission.get("retry_count", 0),
                "handoff_count": mission.get("handoff_count", 0),
                "failure_reason": mission.get("failure_reason"),
                "created_at": mission.get("created_at", _now()),
                "assigned_at": mission.get("assigned_at"),
                "started_at": mission.get("started_at"),
                "completed_at": mission.get("completed_at"),
            },
        )


def get_mission(conn, mission_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM missions WHERE mission_id = %s", (mission_id,))
        return cur.fetchone()


def get_pending_missions(conn) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM missions
            WHERE state = 'PENDING'
            ORDER BY scheduling_priority DESC, priority DESC, created_at ASC
            """
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Dispatch ledger
# ---------------------------------------------------------------------------

def record_dispatch(conn, goal_id: str, mission_id: str, robot_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dispatch_ledger (goal_id, mission_id, robot_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (goal_id) DO NOTHING
            """,
            (goal_id, mission_id, robot_id),
        )


def resolve_dispatch(conn, goal_id: str, nav_outcome: str, goal_status: int) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE dispatch_ledger
            SET resolved_at = NOW(), nav_outcome = %s, goal_status = %s
            WHERE goal_id = %s
            RETURNING *
            """,
            (nav_outcome, goal_status, goal_id),
        )
        return cur.fetchone()


def get_dispatch_by_goal(conn, goal_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM dispatch_ledger WHERE goal_id = %s", (goal_id,))
        return cur.fetchone()


# ---------------------------------------------------------------------------
# Robots
# ---------------------------------------------------------------------------

def upsert_robot(conn, robot: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO robots (
                robot_id, battery_pct, mode, allocation_state,
                current_mission_id, current_zone, pose,
                health_level, estop_active, fault_codes, last_seen_at
            ) VALUES (
                %(robot_id)s, %(battery_pct)s, %(mode)s, %(allocation_state)s,
                %(current_mission_id)s, %(current_zone)s, %(pose)s::jsonb,
                %(health_level)s, %(estop_active)s, %(fault_codes)s,
                %(last_seen_at)s
            )
            ON CONFLICT (robot_id) DO UPDATE SET
                battery_pct         = EXCLUDED.battery_pct,
                mode                = EXCLUDED.mode,
                allocation_state    = EXCLUDED.allocation_state,
                current_mission_id  = EXCLUDED.current_mission_id,
                current_zone        = EXCLUDED.current_zone,
                pose                = EXCLUDED.pose,
                health_level        = EXCLUDED.health_level,
                estop_active        = EXCLUDED.estop_active,
                fault_codes         = EXCLUDED.fault_codes,
                last_seen_at        = EXCLUDED.last_seen_at
            """,
            {
                "robot_id": robot["robot_id"],
                "battery_pct": robot.get("battery_pct", 100.0),
                "mode": robot.get("mode", "IDLE"),
                "allocation_state": robot.get("allocation_state", "IDLE"),
                "current_mission_id": robot.get("current_mission_id"),
                "current_zone": robot.get("current_zone"),
                "pose": json.dumps(robot["pose"]) if robot.get("pose") else None,
                "health_level": robot.get("health_level", 0),
                "estop_active": robot.get("estop_active", False),
                "fault_codes": robot.get("fault_codes", []),
                "last_seen_at": robot.get("last_seen_at", _now()),
            },
        )


def cas_allocation_state(conn, robot_id: str, expected: str, new_state: str) -> bool:
    """Compare-and-set allocation_state; returns True if the update succeeded."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE robots
            SET allocation_state = %s
            WHERE robot_id = %s AND allocation_state = %s
            """,
            (new_state, robot_id, expected),
        )
        return cur.rowcount == 1


def get_robot(conn, robot_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM robots WHERE robot_id = %s", (robot_id,))
        return cur.fetchone()


def get_idle_robots(conn) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM robots WHERE allocation_state = 'IDLE' ORDER BY battery_pct DESC"
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

def upsert_zone(conn, zone: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO zones (zone_id, display_name, polygon, is_mandatory, is_charger_zone)
            VALUES (%(zone_id)s, %(display_name)s, %(polygon)s::jsonb, %(is_mandatory)s, %(is_charger_zone)s)
            ON CONFLICT (zone_id) DO UPDATE SET
                display_name   = EXCLUDED.display_name,
                is_mandatory   = EXCLUDED.is_mandatory,
                is_charger_zone = EXCLUDED.is_charger_zone
            """,
            {
                "zone_id": zone["zone_id"],
                "display_name": zone.get("display_name", zone["zone_id"]),
                "polygon": json.dumps(zone["polygon"]) if zone.get("polygon") else None,
                "is_mandatory": zone.get("is_mandatory", False),
                "is_charger_zone": zone.get("is_charger_zone", False),
            },
        )


def upsert_charger(conn, charger: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chargers (charger_id, zone_id, is_online)
            VALUES (%(charger_id)s, %(zone_id)s, %(is_online)s)
            ON CONFLICT (charger_id) DO UPDATE SET
                zone_id   = EXCLUDED.zone_id,
                is_online = EXCLUDED.is_online
            """,
            {
                "charger_id": charger["charger_id"],
                "zone_id": charger["zone_id"],
                "is_online": charger.get("is_online", True),
            },
        )


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def write_failure(conn, event: dict[str, Any]) -> str:
    fid = event.get("failure_id") or new_id("F")
    # occurred_at: app must supply sim time; fall back to wall-clock only in tests
    occurred_at = event.get("occurred_at")
    if occurred_at is None:
        occurred_at = _now()
    elif isinstance(occurred_at, (int, float)):
        occurred_at = datetime.fromtimestamp(occurred_at, tz=timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO failures (
                failure_id, robot_id, mission_id, zone, event_type,
                nav_outcome, goal_status, health_at_failure, fault_flag,
                distribution, failures_for_this_mission, occurred_at, raw_event
            ) VALUES (
                %(failure_id)s, %(robot_id)s, %(mission_id)s, %(zone)s,
                %(event_type)s, %(nav_outcome)s, %(goal_status)s,
                %(health_at_failure)s::jsonb, %(fault_flag)s,
                %(distribution)s::jsonb, %(failures_for_this_mission)s,
                %(occurred_at)s, %(raw_event)s::jsonb
            )
            ON CONFLICT (failure_id) DO NOTHING
            """,
            {
                "failure_id": fid,
                "robot_id": event["robot_id"],
                "mission_id": event.get("mission_id"),
                "zone": event.get("zone"),
                "event_type": event.get("event_type", "navigation.aborted"),
                "nav_outcome": event.get("nav_outcome"),
                "goal_status": event.get("goal_status"),
                "health_at_failure": json.dumps(event["health_at_failure"]) if event.get("health_at_failure") else None,
                "fault_flag": event.get("fault_flag"),
                "distribution": json.dumps(event["distribution"]) if event.get("distribution") else None,
                "failures_for_this_mission": event.get("failures_for_this_mission", 0),
                "occurred_at": occurred_at,
                "raw_event": json.dumps(event.get("raw_event")) if event.get("raw_event") else None,
            },
        )
    return fid


def get_recent_failures(conn, window_seconds: int = 900, zone: str | None = None) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        if zone:
            cur.execute(
                """
                SELECT * FROM failures
                WHERE occurred_at >= NOW() - INTERVAL '1 second' * %s
                  AND zone = %s
                ORDER BY occurred_at DESC
                """,
                (window_seconds, zone),
            )
        else:
            cur.execute(
                """
                SELECT * FROM failures
                WHERE occurred_at >= NOW() - INTERVAL '1 second' * %s
                ORDER BY occurred_at DESC
                """,
                (window_seconds,),
            )
        return cur.fetchall()


def count_mission_failures(conn, mission_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM failures WHERE mission_id = %s",
            (mission_id,),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Diagnoses
# ---------------------------------------------------------------------------

def write_diagnosis(conn, diagnosis: dict[str, Any]) -> str:
    did = diagnosis.get("diagnosis_id") or new_id("DX")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO diagnoses (
                diagnosis_id, failure_id, cause, scope, persistence,
                affected_zone, confidence, evidence, relied_on_precedents,
                decision_validator_result, decision_validator_notes,
                retrieval_trust_level
            ) VALUES (
                %(diagnosis_id)s, %(failure_id)s, %(cause)s, %(scope)s,
                %(persistence)s, %(affected_zone)s, %(confidence)s,
                %(evidence)s::jsonb, %(relied_on_precedents)s,
                %(decision_validator_result)s, %(decision_validator_notes)s,
                %(retrieval_trust_level)s
            )
            ON CONFLICT (failure_id) DO NOTHING
            """,
            {
                "diagnosis_id": did,
                "failure_id": diagnosis["failure_id"],
                "cause": diagnosis["cause"],
                "scope": diagnosis["scope"],
                "persistence": diagnosis["persistence"],
                "affected_zone": diagnosis.get("affected_zone"),
                "confidence": diagnosis["confidence"],
                "evidence": json.dumps(diagnosis["evidence"]),
                "relied_on_precedents": diagnosis.get("relied_on_precedents", []),
                "decision_validator_result": diagnosis["decision_validator_result"],
                "decision_validator_notes": diagnosis.get("decision_validator_notes"),
                "retrieval_trust_level": diagnosis.get("retrieval_trust_level"),
            },
        )
    return did


# ---------------------------------------------------------------------------
# Fleet analyses
# ---------------------------------------------------------------------------

def write_fleet_analysis(conn, analysis: dict[str, Any]) -> str:
    aid = analysis.get("fleet_analysis_id") or new_id("FA")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fleet_analyses (
                fleet_analysis_id, fleet_health, bottlenecks, charging_pressure,
                confidence, evidence, relied_on_precedents,
                decision_validator_result, decision_validator_notes, retrieval_trust_level
            ) VALUES (
                %(fleet_analysis_id)s, %(fleet_health)s, %(bottlenecks)s::jsonb,
                %(charging_pressure)s, %(confidence)s, %(evidence)s::jsonb,
                %(relied_on_precedents)s, %(decision_validator_result)s,
                %(decision_validator_notes)s, %(retrieval_trust_level)s
            )
            """,
            {
                "fleet_analysis_id": aid,
                "fleet_health": analysis["fleet_health"],
                "bottlenecks": json.dumps(analysis.get("bottlenecks", [])),
                "charging_pressure": analysis["charging_pressure"],
                "confidence": analysis["confidence"],
                "evidence": json.dumps(analysis.get("evidence", [])),
                "relied_on_precedents": analysis.get("relied_on_precedents", []),
                "decision_validator_result": analysis["decision_validator_result"],
                "decision_validator_notes": analysis.get("decision_validator_notes"),
                "retrieval_trust_level": analysis.get("retrieval_trust_level"),
            },
        )
    return aid


# ---------------------------------------------------------------------------
# Strategy runs
# ---------------------------------------------------------------------------

def write_strategy_run(conn, run: dict[str, Any]) -> str:
    rid = run.get("strategy_run_id") or new_id("SR")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO strategy_runs (
                strategy_run_id, incident_diagnosis_id, fleet_analysis_id,
                no_action_reason, confidence, evidence, relied_on_precedents,
                decision_validator_result, decision_validator_notes, retrieval_trust_level
            ) VALUES (
                %(strategy_run_id)s, %(incident_diagnosis_id)s, %(fleet_analysis_id)s,
                %(no_action_reason)s, %(confidence)s, %(evidence)s::jsonb,
                %(relied_on_precedents)s, %(decision_validator_result)s,
                %(decision_validator_notes)s, %(retrieval_trust_level)s
            )
            """,
            {
                "strategy_run_id": rid,
                "incident_diagnosis_id": run.get("incident_diagnosis_id"),
                "fleet_analysis_id": run.get("fleet_analysis_id"),
                "no_action_reason": run.get("no_action_reason"),
                "confidence": run["confidence"],
                "evidence": json.dumps(run.get("evidence", [])),
                "relied_on_precedents": run.get("relied_on_precedents", []),
                "decision_validator_result": run["decision_validator_result"],
                "decision_validator_notes": run.get("decision_validator_notes"),
                "retrieval_trust_level": run.get("retrieval_trust_level"),
            },
        )
    return rid


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def write_policy(conn, policy: dict[str, Any]) -> str:
    pid = policy.get("policy_id") or new_id("POL")
    expires_at = policy.get("expires_at")
    if isinstance(expires_at, (int, float)):
        from datetime import timedelta
        expires_at = _now() + timedelta(seconds=expires_at)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO policies (
                policy_id, type, params, source, is_active,
                expires_at, guardrail_result, guardrail_notes,
                diagnosis_id, strategy_run_id
            ) VALUES (
                %(policy_id)s, %(type)s, %(params)s::jsonb, %(source)s, TRUE,
                %(expires_at)s, %(guardrail_result)s, %(guardrail_notes)s,
                %(diagnosis_id)s, %(strategy_run_id)s
            )
            """,
            {
                "policy_id": pid,
                "type": policy["type"],
                "params": json.dumps(policy.get("params", {})),
                "source": policy.get("source", "agent"),
                "expires_at": expires_at,
                "guardrail_result": policy.get("guardrail_result"),
                "guardrail_notes": policy.get("guardrail_notes"),
                "diagnosis_id": policy.get("diagnosis_id"),
                "strategy_run_id": policy.get("strategy_run_id"),
            },
        )
    return pid


def get_active_policies(conn) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM policies
            WHERE is_active = TRUE AND expires_at > NOW()
            ORDER BY issued_at DESC
            """
        )
        return cur.fetchall()


def deactivate_policy(conn, policy_id: str, reason: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE policies SET is_active = FALSE WHERE policy_id = %s",
            (policy_id,),
        )
        cur.execute(
            """
            INSERT INTO policy_history (
                policy_id, type, params, source, issued_at, expires_at,
                deactivated_at, deactivation_reason
            )
            SELECT policy_id, type, params, source, issued_at, expires_at,
                   NOW(), %s
            FROM policies WHERE policy_id = %s
            ON CONFLICT DO NOTHING
            """,
            (reason, policy_id),
        )


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

def write_outcome(conn, outcome: dict[str, Any]) -> str:
    oid = outcome.get("outcome_id") or new_id("OUT")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO outcomes (
                outcome_id, action_type, action_id, baseline_metrics,
                final_metrics, label, magnitude, window_start, window_end
            ) VALUES (
                %(outcome_id)s, %(action_type)s, %(action_id)s,
                %(baseline_metrics)s::jsonb, %(final_metrics)s::jsonb,
                %(label)s, %(magnitude)s, %(window_start)s, %(window_end)s
            )
            """,
            {
                "outcome_id": oid,
                "action_type": outcome["action_type"],
                "action_id": outcome["action_id"],
                "baseline_metrics": json.dumps(outcome["baseline_metrics"]),
                "final_metrics": json.dumps(outcome["final_metrics"]),
                "label": outcome["label"],
                "magnitude": outcome.get("magnitude"),
                "window_start": outcome["window_start"],
                "window_end": outcome["window_end"],
            },
        )
    return oid


# ---------------------------------------------------------------------------
# pgvector / embeddings
# ---------------------------------------------------------------------------

def write_embedding(conn, record: dict[str, Any]) -> int:
    recorded_at = record.get("recorded_at", _now())
    if isinstance(recorded_at, (int, float)):
        recorded_at = datetime.fromtimestamp(recorded_at, tz=timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO incident_embeddings (
                source_type, source_id, zone, failure_type, scope,
                outcome_label, outcome_id, summary, embedding, recorded_at
            ) VALUES (
                %(source_type)s, %(source_id)s, %(zone)s, %(failure_type)s,
                %(scope)s, %(outcome_label)s, %(outcome_id)s,
                %(summary)s, %(embedding)s::vector, %(recorded_at)s
            )
            RETURNING id
            """,
            {
                "source_type": record["source_type"],
                "source_id": record["source_id"],
                "zone": record.get("zone"),
                "failure_type": record.get("failure_type"),
                "scope": record.get("scope"),
                "outcome_label": record.get("outcome_label"),
                "outcome_id": record.get("outcome_id"),
                "summary": record["summary"],
                "embedding": record["embedding"],
                "recorded_at": recorded_at,
            },
        )
        return cur.fetchone()[0]


def search_similar(
    conn,
    embedding: list[float],
    source_types: list[str] | None = None,
    zone: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return rows ordered by cosine similarity.  Metadata filters applied first."""
    filters: list[str] = []
    filter_params: list[Any] = []

    if source_types:
        filters.append("source_type = ANY(%s)")
        filter_params.append(source_types)
    if zone:
        filters.append("zone = %s")
        filter_params.append(zone)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    # Parameter order: first embedding (similarity display), then filter params,
    # then embedding again (ORDER BY), then limit.
    params: list[Any] = [embedding] + filter_params + [embedding, limit]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT *, 1 - (embedding <=> %s::vector) AS similarity
            FROM incident_embeddings
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Operational metrics (for Operations Strategy Agent input bundle)
# ---------------------------------------------------------------------------

def get_operational_metrics(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM missions WHERE state = 'PENDING'")
        backlog: int = cur.fetchone()[0]

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM robots")
        total: int = cur.fetchone()[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM robots WHERE allocation_state IN ('BUSY', 'RESERVED')"
        )
        busy: int = cur.fetchone()[0]

    utilization = busy / max(total, 1)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT queue_length, occupied_pct, below_low_count
            FROM charging_pressure_metrics
            ORDER BY recorded_at DESC
            LIMIT 1
            """
        )
        cpm = cur.fetchone()

    return {
        "mission_backlog": backlog,
        "robot_utilization": utilization,
        "charging": {
            "queue_len": cpm[0] if cpm else 0,
            "occupied_pct": cpm[1] if cpm else 0.0,
            "below_low_count": cpm[2] if cpm else 0,
        },
    }


# ---------------------------------------------------------------------------
# Fleet metrics (for Fleet State Analysis Agent input bundle — §3a)
# ---------------------------------------------------------------------------

def get_fleet_metrics(conn) -> dict[str, Any]:
    """
    Assemble the fleet_metrics bundle the Fleet State Analysis Agent receives.
    Pulls from: missions, robots, charging_pressure_metrics, failures (recent).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM missions WHERE state = 'PENDING'")
        backlog: int = cur.fetchone()[0]

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM robots")
        total: int = cur.fetchone()[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM robots WHERE allocation_state IN ('BUSY','RESERVED')"
        )
        busy: int = cur.fetchone()[0]

    utilization = busy / max(total, 1)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT queue_length, mean_wait_sec, occupied_pct,
                   below_low_count, below_critical_count
            FROM charging_pressure_metrics
            ORDER BY recorded_at DESC LIMIT 1
            """
        )
        cpm = cur.fetchone()

    charging = {
        "queue_len":           cpm[0] if cpm else 0,
        "mean_wait_sec":       cpm[1] if cpm else 0.0,
        "occupied_pct":        cpm[2] if cpm else 0.0,
        "below_low_count":     cpm[3] if cpm else 0,
        "below_critical_count": cpm[4] if cpm else 0,
    }

    # Recent failure clusters: distinct robots per zone in last 15 min
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT zone, COUNT(DISTINCT robot_id) AS robot_count,
                   COUNT(*) AS failure_count
            FROM failures
            WHERE occurred_at >= NOW() - INTERVAL '900 seconds'
              AND zone IS NOT NULL
            GROUP BY zone
            ORDER BY robot_count DESC
            LIMIT 10
            """
        )
        clusters = cur.fetchall()

    zone_health: dict[str, Any] = {}
    recent_failure_clusters = []
    for c in clusters:
        zone = c["zone"]
        zone_health[zone] = {
            "recent_failures": c["failure_count"],
            "distinct_robots": c["robot_count"],
        }
        if c["robot_count"] >= 2:
            recent_failure_clusters.append({"zone": zone, "robots": c["robot_count"]})

    return {
        "robot_utilization":        utilization,
        "mission_backlog":          backlog,
        "charging":                 charging,
        "zone_health":              zone_health,
        "recent_failure_clusters":  recent_failure_clusters,
    }


# ---------------------------------------------------------------------------
# World state (for Policy Guardrail feasibility check)
# ---------------------------------------------------------------------------

def get_world_state(conn) -> dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT zone_id, is_mandatory, is_charger_zone FROM zones"
        )
        rows = cur.fetchall()

    zones = {
        r["zone_id"]: {
            "is_mandatory":   r["is_mandatory"],
            "is_charger_zone": r["is_charger_zone"],
        }
        for r in rows
    }
    charger_zones = [zid for zid, z in zones.items() if z["is_charger_zone"]]

    # Total charger count — used by the guardrail's charging viability check
    # (Stage 4: reserve_chargers_for_critical must leave at least one for normal ops).
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chargers WHERE is_online = TRUE")
        total_chargers: int = cur.fetchone()[0]

    return {
        "zones":          zones,
        "charger_zones":  charger_zones,
        "total_chargers": total_chargers,
    }
