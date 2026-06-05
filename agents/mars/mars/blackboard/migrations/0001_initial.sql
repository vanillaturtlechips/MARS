-- MARS Blackboard — initial schema
-- Migration: 0001_initial
-- Run automatically by Docker on first start (docker-entrypoint-initdb.d)
-- Also apply manually: psql $DB_DSN -f mars/blackboard/migrations/0001_initial.sql
--
-- Timestamp conventions used throughout this file:
--   SIM TIME  — supplied by the application from the ROS clock (/clock,
--               use_sim_time=true); no DEFAULT so the app is forced to supply it.
--               Used for any value that feeds window/recency calculations.
--   WALL TIME — DEFAULT NOW(); used only for audit/ordering columns (created_at,
--               updated_at) that track when a row was written to Postgres, not
--               when the event happened in the simulation.

-- Extension: the SQL extension is named 'vector', not 'pgvector'.
CREATE EXTENSION IF NOT EXISTS vector;

-------------------------------------------------------------------------------
-- SHARED TRIGGER — keeps updated_at (wall-clock audit) current on UPDATE.
-- updated_at is wall-clock only; it tracks when a row was last touched in
-- Postgres, not when the corresponding event occurred in the simulation.
-------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();   -- wall-clock; correct for audit use
    RETURN NEW;
END;
$$;

-------------------------------------------------------------------------------
-- MISSIONS
-- Full lifecycle of a supervisory mission. "Mission" is supervisory only;
-- dispatch over ROS is a NavigateToPose goal. goal_id links to dispatch_ledger.
--
-- robot_id is nullable: a mission starts PENDING (unassigned) and receives a
-- robot_id only when assigned. The CHECK enforces this invariant.
-------------------------------------------------------------------------------
CREATE TABLE missions (
    mission_id          TEXT        PRIMARY KEY,
    -- NULL while state='PENDING'; required once assigned.
    robot_id            TEXT,
    goal_id             TEXT        UNIQUE,     -- NavigateToPose action goal_id; set at dispatch
    state               TEXT        NOT NULL DEFAULT 'PENDING'
                            CHECK (state IN ('PENDING','ASSIGNED','ACTIVE','COMPLETED','FAILED')),
    priority            INTEGER     NOT NULL DEFAULT 5,
    scheduling_priority INTEGER     NOT NULL DEFAULT 5,
    start_pose          JSONB,
    destination_pose    JSONB,
    zone                TEXT,
    retry_count         INTEGER     NOT NULL DEFAULT 0,
    handoff_count       INTEGER     NOT NULL DEFAULT 0,
    failure_reason      TEXT,
    -- WALL TIME: when this row was created in Postgres (audit).
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- SIM TIME: lifecycle event timestamps supplied by the app from the ROS clock.
    assigned_at         TIMESTAMPTZ,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    -- WALL TIME: updated by the set_updated_at trigger on every UPDATE.
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT mission_robot_required
        CHECK (state = 'PENDING' OR robot_id IS NOT NULL)
);

CREATE INDEX missions_robot_state ON missions (robot_id, state);
CREATE INDEX missions_state ON missions (state);

CREATE TRIGGER missions_updated_at
    BEFORE UPDATE ON missions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-------------------------------------------------------------------------------
-- DISPATCH LEDGER
-- goal_id → mission_id + robot_id, written at NavigateToPose dispatch.
-- Nav outcome (GoalStatus) is correlated back to a mission via this table.
-- GoalStatus: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED  (Humble result is empty)
--
-- dispatched_at / resolved_at: use SIM TIME when the system runs on sim time;
-- supplied by the app (no DEFAULT here except dispatched_at which is set
-- immediately on dispatch — app should pass sim time if available).
-------------------------------------------------------------------------------
CREATE TABLE dispatch_ledger (
    goal_id         TEXT        PRIMARY KEY,
    mission_id      TEXT        NOT NULL REFERENCES missions (mission_id),
    robot_id        TEXT        NOT NULL,
    -- SIM TIME preferred; wall-clock fallback if no sim clock available yet.
    dispatched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,          -- SIM TIME; set when GoalStatus arrives
    nav_outcome     TEXT        CHECK (nav_outcome IN ('succeeded','canceled','aborted')),
    goal_status     INTEGER     CHECK (goal_status IN (4, 5, 6))
);

CREATE INDEX dispatch_ledger_mission ON dispatch_ledger (mission_id);

-------------------------------------------------------------------------------
-- ROBOTS
-- Live structured state. HOT fields (pose, battery, current_zone) are also
-- mirrored in Redis for low-latency Aggregator reads; Postgres is the durable
-- copy updated on state transitions.
-- allocation_state is the shared lock between Scheduler and Charging Service:
-- compare-and-set prevents double-claim (§6a).
-------------------------------------------------------------------------------
CREATE TABLE robots (
    robot_id            TEXT        PRIMARY KEY,
    battery_pct         REAL        NOT NULL DEFAULT 100.0,
    -- Operational mode; CHECK enumerates allowed values.
    mode                TEXT        NOT NULL DEFAULT 'IDLE'
                            CHECK (mode IN ('IDLE','NAVIGATING','CHARGING','STUCK')),
    allocation_state    TEXT        NOT NULL DEFAULT 'IDLE'
                            CHECK (allocation_state IN ('IDLE','RESERVED','BUSY','CHARGING')),
    current_mission_id  TEXT        REFERENCES missions (mission_id),
    current_zone        TEXT,
    pose                JSONB,      -- {x, y, z, qx, qy, qz, qw, frame_id}
    health_level        INTEGER     NOT NULL DEFAULT 0
                            CHECK (health_level IN (0, 1, 2)),  -- 0=OK 1=WARN 2=ERROR
    estop_active        BOOLEAN     NOT NULL DEFAULT FALSE,
    fault_codes         TEXT[]      NOT NULL DEFAULT '{}',
    -- SIM TIME: last time the Aggregator received a health/pose topic for this robot.
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- WALL TIME: updated by trigger.
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER robots_updated_at
    BEFORE UPDATE ON robots
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-------------------------------------------------------------------------------
-- ZONES
-- Static definitions + dynamic health/occupancy.
-- is_mandatory: guardrail must never make this zone unreachable (§6 Stage 4).
-- is_charger_zone: at least one charger zone must remain accessible (§6 Stage 4).
-------------------------------------------------------------------------------
CREATE TABLE zones (
    zone_id             TEXT        PRIMARY KEY,
    display_name        TEXT        NOT NULL,
    polygon             JSONB,      -- list of {x, y} vertices
    is_mandatory        BOOLEAN     NOT NULL DEFAULT FALSE,
    is_charger_zone     BOOLEAN     NOT NULL DEFAULT FALSE,
    current_occupancy   INTEGER     NOT NULL DEFAULT 0,
    health_status       TEXT        NOT NULL DEFAULT 'ok'
                            CHECK (health_status IN ('ok','degraded','blocked')),
    -- WALL TIME: trigger-maintained.
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER zones_updated_at
    BEFORE UPDATE ON zones
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-------------------------------------------------------------------------------
-- CHARGERS
-- One row per physical charger.
-------------------------------------------------------------------------------
CREATE TABLE chargers (
    charger_id          TEXT        PRIMARY KEY,
    zone_id             TEXT        NOT NULL REFERENCES zones (zone_id),
    pose                JSONB,
    is_online           BOOLEAN     NOT NULL DEFAULT TRUE,
    is_occupied         BOOLEAN     NOT NULL DEFAULT FALSE,
    current_robot_id    TEXT,
    -- SIM TIME: when the current robot docked.
    occupied_since      TIMESTAMPTZ,
    -- WALL TIME: trigger-maintained.
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER chargers_updated_at
    BEFORE UPDATE ON chargers
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-------------------------------------------------------------------------------
-- CHARGING SESSIONS
-- History of each charging event; supports outcome evaluation and pressure
-- metric computation (Charging Service → §6b).
-------------------------------------------------------------------------------
CREATE TABLE charging_sessions (
    session_id          TEXT        PRIMARY KEY,
    robot_id            TEXT        NOT NULL,
    charger_id          TEXT        NOT NULL REFERENCES chargers (charger_id),
    -- SIM TIME: supplied by Charging Service from the ROS clock.
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    start_battery_pct   REAL,
    end_battery_pct     REAL,
    target_pct          REAL,       -- target charge level at time of session
    interrupted         BOOLEAN     NOT NULL DEFAULT FALSE,
    interrupt_reason    TEXT,
    -- WALL TIME: when this row was written to Postgres.
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX charging_sessions_robot ON charging_sessions (robot_id, started_at DESC);

-------------------------------------------------------------------------------
-- CHARGING PRESSURE METRICS
-- Emitted periodically by the Charging Service; read by Fleet State Analysis
-- Agent as part of fleet_metrics (§6b → §3a loop).
-------------------------------------------------------------------------------
CREATE TABLE charging_pressure_metrics (
    id                      BIGSERIAL   PRIMARY KEY,
    -- SIM TIME: the sim clock instant this snapshot was taken.
    recorded_at             TIMESTAMPTZ NOT NULL,
    queue_length            INTEGER     NOT NULL,
    mean_wait_sec           REAL,
    p95_wait_sec            REAL,
    occupied_pct            REAL        NOT NULL,
    below_low_count         INTEGER     NOT NULL,
    below_critical_count    INTEGER     NOT NULL,
    -- WALL TIME: when this row was written to Postgres.
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX cpm_time ON charging_pressure_metrics (recorded_at DESC);

-------------------------------------------------------------------------------
-- FAILURES
-- Enriched failure events written by the Aggregator (§1a).
-- fault_flag encodes WHY it is set (battery_critical / estop / diagnostics_error).
-- distribution is the pre-computed zone/robot spread for the Router only — it
-- MUST NOT appear in the Failure Analysis Agent input bundle (the agent derives
-- scope from raw mission_failures per the contracts doc).
-------------------------------------------------------------------------------
CREATE TABLE failures (
    failure_id              TEXT        PRIMARY KEY,
    robot_id                TEXT        NOT NULL,
    mission_id              TEXT,
    zone                    TEXT,
    event_type              TEXT        NOT NULL DEFAULT 'navigation.aborted',
    nav_outcome             TEXT,
    goal_status             INTEGER,
    health_at_failure       JSONB,      -- {battery_pct, estop_active, fault_codes}
    fault_flag              TEXT,       -- null | 'battery_critical' | 'estop' | 'diagnostics_error'
    distribution            JSONB,      -- {per_robot_zone_spread, per_zone_robot_spread} — Router only
    failures_for_this_mission INTEGER   NOT NULL DEFAULT 0,
    -- SIM TIME: when the abort event occurred on the ROS clock.
    occurred_at             TIMESTAMPTZ NOT NULL,
    raw_event               JSONB,
    -- WALL TIME: when this row was written to Postgres.
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX failures_robot   ON failures (robot_id, occurred_at DESC);
CREATE INDEX failures_zone    ON failures (zone, occurred_at DESC);
CREATE INDEX failures_mission ON failures (mission_id);

-------------------------------------------------------------------------------
-- DIAGNOSES
-- Output of the Failure Analysis Agent after passing Decision Validator (§5).
-- Linked 1:1 to a failure (UNIQUE enforces this — one committed diagnosis per
-- failure; DEGRADE/REJECT outputs are logged in decision_validator_result but
-- a row is still written here so all agent runs are auditable).
-------------------------------------------------------------------------------
CREATE TABLE diagnoses (
    diagnosis_id                TEXT        PRIMARY KEY,
    -- UNIQUE: one diagnosis per failure (1:1 design — §5).
    failure_id                  TEXT        NOT NULL UNIQUE REFERENCES failures (failure_id),
    cause                       TEXT        NOT NULL,
    scope                       TEXT        NOT NULL,
    persistence                 TEXT        NOT NULL,
    affected_zone               TEXT,
    confidence                  REAL        NOT NULL,
    evidence                    JSONB       NOT NULL,   -- [{observation, refs}]
    relied_on_precedents        TEXT[]      NOT NULL DEFAULT '{}',
    decision_validator_result   TEXT        NOT NULL,   -- PASS/DEGRADE/REJECT
    decision_validator_notes    TEXT,
    retrieval_trust_level       TEXT,
    -- WALL TIME: when this row was written.
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- No separate index needed; UNIQUE on failure_id creates one implicitly.

-------------------------------------------------------------------------------
-- OUTCOMES
-- Written by the Outcome Evaluator after observing a policy/recovery window.
-- action_type + action_id links back to the policy or failure that was acted on.
-- Every outcome is also embedded in incident_embeddings so RAG retrieves
-- *outcome-labeled* precedent (§7).
-------------------------------------------------------------------------------
CREATE TABLE outcomes (
    outcome_id          TEXT        PRIMARY KEY,
    action_type         TEXT        NOT NULL CHECK (action_type IN ('policy','recovery')),
    action_id           TEXT        NOT NULL,   -- policy_id or failure_id
    baseline_metrics    JSONB       NOT NULL,
    final_metrics       JSONB       NOT NULL,
    label               TEXT        NOT NULL CHECK (label IN ('improved','no_effect','worsened')),
    magnitude           REAL,
    -- SIM TIME: start and end of the observation window on the ROS clock.
    window_start        TIMESTAMPTZ NOT NULL,
    window_end          TIMESTAMPTZ NOT NULL,
    -- WALL TIME: when this row was written.
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX outcomes_action ON outcomes (action_type, action_id);

-------------------------------------------------------------------------------
-- FLEET ANALYSES
-- Output of the Fleet State Analysis Agent after passing Decision Validator.
-- One row per agent run — every run is persisted including degraded/rejected
-- ones (restrained outputs are auditable and retrievable for trend analysis).
-- Parallels the diagnoses table in structure.
-------------------------------------------------------------------------------
CREATE TABLE fleet_analyses (
    fleet_analysis_id           TEXT        PRIMARY KEY,
    fleet_health                TEXT        NOT NULL
                                    CHECK (fleet_health IN ('healthy','strained','degraded','critical')),
    bottlenecks                 JSONB       NOT NULL DEFAULT '[]',   -- list of zone ids
    charging_pressure           TEXT        NOT NULL
                                    CHECK (charging_pressure IN ('low','moderate','high')),
    confidence                  REAL        NOT NULL,
    evidence                    JSONB       NOT NULL,               -- [{observation, refs}]
    relied_on_precedents        TEXT[]      NOT NULL DEFAULT '{}',
    decision_validator_result   TEXT        NOT NULL,               -- PASS/DEGRADE/REJECT
    decision_validator_notes    TEXT,
    retrieval_trust_level       TEXT,
    -- WALL TIME: when this row was written.
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-------------------------------------------------------------------------------
-- STRATEGY RUNS
-- Output of the Operations Strategy Agent after passing Decision Validator.
-- One row per agent run — includes no-action and rejected runs so that
-- restraint is auditable and retrievable (important for RAG and outcome eval).
-- Parallels diagnoses and fleet_analyses in structure.
-------------------------------------------------------------------------------
CREATE TABLE strategy_runs (
    strategy_run_id             TEXT        PRIMARY KEY,
    -- nullable: a run may be triggered by failure-only, fleet-only, or both.
    incident_diagnosis_id       TEXT        REFERENCES diagnoses (diagnosis_id),
    fleet_analysis_id           TEXT        REFERENCES fleet_analyses (fleet_analysis_id),
    no_action_reason            TEXT,       -- set when policy_updates is empty (restraint)
    confidence                  REAL        NOT NULL,
    evidence                    JSONB       NOT NULL,   -- [{observation, refs}]
    relied_on_precedents        TEXT[]      NOT NULL DEFAULT '{}',
    decision_validator_result   TEXT        NOT NULL,   -- PASS/DEGRADE/REJECT
    decision_validator_notes    TEXT,
    retrieval_trust_level       TEXT,
    -- WALL TIME: when this row was written.
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX strategy_runs_diagnosis ON strategy_runs (incident_diagnosis_id);
CREATE INDEX strategy_runs_fleet     ON strategy_runs (fleet_analysis_id);

-------------------------------------------------------------------------------
-- POLICIES
-- Active agent-issued policies. expires_at is always required (§6 Stage 6).
-- Guardrail result recorded for audit.
-- strategy_run_id is a real FK now that strategy_runs exists.
-------------------------------------------------------------------------------
CREATE TABLE policies (
    policy_id           TEXT        PRIMARY KEY,
    type                TEXT        NOT NULL,
    params              JSONB       NOT NULL DEFAULT '{}',
    source              TEXT        NOT NULL DEFAULT 'agent'
                            CHECK (source IN ('agent','operator')),
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
    -- WALL TIME: when this row was written.
    issued_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- expires_at is mandatory — no agent policy may be permanent (§6 Stage 6).
    expires_at          TIMESTAMPTZ NOT NULL,
    guardrail_result    TEXT,       -- ACCEPT/MODIFY/REJECT/DEFER_HUMAN
    guardrail_notes     TEXT,
    -- Links to the agent runs that produced this policy.
    diagnosis_id        TEXT        REFERENCES diagnoses (diagnosis_id),
    strategy_run_id     TEXT        REFERENCES strategy_runs (strategy_run_id)
);

CREATE INDEX policies_active_type ON policies (is_active, type);
CREATE INDEX policies_expires     ON policies (expires_at) WHERE is_active = TRUE;

-- History: append-only audit of every policy that ever existed.
CREATE TABLE policy_history (
    id                  BIGSERIAL   PRIMARY KEY,
    policy_id           TEXT        NOT NULL,
    type                TEXT        NOT NULL,
    params              JSONB       NOT NULL,
    source              TEXT        NOT NULL,
    issued_at           TIMESTAMPTZ NOT NULL,
    expires_at          TIMESTAMPTZ NOT NULL,
    deactivated_at      TIMESTAMPTZ,        -- SIM TIME preferred when sim running
    deactivation_reason TEXT,               -- expired/superseded/operator/guardrail_reject
    effectiveness_label TEXT,               -- improved/no_effect/worsened (from Outcome Evaluator)
    -- WALL TIME: when this history row was written.
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-------------------------------------------------------------------------------
-- INCIDENT EMBEDDINGS (pgvector)
-- Every embedded record carries structured metadata so the Retrieval Validator
-- (§4) can score on metadata_match, recency, coverage_match, and similarity —
-- not cosine similarity alone.
--
-- source_type determines which table source_id points to:
--   failure     → failures.failure_id
--   diagnosis   → diagnoses.diagnosis_id
--   strategy    → strategy_runs.strategy_run_id  (the recommendation)
--   outcome     → outcomes.outcome_id             (outcome-labeled)
--
-- source_id carries no FK constraint intentionally — polymorphic reference.
-- Referential integrity is maintained by the application.
--
-- Crucial design constraints (from coding prompt):
--   1. zone / failure_type / scope / recorded_at present for Retrieval Validator
--   2. outcome_id links the embedding back to its labeled outcome (closes RAG loop)
--
-- Embedding model: OpenAI text-embedding-3-small, 1536 dimensions.
-- See DECISIONS.md §4 for rationale and swap instructions.
-- To switch models: drop inc_emb_hnsw, ALTER COLUMN embedding TYPE vector(N),
-- recreate the index, and re-embed every source row (old vectors don't transfer).
-------------------------------------------------------------------------------
CREATE TABLE incident_embeddings (
    id              BIGSERIAL   PRIMARY KEY,
    source_type     TEXT        NOT NULL
                        CHECK (source_type IN ('failure','diagnosis','strategy','outcome')),
    -- Polymorphic FK — no DB constraint; app maintains integrity.
    source_id       TEXT        NOT NULL,
    -- Metadata for Retrieval Validator scoring (§4).
    zone            TEXT,
    failure_type    TEXT,       -- maps to failure_cause enum
    scope           TEXT,       -- isolated/robot_specific/zone_wide/fleet_wide
    outcome_label   TEXT,       -- improved/no_effect/worsened (from linked outcome)
    outcome_id      TEXT        REFERENCES outcomes (outcome_id),
    summary         TEXT        NOT NULL,
    -- text-embedding-3-small produces 1536-dimensional embeddings.
    -- MUST match EMBEDDING_DIM in config.py and get_embedder() dimension.
    embedding       vector(1536),
    -- SIM TIME: the sim clock instant this record represents (for recency scoring).
    recorded_at     TIMESTAMPTZ NOT NULL,
    -- WALL TIME: when this row was written to Postgres.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index: unlike ivfflat, HNSW needs no training data and works well on
-- incrementally-built tables (which is how the RAG store fills during a run).
CREATE INDEX inc_emb_hnsw ON incident_embeddings
    USING hnsw (embedding vector_cosine_ops);

-- Btree indexes for metadata filtering before vector search.
CREATE INDEX inc_emb_zone_type ON incident_embeddings (zone, failure_type);
CREATE INDEX inc_emb_scope     ON incident_embeddings (scope);
CREATE INDEX inc_emb_time      ON incident_embeddings (recorded_at DESC);

-------------------------------------------------------------------------------
-- MIGRATION BOOKKEEPING
-------------------------------------------------------------------------------
CREATE TABLE schema_migrations (
    version     TEXT        PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_migrations (version) VALUES ('0001_initial');
