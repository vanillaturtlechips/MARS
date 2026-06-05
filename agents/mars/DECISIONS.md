# DECISIONS.md — MARS Open-Parameter Defaults

All numeric and structural choices not specified by the design docs are recorded
here.  If a default turns out wrong, change the env var in `.env` and the config
module will pick it up without a code change.

---

## 1. Embedding provider & dimension

**Updated 2026-06-04 (implement-changes change-set).**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Active LLM provider | `openai` (`gpt-4.1-mini`) | Switched from Anthropic Claude to OpenAI. `OPENAI_MODEL=gpt-4.1-mini` from `.env`. Anthropic kept as a named fallback (`LLM_PROVIDER=anthropic`). |
| Embedding model | `text-embedding-3-small` (OpenAI) | Switched from Voyage AI voyage-3 (1024 dims) to OpenAI text-embedding-3-small (1536 dims) to use a single API provider for both chat and embeddings. |
| `EMBEDDING_DIM` | `1536` | Must match the `vector(N)` column in `0001_initial.sql`. **A mismatch fails at INSERT time.** `text-embedding-3-small` → 1536, `text-embedding-3-large` → 3072. |
| Schema change | `0001_initial.sql` edited in place (1024→1536) | DB not yet applied to any shared instance; fresh apply is correct. Old Voyage vectors (1024-dim) cannot be reused — re-embed all source rows when switching on a live DB. |
| `LLM_TEMPERATURE` | `0.0` | Low temperature for stable, defensible outputs. |

Voyage AI key kept in config as an optional fallback. To revert to Voyage: set `LLM_PROVIDER=voyage` (or blank), set `EMBEDDING_DIM=1024`, update schema column, re-embed.

---

## 2. Battery thresholds

| Variable | Value | Rationale |
|----------|-------|-----------|
| `BATTERY_CRITICAL_PCT` | `15.0` | Conservative floor; robot should be able to reach the nearest charger from any point in the warehouse. Treat as "emergency: interrupt and charge now." |
| `BATTERY_LOW_PCT` | `30.0` | "Finish current mission then charge." Wide gap vs CRITICAL gives enough margin to complete a mid-length mission. |
| `BATTERY_MIN_DISPATCH_PCT` | `20.0` | Flat threshold used by Scheduler (§6a) when no per-robot MissionFeasible service is available. Between LOW and CRITICAL. |
| `BATTERY_TOP_UP_PCT` | `80.0` | Opportunistic trigger: an idle robot below this charges during lulls. |
| `BATTERY_TARGET_CHARGE_PCT` | `80.0` | Under contention the Charging Service charges to this level and releases the charger, saving time at the slow top-of-curve. |

---

## 3. Router thresholds

| Variable | Value | Rationale |
|----------|-------|-----------|
| `FAST_PATH_BUDGET` | `2` | A mission that has failed more than twice is clearly not a transient glitch. Forces the slow path so the Failure Analysis Agent can diagnose. |
| `ROUTER_SCOPE_HINT` | `2` | 2+ distinct robots failing in the same zone in the window is a cheap proxy for an environmental problem. Named "hint" because it is not forwarded to the agent — the agent derives its own scope. |
| `FAILURE_WINDOW_SECONDS` | `900` | 15-minute rolling window for distribution counts and mission failure counts. Wide enough to catch a persistent zone issue; narrow enough not to conflate separate incidents. |

---

## 4. Retrieval Validator weights & floors (§4)

| Variable | Value | Rationale |
|----------|-------|-----------|
| `RV_W_META` | `0.30` | Metadata match is the strongest signal; same zone + failure type is highly predictive. |
| `RV_W_REC` | `0.20` | Recency matters because warehouse layouts / fleet sizes change, but not as much as relevance. |
| `RV_W_COV` | `0.25` | Coverage/scope mismatch is disqualifying (isolated precedent for fleet-wide incident). |
| `RV_W_SIM` | `0.25` | Cosine similarity is necessary but not sufficient — can match semantically related but wrong-scale events. |
| `RV_COV_FLOOR` | `0.4` | Coverage match below 0.4 triggers the mismatch cap. A value of 0.4 corresponds to roughly 2 steps apart in the 4-level scope ladder. |
| `RV_COV_MISMATCH_CAP` | `0.5` | Cap for a coverage-mismatched result. `0.5` is right at the ACCEPT_THRESHOLD so mismatch alone doesn't filter a result, but combined with any other weakness it will. |
| `RV_RECENCY_FLOOR` | `0.3` | Recency score below 0.3 → stale cap applies. At 30-day half-life this is ~53 days old. |
| `RV_STALE_CAP` | `0.4` | Aggressive cap for stale results (fleet configuration may have changed). |
| `RV_ACCEPT_THRESHOLD` | `0.5` | Per-result threshold for surviving the filter. Conservative; err on the side of fewer but higher-quality precedents. |
| `RV_RECENCY_HALF_LIFE_DAYS` | `30` | Exponential decay: a 30-day-old result has 0.5 recency, a 60-day-old has 0.25, etc. Warehouse layouts change on the timescale of months. |

---

## 5. Decision Validator confidence thresholds (tau)

| Variable | Value | Rationale |
|----------|-------|-----------|
| `DV_TAU_DIAGNOSIS` | `0.5` | Diagnosis: medium bar. Below 0.5 = DEGRADE. We accept low-confidence diagnoses with a downgrade rather than silently rejecting them, since some diagnosis (even uncertain) is better than none for the strategy step. |
| `DV_TAU_POLICY_MEDIUM` | `0.6` | Medium-impact policies (avoid_zone, delay_low_priority): higher bar than diagnosis since this directly affects operations. |
| `DV_TAU_POLICY_HIGH` | `0.75` | High-impact (fleet-wide throttles): not in current whitelist but reserved. Requires operator approval via DEFER_HUMAN. |

---

## 6. Strategy Trigger

| Variable | Value | Rationale |
|----------|-------|-----------|
| `STRATEGY_CORRELATION_WINDOW_SECONDS` | `600` | 10 minutes. Fleet analysis runs every 5 min; a 10-min window guarantees at most one full cycle of staleness before the fleet snapshot is treated as uncorrelated. |
| `STRATEGY_BACKLOG_THRESHOLD` | `20` | >20 pending missions triggers a fleet-level strategy evaluation even without a failure event. |
| `STRATEGY_CONGESTION_THRESHOLD` | `0.8` | `congestion` metric (placeholder) above 0.8 signals overload. |

---

## 7. Policy Guardrail bounds (§6)

| Variable | Value | Rationale |
|----------|-------|-----------|
| `POLICY_MAX_DURATION_SEC` | `7200` | 2 hours. No agent-issued policy should stay active longer without re-evaluation. |
| `POLICY_MIN_DURATION_SEC` | `60` | 1 minute. Shorter policies don't give the Scheduler time to react. |
| `POLICY_COOLDOWN_SEC` | `120` | 2-minute hysteresis prevents the same policy type from thrashing. |

---

## 8. Outcome Evaluator

| Variable | Value | Rationale |
|----------|-------|-----------|
| `OUTCOME_WINDOW_SEC` | `900` | 15-minute observation window after a policy or recovery action. Matches the failure-rate rolling window so baselines and finals are comparable. |

---

## 9. Fast-path retry budget (§3)

The `retry_budget` in `slow_disposition()` is hardcoded to `2` (matching `FAST_PATH_BUDGET`).  Both represent "a mission gets 2 attempts before we stop calling it transient." If you change `FAST_PATH_BUDGET`, update `slow_disposition` too — these are intentionally kept in sync.

---

## 10. Hot state split decision (§8)

**HOT (Redis):** robot pose, battery_pct, current_zone, allocation_state, health_level.  Updated every tick by the Aggregator.  30-second TTL so stale robots are evicted automatically.

**COLD (PostgreSQL):** all mission history, incidents, policies, diagnoses, outcomes, embeddings.  The distribution counters for the Router also live in Redis (with a `FAILURE_WINDOW_SECONDS` TTL per key), but they are reconstructed from Postgres on Redis restart.

**Rationale:** The Aggregator ticks at ~1 Hz per robot.  Writing 10 robots × 1 Hz to Postgres would create contention with the analytical writes.  Redis handles high-frequency key/value updates cheaply; Postgres receives only event-level writes (on abort, on state transition).

---

## 11. Vector index: HNSW (replaces ivfflat)

The schema uses **HNSW** (`hnsw (embedding vector_cosine_ops)`) instead of ivfflat.

Reason: ivfflat requires `VACUUM ANALYZE` or a manual `REINDEX` to build useful cluster centroids; built on an empty table (as the RAG store is at system start), it produces degenerate clusters and poor recall during a run.  HNSW builds incrementally as rows are inserted — no training phase, correct recall from the first insert.  Default `m=16, ef_construction=64` are fine at our scale; tune `hnsw.ef_search` at query time if recall needs improvement.

---

## 12. Deployment: ROS2 Humble vs no ROS2

All supervisory Python code imports rclpy only inside concrete adapters (`ros/` module).  The supervisor core, agents, validators, and tests are importable without ROS2 installed.  The mock sim (`sim/`) runs as a plain Python thread — no ROS2 required for the M1 demo.

---

## 13. Sim time vs wall-clock timestamps (schema fix)

The system runs on simulation time (`use_sim_time=true`, Isaac Sim `/clock`).  Any timestamp used for window calculations or recency scoring must come from the ROS clock, not `NOW()`.

**Columns with no DEFAULT (app supplies sim time):**

| Table | Column(s) |
|-------|-----------|
| `failures` | `occurred_at` |
| `charging_pressure_metrics` | `recorded_at` |
| `incident_embeddings` | `recorded_at` |
| `outcomes` | `window_start`, `window_end` |
| `missions` | `assigned_at`, `started_at`, `completed_at` |
| `charging_sessions` | `started_at`, `ended_at` |

**Columns with `DEFAULT NOW()` (wall-clock audit only):**

`created_at` on every table, `updated_at` on tables with the trigger, `issued_at` on `policies`.  These track when rows were written to Postgres, not when events occurred in the simulation.

**App responsibility:** the Aggregator, Orchestrator, Charging Service, and Outcome Evaluator must obtain the current sim time from the ROS clock node (`/clock` topic) before writing any sim-time column.  In tests, supply explicit timestamps.

---

## 14. New agent-output tables (schema fix)

Three tables now store every agent run uniformly:

| Table | Agent | Notes |
|-------|-------|-------|
| `diagnoses` | Failure Analysis Agent | UNIQUE on `failure_id` (1:1 per failure) |
| `fleet_analyses` | Fleet State Analysis Agent | — |
| `strategy_runs` | Operations Strategy Agent | Links to both of the above (nullable) |

`policies.strategy_run_id` is now a real FK → `strategy_runs`.  All three tables record PASS/DEGRADE/REJECT runs, making restraint and rejected outputs auditable and retrievable via RAG.

`incident_embeddings.source_type='strategy'` points `source_id` at `strategy_runs.strategy_run_id` (not `policies.policy_id` as in the original draft).

---

## 15. Failure Analysis Investigator (implement-changes change-set, 2026-06-04)

The Failure Analysis Agent was refactored from a single-shot LLM call with a pre-assembled bundle into a **read-only ReAct tool-calling loop**.

| Decision | Value | Rationale |
|----------|-------|-----------|
| Tool loop driver | `OpenAIInvestigatorClient.chat_with_tools()` | OpenAI native tool calling; maps cleanly to the ReAct pattern. |
| Tools | `query_failures`, `get_zone_state`, `get_robot_history`, `search_incidents`, `get_active_policies` | Five read-only tools; `search_incidents` integrates the Retrieval Validator per-call instead of as a pre-stage. |
| `INVESTIGATOR_MAX_TOOL_CALLS` | `10` | Prevents runaway tool loops; conservative for a warehouse with ≤20 robots. |
| `INVESTIGATOR_MAX_ITERATIONS` | `5` | Max conversation turns before forcing the final step. |
| `INVESTIGATOR_TIMEOUT_SEC` | `30` | Wall-clock limit for the full investigation. |
| Non-convergence fallback | `cause=unknown, confidence=0.1` | Always returns a valid diagnosis so the DV can DEGRADE and the orchestrator falls back to a safe default. |
| Transcript key mapping | `query_failures → mission_failures`, `get_zone_state → zone_state`, etc. | The flattened transcript uses the same dict keys the DV's ref resolver already handles; no changes to `_resolve_ref` or existing evidence refs. |
| Read-only DB role | `mars_reader` (created by `0002_readonly_role.sql`) | Tools physically cannot write; enforcement is at the DB level, not convention. `DB_READONLY_DSN` in config points to this role. |
| Orchestrator slow path | Bundle pre-assembly removed | Orchestrator now passes only `trigger_event` to the investigator; the investigator assembles its own evidence via tool calls. `retrieval_validator_fn` parameter kept (still used by strategy/fleet agents). |
| Output contract | **Unchanged** | Same diagnosis dict shape; `slow_disposition()`, Strategy Trigger Rules, and all downstream consumers are unmodified. |

**Vertical slice verification:** synthetic failure → router (SLOW) → investigator calls `query_failures` tool → transcript populated → diagnosis → DV grounds refs against transcript → PASS → strategy trigger → guardrail → avoid_zone activated → scheduler defers dock missions. All 125 tests pass.
