# MARS — Revised Architecture (v2)

Supervisory multi-agent robot system. This revision adds:

- A **deterministic router** with a **fast-path / slow-path** recovery split.
- A **Retrieval Validator** gating what RAG context the agents are allowed to trust.
- A **Decision Validator** gating agent output before it can drive action.
- A **Policy Guardrail Layer** that validates policy *artifacts* and their *effect on global fleet state*.
- An **Outcome Evaluator** that closes the RAG loop by writing labeled outcomes back to the blackboard.

The original principle holds and is now enforced structurally: **agents reason, services execute, and nothing an agent produces reaches a robot without passing a deterministic validator.**

---

## 0. Implementation Scope vs. Full Design

This document describes the **full supervisory system** as it would exist for a real warehouse. The **project contribution is the AI-agent layer** — multi-agent reasoning, retrieval-augmented diagnosis, and the machinery that makes unreliable agent output safe to act on. Everything outside that layer is environment: it exists in the design for completeness, but is stubbed in the implementation so the agents' behavior stays the focus.

**Scoping razor:** if a component makes the agents' reasoning more visible, interesting, or trustworthy, it is built. If it only makes the warehouse physics realistic, it is stubbed.

### Built (the AI-agent contribution)

```text
Failure Analysis Agent          LLM diagnosis (cause / scope) over raw failures
Operations Strategy Agent       LLM policy recommendation
Fleet State Analysis Agent      periodic fleet assessment        (thin — see note)
Blackboard + pgvector RAG       agent memory + retrieval-augmented diagnosis
Retrieval Validator             gates trust in retrieved precedent
Decision Validator              gates agent output (evidence + confidence)
Policy Guardrail (light)        schema + whitelist + 1 safety check
Aggregator enrichment           health snapshot + zone/robot distribution
Deterministic Router            fast/slow path selection
Charging pressure loop          pressure metrics -> agent -> charging policy -> service
```

### Stubbed (environment — visibly reacts, but not optimized)

```text
Scheduling Service     priority + nearest idle robot + flat battery threshold;
                       no energy math, no ETA optimization, no aging   (§6a is the
                       full design; the build is the ~dozen-line heuristic)
Traffic Coordination   omitted; `avoid_zone` is honored by the scheduler skipping
                       those missions — no real intersection/deadlock logic
Charging Service       deterministic executor: tiers + priority queue + target
                       level + hysteresis (§6b). The pressure->policy LOOP is built
                       (above); the mechanics stay simple
Energy / path planning  NOT in the supervisor — robot-level (Principle 4).
                       Supervisor reads battery %; it never computes route energy
ROS Executor / robots   simulated; a script publishes synthetic ROS2-style events
Outcome Evaluator      minimal or logged-only for the demo (full loop = future work)
Policy expiry / precedence, operator-in-the-loop, workflow durability  = design only
```

### The vertical slice that gets demonstrated end-to-end

```text
simulated failures stream in
  → Aggregator detects a zone pattern (distribution counts)
  → Failure Analysis Agent diagnoses cause/scope via RAG over past incidents
  → Decision Validator checks confidence + evidence
  → Strategy Trigger Rules fire
  → Operations Strategy Agent recommends avoid_zone
  → Policy Guardrail validates it
  → stub Scheduler visibly stops assigning into that zone
  → outcome recorded back to the blackboard
```

This single flow exercises every graded concept — multi-agent coordination, RAG/memory, agent-output validation, and reason/execute separation — while everything it touches outside the agent layer is a stub.

> **Note on the Fleet State Analysis Agent:** kept in the design as the second input to the Strategy Trigger Rules, but implemented thin (a periodic summary). The depth goes into the Failure Analysis Agent's diagnosis + RAG + validation, which is the cleanest single story to demonstrate and defend.

---

## 1. Revised Top-Level Flow

```text
                         ROS2 Topics
                              │
                              ▼
                       Aggregator Layer
        (normalize • health snapshot • distribution counts • §1a)
                              │
                              ▼
                          Event Bus ───────────────────────┐
                              │                             │
                  (per failure event)              (periodic / threshold)
                              ▼                             ▼
                        Orchestrator              Fleet Monitoring Workflow
                              │                             │
                              ▼                   Fleet State Analysis Agent
                 ┌────────────────────────┐      ┌──────────┴────────┐
                 │   Deterministic Router  │      │   Validated RAG    │◄─ Retrieval
                 │   (scope pre-classify)  │      └──────────┬────────┘   Validator
                 └───────────┬────────────┘                 ▼
             fast path │                  │ slow      Decision Validator
                       ▼                  ▼ path             │
            Mission Replanning    Failure Analysis Workflow  │
            (immediate, det.)              │                 │
                       │           Failure Analysis Agent    │
                       │           ┌───────┴────────┐        │
                       │           │  Validated RAG │◄─ Ret. │
                       │           └───────┬────────┘  Valid.│
                       │                   ▼                 │
                       │            Decision Validator       │
                       │          (evidence • confidence)    │
                       │      pass │ degrade │ reject         │
                       │           │         └► safe/human    │
                       │           ▼                          │
                       └────► Mission Replanning              │
                              (diagnosis-informed)            │
                                      │                       │
                                      ▼                       │
                            Strategy Trigger Rules ◄──────────┘
                              (deterministic; correlates
                               failure + fleet analysis)
                                      │
                                      ▼
                          Operations Strategy Agent
                          ┌───────────┴────────┐
                          │   Validated RAG     │◄── Retrieval Validator
                          └───────────┬────────┘
                                      ▼
                            Decision Validator
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │   Policy Guardrail Layer   │
                        │ schema • refs • feasibility│
                        │ conflict • bounds • rate   │
                        └──────────────┬─────────────┘
                                       ▼
                                 Policy Manager
                                       │
                   ┌───────────────────┼───────────────────┐
                   ▼                   ▼                   ▼
              Scheduling         Traffic Coord.         Charging
                   │
                   ▼
              ROS Executor ───► Robots


   Outcome Evaluator ── writes labeled outcomes ──►  PostgreSQL Blackboard
        ▲  (watches metrics over recovery / policy window)      (+ pgvector)
```

Key structural change: the **router**, **two validators**, and the **guardrail** are all deterministic checkpoints surrounding the non-deterministic agents. Agents never touch a service directly.

Note the two independent entry paths. **Failure analysis** is event-driven (per `navigation.aborted` / `mission.failed`) and runs through the Router. **Fleet analysis** is periodic / threshold-driven and bypasses the Router entirely. Both converge at the Strategy Trigger Rules, which is where their outputs are correlated (§3a).

---

## 1a. Aggregator Failure Enrichment (new)

The Router, fast path, and Failure Analysis Agent all assume two things exist on a failure event: the robot's health *at the moment of failure*, and the failure's *distribution* across robots and zones. Neither is in a raw ROS2 abort. The Aggregator computes both before the event reaches the Router, so deterministic consumers never have to infer and the agent reasons over pre-computed structured context instead of re-deriving facts from raw topics.

This is the single owner of the "cheap signals" the Router reads in §2.

### Two independent axes

A failure must be classified on two axes that are easy to conflate:

```text
AXIS 1  transient vs persistent   ── resolved by repetition / count
AXIS 2  robot-internal vs env.    ── resolved by health snapshot + distribution
```

Repetition answers Axis 1 only. A blocked corridor produces repeated failures too, so a high failure count says "persistent," not "robot's fault." Axis 2 needs different evidence.

### Health snapshot (deterministic — answers Axis 2 for the easy cases)

On every failure event, snapshot the robot's state at the failure instant from the health topics. Subscribe to:

```text
/robot_state    battery, health status      (already in v1 topic list)
/diagnostics    component-level fault status (add)
/nav_result     the abort's own result/error code
```

From the snapshot, set a deterministic `fault_flag` when any of these hold — no inference required:

```text
battery below CRITICAL_THRESHOLD
e-stop active
diagnostics ERROR-level entry on a hardware component
```

If `fault_flag` is set, robot-internal is *established* on the first failure. If not, Axis 2 is unresolved and must come from distribution (below) or the agent.

### Distribution counts (deterministic — the cheap direction discriminator)

Maintain, over a rolling window, a cross-tab of where failures land:

```text
per_robot_zone_spread(robot)   # how many DISTINCT zones this robot failed in
per_zone_robot_spread(zone)    # how many DISTINCT robots failed in this zone
```

This is a stronger and cheaper direction signal than same-mission repetition:

```text
one robot, MANY zones        → robot-internal  (robot is the common factor)
many robots, ONE zone        → environmental   (zone is the common factor)
one robot, one zone, healthy → ambiguous → agent's job
```

### Enriched failure event

```json
{
  "event_type": "navigation.aborted",
  "robot_id": "R5",
  "mission_id": "M123",
  "zone": "Receiving Dock",
  "nav_outcome": "aborted",
  "goal_status": 6,

  "health_at_failure": {
    "battery_pct": 14,
    "estop_active": false,
    "diagnostics": [ ... ]
  },
  "fault_flag": null,

  "distribution": {
    "per_robot_zone_spread": 1,
    "per_zone_robot_spread": 4
  },

  "failures_for_this_mission": 2
}
```

### Who consumes what

```text
Router      reads fault_flag + distribution + failures_for_this_mission
            -> path selection only (§2)
Fast path   acts only on fault_flag (handoff + redirect) or first-failure retry
Agent       reads the whole enriched event + history + RAG
            -> infers direction for the ambiguous remainder
```

The agent never subscribes to health topics itself or recomputes these counts; it consumes what the Aggregator attached. Facts that are directly observable are resolved deterministically; only genuine inference reaches the LLM.

### Topic → enriched-event field mapping (Isaac Sim + Nav2 Humble)

All subscriptions are per-robot namespaced (`/robotN/...`). Three sources: **Nav2** (navigation), **Isaac Sim** (sim state/sensors), and **self-published** sim nodes (battery, health — nothing produces these otherwise).

```text
SUBSCRIBED TOPIC                         TYPE                          SRC      -> FIELD
…/navigate_to_pose/_action/status        action_msgs/GoalStatusArray   Nav2     event_type, nav_outcome,
                                                                                 goal_status  (TRIGGER)
…/navigate_to_pose/_action/feedback      nav2_msgs/NavigateToPose_Fb   Nav2     (struggle signal:
                                                                                 number_of_recoveries,
                                                                                 distance_remaining)
/tf  (map -> base_link)                   tf2_msgs/TFMessage            IsaacSim robot pose -> zone,
                                                                                 distribution
…/odom                                    nav_msgs/Odometry             IsaacSim velocity, backup pose
…/battery_state                           sensor_msgs/BatteryState      SELF     health_at_failure.battery_pct
…/robot_health                            mars_msgs/RobotHealth (NEW)   SELF     health_at_failure.estop /
                                                                                 fault_codes -> fault_flag
/clock                                    rosgraph_msgs/Clock           IsaacSim use_sim_time = true
```

`nav_outcome` is read from the action **status** (GoalStatus: 4=succeeded, 5=canceled, 6=aborted), NOT a result error code — on Humble the `NavigateToPose` result is empty. `fault_flag` is derived by the Aggregator from `robot_health.level == ERROR` OR `estop_active` OR `battery_pct < CRITICAL`. "Mission" is supervisory: dispatch is a `NavigateToPose` goal and mission state lives in the blackboard — there is no robot-published mission topic.

---

## 1b. Custom ROS2 Interfaces (`mars_msgs`)

Almost everything uses standard messages. Build **one** custom message; the rest are optional. Put these in a `mars_msgs` interface package.

### RobotHealth.msg  — BUILD THIS

Self-published by the sim's health / fault-injection node. The Aggregator combines it with `sensor_msgs/BatteryState` to set `fault_flag`. It doubles as your scenario generator: publishing a fault here is how you trigger Failure-Analysis demos.

```text
# mars_msgs/msg/RobotHealth
std_msgs/Header header

uint8 LEVEL_OK=0
uint8 LEVEL_WARN=1
uint8 LEVEL_ERROR=2

string    robot_id
uint8     level          # overall health (constants above)
bool      estop_active
string[]  fault_codes    # e.g. ["MOTOR_FAULT","LIDAR_TIMEOUT"]
```

Battery stays in the standard `sensor_msgs/BatteryState` (`percentage` 0–1, `power_supply_status`: 1=CHARGING, 2=DISCHARGING, 4=FULL). Do not duplicate it here.

### MissionCommand.msg / MissionStatus.msg  — OPTIONAL

Only if you want a clean supervisory mission abstraction over Nav2, or a mission topic for a monitor/UI. Otherwise the ROS Executor dispatches a plain `NavigateToPose` goal and mission state stays in the blackboard.

```text
# mars_msgs/msg/MissionCommand   (ROS Executor -> robot; executor turns it into a Nav2 goal)
std_msgs/Header header
uint8 ACTION_ASSIGN=0
uint8 ACTION_CANCEL=1
string                    mission_id
string                    robot_id
uint8                     action
geometry_msgs/PoseStamped start
geometry_msgs/PoseStamped destination
uint8                     priority
uint8                     scheduling_priority
```

```text
# mars_msgs/msg/MissionStatus   (supervisor-published lifecycle; mirrors blackboard state)
std_msgs/Header header
uint8 PENDING=0
uint8 ASSIGNED=1
uint8 ACTIVE=2
uint8 COMPLETED=3
uint8 FAILED=4
string  mission_id
string  robot_id
uint8   state
uint8   retry_count
uint8   handoff_count
string  failure_reason     # set when state == FAILED
```

### MissionFeasible.srv  — OPTIONAL (honors Principle 4)

A service the supervisor calls before dispatch so the robot — which owns path planning and therefore energy — answers feasibility, instead of the supervisor computing route energy. The fallback is a flat battery-% threshold (§6a).

```text
# mars_msgs/srv/MissionFeasible
geometry_msgs/PoseStamped start
geometry_msgs/PoseStamped destination
---
bool    can_complete             # enough energy to finish the mission
bool    can_reach_charger_after  # the non-negotiable floor
float32 estimated_energy_pct
```

---

## 2. Deterministic Router (new)

Sits between Orchestrator and the recovery workflows. Uses only cheap structured signals already maintained by the Aggregator (§1a). Its single job is **path selection**, not diagnosis.

### Signals it reads (all cheap, all deterministic — sourced from §1a)

```text
fault_flag                                # established robot-internal fault
per_zone_robot_spread(zone)               # = distinct_robots_failing_in_zone
failures_for_this_mission(window)         # retry-loop detection
active_policy_on_zone?                     # is this already being handled?
zone_in_known_degraded_set?
```

### Routing logic

```python
def route(failure_event):
    # retry-loop guard: a mission that keeps coming back is not "isolated"
    if failure_event.failures_for_this_mission > FAST_PATH_BUDGET:
        return SLOW_PATH

    # any sign of a pattern -> reason about it
    if failure_event.distribution.per_zone_robot_spread >= ROUTER_SCOPE_HINT \
       or active_policy_on_zone(failure_event) \
       or zone_in_known_degraded_set(failure_event):
        return SLOW_PATH

    return FAST_PATH
```

`distinct_robots_failing_in_zone` is the cheap routing count. It is **not** forwarded to the Failure Analysis Agent — the agent receives raw `mission_failures` and forms its own scope judgment. The two never feed each other.

---

## 3. Fast-Path / Slow-Path Recovery

### Principle
The *current mission's disposition* is latency-sensitive (a stuck robot may block a corridor). *Understanding and systemic response* is not. Split them so a stuck robot never waits on LLM + vector search.

### Fast path (common case: isolated, first-time-ish failure)

```text
navigation.aborted ──► Router ──► FAST
        │
        ▼
Mission Replanning (deterministic, structured state only)
        │
        ▼ disposition (retry | handoff | reschedule | abort)
ROS Executor
        │
        └──► incident still logged to blackboard
             (optional async enrichment — does NOT gate recovery)
```

No LLM, no RAG. Sub-second disposition. The fast path has a **retry budget**; exceeding it forces the slow path so it can never silently mask a real problem by retrying forever.

### Slow path (pattern detected, repeated failure, or fast-path ambiguity)

```text
navigation.aborted ──► Router ──► SLOW
        │
        ├──► (1) immediate provisional safe action
        │        e.g. HOLD / park robot, do NOT retry into suspect zone
        │
        └──► (2) Failure Analysis Agent  (LLM)
                   inputs: raw mission_failures, robot_events,
                           mission/zone state, validated RAG context
                   │
                   ▼
             Decision Validator
              ├─ pass    ──► Mission Replanning (diagnosis-informed)
              ├─ degrade ──► conservative reversible disposition only
              └─ reject  ──► deterministic safe default + flag for operator
                   │
                   ▼
             Strategy Trigger Rules ──► Operations Strategy Agent ──► ...
```

The provisional safe action (1) unblocks the robot immediately; the considered disposition (2) follows once analysis + validation complete. Recovery latency is decoupled from reasoning latency even on the slow path.

### Fast-path disposition (no diagnosis — deterministic signals only)

The fast path acts only on what is *established*, never on what would need inference. The only safe handoff here is on an explicit `fault_flag` (§1a); a bare navigation failure is given one retry and then escalated, because handing it off blind can propagate an environmental block across the fleet.

```python
def fast_disposition(failure):
    # established robot-internal fault -> cause travels with the robot
    if failure.fault_flag:
        redirect_robot(failure.robot_id)      # charge / maintenance
        return "handoff"                       # safe; another robot is the answer

    # bare navigation failure: absorb a transient glitch with ONE retry
    if failure.failures_for_this_mission == 0:
        return "retry"                         # short backoff, same robot

    # persists with no fault flag -> direction is unknown
    return "escalate"                          # Router sends to slow path
```

### Slow-path disposition (diagnosis available, post Decision Validator)

```python
def slow_disposition(failure, diagnosis, policies):
    scope = diagnosis.scope

    if scope in ("zone_wide", "fleet_wide"):       # environmental
        # handoff would propagate the failure -> never handoff here
        if alternate_route_exists(failure.zone, policies):
            return "reschedule"                     # strategy applies avoid_zone
        return "hold_defer"                         # let strategy resolve the zone
        # abort if low-priority AND blockage persists past budget

    if scope in ("isolated", "robot_specific"):
        if looks_transient(failure, diagnosis) and retry_budget_remaining(failure):
            return "retry"
        return "handoff"                            # safe; cause is robot-local

    return "reschedule"
    # both paths respect retry_count AND handoff_count budgets (prevent ping-pong)
```

Two principles encoded here: handoff is only safe when the cause is robot-specific (fast path knows this *only* from `fault_flag`; the agent establishes it otherwise), and retry is gated on transience, not on raw count.

---

## 3a. Fleet Monitoring Workflow & Strategy Trigger Correlation

This workflow is **unchanged in role from v1** but is wrapped in the same validation pattern as the other agents. It is the second of the two inputs into the Strategy Trigger Rules.

### Trigger
Periodic (e.g. every 5 min) **or** a fleet metric crossing a threshold. It is *not* event-driven and does *not* pass through the Deterministic Router — the Router only routes individual failure events.

### Fleet State Analysis Agent

```text
inputs:  robot utilization, mission backlog, traffic metrics,
         charging demand / queues, zone health,
         validated RAG context (historical fleet incidents)

         ┌──────────────┐
         │ Validated RAG │◄── Retrieval Validator   (§4 — same gate as failure agent)
         └──────┬───────┘
                ▼
     Fleet State Analysis Agent  (LLM)
                │
                ▼
        Decision Validator        (§5 — same gate as failure agent)
                │
                ▼
        Strategy Trigger Rules
```

Output (e.g. `fleet_health: degraded`, `bottlenecks: [...]`, `charging_pressure: high`) is treated as agent output and must clear the Decision Validator before it can influence anything.

### The correlation problem at the Strategy Trigger Rules

Failure analysis (event-driven, bursty) and fleet analysis (periodic) arrive on different clocks. The trigger rules must decide whether a given failure analysis and a given fleet analysis describe the **same situation** before acting on them jointly. Treat them as a **time-windowed join**, not independent fire-and-forget inputs:

```python
def evaluate_strategy_trigger(failure_out, fleet_out):
    # correlate within a window; a stale fleet snapshot must not
    # be treated as current context for a fresh incident
    if not within_correlation_window(failure_out, fleet_out):
        fleet_out = None        # act on failure signal alone

    # deterministic thresholds (unchanged intent from v1)
    if failure_out and failure_out.scope in ("zone_wide", "fleet_wide"):
        return TRIGGER
    if fleet_out and (fleet_out.backlog > BACKLOG_T
                      or fleet_out.congestion > CONGESTION_T):
        return TRIGGER
    return NO_TRIGGER
```

This is where the v1 ambiguity ("does the trigger evaluate both jointly or independently?") gets resolved: **jointly when they fall in the same correlation window, independently otherwise.** The Operations Strategy Agent then receives whichever of the two analyses are current.

---

## 4. Retrieval Validator (new — formalizes your design)

Gates whether the agent is allowed to trust retrieved precedent. Runs **per result**, then over the **result set**.

### Per-result dimensions

| Dimension      | Meaning                                                                 | Range |
|----------------|-------------------------------------------------------------------------|-------|
| metadata_match | structured-attribute alignment (zone, robot model, failure type, fleet-size regime, time-of-day) | 0–1 |
| recency        | time-decayed weight; older precedent counts less (layout/fleet changes invalidate it) | 0–1 |
| coverage_match | does the precedent's scope (isolated / zone-wide / fleet-wide) match the current incident's apparent scope? | 0–1 |
| similarity     | raw vector cosine similarity                                            | 0–1 |

### Scoring — weighted score WITH gating (not a flat average)

```python
def retrieval_trust(r):
    base = (W_META * r.metadata_match
          + W_REC  * r.recency
          + W_COV  * r.coverage_match
          + W_SIM  * r.similarity)

    # gating rules: certain mismatches cap trust regardless of base
    if r.coverage_match < COV_FLOOR:      # e.g. isolated precedent, fleet-wide now
        base = min(base, COV_MISMATCH_CAP)
    if r.recency < RECENCY_FLOOR:         # stale precedent from old fleet config
        base = min(base, STALE_CAP)

    return base
```

### Set-level assessment

```text
filter results with retrieval_trust >= ACCEPT_THRESHOLD
support_count   = how many independent precedents survived
set_consistency = do survivors agree on cause/scope/outcome?
```

### What the trust level gates

```text
HIGH   (strong support_count, consistent):  agent may rely on precedent + outcomes
MEDIUM (weak / mixed):                       precedent is a weak prior only; require
                                             corroboration from CURRENT evidence
LOW    (nothing survives filter):            ignore precedent; reason from current
                                             evidence only; tag diagnosis
                                             "no_reliable_precedent"
```

Crucial coupling: when the agent's conclusion leans on precedent, its **output confidence is bounded by retrieval_trust**. The Decision Validator enforces this (§5).

---

## 5. Decision Validator (new — formalizes your design)

Deterministic gate on **agent output** (applies to both the Failure Analysis Agent and the Operations Strategy Agent). This is the boundary that keeps LLM non-determinism from directly driving the fleet.

### Inputs
`evidence`, `confidence`, `retrieval_trust`, the agent's stated `scope`/`action`.

### Checks

```text
1. CONFIDENCE THRESHOLD
   confidence >= tau(action_class)
   higher tau for higher-impact actions (abort mission, fleet-wide policy)

2. EVIDENCE GROUNDING
   evidence is non-empty AND every cited fact is verifiable against the
   blackboard / current event.  ("8 aborts in 15 min" must match real data.)
   -> catches hallucinated evidence

3. EVIDENCE–CONCLUSION CONSISTENCY
   the evidence must actually support the stated scope/cause
   (claims fleet_wide but evidence references one robot -> inconsistent)

4. RETRIEVAL COHERENCE
   if confidence is HIGH but retrieval_trust was LOW and the conclusion
   leaned on precedent -> incoherent -> downgrade
```

### Outcomes

```text
PASS    -> proceed normally
DEGRADE -> proceed but force conservative / reversible action only
           (shorter policy duration, no fleet-wide actions, prefer handoff
            over abort)
REJECT  -> fall back to deterministic safe default AND escalate to operator;
           log full reasoning + evidence for review
```

---

## 6. Policy Guardrail Layer (new — drafted in detail)

Sits between the (validated) Operations Strategy Agent output and Policy Manager activation. The Decision Validator (§5) checks the *reasoning*; the Guardrail checks the *policy artifacts* and their *effect on global fleet state*. Both are required — good reasoning can still produce an unsafe global state.

Each candidate policy passes through ordered stages; failure short-circuits to ACCEPT / MODIFY / REJECT / DEFER_HUMAN.

### Stage 1 — Schema validation
```text
policy.type ∈ POLICY_WHITELIST
required fields present and correctly typed
values within legal ranges (duration <= MAX_DURATION, etc.)
```

### Stage 2 — Referential validation
```text
referenced entities exist and are addressable:
  zone exists, corridor exists, charger_group exists, robot_class valid
```

### Stage 3 — Impact classification
```text
each policy.type carries an impact tier:
  LOW     prefer_alternate_route          (advisory)
  MEDIUM  avoid_zone, delay_low_priority
  HIGH    fleet-wide throttles, charger reservations affecting all robots
HIGH tier -> requires elevated confidence OR operator approval (DEFER_HUMAN)
```

### Stage 4 — Feasibility / safety invariants (the critical stage)
Check the proposed policy against current world state. Reject anything that violates a global invariant:

```text
REACHABILITY    after applying, every robot still has a path to:
                  - at least one charger
                  - all mandatory/critical zones
                no fleet partitioning, no stranding low-battery robots
                behind an avoid_zone

CHARGING        enough reachable chargers remain for projected demand;
                reservations don't starve normal operations

THROUGHPUT      no corridor is funneled below capacity (deadlock risk)

LIVENESS        all critical / HIGH-priority missions remain feasible
```

This is where "avoid_zone the only corridor to the chargers" gets caught and rejected.

### Stage 5 — Conflict resolution / precedence
```text
compare against currently ACTIVE policies:
  detect contradictions (avoid_zone X vs prefer_route_through X)
  precedence order:
     operator_override > safety_invariant > critical_mission > efficiency
     on equal tier: newer supersedes older
  -> merge, supersede, or reject
```

### Stage 6 — Bounds & expiry normalization
```text
clamp duration to [MIN, MAX]
require expiry on ALL agent-issued policies (no permanent agent policies)
normalize/round expiry timestamps
```

### Stage 7 — Rate limiting / hysteresis
```text
prevent thrash: a policy on entity X cannot be re-applied/flipped within
COOLDOWN of its last change; require minimum dwell time before reversal
```

### Per-policy outcome
```text
ACCEPT      -> Policy Manager activates + publishes to consumers
MODIFY      -> activate clamped/normalized version, log the modification
REJECT      -> drop, log reason, surface to operator dashboard
DEFER_HUMAN -> queue for operator approval, do not activate yet
```

---

## 6a. Scheduling Service (specified)

Mission-to-robot assignment. A pure executor — it consumes active policies, it never reasons about strategy. v1 named it; this is the spec.

### Triggers
- A scheduling-relevant policy is activated by the Operations Strategy Agent (`avoid_zone`, `delay_low_priority_missions`, ...).
- Periodic sweep of the pending queue.
- A mission re-enters the pending pool from the recovery path (handoff / reschedule).

Two trigger sources can fire concurrently, so assignment must be **single-flight over the resources it touches** (see state machine below).

### Inputs
```text
pending missions   priority, scheduling_priority, start + destination,
                   scheduling_status, wait_time, recovery exclusions
idle robots        location, battery
busy robots        current mission, location, battery, time_remaining
active policies     from Policy Manager
routing            policy-constrained route/ETA (same routing Traffic enforces)
```

### Robot allocation state machine (prevents double-assignment & charging contention)

A robot has **one owner of its allocation state**. The Scheduler and the Charging Service cannot both claim the same robot; transitions are atomic.

```text
        ┌────────► IDLE ◄─────────┐
        │           │             │
   charge done   reserve      (charging
        │           ▼          summons)
   CHARGING ◄── RESERVED ──► dispatched ──► BUSY ──┐
        ▲                                          │
        └──────────── mission complete ────────────┘
```

A robot in `RESERVED`/`BUSY`/`CHARGING` is not a scheduling candidate. The Charging Service summoning a robot and the Scheduler dispatching it are mutually exclusive on the same record (compare-and-set on `allocation_state`).

### Assignment pipeline — filter, then rank

Battery and exclusions are **feasibility filters**, not final tie-breakers, so ranking only ever sees valid robots.

```text
1. POLICY FILTER       drop / defer missions per active policy
                       (avoid_zone -> remove missions routing through it;
                        delay_low_priority -> hold, with a release condition)

2. ORDER QUEUE         scheduling_priority -> priority -> aging -> oldest-first
                       aging: effective priority rises with wait_time
                       (or a hard max_wait that force-promotes)

3. CANDIDATE SET       robots in IDLE (+ eligible BUSY, see below), minus:
                         - locked / charging robots
                         - recovery exclusions (the robot that just failed
                           this mission; the diagnosed-bad zone)

4. FEASIBILITY FILTER  battery >= energy(travel_to_start)
                                 + energy(execute_to_destination)
                                 + reserve(reach charger from end)   [HARD]
                       interruptible exception relaxes "execute_to_destination"
                       ONLY — never the charger reserve (see below)

5. RANK                completion_time =
                         travel(robot -> mission_start) + execute(start -> dest)
                         busy robot: time_remaining + that
                       all distances on POLICY-CONSTRAINED routes

6. ASSIGN (atomic)     compare-and-set allocation_state IDLE->RESERVED,
                       set scheduling_status, emit to ROS Executor
```

### Battery: the non-negotiable is reaching a charger

The relaxable constraint is *mission completion*; the constraint that can **never** be relaxed is *the robot's ability to reach a charger from wherever it ends up*. The interruptible exception (low-priority, interruptible mission, no fully-charged robot available) may dispatch a robot that will break off mid-mission — but only if it still clears the charger floor, there is a safe break point, and resume cost is low. Note this deliberately creates an interrupted mission that re-enters the recovery path, so it is a last resort, not a default. Often the correct action is to leave the mission pending until a charged robot frees up.

> **Scope / Principle 4 note:** the energy terms above require route knowledge, and routing is robot-level — so the *supervisor must not compute them*. In the full design these are obtained by **querying the robot** ("can you complete mission M and still reach a charger?") and letting the robot's planner answer. In the **implementation**, this collapses to a flat battery-% threshold on robot-reported state: `battery_pct >= MIN_DISPATCH_PCT`. No energy math runs in the supervisor.

### Busy-robot consideration
A `BUSY` robot is a candidate when no idle robot is feasible, or when `time_remaining + travel_to_start` beats the best idle robot's completion time. Decide commit semantics explicitly: either **reserve** the busy robot (commit now, risk staleness) or **leave the mission pending** and re-evaluate next cycle (risk indecision) — do not leave it implicit. The busy-robot estimate must also account for the robot needing to charge after its current mission.

### Batch vs greedy
On the periodic sweep the Scheduler holds the whole pending list, so assign the batch jointly rather than purely one-at-a-time greedy — per-mission greed can let a high-priority mission grab the only robot that was also the sole good fit for a soon-to-be-critical mission. Event-triggered single-mission assignment stays greedy by nature.

---

## 6b. Charging Service (specified)

A deterministic executor. The **agent-relevant part is the pressure-up / policy-down loop** (below); everything else is service mechanics that exist only to give that loop knobs to turn.

### The loop (the part that matters for this project)

```text
Charging Service ── emits pressure metrics ──► Blackboard
   (queue length, mean/95p wait, chargers_occupied_pct,
    count of robots below LOW)
                          │
                          ▼
        Fleet State Analysis Agent  →  charging_pressure: high
                          │
                          ▼
        Strategy Trigger Rules → Operations Strategy Agent
                          │   recommends a CHARGING policy
                          ▼
        Decision Validator → Policy Guardrail → Policy Manager
                          │
                          ▼
        Charging Service consumes policy, changes behavior
                          │
                          ▼
        Outcome Evaluator  (did pressure drop?) → labeled outcome → Blackboard
```

This mirrors the failure and (former) congestion loops exactly. The intelligence lives in the agent and the policy; the service only obeys. Two consistency points:

- The charging policy is **not special** — it rides the same Decision Validator and Policy Guardrail as every other agent output. The guardrail's charging-viability invariant (§6, Stage 4) is already what stops a `reserve_chargers_for_critical` policy from reserving so many chargers that normal robots can't charge at all.
- The only genuinely new pieces are the **charging-pressure metric emission** (upward) and the **charging-specific policy types** the agent may issue. Detection, validation, guarding, and outcome recording are all the existing path.

### Example charging policies the agent can issue
```text
reserve_chargers_for_critical    keep N chargers free for CRITICAL-tier robots
lower_target_charge_level         charge to a lower target under pressure
pre_charge_for_demand_spike       top up idle robots ahead of a known inbound batch
```
These are no-ops unless the service exposes the matching knobs — which is why the mechanics below have to exist.

### Service mechanics (deterministic — the knobs the policy turns)

**Battery tiers.** The trigger stays event-driven (`battery_low`, `battery_critical`), but the two are handled differently:

```text
OK        eligible for any mission
LOW       finish current mission, THEN charge; ineligible for new LONG missions
CRITICAL  go to a charger now (interrupt if necessary) — the safety floor
```

CRITICAL is not a fixed % — it is "enough to *reach* a charger + margin," which depends on distance to the nearest charger. That distance is routing, so the judgment is robot-level (Principle 4): the robot answers "can I still reach a charger?" and "can I finish this mission and then reach one?"; the supervisor reacts to the reported tier. (Fixed conservative % is the simpler fallback.)

**Priority queue.** Chargers < robots, so charging requests queue. The queue is ordered by tier (CRITICAL ahead of LOW) — this ordering is exactly what `reserve_chargers_for_critical` reshapes. "Go charge" resolves to a *specific* charger (nearest free / soonest-free); the supervisor picks which charger, the robot plans the route.

**Target charge level.** Don't always charge to 100% — the top of the curve is slow and hogs a scarce charger. Under contention, charge to a target (e.g. 80%) and release; top off fully only when nobody is waiting. This target is the knob `lower_target_charge_level` turns.

**Opportunistic trigger.** Beyond the low/critical events, an idle robot near a free charger that is below a `TOP_UP_LEVEL` charges during lulls — spending otherwise-wasted idle time to avoid a future mid-shift interruption.

**Hysteresis.** Separate the "start charging" level from the "ok to dispatch" level (start < dispatch). Without the gap a robot charges just past LOW, gets dispatched, immediately drops back below LOW, and returns to the charger — the same ping-pong guard used elsewhere.

**Shared allocation state.** Charging is just another writer of `allocation_state` (§6a). Transitioning a robot to `CHARGING` makes it un-dispatchable; the compare-and-set on that field is what prevents the Scheduler and Charging Service from both claiming it. No separate lock.

**Charger faults.** A charger going offline mid-charge re-queues its robot.

---

## 7. Outcome Evaluator (new — closes the RAG loop)

Without this, the vector store fills with recommendations that have no outcome labels, and "retrieve past strategy outcomes" returns unlabeled noise.


```text
On policy activation OR recovery action:
  register a watch over the relevant window (zone metrics, backlog,
  charging queue, abort rate, mission completion)

At window end:
  measure deltas vs. pre-action baseline
  label the outcome: improved | no_effect | worsened
  write {action, context, outcome, magnitude} to the blackboard
        (structured + embedded for pgvector)
```

These labeled records are exactly what the Retrieval Validator's `coverage_match` and the agents' `outcome`-aware reasoning consume on the next incident.

---

## 8. Blackboard Note (storage split)

Don't co-locate high-frequency hot state with the analytical/vector store — they have opposite access patterns and will contend.

```text
HOT STATE  (in-memory / Redis):  robot pose, battery, current mission,
                                 live zone occupancy
HISTORY + VECTORS (PostgreSQL + pgvector): mission history, incidents,
                                 outcomes, policy effectiveness, embeddings
```

The Aggregator updates hot state continuously; only events and resolved records flow to Postgres.

---

## 9. What Changed vs v1 — summary

```text
+ Aggregator enrichment: health snapshot + robot/zone distribution on each failure
+ Two-axis failure model: transient/persistent (count) vs robot/env (health+distribution)
+ Deterministic Router with fast/slow split (recovery latency decoupled from reasoning)
+ Failure analysis no longer runs on every abort (symmetric gating with strategy)
+ Fast-path handoff only on explicit fault_flag; bare nav failure = retry once then escalate
+ Disposition logic now conditioned on cause/scope, not just retry count
+ affected_robots removed from agent input; agent reasons from raw mission_failures
+ Retrieval Validator: metadata/recency/coverage/similarity with gating (not flat avg)
+ Decision Validator on all agent output (evidence + confidence + retrieval coherence)
+ All three agents (Failure, Fleet State, Strategy) wrapped in the same two validators
+ Strategy Trigger Rules now do a time-windowed join of failure + fleet analysis
+ Policy Guardrail Layer: schema -> refs -> impact -> feasibility -> conflict -> bounds -> rate
+ Scheduling Service specified: robot allocation state machine + filter-then-rank pipeline
+ Battery feasibility includes a hard reach-a-charger reserve; interruptible exception gated
+ Charging Service specified: tiers + priority queue + target level + hysteresis + opportunistic
+ Charging pressure->policy loop (pressure metrics -> agent -> charging policy -> service)
+ Outcome Evaluator closes the RAG loop (labeled effectiveness history)
+ Hot state separated from analytical/vector store
+ Topic->field mapping pinned to Isaac Sim + Nav2 Humble; outcome from GoalStatus (result empty)
+ Custom mars_msgs interfaces: RobotHealth (build) + optional MissionCommand/Status, MissionFeasible.srv
```
