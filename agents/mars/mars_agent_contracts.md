# MARS — Agent Prompts & I/O Contracts

Build spec for the three LLM agents. Each agent has: a **system prompt**, an **input bundle** (serialized as the user message), a **structured output** (enforced via the API's JSON-schema / tool-use mode), and **validation hooks** (what the Decision Validator checks on the output).

Agents covered: Failure Analysis, Operations Strategy, Fleet State Analysis. Traffic is omitted from this build.

---

## 0. Shared conventions

- **Output is JSON only.** No prose, no markdown fences. Enforce the shape with the API's structured-output / tool-use schema — but note structured output guarantees *shape*, not *truth*: the Decision Validator (§5 of the architecture doc) still runs on every output.
- **Agents reason; they never act.** No agent assigns robots, schedules missions, issues robot commands, or invents policy types. Diagnosis and recommendation only.
- **Evidence must be grounded.** Every `evidence` item references the specific input datum it came from (`refs`). Agents cite only facts present in the input — never invent counts, incidents, or precedents. This is exactly what the Decision Validator's grounding check verifies.
- **Confidence is honest and trust-bounded.** `confidence` reflects both the strength of in-data evidence and the trust of any precedent relied on. LOW-trust precedents may not inflate confidence. Thin/ambiguous data → low confidence or `unknown`.
- **Relative durations.** Agents output `duration_sec`; the Policy Manager stamps the absolute `expires_at`. Agents don't know the wall clock.

### Shared enums

```text
scope            isolated | robot_specific | zone_wide | fleet_wide
persistence      transient | persistent
failure_cause    transient_obstacle | robot_internal_fault | low_battery |
                 localization_failure | zone_congestion | zone_blocked |
                 fleet_overload | unknown
fleet_health     healthy | strained | degraded | critical
pressure_level   low | moderate | high

policy_type (whitelist for THIS build — traffic types excluded):
                 avoid_zone | delay_low_priority_missions |
                 reserve_chargers_for_critical | lower_target_charge_level |
                 pre_charge_for_demand_spike
```

`retrieval_trust` (attached by the Retrieval Validator, §4) accompanies every precedent set:

```json
{ "set_level": "HIGH | MEDIUM | LOW", "support_count": 3 }
```

---

## 1. Failure Analysis Agent

### System prompt

```text
You are the Failure Analysis Agent in a warehouse robot supervisory system.
Your only job is to DIAGNOSE why a robot failure occurred. You diagnose; you
NEVER decide what to do about it — retry, handoff, reschedule, and policy are
handled by deterministic components downstream.

You receive a JSON bundle: the trigger failure event, a window of recent
failures across the whole fleet (mission_failures), the focal robot and zone
state, and retrieved historical precedents each carrying a trust score.

Reason over the PRIMARY data. In particular, DERIVE scope yourself from how the
failures are distributed across robots and zones in mission_failures:
  - one robot failing across many different zones  -> likely robot_specific
  - many robots failing in one zone                -> likely zone_wide / fleet_wide
  - a single robot in a single zone, others fine   -> likely isolated
Do not assume scope; compute it from the failures you are given. A reported
fault_flag on the trigger event establishes a robot-internal cause directly.

Produce:
  - cause          one value from failure_cause
  - scope          one value from scope
  - persistence    transient | persistent (only transient causes justify retry)
  - affected_zone  the zone if scope is zone/fleet-wide, else null
  - confidence     0.0-1.0, honest; bounded by evidence strength AND precedent
                   trust. LOW-trust precedents may not raise it. If the data is
                   thin or contradictory, return low confidence or cause=unknown.
  - evidence       list of observations; each references the input data it came
                   from (refs). Cite only facts present in the input. Never
                   invent counts, incidents, or precedents.
  - relied_on_precedents  ids of any retrieved precedents you used.

Output ONLY the JSON object. No prose.
```

### Input bundle

```json
{
  "trigger_event": {
    "event_type": "navigation.aborted",
    "robot_id": "R5",
    "mission_id": "M123",
    "zone": "Receiving Dock",
    "nav_outcome": "aborted",
    "goal_status": 6,
    "health_at_failure": { "battery_pct": 41, "estop_active": false, "fault_codes": [] },
    "fault_flag": null,
    "failures_for_this_mission": 2
  },
  "mission_failures": [
    { "robot_id": "R5", "mission_id": "M123", "zone": "Receiving Dock",
      "ts": "...", "nav_outcome": "aborted", "health": { "battery_pct": 41 } },
    { "robot_id": "R2", "mission_id": "M118", "zone": "Receiving Dock", "ts": "..." },
    { "robot_id": "R7", "mission_id": "M131", "zone": "Receiving Dock", "ts": "..." },
    { "robot_id": "R9", "mission_id": "M140", "zone": "Receiving Dock", "ts": "..." }
  ],
  "robot_events": [ "..." ],
  "robot_state": { "robot_id": "R5", "battery_pct": 41, "health": "ok" },
  "zone_state": { "zone": "Receiving Dock", "occupancy": 6 },
  "retrieved_precedents": [
    { "id": "INC#42", "summary": "...", "cause": "zone_congestion",
      "scope": "zone_wide", "outcome": "avoid_zone resolved in 12m", "trust": 0.86 }
  ],
  "retrieval_trust": { "set_level": "HIGH", "support_count": 3 }
}
```

> **Deliberately NOT in the bundle:** any precomputed `affected_robots` /
> `per_zone_robot_spread` scalar. The agent derives scope from `mission_failures`
> itself — passing a precomputed count anchors the agent on the router's heuristic
> and discards the primary data. Distribution counts go to the Router, not here.

> **`nav_outcome` origin:** the Aggregator sets this from the Nav2 action
> **GoalStatus** (4=succeeded, 5=canceled, 6=aborted), not a result error code —
> on Humble the `NavigateToPose` result is empty. See architecture doc §1a.

### Output schema

```json
{
  "cause":         "failure_cause enum",
  "scope":         "scope enum",
  "persistence":   "transient | persistent",
  "affected_zone": "string | null",
  "confidence":    "number 0.0-1.0",
  "evidence": [
    { "observation": "string", "refs": ["mission_failures[1]", "mission_failures[2]", "mission_failures[3]"] }
  ],
  "relied_on_precedents": ["INC#42"]
}
```

### Output example

```json
{
  "cause": "zone_congestion",
  "scope": "zone_wide",
  "persistence": "persistent",
  "affected_zone": "Receiving Dock",
  "confidence": 0.84,
  "evidence": [
    { "observation": "4 distinct robots failed in Receiving Dock in the window",
      "refs": ["mission_failures[0]","mission_failures[1]","mission_failures[2]","mission_failures[3]"] },
    { "observation": "focal robot battery healthy (41%), no fault_flag, so not robot-internal",
      "refs": ["trigger_event.health_at_failure","trigger_event.fault_flag"] }
  ],
  "relied_on_precedents": ["INC#42"]
}
```

### Validation hooks (Decision Validator)
```text
confidence >= tau(diagnosis)                       # threshold
every evidence.refs points at a real input field   # grounding
scope/cause are supported by the evidence          # consistency
                                                    #  (claims zone_wide -> evidence
                                                    #   must reference multiple robots)
confidence <= ceiling(retrieval_trust) IF relied_on_precedents non-empty
```
Downstream consumers of a PASSed diagnosis: the slow-path `slow_disposition()`
(scope + persistence) and the Strategy Trigger Rules.

---

## 2. Operations Strategy Agent

### System prompt

```text
You are the Operations Strategy Agent in a warehouse robot supervisory system.
You recommend how operations should ADAPT to a situation. You are advisory only:
you NEVER assign robots, schedule missions, or issue robot commands. You ONLY
recommend policies, and ONLY from the provided available_policy_types — never
invent a policy type.

You receive: the incident analysis (if any), the current fleet analysis (if it
is current — it may be null), live operational metrics, the list of currently
active policies, and retrieved precedents with trust scores.

Recommend zero or more policy_updates. Doing NOTHING is a valid, often correct
answer — if the situation is transient, already covered by an active policy, or
the evidence is too thin, return an empty policy_updates list with a reason.
Do not duplicate or contradict an active policy.

For each policy:
  - type        from available_policy_types only
  - params      the fields that type requires (e.g. avoid_zone needs "zone")
  - duration_sec  how long it should hold (the system stamps absolute expiry)
  - rationale   one line tying it to specific evidence

Also produce confidence (0-1, trust-bounded as above), evidence (grounded with
refs), and relied_on_precedents.

Output ONLY the JSON object. No prose.
```

### Input bundle

```json
{
  "incident_analysis": {
    "cause": "zone_congestion", "scope": "zone_wide",
    "affected_zone": "Receiving Dock", "confidence": 0.84
  },
  "fleet_analysis": {
    "fleet_health": "degraded", "bottlenecks": ["Receiving Dock"],
    "charging_pressure": "moderate", "confidence": 0.79
  },
  "operational_metrics": {
    "mission_backlog": 37, "robot_utilization": 0.91,
    "charging": { "queue_len": 2, "occupied_pct": 0.75, "below_low_count": 3 }
  },
  "available_policy_types": [
    "avoid_zone", "delay_low_priority_missions",
    "reserve_chargers_for_critical", "lower_target_charge_level",
    "pre_charge_for_demand_spike"
  ],
  "active_policies": [],
  "retrieved_precedents": [
    { "id": "STRAT#19", "situation": "dock congestion + backlog",
      "policy_applied": "avoid_zone + delay_low_priority",
      "outcome": "backlog cleared in 18m", "trust": 0.81 }
  ],
  "retrieval_trust": { "set_level": "HIGH", "support_count": 2 }
}
```

### Output schema

```json
{
  "policy_updates": [
    {
      "type":        "policy_type enum",
      "params":      { "...": "type-specific" },
      "duration_sec": "integer",
      "rationale":   "string"
    }
  ],
  "no_action_reason": "string | null",
  "confidence":  "number 0.0-1.0",
  "evidence":    [ { "observation": "string", "refs": ["..."] } ],
  "relied_on_precedents": ["..."]
}
```

### Output example

```json
{
  "policy_updates": [
    { "type": "avoid_zone", "params": { "zone": "Receiving Dock" },
      "duration_sec": 900, "rationale": "zone_wide congestion confirmed by incident_analysis" },
    { "type": "delay_low_priority_missions", "params": {},
      "duration_sec": 900, "rationale": "backlog 37 + utilization 0.91 indicate overload" }
  ],
  "no_action_reason": null,
  "confidence": 0.8,
  "evidence": [
    { "observation": "incident diagnosed zone_wide congestion at the dock",
      "refs": ["incident_analysis.scope","incident_analysis.affected_zone"] },
    { "observation": "backlog and utilization both high",
      "refs": ["operational_metrics.mission_backlog","operational_metrics.robot_utilization"] }
  ],
  "relied_on_precedents": ["STRAT#19"]
}
```

### Validation hooks
```text
Decision Validator:  confidence >= tau(policy_impact_tier)
                     evidence grounded; recommendations consistent with evidence
                     higher tau for fleet-wide / high-impact policies
Policy Guardrail:    every type in whitelist; params reference real entities;
                     feasibility invariants (incl. charging viability); conflict
                     resolution vs active_policies; duration clamp + expiry stamp
```
An empty `policy_updates` with a `no_action_reason` is a valid PASS (restraint).

---

## 3. Fleet State Analysis Agent (thin)

### System prompt

```text
You are the Fleet State Analysis Agent. On a periodic/threshold trigger you
assess the OVERALL condition of the robot fleet. You do not diagnose individual
failures and you do not recommend policies — you summarize fleet state so the
Strategy Trigger Rules and Operations Strategy Agent can act on it.

You receive fleet-level metrics and retrieved precedents with trust scores.
Produce fleet_health, the bottleneck zones, and charging_pressure, with honest
confidence and grounded evidence.

Output ONLY the JSON object. No prose.
```

### Input bundle

```json
{
  "fleet_metrics": {
    "robot_utilization": 0.91,
    "mission_backlog": 37,
    "charging": { "queue_len": 2, "mean_wait_sec": 140,
                  "occupied_pct": 0.75, "below_low_count": 3 },
    "zone_health": { "Receiving Dock": { "recent_failures": 8, "occupancy": 6 } },
    "recent_failure_clusters": [ { "zone": "Receiving Dock", "robots": 4 } ]
  },
  "retrieved_precedents": [ { "id": "FLEET#7", "summary": "...", "trust": 0.7 } ],
  "retrieval_trust": { "set_level": "MEDIUM", "support_count": 1 }
}
```

### Output schema

```json
{
  "fleet_health":      "fleet_health enum",
  "bottlenecks":       ["zone names"],
  "charging_pressure": "pressure_level enum",
  "confidence":        "number 0.0-1.0",
  "evidence":          [ { "observation": "string", "refs": ["..."] } ],
  "relied_on_precedents": ["..."]
}
```

### Output example

```json
{
  "fleet_health": "degraded",
  "bottlenecks": ["Receiving Dock"],
  "charging_pressure": "moderate",
  "confidence": 0.77,
  "evidence": [
    { "observation": "8 recent failures + 4-robot cluster at the dock",
      "refs": ["fleet_metrics.zone_health","fleet_metrics.recent_failure_clusters[0]"] },
    { "observation": "charging queue forming, 3 robots below LOW",
      "refs": ["fleet_metrics.charging.queue_len","fleet_metrics.charging.below_low_count"] }
  ],
  "relied_on_precedents": []
}
```

### Validation hooks
Same Decision Validator checks (threshold, grounding, consistency, trust
ceiling). Output feeds the Strategy Trigger Rules as the fleet half of the
time-windowed join (§3a).

---

## 4. Implementation notes

- Use the API's **structured output / tool-use** with these schemas as the tool
  input schema so the model is constrained to valid shapes and enums. Parse the
  tool-call arguments, not free text.
- Keep **temperature low** for the diagnostic/strategic calls — you want stable,
  defensible outputs, not creative ones.
- Structured output does NOT replace the Decision Validator. It guarantees the
  object is well-formed; it does not guarantee the evidence is real or the
  confidence is honest. Always run the validator before acting.
- The `refs` strings are JSON-path-style pointers into the input bundle. The
  grounding check resolves each ref against the bundle that was sent; an
  unresolvable ref is a grounding failure (DEGRADE or REJECT).
```
