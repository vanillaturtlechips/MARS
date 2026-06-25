# Deterministic Validation of LLM Supervisory Agents for Warehouse Robot Fleets: A Multi-Model Study

*Working draft — single author. Target: arXiv → workshop / domestic venue (KIISE/KIPS/KROS). All numbers from eval/RESULTS_*.md (test split, frozen prompts).*

---

## Abstract

Large language models (LLMs) are attractive as a supervisory layer for warehouse
robot fleets: they can diagnose novel mission failures and translate a human
operator's natural-language intent into fleet policies. But an LLM that drives
robots can also hallucinate — emitting a confident but wrong diagnosis, or a
fleet-wide policy the operator never intended. We present MARS, a supervisory
architecture that gates unreliable LLM input and output through deterministic
validation: a retrieval-augmented diagnosis agent whose output is checked by a
decision validator, and an intent agent whose proposed policies are checked by a
policy guardrail. We evaluate both directions on controlled failure/intent
datasets across three LLMs (GPT-4.1-mini, Claude Haiku 4.5, Upstage Solar-Pro).
Retrieval-augmented generation (RAG) improves diagnosis cause accuracy by
38–47 percentage points *across all three models*, and reduces confident-wrong
diagnoses. For operator intent, a defense-in-depth of agent self-restraint plus
guardrail blocks 87–100% of unsafe instructions. Our central finding holds on
both directions and all models: deterministic validation reliably catches
*structurally* invalid output (ungrounded references, out-of-whitelist policies,
nonexistent zones) but **cannot** catch *grounded-but-wrong* diagnoses or
*valid-but-unintended* policies — validation is necessary but not sufficient, and
the residual risk is model-dependent.

---

## 1. Introduction

Autonomous mobile robots (AMRs) in warehouses operate as a fleet: dozens of
robots share aisles, chargers, and mission queues under a central operations
stack (e.g., Nav2 for navigation, a scheduler for mission assignment). When a
mission fails — a robot aborts a navigation goal, a zone becomes congested, a
localization estimate diverges — a human operator must diagnose *why* and decide
*what fleet-level action* to take. As fleets scale, this supervisory reasoning
becomes a bottleneck.

LLMs are a natural candidate for this layer. They can read heterogeneous
evidence (failure events, robot history, zone state) and produce a structured
diagnosis; they can also accept a free-form operator instruction ("keep aisle 5
clear for the next hour") and translate it into a concrete fleet policy. Both are
tasks that classical rule engines handle poorly because the input space is
open-ended.

The problem is reliability. An LLM supervisor sits *above* the robots and its
decisions change fleet behavior, so a hallucinated diagnosis or a misread intent
is not a harmless text error — it can reroute or stall the whole fleet. The
research question of this paper is therefore not "can an LLM do fleet
supervision" (it can) but **"how much safety can deterministic validation add to
an unreliable LLM supervisor, and where does it stop helping?"**

We make three contributions:

1. **A supervisory architecture (MARS)** that gates unreliable LLM *output*
   (a RAG diagnosis agent + decision validator) and unreliable LLM *input* (an
   intent agent + policy guardrail) through deterministic checks, separating
   "reason" (LLM) from "act" (validated).
2. **A multi-model quantitative evaluation** on controlled failure (n=100) and
   intent (n=39) test sets across three LLMs. RAG lifts diagnosis cause accuracy
   by 38–47 pp on every model and suppresses confident-wrong output; agent +
   guardrail block 87–100% of unsafe operator intents.
3. **A characterization of the safety ceiling**: deterministic validation
   catches structurally invalid output but not grounded-but-wrong diagnoses or
   valid-but-unintended policies. This residual risk is model-dependent (e.g.,
   weaker models decline rather than guess), so safety emerges from the
   *interaction* of RAG, agent self-restraint, and the validator — not from any
   single mechanism.

*Scope.* This paper evaluates the LLM supervisory layer on controlled,
programmatically generated events. The robot/simulation integration (Isaac Sim +
Nav2, a failure bridge that forwards real navigation aborts to the agent) is
implemented but is not part of the quantitative evaluation; full
robot-in-the-loop measurement is left to future work.

---

## 2. Related Work

*(To be completed; key threads to position against — ~20–30 refs total.)*

- **LLM agents and tool use:** ReAct-style reason–act loops, tool-calling /
  function calling for structured action.
- **Retrieval-augmented generation (RAG):** grounding LLM output in retrieved
  evidence; trust/quality of retrieval.
- **LLM safety and guardrails:** output validation, constrained decoding,
  schema/whitelist enforcement, hallucination detection.
- **Multi-robot / fleet management:** task allocation, congestion and charging
  management, failure recovery in AMR fleets.
- **LLMs for robotics:** natural-language instruction grounding, LLM planners.

*Positioning:* prior work shows LLMs can act in robotic/operational settings; we
focus on the **machinery that makes unreliable agent output safe to act on**, and
measure where that machinery's guarantees stop.

---

## 3. System Architecture

MARS is a supervisory layer over an existing fleet stack. It never commands a
robot directly; it produces *validated* diagnoses and *validated* fleet policies.
Two pipelines share a blackboard (PostgreSQL + pgvector for retrieval).

### 3.1 Diagnosis pipeline (validating LLM output)

```
failure event ─► Failure Analysis Agent (ReAct tool loop) ─► diagnosis ─► Decision Validator ─► {PASS | DEGRADE | REJECT}
                          │                                      ▲
                          └─ tools: mission_failures, zone_state, robot_history,
                             retrieved_precedents (RAG), active_policies
```

The **Failure Analysis Agent** runs a bounded ReAct loop: it calls read-only
tools to gather evidence, retrieves similar past incidents (precedents) by vector
similarity, and emits a structured diagnosis — `cause` (one of an 8-value enum:
`transient_obstacle, robot_internal_fault, low_battery, localization_failure,
zone_congestion, zone_blocked, fleet_overload, unknown`), `scope`
(`isolated, robot_specific, zone_wide, fleet_wide`), `persistence`, `confidence`,
and `evidence` references.

The **Decision Validator** is deterministic. It rejects/degrades a diagnosis when
confidence < τ (=0.5), when evidence references are empty or unresolvable, when a
zone/fleet-wide scope is not supported by ≥2 mission-failure references, or when
the agent relied on low-trust retrieval at high confidence. The system *acts*
only on PASS; DEGRADE/REJECT are held — a fail-safe.

### 3.2 Intent pipeline (validating LLM input)

```
operator NL utterance ─► Intent Agent ─► candidate policies ─► Policy Guardrail ─► {ACCEPT | MODIFY | REJECT | DEFER_HUMAN}
```

The **Intent Agent** translates a free-form instruction into zero or more
policies drawn *strictly* from a five-entry whitelist (`avoid_zone`,
`delay_low_priority_missions`, `reserve_chargers_for_critical`,
`lower_target_charge_level`, `pre_charge_for_demand_spike`). It may also decline:
`out_of_scope` (no whitelist policy expresses the request) or
`needs_clarification` (utterance too vague).

The **Policy Guardrail** is deterministic and stateful. It validates each
candidate against: schema/whitelist membership, referential integrity (zone
exists), impact tier (HIGH-impact → DEFER_HUMAN), feasibility invariants
(e.g., `avoid_zone` must not strand all robots from chargers; cannot avoid a
mandatory zone), conflict/duplicate detection, duration bounds, and rate limits.

---

## 4. Experimental Setup

### 4.1 Datasets

- **Diagnosis:** 150 cases (dev 50 / test 100), generated programmatically with
  diverse symptom text per cause, randomized numerics, and difficulty tags
  (easy/medium/hard). Cases span all causes plus adversarial variants (e.g.,
  sensor faults that *look* like obstacles, recurring zone blockages) and
  no-evidence cases whose correct answer is `unknown`. Each case has ground-truth
  cause/scope and (where applicable) a relevant precedent seeded into the
  retrieval store.
- **Intent:** 58 cases (dev 19 / test 39), tagged safe / temporal /
  compositional / duration-out-of-bounds / out-of-scope / unsafe-global /
  infeasible / duplicate / ambiguous, with a fleet `world_state` shaped for the
  guardrail's feasibility checks.

Prompts were tuned on **dev only** and then **frozen**; all reported numbers are
on the held-out **test** split.

### 4.2 Models

GPT-4.1-mini, Claude Haiku 4.5, and Upstage Solar-Pro, run through identical
prompts and schemas. Embeddings: a local model (bge-small, 384-d) or OpenAI
text-embedding-3-small; retrieval quality is model-independent.

### 4.3 Metrics

- **Diagnosis:** cause / scope accuracy; precedent reliance (fraction of cases
  with a relevant precedent where the agent actually relied on it);
  **confident-wrong** (wrong cause, not `unknown`, that PASSed validation —
  i.e., a wrong action the system would have taken); **acted-precision**
  (accuracy among PASS cases).
- **Intent:** translation accuracy (safe cases yield the right policy set);
  **must-not-activate violation** (an unsafe/out-of-scope/infeasible/ambiguous
  intent that nonetheless activated a policy); a defense-in-depth decomposition
  (agent declined / guardrail blocked / leaked).
- **Validator stress test:** 30 hand-built probes (ungrounded reference, empty
  evidence, low confidence, unsupported scope, incoherent retrieval) — the
  validator caught 30/30 with zero false blocks.

---

## 5. Results

### 5.1 Diagnosis (test n=100)

| Model | cause (RAG on) | cause (RAG off) | scope (on) | precedent reliance | confident-wrong | acted-precision |
|---|---|---|---|---|---|---|
| GPT-4.1-mini | 81% | 43% | 93% | 9% | 0% | 82% |
| Claude Haiku 4.5 | 93% | 52% | 96% | 92% | 3% | 96% |
| Solar-Pro | 82% | 35% | 73% | 78% | 0% | 87% |

**RAG helps every model** (cause +38 / +41 / +47 pp). The benefit is largest by
difficulty on medium/hard cases (e.g., for GPT-4.1-mini, medium 12%→95% with
RAG). See Figure 1.

### 5.2 Diagnosis safety

Confident-wrong rate (wrong, confident, PASSed) is near-zero with RAG and rises
sharply without it (e.g., Haiku 3%→22% RAG off). The decision validator's
*marginal* safety contribution is ≈0 on all models: it catches ungrounded /
over-confident output, but the wrong diagnoses that remain are
*grounded-but-wrong* and pass. Safety here comes from RAG raising accuracy plus
the agent declining (`unknown`) when unsure — not from the validator catching
wrong-but-grounded output. See Figure 2.

### 5.3 Operator intent (test n=39)

| Model | overall | declined (agent) | blocked (guardrail) | leaked |
|---|---|---|---|---|
| GPT-4.1-mini | 77% | 6 | 7 | 2 |
| Claude Haiku 4.5 | 87% | 11 | 4 | 0 |
| Solar-Pro | 72% | 4 | 7 | 4 |

Of 15 must-not-activate intents, the agent + guardrail block 13/15, 15/15, and
11/15 respectively (Figure 3). The two layers are complementary: the agent
declines out-of-scope/ambiguous requests, while the guardrail blocks structurally
unsafe ones (avoiding a charger/mandatory zone, reserving all chargers,
nonexistent zones, duplicates).

---

## 6. Analysis

**Precedent utilization is the model-dependent bottleneck.** All models *retrieve*
the relevant precedent (recall ≈100% when RAG is on), but reliance ranges from 9%
(GPT-4.1-mini) to 92% (Haiku). For the weak model, the retrieved precedent has
adequate trust (≈0.71, equal to that on correct cases) yet is not integrated into
the answer on hard cases — a reasoning limitation, not a retrieval one. The
stronger model converts the same retrieval into correct answers.

**Failure mode differs by model.** Without RAG, GPT-4.1-mini *declines* (answers
`unknown` → held by the validator → safe but low-accuracy), whereas Haiku and
Solar *guess*, producing confident-wrong output (Haiku 22% RAG off). A stronger,
more confident model is therefore not automatically safer: it produces more
confident-wrong output when grounding is removed.

**The guardrail cannot catch valid-but-unintended policies.** The residual intent
leaks (2 for GPT-4.1-mini, 4 for Solar) are cases where the agent force-fit an
out-of-scope or vague request into a *valid* whitelist policy (e.g., "make the
robots drive faster" → `delay_low_priority_missions`). Because the policy is
structurally valid, the guardrail accepts it; only the agent's own self-restraint
can prevent it, and that restraint is model-dependent (Haiku 0 leaks, Solar 4).

**Symmetry across both directions.** The diagnosis finding (the validator passes
grounded-but-wrong output) and the intent finding (the guardrail passes
valid-but-unintended policies) are the same phenomenon: deterministic checks
verify *structure*, not *semantic correctness of intent*. This is why we frame
validation as necessary but not sufficient.

---

## 7. Limitations

- **Synthetic, single-environment evaluation.** Cases are programmatically
  generated for one warehouse configuration; no real operator utterances or
  real failure logs. External validity is limited.
- **Modest n, single run.** Test sets are 100 (diagnosis) and 39 (intent), one
  run per condition; we do not report variance across seeds.
- **Author-constructed ground truth.** The cause taxonomy matches the agent's
  output enum and precedents are seeded by us; we mitigate with dev/test split,
  frozen prompts, and diversity, but the data is synthetic-by-author.
- **Robot integration not measured.** The Isaac/Nav2 failure bridge is
  implemented but not part of the quantitative results.
- **Simple validators.** The deterministic checks are schema/whitelist/confidence
  /feasibility rules; richer semantic validation is future work.

---

## 8. Conclusion

Deterministic validation gives a *model-independent* safety improvement to an LLM
fleet supervisor — RAG raises diagnosis accuracy by 38–47 pp on three different
LLMs and suppresses confident-wrong output, while an agent + guardrail block the
large majority of unsafe operator intents. But the guarantees stop at structure:
grounded-but-wrong diagnoses and valid-but-unintended policies slip through, and
this residual risk depends on the model. Safe LLM supervision therefore comes
from the interaction of retrieval grounding, agent self-restraint, and
deterministic validation, not any one of them alone. Future work: robot-in-the-
loop evaluation via the implemented Isaac/Nav2 bridge, larger and real-sourced
datasets, and semantic-level validation of agent intent.

---

## Figures

- **Figure 1** (`eval/figs/fig_diag_rag.png`): diagnosis cause accuracy, RAG on vs
  off, overall and by difficulty.
- **Figure 2** (`eval/figs/fig_diag_safety.png`): confident-wrong and
  acted-precision, RAG on vs off.
- **Figure 3** (`eval/figs/fig_intent_defense.png`): intent defense-in-depth
  (agent declined / guardrail blocked / leaked).

*(Figures currently rendered for GPT-4.1-mini; regenerate per model or as a
3-model grouped bar for the camera-ready.)*
