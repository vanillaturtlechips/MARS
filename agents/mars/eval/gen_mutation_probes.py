"""Mutation-probe generator for P1/P2 — derived from docs/CONTRACT_C.md ONLY.

Builds contract-valid originals BY CONSTRUCTION (never by running a validator),
then applies one mutation operator per probe (or two for composites), each of
which falsifies exactly one contract predicate. Boundary non-violations are
emitted as `expect: accept` probes to sharpen P2.

Independence rule: this file imports enums / operating parameters from
mars.config and the agents' declared output schemas (those ARE the contract).
It must never import mars.validators.* or mars.guardrail.* — run_mutations.py
is the only place the validator is invoked.

    python3 -m eval.gen_mutation_probes            # writes eval/mutation_probes.json
    python3 -m eval.gen_mutation_probes --n 120    # per-operator count
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from mars.config import (
    DV_TAU_DIAGNOSIS,
    POLICY_COOLDOWN_SEC,
    POLICY_MAX_DURATION_SEC,
    POLICY_MIN_DURATION_SEC,
    POLICY_WHITELIST,
)

# ---- C1 enums (mars_agent_contracts.md §0 shared enums) --------------------
CAUSES = ["transient_obstacle", "robot_internal_fault", "low_battery",
          "localization_failure", "zone_congestion", "zone_blocked",
          "fleet_overload", "unknown"]
SCOPES = ["isolated", "robot_specific", "zone_wide", "fleet_wide"]
PERSISTENCE = ["transient", "persistent"]
TRUST_LEVELS = ["HIGH", "MEDIUM", "LOW"]
TRUST_CEIL_LOW = 0.7                                  # C5.2 ceil(LOW)
TAU = DV_TAU_DIAGNOSIS                                # C5.1

ZONES = ["receiving_dock", "aisle_1", "aisle_2", "aisle_3", "aisle_5",
         "cold_zone", "shipping_dock", "staging", "pack_station", "returns"]
FAULT_FLAGS = ["battery_critical", "estop", "diagnostics_error", None]


# ===========================================================================
# Diagnosis side: bundle B, valid original d_dx, path enumeration (C4.2 grammar)
# ===========================================================================

def make_bundle(rng: random.Random, zone: str, n_fail: int, n_prec: int) -> dict:
    """An input bundle with the keys the transcript builder emits (§1 input bundle)."""
    def mf(i):
        return {"robot_id": f"R{i+1}", "mission_id": f"M-{rng.randint(100, 999)}",
                "zone": zone, "goal_status": rng.choice([5, 6]),
                "nav_outcome": rng.choice(["aborted", "canceled"]),
                "fault_flag": rng.choice(FAULT_FLAGS),            # may be None (B-dx-4)
                "health_at_failure": {"battery_pct": round(rng.uniform(5, 95), 1),
                                      "estop_active": rng.random() < 0.1,
                                      "fault_codes": []}}
    return {
        "trigger_event": {"robot_id": "R1", "zone": zone, "goal_status": 6,
                          "event_type": "navigation.aborted",
                          "health_at_failure": {"battery_pct": round(rng.uniform(5, 95), 1),
                                                "estop_active": False, "fault_codes": []}},
        "mission_failures": [mf(i) for i in range(n_fail)],
        "zone_state": {"zone": zone, "recent_failures": n_fail,
                       "robots_present": rng.randint(0, 4), "last_cleared": None},
        "robot_history": [{"robot_id": "R1", "event_type": "navigation.aborted",
                           "zone": rng.choice(ZONES)} for _ in range(rng.randint(0, 3))],
        "retrieved_precedents": [{"incident_id": f"INC-{k+1}", "similarity": round(rng.uniform(0.5, 0.95), 2),
                                  "trust": round(rng.uniform(0.3, 0.9), 2)} for k in range(n_prec)],
        "active_policies": [{"type": "avoid_zone", "params": {"zone": rng.choice(ZONES)}}
                            for _ in range(rng.randint(0, 2))],
    }


def enumerate_paths(obj, prefix: str = "") -> list[str]:
    """All paths resolvable under the C4.2 grammar (key / key.sub / key[i] / key[i].sub).
    Grammar is taken from CONTRACT_C.md §1 C4.2, not from the validator."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.append(p)
            out.extend(enumerate_paths(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.append(p)
            out.extend(enumerate_paths(v, p))
    return out


def make_valid_diagnosis(rng: random.Random, bundle: dict, trust: str) -> dict:
    """Construct d_dx satisfying every predicate of C for (bundle, trust) by spec."""
    scope = rng.choice(SCOPES)
    n_fail = len(bundle["mission_failures"])
    if scope in ("zone_wide", "fleet_wide") and n_fail < 2:
        scope = rng.choice(["isolated", "robot_specific"])
    paths = enumerate_paths(bundle)
    mission_paths = [p for p in paths if p.startswith("mission_failures[")]
    other_paths = [p for p in paths if not p.startswith("mission_failures[")]
    refs: list[str] = []
    if scope in ("zone_wide", "fleet_wide"):
        # C4.3: >= 2 distinct mission_failures entries
        idx = rng.sample(range(n_fail), 2)
        refs += [f"mission_failures[{i}].{rng.choice(['zone', 'goal_status', 'robot_id'])}" for i in idx]
    refs += rng.sample(other_paths, k=min(len(other_paths), rng.randint(1, 3)))
    if mission_paths and rng.random() < 0.5:
        refs.append(rng.choice(mission_paths))
    relied: list[str] = []
    n_prec = len(bundle["retrieved_precedents"])
    if n_prec and rng.random() < 0.4:
        relied = [bundle["retrieved_precedents"][0]["incident_id"]]
    # C5.1 and C5.2 together
    hi = TRUST_CEIL_LOW if (relied and trust == "LOW") else 1.0
    conf = round(rng.uniform(TAU, hi), 3)
    # split refs across 1..2 evidence items (all non-empty)
    k = 1 if len(refs) < 2 else rng.randint(1, 2)
    chunks = [refs[i::k] for i in range(k)]
    evidence = [{"observation": f"obs-{j}", "refs": c} for j, c in enumerate(chunks) if c]
    return {"cause": rng.choice(CAUSES), "scope": scope,
            "persistence": rng.choice(PERSISTENCE), "affected_zone": bundle["trigger_event"]["zone"],
            "confidence": conf, "evidence": evidence, "relied_on_precedents": relied}


def _set_ref(d: dict, new: str, where=(0, 0)):
    i, j = where
    d["evidence"][i]["refs"][j] = new


# ---- diagnosis mutation operators (CONTRACT_C.md §4) ------------------------
# each: (d, bundle, trust, rng) -> (d', bundle', trust') ; falsifies exactly one predicate

def d_c1_1(d, b, t, rng):
    d["cause"] = rng.choice(["obstacle", "TRANSIENT_OBSTACLE", "battery", "", "zone_blocked "])
    return d, b, t

def d_c1_2(d, b, t, rng):
    if rng.random() < 0.5:
        d["scope"] = rng.choice(["global", "zone", "ZONE_WIDE", ""])
    else:
        d["persistence"] = rng.choice(["recurring", "permanent", "Transient"])
    return d, b, t

def d_c1_3(d, b, t, rng):
    d["confidence"] = rng.choice([1.001, 1.5, -0.01, -1.0, "0.8", None])
    return d, b, t

def d_c1_4(d, b, t, rng):
    c = rng.choice(["drop_cause", "drop_scope", "drop_confidence", "drop_evidence", "drop_refs", "refs_str"])
    if c.startswith("drop_") and c != "drop_refs":
        d.pop(c[5:], None)
    elif c == "drop_refs":
        d["evidence"][0].pop("refs")
    else:
        d["evidence"][0]["refs"] = "mission_failures[0].zone"
    return d, b, t

def d_c4_1(d, b, t, rng):
    d["evidence"] = []
    return d, b, t

def d_c4_2a(d, b, t, rng):
    _set_ref(d, rng.choice(["fleet_metrics.total", "sensor_log.lidar", "missions[0].id", "world.zone"]))
    return d, b, t

def d_c4_2b(d, b, t, rng):
    n = len(b["mission_failures"])
    _set_ref(d, f"mission_failures[{n + rng.randint(0, 5)}].zone")
    return d, b, t

def d_c4_2c(d, b, t, rng):
    _set_ref(d, rng.choice(["trigger_event.mission_id", "zone_state.nonexistent",
                            "mission_failures[0].battery_pct", "trigger_event.health_at_failure.temp"]))
    return d, b, t

def d_c4_2d(d, b, t, rng):
    _set_ref(d, rng.choice(["mission_failures[].zone", "mission_failures[x].zone", "mission_failures[-1].zone"]))
    return d, b, t

def d_c4_2e(d, b, t, rng):
    for item in d["evidence"]:
        item["refs"] = [f"ghost_{k}.field" for k in range(len(item["refs"]))]
    return d, b, t

def d_c4_3(d, b, t, rng):
    d["scope"] = rng.choice(["zone_wide", "fleet_wide"])
    keep = rng.randint(0, 1)                      # <= 1 mission_failures ref survives
    seen = 0
    for item in d["evidence"]:
        new = []
        for r in item["refs"]:
            if r.startswith("mission_failures["):
                if seen < keep:
                    new.append(r); seen += 1
                else:
                    new.append("zone_state.recent_failures")
            else:
                new.append(r)
        item["refs"] = new
    return d, b, t

def d_c5_1(d, b, t, rng):
    delta = rng.choice([0.001, 0.01, 0.05, 0.1, 0.3, TAU])
    d["confidence"] = round(max(0.0, TAU - delta), 3)
    return d, b, t

def d_c5_2(d, b, t, rng):
    if not b["retrieved_precedents"]:
        b["retrieved_precedents"].append({"incident_id": "INC-1", "similarity": 0.8, "trust": 0.3})
    d["relied_on_precedents"] = [b["retrieved_precedents"][0]["incident_id"]]
    delta = rng.choice([0.001, 0.01, 0.05, 0.1, 0.29])
    d["confidence"] = round(min(1.0, TRUST_CEIL_LOW + delta), 3)
    return d, b, "LOW"

DX_OPS = {
    "D-C1.1": (d_c1_1, "C1.1"), "D-C1.2": (d_c1_2, "C1.2"), "D-C1.3": (d_c1_3, "C1.3"),
    "D-C1.4": (d_c1_4, "C1.4"), "D-C4.1": (d_c4_1, "C4.1"),
    "D-C4.2a": (d_c4_2a, "C4.2"), "D-C4.2b": (d_c4_2b, "C4.2"), "D-C4.2c": (d_c4_2c, "C4.2"),
    "D-C4.2d": (d_c4_2d, "C4.2"), "D-C4.2e": (d_c4_2e, "C4.2"),
    "D-C4.3": (d_c4_3, "C4.3"), "D-C5.1": (d_c5_1, "C5.1"), "D-C5.2": (d_c5_2, "C5.2"),
}
# composites: pairs whose effects do not cancel (schema ops excluded: they mask everything)
DX_COMPOSITES = [("D-C4.2a", "D-C5.1"), ("D-C4.1", "D-C5.1"), ("D-C4.3", "D-C5.2"),
                 ("D-C4.2b", "D-C4.3"), ("D-C5.1", "D-C5.2")]


# ---- diagnosis boundary non-violations (must accept) -------------------------

def b_dx_1(d, b, t, rng):
    d["confidence"] = TAU; return d, b, t

def b_dx_2(d, b, t, rng):
    if not b["retrieved_precedents"]:
        b["retrieved_precedents"].append({"incident_id": "INC-1", "similarity": 0.8, "trust": 0.3})
    d["relied_on_precedents"] = [b["retrieved_precedents"][0]["incident_id"]]
    d["confidence"] = TRUST_CEIL_LOW
    return d, b, "LOW"

def b_dx_3(d, b, t, rng):
    while len(b["mission_failures"]) < 2:
        b["mission_failures"].append(copy.deepcopy(b["mission_failures"][0]))
    b["zone_state"]["recent_failures"] = len(b["mission_failures"])
    d["scope"] = "zone_wide"
    d["evidence"] = [{"observation": "two robots", "refs": ["mission_failures[0].zone", "mission_failures[1].zone"]}]
    return d, b, t

def b_dx_4(d, b, t, rng):
    b["mission_failures"][0]["fault_flag"] = None
    _set_ref(d, "mission_failures[0].fault_flag")
    return d, b, t

def b_dx_5(d, b, t, rng):
    n = len(b["mission_failures"])
    d["evidence"][0]["refs"].append(f"mission_failures[{n-1}].robot_id")   # append: keep C4.3 intact
    return d, b, t

def b_dx_6(d, b, t, rng):
    if not b["retrieved_precedents"]:
        b["retrieved_precedents"].append({"incident_id": "INC-1", "similarity": 0.9, "trust": 0.9})
    d["relied_on_precedents"] = [b["retrieved_precedents"][0]["incident_id"]]
    d["confidence"] = 0.95
    return d, b, rng.choice(["MEDIUM", "HIGH"])

DX_BOUNDARY = {"B-dx-1": b_dx_1, "B-dx-2": b_dx_2, "B-dx-3": b_dx_3,
               "B-dx-4": b_dx_4, "B-dx-5": b_dx_5, "B-dx-6": b_dx_6}


# ===========================================================================
# Policy side: world W, valid original d_pol
# ===========================================================================

def make_world(rng: random.Random) -> dict:
    zones = {z: {} for z in rng.sample(ZONES, rng.randint(5, len(ZONES)))}
    chargers = rng.sample(["charge_bay", "charge_bay_2"], rng.choice([1, 1, 2]))
    for c in chargers:
        zones[c] = {"is_charger_zone": True}
    zones["main_corridor"] = {"is_mandatory": True}
    return {"zones": zones, "charger_zones": chargers,
            "total_chargers": rng.randint(2, 4),
            "_open": [z for z, meta in zones.items() if not meta]}      # helper, stripped later


def make_valid_policy(rng: random.Random, world: dict) -> tuple[dict, list, dict]:
    ptype = rng.choice(POLICY_WHITELIST)
    params: dict = {}
    if ptype == "avoid_zone":
        params = {"zone": rng.choice(world["_open"])}
    elif ptype == "reserve_chargers_for_critical":
        params = {"reserve_count": rng.randint(1, world["total_chargers"] - 1)}
    pol = {"type": ptype, "params": params,
           "duration_sec": rng.randint(POLICY_MIN_DURATION_SEC, POLICY_MAX_DURATION_SEC),
           "rationale": "operator request"}
    # active policies that do not collide (C3.4)
    active = []
    if rng.random() < 0.5:
        other = rng.choice([z for z in world["_open"] if z != params.get("zone")])
        active.append({"policy_id": "P-old", "type": "avoid_zone", "params": {"zone": other}})
    # last_applied is stored as SECONDS AGO (relative); run_mutations.py converts
    # to absolute time at run time so cooldown probes don't age out between
    # generation and execution.
    last_applied = {ptype: POLICY_COOLDOWN_SEC + rng.randint(1, 3600)}  # C5.4 satisfied
    return pol, active, last_applied


# ---- policy mutation operators ---------------------------------------------
# each: (pol, active, world, last_applied, rng) -> same tuple

def p_c1_5(p, a, w, l, rng):
    p["type"] = rng.choice(["avoid_zones", "fleet_wide_throttle", "AVOID_ZONE", "", "prefer_alternate_route"])
    return p, a, w, l

def p_c1_6a(p, a, w, l, rng):
    c = rng.choice(["drop", "zero", "neg"])
    if c == "drop": p.pop("duration_sec")
    elif c == "zero": p["duration_sec"] = 0
    else: p["duration_sec"] = -rng.randint(1, 500)
    return p, a, w, l

def p_c1_6b(p, a, w, l, rng):
    p.pop("params")
    return p, a, w, l

def p_c1_7(p, a, w, l, rng):
    p["type"] = rng.choice(["avoid_zone", "reserve_chargers_for_critical"])
    p["params"] = {}
    return p, a, w, l

def p_c2_1(p, a, w, l, rng):
    p["type"] = "avoid_zone"
    real = rng.choice(w["_open"])
    p["params"] = {"zone": rng.choice([real.upper(), real + " ", real[:-1], "aisle_9", "warehouse"])}
    return p, a, w, l

def p_c3_1(p, a, w, l, rng):
    # make a single charger zone, then avoid it
    only = w["charger_zones"][0]
    for c in w["charger_zones"][1:]:
        w["zones"].pop(c)
    w["charger_zones"] = [only]
    p["type"] = "avoid_zone"; p["params"] = {"zone": only}
    return p, a, w, l

def p_c3_2(p, a, w, l, rng):
    p["type"] = "avoid_zone"; p["params"] = {"zone": "main_corridor"}
    return p, a, w, l

def p_c3_3(p, a, w, l, rng):
    p["type"] = "reserve_chargers_for_critical"
    p["params"] = {"reserve_count": w["total_chargers"] + rng.randint(0, 3)}
    return p, a, w, l

def p_c3_4(p, a, w, l, rng):
    a.append({"policy_id": "P-dup", "type": p["type"], "params": copy.deepcopy(p["params"])})
    return p, a, w, l

def p_c5_3(p, a, w, l, rng):
    if rng.random() < 0.5:
        p["duration_sec"] = POLICY_MIN_DURATION_SEC - rng.choice([1, 5, 30, POLICY_MIN_DURATION_SEC - 1])
    else:
        p["duration_sec"] = POLICY_MAX_DURATION_SEC + rng.choice([1, 60, 3600, 86400])
    return p, a, w, l

def p_c5_4(p, a, w, l, rng):
    l[p["type"]] = POLICY_COOLDOWN_SEC - rng.choice([1, 10, 60, POLICY_COOLDOWN_SEC - 1])   # seconds ago
    return p, a, w, l

POL_OPS = {
    "P-C1.5": (p_c1_5, "C1.5"), "P-C1.6a": (p_c1_6a, "C1.6"), "P-C1.6b": (p_c1_6b, "C1.6"),
    "P-C1.7": (p_c1_7, "C1.7"), "P-C2.1": (p_c2_1, "C2.1"),
    "P-C3.1": (p_c3_1, "C3.1"), "P-C3.2": (p_c3_2, "C3.2"), "P-C3.3": (p_c3_3, "C3.3"),
    "P-C3.4": (p_c3_4, "C3.4"), "P-C5.3": (p_c5_3, "C5.3"), "P-C5.4": (p_c5_4, "C5.4"),
}
POL_COMPOSITES = [("P-C2.1", "P-C5.3"), ("P-C3.2", "P-C5.4"), ("P-C3.3", "P-C5.3"), ("P-C3.4", "P-C5.4")]


# ---- policy boundary non-violations (must accept) --------------------------

def b_pol_1(p, a, w, l, rng):
    p["duration_sec"] = rng.choice([POLICY_MIN_DURATION_SEC, POLICY_MAX_DURATION_SEC]); return p, a, w, l

def b_pol_2(p, a, w, l, rng):
    p["type"] = "reserve_chargers_for_critical"
    p["params"] = {"reserve_count": w["total_chargers"] - 1}; return p, a, w, l

def b_pol_3(p, a, w, l, rng):
    if len(w["charger_zones"]) < 2:
        extra = "charge_bay" if "charge_bay" not in w["charger_zones"] else "charge_bay_2"
        w["zones"][extra] = {"is_charger_zone": True}; w["charger_zones"].append(extra)
    p["type"] = "avoid_zone"; p["params"] = {"zone": w["charger_zones"][0]}; return p, a, w, l

def b_pol_4(p, a, w, l, rng):
    l[p["type"]] = POLICY_COOLDOWN_SEC; return p, a, w, l   # exactly at cooldown (seconds ago)

def b_pol_5(p, a, w, l, rng):
    p["type"] = "avoid_zone"; p["params"] = {"zone": w["_open"][0]}
    a.append({"policy_id": "P-other", "type": "avoid_zone", "params": {"zone": w["_open"][1]}}); return p, a, w, l

POL_BOUNDARY = {"B-pol-1": b_pol_1, "B-pol-2": b_pol_2, "B-pol-3": b_pol_3,
                "B-pol-4": b_pol_4, "B-pol-5": b_pol_5}


# ===========================================================================
# Driver
# ===========================================================================

def gen(n_per_op: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    probes: list[dict] = []
    pid = 0

    def emit(kind, op, predicate, expect, payload, origin):
        nonlocal pid
        pid += 1
        probes.append({"probe_id": f"MP-{pid:05d}", "kind": kind, "op": op,
                       "predicate": predicate, "expect": expect, **payload, "origin": origin})

    # ---- diagnosis
    def fresh_dx():
        zone = rng.choice(ZONES)
        b = make_bundle(rng, zone, n_fail=rng.randint(1, 5), n_prec=rng.randint(0, 3))
        t = rng.choice(TRUST_LEVELS)
        d = make_valid_diagnosis(rng, b, t)
        return d, b, t

    n_orig = 0
    for op, (fn, pred) in DX_OPS.items():
        for _ in range(n_per_op):
            d, b, t = fresh_dx()
            origin = {"diagnosis": copy.deepcopy(d), "bundle": copy.deepcopy(b), "trust": t}
            d2, b2, t2 = fn(copy.deepcopy(d), copy.deepcopy(b), t, rng)
            emit("dx", op, pred, "non_accept", {"diagnosis": d2, "bundle": b2, "trust": t2}, origin)
            n_orig += 1
    for op1, op2 in DX_COMPOSITES:
        for _ in range(n_per_op // 2):
            d, b, t = fresh_dx()
            origin = {"diagnosis": copy.deepcopy(d), "bundle": copy.deepcopy(b), "trust": t}
            d2, b2, t2 = DX_OPS[op1][0](copy.deepcopy(d), copy.deepcopy(b), t, rng)
            d2, b2, t2 = DX_OPS[op2][0](d2, b2, t2, rng)
            emit("dx", f"{op1}+{op2}", f"{DX_OPS[op1][1]}+{DX_OPS[op2][1]}", "non_accept",
                 {"diagnosis": d2, "bundle": b2, "trust": t2}, origin)
    for op, fn in DX_BOUNDARY.items():
        for _ in range(n_per_op // 2):
            d, b, t = fresh_dx()
            origin = {"diagnosis": copy.deepcopy(d), "bundle": copy.deepcopy(b), "trust": t}
            d2, b2, t2 = fn(copy.deepcopy(d), copy.deepcopy(b), t, rng)
            emit("dx", op, "boundary", "accept", {"diagnosis": d2, "bundle": b2, "trust": t2}, origin)

    # ---- policy
    def fresh_pol():
        w = make_world(rng)
        p, a, l = make_valid_policy(rng, w)
        return p, a, w, l

    def strip(w):
        w = copy.deepcopy(w); w.pop("_open", None); return w

    for op, (fn, pred) in POL_OPS.items():
        for _ in range(n_per_op):
            p, a, w, l = fresh_pol()
            origin = {"policy": copy.deepcopy(p), "active": copy.deepcopy(a), "world": strip(w), "last_applied": dict(l)}
            p2, a2, w2, l2 = fn(copy.deepcopy(p), copy.deepcopy(a), copy.deepcopy(w), dict(l), rng)
            emit("pol", op, pred, "non_accept", {"policy": p2, "active": a2, "world": strip(w2), "last_applied": l2}, origin)
    for op1, op2 in POL_COMPOSITES:
        for _ in range(n_per_op // 2):
            p, a, w, l = fresh_pol()
            origin = {"policy": copy.deepcopy(p), "active": copy.deepcopy(a), "world": strip(w), "last_applied": dict(l)}
            p2, a2, w2, l2 = POL_OPS[op1][0](copy.deepcopy(p), copy.deepcopy(a), copy.deepcopy(w), dict(l), rng)
            p2, a2, w2, l2 = POL_OPS[op2][0](p2, a2, w2, l2, rng)
            emit("pol", f"{op1}+{op2}", f"{POL_OPS[op1][1]}+{POL_OPS[op2][1]}", "non_accept",
                 {"policy": p2, "active": a2, "world": strip(w2), "last_applied": l2}, origin)
    for op, fn in POL_BOUNDARY.items():
        for _ in range(n_per_op // 2):
            p, a, w, l = fresh_pol()
            origin = {"policy": copy.deepcopy(p), "active": copy.deepcopy(a), "world": strip(w), "last_applied": dict(l)}
            p2, a2, w2, l2 = fn(copy.deepcopy(p), copy.deepcopy(a), copy.deepcopy(w), dict(l), rng)
            emit("pol", op, "boundary", "accept", {"policy": p2, "active": a2, "world": strip(w2), "last_applied": l2}, origin)
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="probes per single operator (composites/boundary: n/2)")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=str(Path(__file__).parent / "mutation_probes.json"))
    a = ap.parse_args()
    probes = gen(a.n, a.seed)
    Path(a.out).write_text(json.dumps(probes, indent=1, default=str))
    from collections import Counter
    c = Counter((p["kind"], p["expect"]) for p in probes)
    print(f"wrote {a.out}: {len(probes)} probes  {dict(c)}")
    print(f"  operators: dx {len(DX_OPS)} single + {len(DX_COMPOSITES)} composite + {len(DX_BOUNDARY)} boundary; "
          f"pol {len(POL_OPS)} single + {len(POL_COMPOSITES)} composite + {len(POL_BOUNDARY)} boundary")


if __name__ == "__main__":
    main()
