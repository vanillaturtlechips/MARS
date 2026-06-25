"""Generate ~150 diverse diagnosis_cases (eval/diagnosis_cases.yaml) — 중강 P1.

Diversity (so it isn't "9 templates x repeat"):
  - per-cause incident TEXT pools (varied phrasings) — RAG must match meaning,
    not a fixed sentence.
  - randomized numerics (battery, time offsets, robot counts, zones) — fixed
    seed for reproducibility.
  - difficulty tag: easy (clear evidence) | medium (needs precedent) |
    hard (symptom != cause, distractors).
  - balanced cause/scope distribution.
  - dev/test split (~1/3 dev) stratified within each scenario block.

Labels use the agent's enums (failure_analysis _OUTPUT_SCHEMA):
  cause: transient_obstacle|robot_internal_fault|low_battery|localization_failure|
         zone_congestion|zone_blocked|fleet_overload|unknown
  scope: isolated|robot_specific|zone_wide|fleet_wide   persistence: transient|persistent

    python3 eval/gen_diagnosis_cases.py
"""
from __future__ import annotations
import random
from pathlib import Path
import yaml

RNG = random.Random(42)

ZONES = ["receiving_dock", "shipping_dock", "aisle_1", "aisle_2", "aisle_3",
         "aisle_5", "cold_zone", "staging", "pack_station", "returns", "qc_bay"]
ROBOTS = [f"R{i}" for i in range(1, 13)] + ["iw_hub", "amr_07", "tug_3"]

# incident text variants per cause — varied surface form, same root cause.
TEXTS = {
    "low_battery": [
        "E-stop latched only AFTER the pack dropped below the cutoff; the stop is a symptom of depletion.",
        "Robot halted mid-route as state-of-charge hit the reserve floor; not a hardware fault.",
        "Voltage sag under load triggered a protective stop; root cause was an under-charged battery.",
        "Mission aborted when battery fell under the dispatch minimum; charging resolved it.",
    ],
    "robot_internal_fault": [
        "Drive motor over-current tripped the controller; a failing bearing was replaced.",
        "Wheel encoder drift caused repeated heading errors until the encoder was swapped.",
        "Lidar returns corrupted by a degraded sensor produced phantom obstacles and aborts.",
        "IMU bias fault made the robot mis-track; resolved after a sensor recalibration.",
        "A stuck caster wheel raised motor current and forced a safety stop.",
    ],
    "localization_failure": [
        "AMCL covariance spiked near a featureless wall and the robot lost its pose.",
        "Map-to-odom transform diverged after a long straight corridor; relocalization failed.",
        "Particle filter collapsed in a sparse-feature aisle, aborting the goal.",
        "Pose jumped after passing reflective racking; localization never recovered.",
    ],
    "zone_blocked": [
        "A dropped pallet blocked the lane; several robots aborted until it was cleared.",
        "A forklift parked across the aisle entrance during shift change, stopping traffic.",
        "Spilled cargo on the floor forced every robot through that zone to abort.",
        "A maintenance cart left overnight obstructed the corridor.",
        "Stacked returns overflowed into the lane and blocked the path.",
    ],
    "zone_congestion": [
        "Several robots jammed at the intersection, mutually blocking until one rerouted.",
        "Peak-hour pile-up in the narrow aisle caused a standoff among AMRs.",
        "Two-way traffic in a single-lane segment deadlocked the robots.",
        "Convergent routes at the pack station created gridlock.",
    ],
    "fleet_wide": [
        "Fleet-wide localization degraded after a map-server restart; aborts across all zones.",
        "A bad global costmap update propagated to every robot, causing widespread aborts.",
        "Network time desync stalled navigation fleet-wide until clocks resynced.",
    ],
}
DISTRACTORS = [
    ("low_battery", "An unrelated battery cutoff happened mid-aisle last week."),
    ("robot_internal_fault", "A single robot gearbox fault, unrelated to this zone."),
    ("zone_congestion", "Minor congestion cleared on its own earlier, unrelated."),
]

cases: list[dict] = []
n = 0


def health(batt=70, estop=False, faults=None):
    return {"battery_pct": batt, "estop_active": estop, "fault_codes": faults or []}


def trig(robot, zone, batt=70, estop=False, faults=None, zspread=1, rspread=1):
    return {"event_type": "navigation.aborted", "robot_id": robot,
            "mission_id": f"M-{robot}-{RNG.randint(100,999)}", "goal_id": f"G-{RNG.randint(100,999)}",
            "zone": zone, "goal_status": 6, "nav_outcome": "aborted",
            "health_at_failure": health(batt, estop, faults),
            "distribution": {"per_robot_zone_spread": rspread, "per_zone_robot_spread": zspread}}


def fail(robot, zone, off, batt=70, estop=False, faults=None, flag=None):
    return {"robot_id": robot, "mission_id": f"M-{robot}-{RNG.randint(100,999)}", "zone": zone,
            "event_type": "navigation.aborted", "nav_outcome": "aborted", "goal_status": 6,
            "health_at_failure": health(batt, estop, faults), "fault_flag": flag,
            "ts_offset_sec": off}


def add(tags, seed, trigger, cause, scope, persistence, prec_ids, desc, exp_val="PASS"):
    global n
    n += 1
    cases.append({
        "case_id": f"DC-{n:03d}", "split": "dev" if n % 3 == 0 else "test",
        "description": desc, "tags": tags,
        "seed_state": {"failures": seed.get("failures", []),
                       "incidents": seed.get("incidents", []),
                       "active_policies": seed.get("active_policies", [])},
        "trigger_event": trigger,
        "ground_truth": {"cause": cause, "scope": scope, "persistence": persistence,
                         "relevant_precedent_ids": prec_ids, "expected_validation": exp_val},
    })


def pick_zone():
    return RNG.choice(ZONES)


def text_for(cause):
    return RNG.choice(TEXTS[cause])


def prec(cause, iid):
    return {"incident_id": iid, "true_cause": cause, "text": text_for(cause), "is_relevant": True}


def distractor(iid):
    c, t = RNG.choice(DISTRACTORS)
    return {"incident_id": iid, "true_cause": c, "text": t, "is_relevant": False}


# A. isolated / unknown — no evidence -> decline (DEGRADE). (18)
for i in range(18):
    add(["isolated", "easy", "thin_evidence"], {}, trig(RNG.choice(ROBOTS), pick_zone()),
        "unknown", "isolated", "transient", [],
        "single abort, no evidence -> unknown (DEGRADE)", exp_val="DEGRADE")

# B. isolated / low_battery — estop symptom (hard adversarial). (18)
for i in range(18):
    seed = {"incidents": [prec("low_battery", f"INC-B{i}")]}
    add(["isolated", "hard", "adversarial"], seed,
        trig(RNG.choice(ROBOTS), pick_zone(), batt=RNG.randint(5, 14), estop=True, faults=["ESTOP"]),
        "low_battery", "isolated", "persistent", [f"INC-B{i}"],
        "low battery; e-stop is a symptom (precedent needed)")

# C. isolated / robot_internal_fault — explicit fault code (easy). (15)
for i in range(15):
    fc = RNG.choice(["MOTOR_OVERCURRENT", "ENCODER_FAULT", "IMU_FAULT", "DRIVE_TEMP"])
    add(["isolated", "easy"], {}, trig(RNG.choice(ROBOTS), pick_zone(), batt=RNG.randint(60, 90), faults=[fc]),
        "robot_internal_fault", "isolated", "persistent", [],
        f"explicit fault code {fc} -> robot_internal_fault")

# D. isolated / robot_internal_fault — sensor degradation (hard, distractor). (15)
for i in range(15):
    seed = {"incidents": [prec("robot_internal_fault", f"INC-S{i}"), distractor(f"INC-SX{i}")]}
    add(["isolated", "hard", "adversarial"], seed,
        trig(RNG.choice(ROBOTS), pick_zone(), batt=RNG.randint(60, 88)),
        "robot_internal_fault", "isolated", "persistent", [f"INC-S{i}"],
        "sensor degradation -> robot_internal_fault (distractor present)")

# E. isolated / localization_failure (medium). (15)
for i in range(15):
    seed = {"incidents": [prec("localization_failure", f"INC-L{i}")]}
    add(["isolated", "medium"], seed, trig(RNG.choice(ROBOTS), pick_zone(), batt=RNG.randint(55, 85)),
        "localization_failure", "isolated", "persistent", [f"INC-L{i}"],
        "localization drift (precedent helps)")

# F. robot_specific — one robot across many zones (medium). (15)
for i in range(15):
    r = RNG.choice(ROBOTS)
    zs = RNG.sample(ZONES, 3)
    seed = {"failures": [fail(r, zs[0], RNG.randint(400, 700)), fail(r, zs[1], RNG.randint(150, 380)),
                         fail(r, zs[2], RNG.randint(60, 140))],
            "incidents": [prec("robot_internal_fault", f"INC-RS{i}")]}
    add(["robot_specific", "medium"], seed,
        trig(r, pick_zone(), batt=RNG.randint(60, 85), rspread=RNG.randint(3, 5), zspread=1),
        "robot_internal_fault", "robot_specific", "persistent", [f"INC-RS{i}"],
        "one robot fails across many zones -> robot_specific")

# G. zone_wide / zone_blocked — obstacle, multiple robots (medium). (18)
for i in range(18):
    z = pick_zone()
    k = RNG.randint(2, 4)
    seed = {"failures": [fail(RNG.choice(ROBOTS), z, RNG.randint(60, 500)) for _ in range(k)],
            "incidents": [prec("zone_blocked", f"INC-O{i}")]}
    add(["zone_wide", "medium"], seed, trig(RNG.choice(ROBOTS), z, zspread=k + 1),
        "zone_blocked", "zone_wide", "persistent", [f"INC-O{i}"],
        "multiple robots blocked in same zone -> zone_blocked")

# H. zone_wide / zone_blocked — recurring + distractor (hard). (12)
for i in range(12):
    z = pick_zone()
    seed = {"failures": [fail(RNG.choice(ROBOTS), z, o) for o in
                         RNG.sample(range(60, 700), 3)],
            "incidents": [prec("zone_blocked", f"INC-R{i}"), distractor(f"INC-RX{i}")]}
    add(["zone_wide", "hard", "novel_cause"], seed, trig(RNG.choice(ROBOTS), z, zspread=4),
        "zone_blocked", "zone_wide", "persistent", [f"INC-R{i}"],
        "recurring same-zone blockage (distractor present)")

# I. zone_wide / zone_congestion (medium). (12)
for i in range(12):
    z = RNG.choice(["aisle_1", "aisle_2", "aisle_3", "pack_station"])
    seed = {"failures": [fail(RNG.choice(ROBOTS), z, RNG.randint(40, 120)) for _ in range(RNG.randint(2, 3))],
            "incidents": [prec("zone_congestion", f"INC-C{i}")]}
    add(["zone_wide", "medium"], seed, trig(RNG.choice(ROBOTS), z, zspread=3),
        "zone_congestion", "zone_wide", "persistent", [f"INC-C{i}"],
        "robots jammed in narrow zone -> zone_congestion")

# J. fleet_wide — across many zones AND robots (hard). (12)
for i in range(12):
    zs = RNG.sample(ZONES, 4)
    seed = {"failures": [fail(RNG.choice(ROBOTS), z, RNG.randint(60, 350)) for z in zs],
            "incidents": [prec("fleet_wide", f"INC-F{i}")]}
    t = trig(RNG.choice(ROBOTS), RNG.choice(zs), zspread=2, rspread=4)
    add(["fleet_wide", "hard"], seed, t,
        "localization_failure", "fleet_wide", "persistent", [f"INC-F{i}"],
        "aborts span many zones and robots -> fleet_wide")


def main():
    out = Path(__file__).parent / "diagnosis_cases.yaml"
    out.write_text(yaml.safe_dump(cases, sort_keys=False, allow_unicode=True, width=100))
    from collections import Counter
    print(f"wrote {out} — {len(cases)} cases")
    print("split:", dict(Counter(c["split"] for c in cases)))
    print("scope:", dict(Counter(c["ground_truth"]["scope"] for c in cases)))
    print("cause:", dict(Counter(c["ground_truth"]["cause"] for c in cases)))
    print("difficulty:", dict(Counter(t for c in cases for t in c["tags"]
                                       if t in ("easy", "medium", "hard"))))


if __name__ == "__main__":
    main()
