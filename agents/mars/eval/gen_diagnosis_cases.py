"""Generate 60 diagnosis_cases (eval/diagnosis_cases.yaml) — 중강 Part 1.

End-to-end cases: seed_state (failures + incident texts + policies) + trigger →
FailureAnalysisAgent.analyze() → compare cause/scope to ground_truth.

Labels use the agent's ACTUAL output enums (failure_analysis _OUTPUT_SCHEMA):
  cause: transient_obstacle | robot_internal_fault | low_battery |
         localization_failure | zone_congestion | zone_blocked | fleet_overload | unknown
  scope: isolated | robot_specific | zone_wide | fleet_wide
  persistence: transient | persistent

dev/test split (stratified, every 3rd case -> dev) so prompt tuning uses dev only
and the headline numbers are reported on test (no overfit to the eval set).

    python3 eval/gen_diagnosis_cases.py     # writes eval/diagnosis_cases.yaml
"""
from __future__ import annotations
from pathlib import Path
import yaml

ZONES = ["receiving_dock", "aisle_1", "aisle_3", "aisle_5", "cold_zone",
         "shipping_dock", "staging", "pack_station"]


def health(batt=70, estop=False, faults=None):
    return {"battery_pct": batt, "estop_active": estop, "fault_codes": faults or []}


def trig(robot, zone, batt=70, estop=False, faults=None, zspread=1, rspread=1):
    return {"event_type": "navigation.aborted", "robot_id": robot,
            "mission_id": f"M-{robot}", "goal_id": f"G-{robot}", "zone": zone,
            "goal_status": 6, "nav_outcome": "aborted",
            "health_at_failure": health(batt, estop, faults),
            "distribution": {"per_robot_zone_spread": rspread,
                             "per_zone_robot_spread": zspread}}


def fail(robot, zone, off, batt=70, estop=False, faults=None, flag=None):
    return {"robot_id": robot, "mission_id": f"M-{robot}", "zone": zone,
            "event_type": "navigation.aborted", "nav_outcome": "aborted",
            "goal_status": 6, "health_at_failure": health(batt, estop, faults),
            "fault_flag": flag, "ts_offset_sec": off}


def inc(iid, cause, text, relevant):
    return {"incident_id": iid, "true_cause": cause, "text": text, "is_relevant": relevant}


cases: list[dict] = []
n = 0


def add(tags, seed, trigger, cause, scope, persistence, prec_ids, desc, exp_val="PASS"):
    global n
    n += 1
    split = "dev" if n % 3 == 0 else "test"   # ~1/3 dev, stratified within blocks
    cases.append({
        "case_id": f"DC-{n:03d}", "split": split, "description": desc, "tags": tags,
        "seed_state": {"failures": seed.get("failures", []),
                       "incidents": seed.get("incidents", []),
                       "active_policies": seed.get("active_policies", [])},
        "trigger_event": trigger,
        "ground_truth": {"cause": cause, "scope": scope, "persistence": persistence,
                         "relevant_precedent_ids": prec_ids,
                         "expected_validation": exp_val},
    })


def Z(i):
    return ZONES[i % len(ZONES)]


# A. unknown/isolated — no evidence -> decline (DEGRADE). (6)
for i in range(6):
    add(["normal", "thin_evidence"], {}, trig(f"R{i+1}", Z(i)),
        "unknown", "isolated", "transient", [],
        "single abort, no evidence -> unknown/isolated (DEGRADE)", exp_val="DEGRADE")

# B. low_battery isolated — adversarial: estop symptom, battery is cause. (8)
for i in range(8):
    seed = {"incidents": [inc(f"INC-B{i}", "low_battery",
            "E-stop latched AFTER battery fell below cutoff; root cause is depletion, the e-stop is a symptom.", True)]}
    add(["adversarial"], seed, trig(f"R{i+1}", Z(i), batt=9 + i % 3, estop=True, faults=["ESTOP"]),
        "low_battery", "isolated", "persistent", [f"INC-B{i}"],
        "low battery (estop is a symptom) — needs evidence/precedent")

# C. robot_internal_fault isolated — explicit motor/hardware fault code. (6)
for i in range(6):
    add(["normal"], {}, trig(f"R{i+1}", Z(i+1), batt=80, faults=["MOTOR_OVERCURRENT"]),
        "robot_internal_fault", "isolated", "persistent", [],
        "motor fault code -> robot_internal_fault isolated")

# D. localization_failure isolated. (6)
for i in range(6):
    seed = {"incidents": [inc(f"INC-L{i}", "localization_failure",
            "AMCL pose covariance spiked near a featureless wall; robot lost localization and aborted.", True)]}
    add(["normal"], seed, trig(f"R{i+1}", Z(i+2), batt=75),
        "localization_failure", "isolated", "persistent", [f"INC-L{i}"],
        "localization drift near featureless area")

# E. robot_internal_fault isolated — sensor degradation (adversarial + distractor). (6)
for i in range(6):
    seed = {"incidents": [
        inc(f"INC-S{i}", "robot_internal_fault",
            "Lidar returns corrupted by reflective shrink-wrap; phantom obstacles from a degraded sensor caused aborts.", True),
        inc(f"INC-SX{i}", "low_battery", "Unrelated battery cutoff mid-aisle.", False)]}
    add(["adversarial"], seed, trig(f"R{i+1}", Z(i), batt=78),
        "robot_internal_fault", "isolated", "persistent", [f"INC-S{i}"],
        "sensor degradation -> robot_internal_fault (distractor present)")

# F. robot_specific — ONE robot failing across MANY zones (per spec). (6)
for i in range(6):
    r = f"R{i+1}"
    seed = {"failures": [fail(r, "aisle_1", 500), fail(r, "aisle_5", 300), fail(r, "dock", 120)],
            "incidents": [inc(f"INC-RS{i}", "robot_internal_fault",
                "One robot kept aborting in different zones — a drifting wheel encoder, not the environment.", True)]}
    t = trig(r, Z(i), batt=72, rspread=4, zspread=1)
    add(["robot_specific"], seed, t,
        "robot_internal_fault", "robot_specific", "persistent", [f"INC-RS{i}"],
        "one robot fails across many zones -> robot_specific")

# G. zone_blocked zone_wide — dropped pallet, 2 priors same zone. (8)
for i in range(8):
    z = Z(i)
    seed = {"failures": [fail(f"R{i*2+1}", z, 400), fail(f"R{i*2+2}", z, 180)],
            "incidents": [inc(f"INC-O{i}", "zone_blocked",
                "A dropped pallet blocked the lane; several robots aborted until it was removed.", True)]}
    add(["zone_wide"], seed, trig(f"RX{i}", z, zspread=3),
        "zone_blocked", "zone_wide", "persistent", [f"INC-O{i}"],
        "multiple robots blocked same zone -> zone_blocked zone_wide")

# H. zone_blocked zone_wide — recurring blockage, 3 priors + distractor. (6)
for i in range(6):
    z = Z(i)
    seed = {"failures": [fail(f"A{i}", z, 600), fail(f"B{i}", z, 300), fail(f"C{i}", z, 120)],
            "incidents": [
                inc(f"INC-R{i}", "zone_blocked",
                    "Pallet repeatedly left at the zone entrance each afternoon; many AMRs aborted until cleared.", True),
                inc(f"INC-RX{i}", "robot_internal_fault", "A single robot gearbox fault, unrelated.", False)]}
    add(["zone_wide", "novel_cause"], seed, trig(f"R{i+1}", z, zspread=4),
        "zone_blocked", "zone_wide", "persistent", [f"INC-R{i}"],
        "repeated same-zone failures -> zone_blocked (distractor present)")

# I. zone_congestion zone_wide — robots jammed at intersection. (4)
for i in range(4):
    z = "aisle_3"
    seed = {"failures": [fail(f"D{i}1", z, 90), fail(f"D{i}2", z, 60)],
            "incidents": [inc(f"INC-C{i}", "zone_congestion",
                "Several robots jammed at the aisle intersection; mutual blocking until one rerouted.", True)]}
    add(["zone_wide"], seed, trig(f"R{i+1}", z, zspread=3),
        "zone_congestion", "zone_wide", "persistent", [f"INC-C{i}"],
        "robots jammed same narrow zone -> zone_congestion")

# J. fleet_wide — aborts across many zones (distribution signal). (4)
for i in range(4):
    seed = {"failures": [fail("F1", "aisle_1", 300), fail("F2", "aisle_5", 250),
                         fail("F3", "shipping_dock", 150), fail("F4", "cold_zone", 90)],
            "incidents": [inc(f"INC-F{i}", "localization_failure",
                "Fleet-wide localization degraded after a map-server restart; aborts across all zones.", True)]}
    t = trig(f"R{i+1}", "staging", zspread=1, rspread=4)
    add(["fleet_wide"], seed, t,
        "localization_failure", "fleet_wide", "persistent", [f"INC-F{i}"],
        "aborts across many zones -> fleet_wide")


def main():
    out = Path(__file__).parent / "diagnosis_cases.yaml"
    out.write_text(yaml.safe_dump(cases, sort_keys=False, allow_unicode=True, width=100))
    from collections import Counter
    print(f"wrote {out} — {len(cases)} cases")
    print("split:", dict(Counter(c["split"] for c in cases)))
    print("scope:", dict(Counter(c["ground_truth"]["scope"] for c in cases)))
    print("cause:", dict(Counter(c["ground_truth"]["cause"] for c in cases)))


if __name__ == "__main__":
    main()
