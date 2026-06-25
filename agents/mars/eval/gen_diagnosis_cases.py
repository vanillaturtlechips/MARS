"""Generate 30 diagnosis_cases (eval/diagnosis_cases.yaml) — 중강 Part 1.

End-to-end cases: seed_state (failures + incident texts + policies) + trigger →
FailureAnalysisAgent.analyze() → compare cause/scope to ground_truth. Covers the
cause taxonomy, scopes, adversarial (symptom != cause), and RAG-relevant vs
distractor precedents.

    python3 eval/gen_diagnosis_cases.py     # writes eval/diagnosis_cases.yaml
"""
from __future__ import annotations
from pathlib import Path
import yaml

ZONES = ["receiving_dock", "aisle_1", "aisle_3", "aisle_5", "cold_zone",
         "shipping_dock", "staging", "pack_station"]


def health(batt=70, estop=False, faults=None):
    return {"battery_pct": batt, "estop_active": estop, "fault_codes": faults or []}


def trig(robot, zone, batt=70, estop=False, faults=None, spread=1):
    return {"event_type": "navigation.aborted", "robot_id": robot,
            "mission_id": f"M-{robot}", "goal_id": f"G-{robot}", "zone": zone,
            "goal_status": 6, "nav_outcome": "aborted",
            "health_at_failure": health(batt, estop, faults),
            "distribution": {"per_robot_zone_spread": 1, "per_zone_robot_spread": spread}}


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
    cases.append({
        "case_id": f"DC-{n:03d}", "description": desc, "tags": tags,
        "seed_state": {"failures": seed.get("failures", []),
                       "incidents": seed.get("incidents", []),
                       "active_policies": seed.get("active_policies", [])},
        "trigger_event": trigger,
        "ground_truth": {"cause": cause, "scope": scope, "persistence": persistence,
                         "relevant_precedent_ids": prec_ids,
                         "expected_validation": exp_val},
    })


# 1-3 isolated transient (no history, no precedent)
for i in range(3):
    z = ZONES[i]
    add(["normal", "thin_evidence"], {}, trig(f"R{i+1}", z),
        "transient_obstacle", "isolated", "transient", [],
        "single abort, no history/precedent -> isolated transient")

# 4-7 battery_depletion isolated (adversarial: estop symptom, real cause battery)
for i in range(4):
    z = ZONES[i]
    seed = {"incidents": [inc(f"INC-B{i}", "low_battery",
            "estop latched after battery dropped below cutoff; root cause depletion not e-stop hardware.", True)]}
    add(["adversarial"], seed, trig(f"R{i+1}", z, batt=10, estop=True, faults=["ESTOP"]),
        "low_battery", "isolated", "persistent", [f"INC-B{i}"],
        "estop symptom but battery is the cause (needs precedent/evidence)")

# 8-10 hardware_fault isolated
for i in range(3):
    z = ZONES[i+1]
    add(["normal"], {}, trig(f"R{i+1}", z, batt=80, faults=["MOTOR_OVERCURRENT"]),
        "robot_internal_fault", "isolated", "persistent", [],
        "motor fault code -> hardware_fault isolated")

# 11-13 localization_loss isolated
for i in range(3):
    z = ZONES[i+2]
    seed = {"incidents": [inc(f"INC-L{i}", "localization_failure",
            "AMCL pose covariance spiked near featureless wall; robot lost localization and aborted.", True)]}
    add(["normal"], seed, trig(f"R{i+1}", z, batt=75),
        "localization_failure", "isolated", "persistent", [f"INC-L{i}"],
        "localization drift near featureless area")

# 14-16 sensor_degradation isolated (reflective surface precedent + distractor)
for i in range(3):
    z = ZONES[i]
    seed = {"incidents": [
        inc(f"INC-S{i}", "robot_internal_fault",
            "Lidar returns corrupted by reflective shrink-wrap; phantom obstacles caused aborts.", True),
        inc(f"INC-SX{i}", "low_battery", "Unrelated battery cutoff mid-aisle.", False)]}
    add(["adversarial"], seed, trig(f"R{i+1}", z, batt=78),
        "robot_internal_fault", "isolated", "persistent", [f"INC-S{i}"],
        "reflective surface -> sensor_degradation (distractor present)")

# 17-20 static_obstacle zone (2 recent failures same zone + precedent)
for i in range(4):
    z = ZONES[i]
    seed = {"failures": [fail(f"R{i*2+1}", z, 400), fail(f"R{i*2+2}", z, 180)],
            "incidents": [inc(f"INC-O{i}", "zone_blocked",
                "Dropped pallet blocked the lane; robots aborted until removed.", True)]}
    add(["zone_wide"], seed, trig(f"RX{i}", z, spread=3),
        "zone_blocked", "zone_wide", "persistent", [f"INC-O{i}"],
        "two prior + this abort same zone -> static_obstacle zone_wide")

# 21-25 recurring_blockage zone_wide (3 prior + relevant precedent + distractor)
for i in range(5):
    z = ZONES[i % len(ZONES)]
    seed = {"failures": [fail(f"A{i}", z, 600), fail(f"B{i}", z, 300), fail(f"C{i}", z, 120)],
            "incidents": [
                inc(f"INC-R{i}", "zone_blocked",
                    "Pallet repeatedly left at zone entrance during afternoon shift; multiple AMRs aborted until cleared.", True),
                inc(f"INC-RX{i}", "robot_internal_fault", "Single robot gearbox fault, unrelated.", False)]}
    add(["zone_wide", "novel_cause"], seed, trig(f"R{i+1}", z, spread=4),
        "zone_blocked", "zone_wide", "persistent", [f"INC-R{i}"],
        "repeated same-zone failures -> recurring_blockage (distractor present)")

# 26-28 congestion_deadlock zone_wide (multiple robots, narrow zone)
for i in range(3):
    z = "aisle_3"
    seed = {"failures": [fail(f"D{i}1", z, 90), fail(f"D{i}2", z, 60)],
            "incidents": [inc(f"INC-C{i}", "zone_congestion",
                "Three robots jammed at the aisle intersection; mutual blocking until one was rerouted.", True)]}
    add(["zone_wide"], seed, trig(f"R{i+1}", z, spread=3),
        "zone_congestion", "zone_wide", "persistent", [f"INC-C{i}"],
        "multiple robots jammed same narrow zone -> congestion_deadlock")

# 29-30 fleet_wide (failures across many zones)
for i in range(2):
    seed = {"failures": [fail("F1", "aisle_1", 300), fail("F2", "aisle_5", 250),
                         fail("F3", "shipping_dock", 150), fail("F4", "cold_zone", 90)],
            "incidents": [inc(f"INC-F{i}", "localization_failure",
                "Fleet-wide localization degraded after a map server restart; aborts across all zones.", True)]}
    t = trig(f"R{i+1}", "staging", spread=1)
    t["distribution"] = {"per_robot_zone_spread": 4, "per_zone_robot_spread": 1}
    add(["fleet_wide"], seed, t,
        "localization_failure", "fleet_wide", "persistent", [f"INC-F{i}"],
        "aborts across many zones -> fleet_wide")


def main():
    # drop any accidental None entries
    clean = [c for c in cases if c]
    out = Path(__file__).parent / "diagnosis_cases.yaml"
    out.write_text(yaml.safe_dump(clean, sort_keys=False, allow_unicode=True, width=100))
    from collections import Counter
    print(f"wrote {out} — {len(clean)} cases")
    print("scope:", dict(Counter(c["ground_truth"]["scope"] for c in clean)))
    print("cause:", dict(Counter(c["ground_truth"]["cause"] for c in clean)))


if __name__ == "__main__":
    main()
