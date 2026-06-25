"""Generate 30 validator probes (eval/validator_probes.yaml) for the 중강 claim.

Each probe isolates ONE defect so its expected_verdict is deterministic against
decision_validator.validate_diagnosis (tau_diagnosis=0.5). The expected_verdict
is asserted from the validator's documented rules — NOT by calling the validator
(that would be circular). run_validator.py then checks the real validator matches.

Validator rules (sequential):
  conf < 0.5            -> DEGRADE
  evidence empty        -> DEGRADE
  any ref unresolvable  -> REJECT
  scope zone/fleet_wide & <2 mission_failures refs -> DEGRADE (if still PASS)
  relied & trust LOW & conf>0.7 -> DEGRADE

NOTE: pure overconfidence with grounded evidence is intentionally PASS — the
validator trusts grounded high-confidence diagnoses. That boundary is a
documented limitation, not a bug, and is exercised by the `none` probes.

    python3 eval/gen_validator_probes.py        # writes eval/validator_probes.yaml
"""
from __future__ import annotations

from pathlib import Path
import yaml

CAUSES = ["static_obstacle", "congestion_deadlock", "recurring_blockage",
          "localization_loss", "battery_depletion", "hardware_fault",
          "sensor_degradation", "transient_glitch"]
ZONES = ["receiving_dock", "aisle_1", "aisle_3", "aisle_5", "cold_zone",
         "shipping_dock", "staging"]


def _mf(robot, zone, status=6):
    return {"robot_id": robot, "zone": zone, "goal_status": status}


def base_bundle(zone, n_failures=2):
    return {
        "trigger_event": {"robot_id": "R1", "zone": zone},
        "mission_failures": [_mf(f"R{i+1}", zone) for i in range(n_failures)],
        "zone_state": {"zone": zone, "recent_failures": n_failures},
        "retrieved_precedents": [{"incident_id": "INC-1", "similarity": 0.8}],
        "active_policies": [],
    }


probes: list[dict] = []
pid = 0


def add(defect, expected, bundle, diagnosis, trust, desc):
    global pid
    pid += 1
    probes.append({
        "probe_id": f"VP-{pid:03d}",
        "description": desc,
        "defect_type": defect,
        "bundle": bundle,
        "retrieval_trust": trust,
        "diagnosis": diagnosis,
        "expected_verdict": expected,
    })


# ---- none -> PASS (8): grounded, conf>=0.5, scope consistent --------------
for i in range(8):
    z = ZONES[i % len(ZONES)]
    cause = CAUSES[i % len(CAUSES)]
    zone_wide = (i % 2 == 1)
    b = base_bundle(z, n_failures=2 if zone_wide else 1)
    if zone_wide:
        ev = [{"observation": "two robots aborted in zone",
               "refs": ["mission_failures[0].zone", "mission_failures[1].zone"]}]
        scope = "zone_wide"
    else:
        ev = [{"observation": "robot aborted",
               "refs": ["mission_failures[0].zone", "trigger_event.robot_id"]}]
        scope = "isolated"
    conf = 0.6 + 0.03 * i              # 0.6..0.81 (some "overconfident" but grounded -> PASS)
    add("none", "PASS", b,
        {"cause": cause, "scope": scope, "persistence": "persistent",
         "confidence": round(conf, 2), "evidence": ev, "relied_on_precedents": []},
        {"set_level": "HIGH"},
        f"grounded {scope} diagnosis, conf {conf:.2f} -> PASS")

# ---- ungrounded_ref -> REJECT (7): a ref that does not resolve ------------
bad_refs = [
    "mission_failures[5].robot_id",   # index out of range
    "zone_state.nonexistent_field",
    "trigger_event.mission_id",       # not in bundle trigger_event
    "retrieved_precedents[3].incident_id",
    "fleet_metrics.total",            # whole top-level key missing
    "mission_failures[0].battery_pct",  # subfield missing
    "active_policies[0].type",        # empty list -> index OOR
]
for i, ref in enumerate(bad_refs):
    z = ZONES[i % len(ZONES)]
    b = base_bundle(z, n_failures=1)
    add("ungrounded_ref", "REJECT", b,
        {"cause": CAUSES[i % len(CAUSES)], "scope": "isolated",
         "persistence": "persistent", "confidence": 0.7,
         "evidence": [{"observation": "cites missing evidence", "refs": [ref]}],
         "relied_on_precedents": []},
        None,                          # trust None -> step4 cannot overwrite REJECT
        f"unresolvable ref {ref!r} -> REJECT")

# ---- empty_evidence -> DEGRADE (4) ---------------------------------------
for i in range(4):
    z = ZONES[i % len(ZONES)]
    b = base_bundle(z, n_failures=1)
    add("empty_evidence", "DEGRADE", b,
        {"cause": CAUSES[i % len(CAUSES)], "scope": "isolated",
         "persistence": "persistent", "confidence": 0.7,
         "evidence": [], "relied_on_precedents": []},
        {"set_level": "HIGH"},
        "evidence empty -> DEGRADE")

# ---- low_confidence -> DEGRADE (4): conf < 0.5, otherwise valid -----------
for i in range(4):
    z = ZONES[i % len(ZONES)]
    b = base_bundle(z, n_failures=1)
    conf = 0.2 + 0.07 * i              # 0.20..0.41 < 0.5
    add("low_confidence", "DEGRADE", b,
        {"cause": CAUSES[i % len(CAUSES)], "scope": "isolated",
         "persistence": "transient", "confidence": round(conf, 2),
         "evidence": [{"observation": "weak signal",
                       "refs": ["mission_failures[0].zone"]}],
         "relied_on_precedents": []},
        {"set_level": "HIGH"},
        f"confidence {conf:.2f} < tau 0.5 -> DEGRADE")

# ---- scope_unsupported -> DEGRADE (4): zone_wide but <2 mission refs ------
for i in range(4):
    z = ZONES[i % len(ZONES)]
    b = base_bundle(z, n_failures=1)
    add("scope_unsupported", "DEGRADE", b,
        {"cause": "recurring_blockage", "scope": "zone_wide",
         "persistence": "recurring", "confidence": 0.7,
         "evidence": [{"observation": "claims zone-wide on one robot",
                       "refs": ["mission_failures[0].zone", "zone_state.recent_failures"]}],
         "relied_on_precedents": []},
        {"set_level": "MEDIUM"},
        "scope=zone_wide but <2 mission_failures refs -> DEGRADE")

# ---- retrieval_incoherent -> DEGRADE (3): relied + LOW trust + conf>0.7 ---
for i in range(3):
    z = ZONES[i % len(ZONES)]
    b = base_bundle(z, n_failures=1)
    add("retrieval_incoherent", "DEGRADE", b,
        {"cause": CAUSES[i % len(CAUSES)], "scope": "isolated",
         "persistence": "persistent", "confidence": 0.9,
         "evidence": [{"observation": "leans on weak precedent",
                       "refs": ["retrieved_precedents[0].incident_id"]}],
         "relied_on_precedents": ["INC-1"]},
        {"set_level": "LOW"},
        "relied + LOW trust + conf>0.7 -> DEGRADE")


def main():
    out = Path(__file__).parent / "validator_probes.yaml"
    out.write_text(yaml.safe_dump(probes, sort_keys=False, allow_unicode=True, width=100))
    from collections import Counter
    c = Counter(p["defect_type"] for p in probes)
    print(f"wrote {out} — {len(probes)} probes")
    for k, v in c.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
