"""Run validator_probes through the REAL decision_validator and score it.

Headline of the 중강 claim: does the deterministic validator catch unsafe
(ungrounded / unsupported / low-confidence / incoherent) diagnoses while passing
valid ones? This runs each probe through validate_diagnosis() and compares the
verdict to the probe's expected_verdict.

    cd agents/mars
    source .venv/bin/activate          # needs python-dotenv, pydantic, pyyaml
    python3 -m eval.run_validator

Metrics:
  - exact verdict accuracy (per defect_type)
  - block precision/recall  (block = DEGRADE|REJECT; "should block" = defect != none)
  - false-block rate        (valid 'none' probes wrongly blocked)
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import yaml

from mars.validators.decision_validator import validate_diagnosis, DVResult


def main(path: str | None = None):
    probes_path = Path(path) if path else Path(__file__).parent / "validator_probes.yaml"
    probes = yaml.safe_load(probes_path.read_text())

    rows = []
    for p in probes:
        verdict, notes = validate_diagnosis(
            p["diagnosis"], p["bundle"], p.get("retrieval_trust"))
        actual = verdict.value
        expected = p["expected_verdict"]
        rows.append({
            "probe_id": p["probe_id"], "defect": p["defect_type"],
            "expected": expected, "actual": actual, "match": actual == expected,
            "notes": notes,
        })

    # exact-match accuracy
    n = len(rows)
    correct = sum(r["match"] for r in rows)

    # per defect_type
    by_defect = defaultdict(lambda: [0, 0])
    for r in rows:
        by_defect[r["defect"]][1] += 1
        by_defect[r["defect"]][0] += r["match"]

    # block precision/recall (block = DEGRADE|REJECT)
    def is_block(v):
        return v in ("DEGRADE", "REJECT")

    tp = sum(1 for r in rows if r["defect"] != "none" and is_block(r["actual"]))
    fn = sum(1 for r in rows if r["defect"] != "none" and not is_block(r["actual"]))
    fp = sum(1 for r in rows if r["defect"] == "none" and is_block(r["actual"]))
    tn = sum(1 for r in rows if r["defect"] == "none" and not is_block(r["actual"]))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    n_none = sum(1 for r in rows if r["defect"] == "none")
    false_block = fp / n_none if n_none else 0.0

    print("=" * 60)
    print(f"validator probes: {n}   exact-verdict accuracy: {correct}/{n} "
          f"({100*correct/n:.1f}%)")
    print("-" * 60)
    print("per defect_type (correct/total):")
    for d, (c, t) in sorted(by_defect.items()):
        print(f"  {d:22s} {c}/{t}")
    print("-" * 60)
    print(f"block precision: {prec:.3f}   recall: {rec:.3f}")
    print(f"false-block rate (valid wrongly blocked): {false_block:.3f}  ({fp}/{n_none})")
    print("=" * 60)

    mism = [r for r in rows if not r["match"]]
    if mism:
        print("MISMATCHES (expected != actual):")
        for r in mism:
            print(f"  {r['probe_id']} [{r['defect']}] expected={r['expected']} "
                  f"actual={r['actual']}  notes={r['notes']}")
        sys.exit(1)
    print("all probes matched expected verdicts ✓")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
