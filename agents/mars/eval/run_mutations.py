"""P1/P2 mutation test: run eval/mutation_probes.json through the runtime validator V.

V is the whole runtime gate, not just the Python validator:
  diagnosis: V_dx = schema(_OUTPUT_SCHEMA, enforced by the API's structured output at
             runtime, reproduced here with jsonschema) ∘ validate_diagnosis
  policy:    V_pol = schema(policy item) ∘ guardrail.check

Reports, per operator: non-accept rate (P1 for `expect: non_accept`), accept rate
(P2 for `expect: accept`), verdict distribution, and lists every violation.
Also checks P2 on every probe's ORIGIN (the unmutated, valid-by-construction input).

    python3 -m eval.run_mutations              # exit 1 on any P1/P2 violation
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import jsonschema

from mars.agents.failure_analysis import _OUTPUT_SCHEMA as DX_SCHEMA
from mars.agents.intent_agent import _OUTPUT_SCHEMA as INTENT_SCHEMA
from mars.validators.decision_validator import validate_diagnosis
from mars.guardrail.guardrail import check as guardrail_check

POL_ITEM_SCHEMA = INTENT_SCHEMA["properties"]["policy_updates"]["items"]


def v_dx(d, bundle, trust):
    try:
        jsonschema.validate(d, DX_SCHEMA)
    except jsonschema.ValidationError as e:
        return "SCHEMA_REJECT", f"schema: {e.message[:80]}"
    verdict, notes = validate_diagnosis(d, bundle, {"set_level": trust} if trust else None)
    return verdict.value, notes


def v_pol(p, active, world, last_applied):
    # probes store last_applied as seconds-ago; guardrail wants absolute epoch
    now = time.time()
    last_applied = {k: now - v for k, v in (last_applied or {}).items()}
    try:
        jsonschema.validate(p, POL_ITEM_SCHEMA)
    except jsonschema.ValidationError as e:
        return "SCHEMA_REJECT", f"schema: {e.message[:80]}"
    g, _modified, notes = guardrail_check(p, active, world, last_applied=last_applied)
    return g.value, notes


ACCEPT = {"dx": {"PASS"}, "pol": {"ACCEPT"}}


def run_one(pr):
    if pr["kind"] == "dx":
        return v_dx(pr["diagnosis"], pr["bundle"], pr["trust"])
    return v_pol(pr["policy"], pr["active"], pr["world"], pr["last_applied"])


def run_origin(pr):
    o = pr["origin"]
    if pr["kind"] == "dx":
        return v_dx(o["diagnosis"], o["bundle"], o["trust"])
    return v_pol(o["policy"], o["active"], o["world"], o["last_applied"])


def main(path: str | None = None):
    p = Path(path) if path else Path(__file__).parent / "mutation_probes.json"
    probes = json.loads(p.read_text())

    by_op = defaultdict(lambda: {"n": 0, "ok": 0, "verdicts": Counter(), "fails": []})
    origin_fail = []
    for pr in probes:
        verdict, notes = run_one(pr)
        accepted = verdict in ACCEPT[pr["kind"]]
        ok = (not accepted) if pr["expect"] == "non_accept" else accepted
        s = by_op[pr["op"]]
        s["n"] += 1; s["ok"] += ok; s["verdicts"][verdict] += 1
        if not ok:
            s["fails"].append((pr["probe_id"], verdict, notes))
        # P2 on the unmutated origin
        ov, on = run_origin(pr)
        if ov not in ACCEPT[pr["kind"]]:
            origin_fail.append((pr["probe_id"], pr["op"], ov, on))

    n = len(probes)
    total_ok = sum(s["ok"] for s in by_op.values())
    print("=" * 78)
    print(f"mutation probes: {n}   P1/P2 satisfied: {total_ok}/{n} ({100*total_ok/n:.2f}%)")
    print(f"origin (unmutated) P2: {n - len(origin_fail)}/{n} accepted")
    print("-" * 78)
    print(f"{'operator':22s} {'expect':10s} {'ok/n':>9s}  verdicts")
    for op in sorted(by_op, key=lambda k: (k.startswith('B-'), k)):
        s = by_op[op]
        exp = "accept" if op.startswith("B-") else "non_accept"
        flag = "" if s["ok"] == s["n"] else "  <-- VIOLATION"
        print(f"{op:22s} {exp:10s} {s['ok']:4d}/{s['n']:<4d}  {dict(s['verdicts'])}{flag}")
    print("-" * 78)
    # P1 / P2 summary by predicate class
    p1 = [(s["ok"], s["n"]) for op, s in by_op.items() if not op.startswith("B-")]
    p2 = [(s["ok"], s["n"]) for op, s in by_op.items() if op.startswith("B-")]
    print(f"P1 (mutants non-accepted): {sum(a for a,_ in p1)}/{sum(b for _,b in p1)}")
    print(f"P2 (boundary accepted):    {sum(a for a,_ in p2)}/{sum(b for _,b in p2)}")
    print("=" * 78)

    bad = False
    for op, s in by_op.items():
        if s["fails"]:
            bad = True
            print(f"\n[{op}] {len(s['fails'])} violations, e.g.:")
            for pid, v, notes in s["fails"][:3]:
                print(f"   {pid}: verdict={v}  notes={notes}")
    if origin_fail:
        bad = True
        print(f"\nORIGIN P2 violations ({len(origin_fail)}), e.g.:")
        for pid, op, v, notes in origin_fail[:5]:
            print(f"   {pid} [{op}]: {v}  {notes}")
    out = Path(__file__).parent / "results_mutations.json"
    out.write_text(json.dumps({op: {"n": s["n"], "ok": s["ok"], "verdicts": dict(s["verdicts"]),
                                    "fails": s["fails"]} for op, s in by_op.items()}, indent=1))
    print(f"\nsaved -> {out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
