"""Re-aggregate results_<split>.json to dissect the 3 known limitations. No LLM.

  1. fleet_wide / sensor-adversarial: precedent retrieved but not used —
     is it a LOW trust score (discounted) or used-but-wrong?
  2. over-block: correct diagnoses that got DEGRADE/REJECT — which DV rule fired
     (low confidence vs scope ref<2)?
  3. confident-wrong residue: wrong & confident & PASS (validator gap).

    python3 -m eval.analyze_limits            # results_test.json
    python3 -m eval.analyze_limits dev
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path


def section(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def fleet_sensor(rows):
    section("1. precedent retrieved-but-not-used (fleet_wide / sensor)")
    # wrong cases that HAD a relevant precedent retrieved
    bad = [r for r in rows if not r["cause_ok"] and r.get("relevant_retrieved")]
    print(f"wrong despite relevant precedent retrieved: {len(bad)}")
    declined = [r for r in bad if r["pred_cause"] == "unknown"]
    usedwrong = [r for r in bad if r["pred_cause"] != "unknown"]
    print(f"  declined (pred=unknown): {len(declined)}   used-but-wrong: {len(usedwrong)}")
    trusts = [r["relevant_trust"] for r in bad if r.get("relevant_trust") is not None]
    if trusts:
        mean_t = sum(trusts) / len(trusts)
        lo = sum(t < 0.5 for t in trusts)
        print(f"  relevant-precedent trust: min {min(trusts):.2f} mean {mean_t:.2f} "
              f"max {max(trusts):.2f}  (<0.5: {lo}/{len(trusts)})")
        # data-driven verdict (don't assume LOW trust)
        if lo > len(trusts) / 2:
            print("  -> mostly LOW trust: retrieval validator discounts it (retrieval-side bottleneck).")
        else:
            print("  -> trust is ADEQUATE yet the agent still declined (pred=unknown):")
            print("     NOT a retrieval/trust problem -> model fails to INTEGRATE an available")
            print("     precedent on hard cases (reasoning limit; test a stronger model).")
    # compare to trust of CORRECT cases that used precedent
    good_trust = [r["relevant_trust"] for r in rows
                  if r["cause_ok"] and r.get("relevant_trust") is not None]
    if good_trust:
        print(f"  (correct cases relevant-trust mean: {sum(good_trust)/len(good_trust):.2f})")


def over_block(rows):
    section("2. over-block: correct diagnosis but DEGRADE/REJECT")
    ob = [r for r in rows if r["cause_ok"] and r["verdict"] != "PASS"]
    print(f"over-blocked: {len(ob)}")
    lowconf = [r for r in ob if (r.get("confidence") or 1) < 0.5]
    print(f"  due to low confidence (<0.5): {len(lowconf)}")
    # note keywords
    kw = Counter()
    for r in ob:
        nt = (r.get("dv_notes") or "").lower()
        if "confidence" in nt: kw["confidence<tau"] += 1
        if "mission_failures" in nt or "scope" in nt: kw["scope ref<2"] += 1
        if "unresolvable" in nt: kw["ungrounded ref"] += 1
        if "empty" in nt: kw["empty evidence"] += 1
    print(f"  DV-note reasons: {dict(kw)}")


def residue(rows):
    section("3. confident-wrong residue (validator gap)")
    cw = [r for r in rows if not r["cause_ok"] and r["pred_cause"] != "unknown"
          and r["verdict"] == "PASS"]
    print(f"wrong & confident & PASS: {len(cw)}")
    for r in cw[:20]:
        print(f"  {r['case']}: pred {r['pred_cause']} / gt {r['gt_cause']}  (grounded-but-wrong)")
    print("  -> validator passes grounded-but-wrong; safety here = agent's decline, not DV.")


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    path = Path(__file__).parent / f"results_{split}.json"
    if not path.exists():
        print(f"{path} not found — run run_diagnosis first."); sys.exit(1)
    data = json.loads(path.read_text())
    rows = data.get("rag_on", [])
    rows = [r for r in rows if "err" not in r]
    print(f"analyzing RAG ON, n={len(rows)} ({path.name})")
    fleet_sensor(rows)
    over_block(rows)
    residue(rows)


if __name__ == "__main__":
    main()
