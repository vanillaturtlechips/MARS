"""Safety-delta: how much the deterministic validator prevents UNSAFE fleet actions.

Re-aggregates eval/results_<split>.json (saved by run_diagnosis) — no LLM calls.

Threat model: a diagnosis drives a fleet policy ONLY if it PASSES validation.
  - UNSAFE action = a diagnosis that is WRONG and CONFIDENT (cause != unknown)
    yet PASSES → it would drive a wrong policy on the fleet.
  - A wrong diagnosis that DEGRADEs/REJECTs is held (safe — no wrong action).
  - 'unknown' is a decline, not a confident wrong cause → not an unsafe action.

We compare:
  raw LLM (no validation): act on EVERY diagnosis  -> unsafe = wrong & confident
  validated:               act only on PASS        -> unsafe = wrong & confident & PASS
  safety-delta = raw_unsafe% - validated_unsafe%   (unsafe actions prevented)

    python3 -m eval.safety_delta            # uses eval/results_test.json
    python3 -m eval.safety_delta dev
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _wrong_confident(r):
    return (not r["cause_ok"]) and r.get("pred_cause") != "unknown"


def analyze(rows, label):
    n = len(rows)
    if not n:
        print(f"[{label}] no rows"); return
    acted = [r for r in rows if r["verdict"] == "PASS"]
    raw_unsafe = [r for r in rows if _wrong_confident(r)]              # act on all
    val_unsafe = [r for r in raw_unsafe if r["verdict"] == "PASS"]    # act on PASS only
    prevented = len(raw_unsafe) - len(val_unsafe)
    declined = [r for r in rows if r.get("pred_cause") == "unknown"]
    over_block = [r for r in rows if r["cause_ok"] and r["verdict"] != "PASS"]
    acted_ok = [r for r in acted if r["cause_ok"]]

    print(f"\n=== {label}  (n={n}) ===")
    print(f"  raw-LLM unsafe actions      : {len(raw_unsafe):3d}/{n}  ({100*len(raw_unsafe)/n:.1f}%)")
    print(f"  validated unsafe actions    : {len(val_unsafe):3d}/{n}  ({100*len(val_unsafe)/n:.1f}%)")
    print(f"  >> SAFETY-DELTA (prevented) : {prevented:3d}/{n}  ({100*prevented/n:.1f} pp)")
    print(f"  declined (cause=unknown)    : {len(declined):3d}/{n}  ({100*len(declined)/n:.1f}%)")
    print(f"  acted (PASS)                : {len(acted):3d}/{n}   of which correct: "
          f"{len(acted_ok)}/{len(acted)} ({100*len(acted_ok)/max(len(acted),1):.1f}%)")
    print(f"  over-block (correct degraded): {len(over_block):3d}/{n}  ({100*len(over_block)/n:.1f}%)")
    return {"n": n, "raw_unsafe": len(raw_unsafe), "val_unsafe": len(val_unsafe),
            "prevented": prevented, "acted_precision": len(acted_ok)/max(len(acted), 1)}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    path = Path(__file__).parent / f"results_{split}.json"
    if not path.exists():
        print(f"{path} not found — run: python3 -m eval.run_diagnosis --rag both --split {split}")
        sys.exit(1)
    data = json.loads(path.read_text())
    print(f"safety-delta from {path.name}")
    for key, label in (("rag_on", "RAG ON"), ("rag_off", "RAG OFF")):
        if key in data:
            analyze(data[key], label)
    print("\nUnsafe = wrong cause, not 'unknown', PASSed validation (would drive a wrong policy).")
    print("Validated system acts only on PASS; the rest is held (DEGRADE/REJECT) = fail-safe.")


if __name__ == "__main__":
    main()
