"""Consolidate per-model result JSONs into the multi-model comparison tables.

No LLM. Reads results_test[_<model>].json (diagnosis) and
results_intent_test[_<model>].json (intent).

    python3 -m eval.multimodel_table
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
DIAG = {"gpt-4.1-mini": "results_test.json", "haiku": "results_test_haiku.json",
        "solar": "results_test_solar.json"}
INTENT = {"gpt-4.1-mini": "results_intent_test.json", "haiku": "results_intent_test_haiku.json",
          "solar": "results_intent_test_solar.json"}


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def acc(rows, key):
    rows = [r for r in rows if "err" not in r]
    return 100 * sum(r[key] for r in rows) / max(len(rows), 1)


def diag_row(model, fn):
    d = load(fn)
    if not d:
        return None
    on, off = d["rag_on"], d["rag_off"]
    onok = [r for r in on if "err" not in r]
    cw = 100 * sum(1 for r in onok if not r["cause_ok"] and r["pred_cause"] != "unknown") / len(onok)
    acted = [r for r in onok if r["verdict"] == "PASS"]
    aprec = 100 * sum(r["cause_ok"] for r in acted) / max(len(acted), 1)
    rel = [r for r in onok if r.get("has_relevant")]
    relied = 100 * sum(r.get("relied_relevant") for r in rel) / max(len(rel), 1)
    return dict(model=model, cause_on=acc(on, "cause_ok"), cause_off=acc(off, "cause_ok"),
                scope_on=acc(on, "scope_ok"), relied=relied, cw=cw, aprec=aprec)


def intent_row(model, fn):
    rows = load(fn)
    if not rows:
        return None
    ok = [r for r in rows if "err" not in r]
    overall = 100 * sum(r["correct"] for r in ok) / len(ok)
    mn = [r for r in ok if r.get("must_not")]
    viol = sum(1 for r in mn if r["n_activated"] > 0)
    declined = sum(1 for r in mn if r["agent_declined"])
    blocked = sum(1 for r in mn if (not r["agent_declined"]) and r["n_activated"] == 0)
    return dict(model=model, overall=overall, n_mustnot=len(mn),
                declined=declined, blocked=blocked, leaked=viol)


def main():
    print("=" * 78)
    print("DIAGNOSIS (test n=100)  — cause/scope %, precedent reliance, safety")
    print("=" * 78)
    print(f"{'model':14s} {'cause_on':>9} {'cause_off':>10} {'scope_on':>9} "
          f"{'relied%':>8} {'conf-wrong':>11} {'acted-prec':>11}")
    for m, fn in DIAG.items():
        r = diag_row(m, fn)
        if r:
            print(f"{r['model']:14s} {r['cause_on']:8.0f}% {r['cause_off']:9.0f}% "
                  f"{r['scope_on']:8.0f}% {r['relied']:7.0f}% {r['cw']:10.0f}% {r['aprec']:10.0f}%")
    print()
    print("=" * 78)
    print("INTENT (test n=39)  — overall %, defense-in-depth on 15 must-not-activate")
    print("=" * 78)
    print(f"{'model':14s} {'overall':>8} {'mustnot':>8} {'declined':>9} {'blocked':>8} {'leaked':>7}")
    for m, fn in INTENT.items():
        r = intent_row(m, fn)
        if r:
            print(f"{r['model']:14s} {r['overall']:7.0f}% {r['n_mustnot']:8d} "
                  f"{r['declined']:9d} {r['blocked']:8d} {r['leaked']:7d}")
    print()
    print("Reading: RAG lifts cause across ALL models; failure mode differs (mini declines,")
    print("haiku/solar guess); intent leaks (guardrail-uncatchable force-fit) are model-dependent.")


if __name__ == "__main__":
    main()
