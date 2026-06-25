"""Generate paper figures from results JSON (no LLM). Agg backend, headless.

    python3 -m eval.make_figs            # -> eval/figs/*.png

Reads results_test.json (diagnosis) + results_intent_test.json (intent).
ASCII labels (default font can't render Korean).
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def _load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def acc(rows, key="cause_ok"):
    rows = [r for r in rows if "err" not in r]
    return 100 * sum(r[key] for r in rows) / max(len(rows), 1)


def fig_diag_rag(diag):
    on, off = diag["rag_on"], diag["rag_off"]
    # overall + by difficulty
    diffs = ["easy", "medium", "hard"]
    def by(rows, d): return [r for r in rows if "err" not in r and r.get("difficulty") == d]
    labels = ["overall"] + diffs
    on_v = [acc(on)] + [acc(by(on, d)) for d in diffs]
    off_v = [acc(off)] + [acc(by(off, d)) for d in diffs]
    x = range(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w/2 for i in x], on_v, w, label="RAG on", color="tab:blue")
    ax.bar([i + w/2 for i in x], off_v, w, label="RAG off", color="tab:gray")
    for i, v in enumerate(on_v): ax.text(i - w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for i, v in enumerate(off_v): ax.text(i + w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("cause accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Diagnosis cause accuracy: RAG ablation (test n=100)")
    ax.legend()
    f = FIGS / "fig_diag_rag.png"; fig.savefig(f, dpi=130, bbox_inches="tight")
    print("wrote", f)


def fig_safety(diag):
    on, off = diag["rag_on"], diag["rag_off"]
    def cw(rows):  # confident-wrong rate
        rows = [r for r in rows if "err" not in r]
        return 100 * sum(1 for r in rows if not r["cause_ok"] and r["pred_cause"] != "unknown") / len(rows)
    def acted_prec(rows):
        rows = [r for r in rows if "err" not in r]
        acted = [r for r in rows if r["verdict"] == "PASS"]
        return 100 * sum(r["cause_ok"] for r in acted) / max(len(acted), 1)
    metrics = ["confident-wrong", "acted-precision"]
    on_v = [cw(on), acted_prec(on)]; off_v = [cw(off), acted_prec(off)]
    x = range(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar([i - w/2 for i in x], on_v, w, label="RAG on", color="tab:green")
    ax.bar([i + w/2 for i in x], off_v, w, label="RAG off", color="tab:gray")
    for i, v in enumerate(on_v): ax.text(i - w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for i, v in enumerate(off_v): ax.text(i + w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Diagnosis safety (test n=100)")
    ax.legend()
    f = FIGS / "fig_diag_safety.png"; fig.savefig(f, dpi=130, bbox_inches="tight")
    print("wrote", f)


def fig_intent_defense(intent):
    rows = [r for r in intent if "err" not in r]
    mn = [r for r in rows if r.get("must_not")]
    agent = sum(1 for r in mn if r["agent_declined"])
    blocked = sum(1 for r in mn if (not r["agent_declined"]) and r["n_activated"] == 0)
    leaked = sum(1 for r in mn if r["n_activated"] > 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(["unsafe intents"], [agent], label=f"agent declined ({agent})", color="tab:green")
    ax.bar(["unsafe intents"], [blocked], bottom=[agent],
           label=f"guardrail blocked ({blocked})", color="tab:blue")
    ax.bar(["unsafe intents"], [leaked], bottom=[agent + blocked],
           label=f"leaked ({leaked})", color="tab:red")
    ax.set_ylabel("must-not-activate cases")
    ax.set_title(f"Intent defense-in-depth (n={len(mn)}): blocked {agent+blocked}/{len(mn)}")
    ax.legend(loc="lower right")
    f = FIGS / "fig_intent_defense.png"; fig.savefig(f, dpi=130, bbox_inches="tight")
    print("wrote", f)


def main():
    diag = _load("results_test.json")
    intent = _load("results_intent_test.json")
    if diag:
        fig_diag_rag(diag); fig_safety(diag)
    if intent:
        fig_intent_defense(intent)
    print(f"figs -> {FIGS}")


if __name__ == "__main__":
    main()
