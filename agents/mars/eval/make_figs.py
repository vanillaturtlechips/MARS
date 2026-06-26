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


DIAG_MODELS = {"gpt-4.1-mini": "results_test.json", "haiku": "results_test_haiku.json",
               "solar": "results_test_solar.json"}
INTENT_MODELS = {"gpt-4.1-mini": "results_intent_test.json",
                 "haiku": "results_intent_test_haiku.json",
                 "solar": "results_intent_test_solar.json"}


def _cause(rows):
    rows = [r for r in rows if "err" not in r]
    return 100 * sum(r["cause_ok"] for r in rows) / max(len(rows), 1)


def _confwrong(rows):
    rows = [r for r in rows if "err" not in r]
    return 100 * sum(1 for r in rows if not r["cause_ok"] and r["pred_cause"] != "unknown") / max(len(rows), 1)


def fig_mm_rag():
    """Grouped bars: cause accuracy RAG on vs off, per model."""
    models, on, off = [], [], []
    for m, fn in DIAG_MODELS.items():
        d = _load(fn)
        if not d:
            continue
        models.append(m); on.append(_cause(d["rag_on"])); off.append(_cause(d["rag_off"]))
    x = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([i - w/2 for i in x], on, w, label="RAG on", color="tab:blue")
    ax.bar([i + w/2 for i in x], off, w, label="RAG off", color="tab:gray")
    for i, v in enumerate(on): ax.text(i - w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for i, v in enumerate(off): ax.text(i + w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylabel("cause accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Diagnosis cause accuracy by model: RAG ablation (test n=100)")
    ax.legend()
    f = FIGS / "fig_mm_rag.png"; fig.savefig(f, dpi=130, bbox_inches="tight"); print("wrote", f)


def fig_mm_safety():
    """Grouped bars: confident-wrong RAG on vs off, per model."""
    models, on, off = [], [], []
    for m, fn in DIAG_MODELS.items():
        d = _load(fn)
        if not d:
            continue
        models.append(m); on.append(_confwrong(d["rag_on"])); off.append(_confwrong(d["rag_off"]))
    x = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([i - w/2 for i in x], on, w, label="RAG on", color="tab:green")
    ax.bar([i + w/2 for i in x], off, w, label="RAG off", color="tab:red")
    for i, v in enumerate(on): ax.text(i - w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    for i, v in enumerate(off): ax.text(i + w/2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylabel("confident-wrong rate (%)"); ax.set_ylim(0, max(max(off), 1) + 6)
    ax.set_title("Unsafe (confident-wrong) diagnoses by model: RAG ablation")
    ax.legend()
    f = FIGS / "fig_mm_safety.png"; fig.savefig(f, dpi=130, bbox_inches="tight"); print("wrote", f)


def fig_mm_intent():
    """Stacked bars per model: agent declined / guardrail blocked / leaked."""
    models, agent, blocked, leaked = [], [], [], []
    for m, fn in INTENT_MODELS.items():
        rows = _load(fn)
        if not rows:
            continue
        mn = [r for r in rows if "err" not in r and r.get("must_not")]
        models.append(m)
        agent.append(sum(1 for r in mn if r["agent_declined"]))
        blocked.append(sum(1 for r in mn if (not r["agent_declined"]) and r["n_activated"] == 0))
        leaked.append(sum(1 for r in mn if r["n_activated"] > 0))
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x, agent, label="agent declined", color="tab:green")
    ax.bar(x, blocked, bottom=agent, label="guardrail blocked", color="tab:blue")
    ax.bar(x, leaked, bottom=[a + b for a, b in zip(agent, blocked)], label="leaked", color="tab:red")
    for i in x:
        if leaked[i]:
            ax.text(i, agent[i] + blocked[i] + leaked[i] + 0.2, f"leak {leaked[i]}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylabel("must-not-activate cases (of 15)")
    ax.set_title("Intent defense-in-depth by model (test, 15 unsafe intents)")
    ax.legend(loc="lower right")
    f = FIGS / "fig_mm_intent.png"; fig.savefig(f, dpi=130, bbox_inches="tight"); print("wrote", f)


def fig_gaming():
    """Honest vs acceptance-incentivized (gamed) agent: DEGRADE collapse but
    confident-wrong stays ~0."""
    h = _load("results_test.json")
    g = _load("results_test_gamed.json")
    if not (h and g):
        return
    def m(d):
        rows = [r for r in d["rag_on"] if "err" not in r]
        deg = sum(1 for r in rows if r["verdict"] == "DEGRADE")
        cw = sum(1 for r in rows if not r["cause_ok"] and r["pred_cause"] != "unknown"
                 and r["verdict"] == "PASS")
        return deg, cw
    hd, hcw = m(h); gd, gcw = m(g)
    labels = ["DEGRADE\n(held, low-conf)", "confident-wrong\n(unsafe PASS)"]
    honest = [hd, hcw]; gamed = [gd, gcw]
    x = range(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar([i - w/2 for i in x], honest, w, label="honest agent", color="tab:blue")
    ax.bar([i + w/2 for i in x], gamed, w, label="acceptance-incentivized", color="tab:orange")
    for i, v in enumerate(honest): ax.text(i - w/2, v + 0.3, str(v), ha="center", fontsize=9)
    for i, v in enumerate(gamed): ax.text(i + w/2, v + 0.3, str(v), ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("cases (of 100)")
    ax.set_title("Gaming the confidence gate (GPT-4.1-mini, RAG on)")
    ax.legend()
    f = FIGS / "fig_gaming.png"; fig.savefig(f, dpi=130, bbox_inches="tight"); print("wrote", f)


def main():
    diag = _load("results_test.json")
    intent = _load("results_intent_test.json")
    if diag:
        fig_diag_rag(diag); fig_safety(diag)
    if intent:
        fig_intent_defense(intent)
    # multi-model figures
    fig_mm_rag(); fig_mm_safety(); fig_mm_intent()
    # gaming experiment
    fig_gaming()
    print(f"figs -> {FIGS}")


if __name__ == "__main__":
    main()
