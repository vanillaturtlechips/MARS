"""Run diagnosis_cases through the REAL FailureAnalysisAgent and score it (중강 P1).

For each case: reset DB → seed failures + incident embeddings (+ policies) →
run agent.analyze(trigger) → compare cause/scope to ground_truth → run the
Decision Validator on the output. Supports a RAG ablation (precedents present
vs absent) so we can measure RAG's contribution.

Needs: Postgres+pgvector up (docker compose up -d), .env with ANTHROPIC_API_KEY,
LLM_PROVIDER=anthropic, EMBEDDING_PROVIDER=local, EMBEDDING_DIM=384.

    cd agents/mars && source .venv/bin/activate
    python3 -m eval.run_diagnosis --rag both --limit 0     # 0 = all

WARNING: calls the Anthropic API once per case (per rag mode) — costs tokens.
Use --limit N to smoke-test on a few first.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

import mars.blackboard.queries as Q
from mars.blackboard.db import connect
from mars.llm.client import get_investigator_client, get_embedder, MockEmbedder
from mars.agents.tools import InvestigatorTools
from mars.agents.failure_analysis import FailureAnalysisAgent
from mars.validators.decision_validator import validate_diagnosis

RESET_TABLES = ["failures", "incident_embeddings", "policies",
                "diagnoses", "outcomes"]


def _reset(conn):
    with conn.cursor() as cur:
        cur.execute("TRUNCATE " + ", ".join(RESET_TABLES) + " RESTART IDENTITY CASCADE")
    conn.commit()


def _seed(conn, case, embedder, rag_on: bool):
    now = datetime.now(timezone.utc)
    for f in case["seed_state"].get("failures", []):
        ev = dict(f)
        ev["occurred_at"] = now - timedelta(seconds=f.get("ts_offset_sec", 60))
        Q.write_failure(conn, ev)
    if rag_on:
        for ic in case["seed_state"].get("incidents", []):
            Q.write_embedding(conn, {
                "source_type": "outcome", "source_id": ic["incident_id"],
                "zone": case["trigger_event"].get("zone"),
                "failure_type": ic.get("true_cause"), "scope": None,
                "outcome_label": None, "outcome_id": None,
                "summary": ic["text"], "embedding": embedder.embed(ic["text"]),
                "recorded_at": now - timedelta(seconds=300),
            })
    conn.commit()


def run_mode(cases, rag_on: bool, limit: int):
    conn = connect(autocommit=False)
    embedder = get_embedder() if rag_on else MockEmbedder(dim=384)
    client = get_investigator_client()
    rows = []
    sel = cases[:limit] if limit else cases
    tag = "RAG_ON" if rag_on else "RAG_OFF"
    for i, case in enumerate(sel, 1):
        print(f"  [{tag} {i}/{len(sel)}] {case['case_id']} ...", flush=True)
        _reset(conn)
        _seed(conn, case, embedder, rag_on)
        tools = InvestigatorTools(conn, embedder)
        agent = FailureAnalysisAgent(client, tools)
        try:
            dx = agent.analyze(case["trigger_event"])
        except Exception as e:  # noqa: BLE001
            rows.append({"case": case["case_id"], "err": str(e)})
            continue
        gt = case["ground_truth"]
        bundle = dx.get("_tool_transcript", {})
        verdict, _ = validate_diagnosis(dx, bundle)
        rows.append({
            "case": case["case_id"],
            "cause_ok": dx.get("cause") == gt["cause"],
            "scope_ok": dx.get("scope") == gt["scope"],
            "pred_cause": dx.get("cause"), "gt_cause": gt["cause"],
            "pred_scope": dx.get("scope"), "gt_scope": gt["scope"],
            "verdict": verdict.value,
        })
    conn.close()
    return rows


def summarize(tag, rows):
    ok = [r for r in rows if "err" not in r]
    n = len(ok)
    cause = sum(r["cause_ok"] for r in ok)
    scope = sum(r["scope_ok"] for r in ok)
    errs = [r for r in rows if "err" in r]
    print(f"\n=== {tag}  (n={n}, errors={len(errs)}) ===")
    print(f"  cause accuracy: {cause}/{n} ({100*cause/n:.1f}%)" if n else "  no cases")
    print(f"  scope accuracy: {scope}/{n} ({100*scope/n:.1f}%)" if n else "")
    print(f"  verdicts: {dict(Counter(r['verdict'] for r in ok))}")
    # per-case pred vs gt (so accuracy is debuggable, not a black box)
    print("  case        cause(pred/gt)                    scope(pred/gt)        verdict")
    for r in ok:
        cflag = "✓" if r["cause_ok"] else "✗"
        sflag = "✓" if r["scope_ok"] else "✗"
        print(f"  {r['case']:10s} {cflag} {str(r['pred_cause']):20s}/{str(r['gt_cause']):20s} "
              f"{sflag} {str(r['pred_scope']):12s}/{str(r['gt_scope']):12s} {r['verdict']}")
    if errs:
        print(f"  ERRORS: {[(e['case'], e['err'][:60]) for e in errs[:5]]}")
    return cause, scope, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(Path(__file__).parent / "diagnosis_cases.yaml"))
    ap.add_argument("--rag", choices=["on", "off", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    a = ap.parse_args()
    cases = yaml.safe_load(Path(a.cases).read_text())
    print(f"loaded {len(cases)} diagnosis cases")

    if a.rag in ("on", "both"):
        summarize("RAG ON", run_mode(cases, True, a.limit))
    if a.rag in ("off", "both"):
        summarize("RAG OFF", run_mode(cases, False, a.limit))


if __name__ == "__main__":
    main()
