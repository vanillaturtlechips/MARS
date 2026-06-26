"""Render the two architecture pipelines as clean vector diagrams (PNG via dot).

Replaces the ASCII blocks in the paper. Output: eval/figs/fig_arch_diag.png,
fig_arch_intent.png.

    python3 -m eval.make_arch_figs   (needs graphviz `dot`)
"""
import subprocess
from pathlib import Path

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

DIAG = r'''
digraph G {
  rankdir=LR; bgcolor="white";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        fillcolor="#eef3fb", color="#3a6ea5", penwidth=1.3, margin="0.12,0.08"];
  edge [fontname="Helvetica", fontsize=9, color="#444444"];

  ev   [label="Failure\nevent", fillcolor="#f5f5f5", color="#888888"];
  agent[label="Failure Analysis Agent\n(ReAct tool loop)"];
  dx   [label="Diagnosis\n(cause, scope,\nconfidence, evidence)", fillcolor="#f5f5f5", color="#888888"];
  val  [label="Decision\nValidator", fillcolor="#fdeee6", color="#c8722e"];
  out  [label="PASS / DEGRADE / REJECT", shape=note, fillcolor="#eaf6ec", color="#4a8b5c"];
  tools[label="read-only tools:\nmission_failures, zone_state,\nrobot_history, retrieved\nprecedents (RAG), policies",
        shape=folder, fillcolor="#fbf8e8", color="#b3a233", fontsize=9];

  ev -> agent -> dx -> val -> out;
  tools -> agent [style=dashed, dir=both, constraint=false];
}
'''

INTENT = r'''
digraph G {
  rankdir=LR; bgcolor="white";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        fillcolor="#eef3fb", color="#3a6ea5", penwidth=1.3, margin="0.12,0.08"];
  edge [fontname="Helvetica", fontsize=9, color="#444444"];

  utt  [label="Operator\nNL utterance", fillcolor="#f5f5f5", color="#888888"];
  ia   [label="Intent Agent\n(whitelist only;\nmay decline)"];
  pol  [label="Candidate\npolicies", fillcolor="#f5f5f5", color="#888888"];
  gr   [label="Policy Guardrail\n(7 ordered stages)", fillcolor="#fdeee6", color="#c8722e"];
  out  [label="ACCEPT / MODIFY /\nREJECT / DEFER_HUMAN", shape=note, fillcolor="#eaf6ec", color="#4a8b5c"];

  utt -> ia -> pol -> gr -> out;
}
'''


def render(dot_src, name):
    p = subprocess.run(["dot", "-Tpng", "-Gdpi=200", "-o", str(FIGS / name)],
                       input=dot_src.encode(), capture_output=True)
    if p.returncode != 0:
        print("ERR", name, p.stderr.decode()[:200])
    else:
        print("wrote", FIGS / name)


if __name__ == "__main__":
    render(DIAG, "fig_arch_diag.png")
    render(INTENT, "fig_arch_intent.png")
