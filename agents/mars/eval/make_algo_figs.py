"""Render Algorithm 1 & 2 as clean bordered monospace boxes (PNG).

Replaces the broken blockquote+codeblock pseudocode in the paper. ASCII-only
operators so nothing depends on glyph availability.

    python3 -m eval.make_algo_figs
Output: eval/figs/fig_algo1.png, fig_algo2.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).parent / "figs"; FIGS.mkdir(exist_ok=True)

ALGO1 = """Algorithm 1  Decision Validator (diagnosis)
------------------------------------------------------------
Input : diagnosis d (cause, scope, confidence, evidence,
        relied_on_precedents); input bundle B;
        retrieval set-level t
Output: PASS | DEGRADE | REJECT
------------------------------------------------------------
1  r <- PASS
2  if d.confidence < tau_diag:            r <- DEGRADE
3  if d.evidence is empty:                r <- DEGRADE
4  for each ref in d.evidence.refs:
5      if ref does not resolve in B:      r <- REJECT
6  if d.scope in {zone_wide, fleet_wide}
7     and (#refs to mission_failures) < 2: r <- max(r, DEGRADE)
8  if d.relied_on_precedents != {} and t = LOW
9     and d.confidence > 0.7:             r <- max(r, DEGRADE)
10 return r"""

ALGO2 = """Algorithm 2  Policy Guardrail
------------------------------------------------------------
Input : candidate policy p; active policies A;
        world state W; last-applied times L
Output: ACCEPT | MODIFY | REJECT | DEFER_HUMAN
------------------------------------------------------------
1  if p.type not in WHITELIST
      or p.duration missing:        return REJECT
2  if p.zone set and p.zone not in W.zones:
                                     return REJECT
3  if impact_tier(p.type) = HIGH:   return DEFER_HUMAN
4  if p violates liveness(W):       return REJECT
5  if exists a in A with same type and params:
                                     return REJECT
6  p.duration <- clamp(p.duration, 60, 7200)
7  if now - L[p.type] < cooldown:   return REJECT
8  return MODIFY if adjusted else ACCEPT"""


def render(text, name):
    nlines = text.count("\n") + 1
    fig, ax = plt.subplots(figsize=(6.2, 0.23 * nlines + 0.3))
    ax.axis("off")
    ax.text(0.015, 0.97, text, family="monospace", fontsize=9.5, va="top", ha="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.patch.set_edgecolor("#333333"); fig.patch.set_linewidth(1.2)
    f = FIGS / name
    fig.savefig(f, dpi=200, bbox_inches="tight", edgecolor="#333333")
    print("wrote", f)


if __name__ == "__main__":
    render(ALGO1, "fig_algo1.png")
    render(ALGO2, "fig_algo2.png")
