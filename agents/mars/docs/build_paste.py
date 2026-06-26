"""Produce a paste-ready plain-text version of paper_ko.md for the KROS Word form.

- Markdown markers stripped (**bold**, *italic*, `code`)
- Tables -> TAB-separated rows (in Word: paste, then Insert > Table > Convert
  Text to Table, separator = Tab)
- Algorithms/code -> plain monospace text (put in a box/figure in Word)
- Figures -> explicit "[INSERT image: path]" markers with caption

    python3 docs/build_paste.py
Output: /home/user/Downloads/MARS_paste_ready.txt
"""
import re
from pathlib import Path

KO = Path(__file__).parent / "paper_ko.md"
OUT = "/home/user/Downloads/MARS_paste_ready.txt"
FIGDIR = Path(__file__).parent.parent / "eval" / "figs"
FIGCAP = {
    "fig_mm_rag.png": "Fig. 1. Diagnosis cause accuracy by model, RAG on vs. off (test, n=100).",
    "fig_mm_safety.png": "Fig. 2. Confident-wrong (unsafe) rate by model, RAG on vs. off.",
    "fig_mm_intent.png": "Fig. 3. Intent defense-in-depth by model over 15 unsafe intents.",
    "fig_gaming.png": "Fig. 4. Honest vs. acceptance-incentivized agent (GPT-4.1-mini, RAG on).",
}


def strip_md(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = t.replace("`", "")
    return t


def main():
    lines = KO.read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        # figures section: replace with insert markers
        if s == "## Figures (캡션 영문 — paper_en.md와 동일)" or s.startswith("## Figures"):
            out.append("\n==================== FIGURES ====================")
            for fn, cap in FIGCAP.items():
                out.append(f"\n[INSERT IMAGE: eval/figs/{fn}]")
                out.append(cap)
            # skip until References
            while i < len(lines) and not lines[i].strip().startswith("## References"):
                i += 1
            continue
        if not s:
            out.append(""); i += 1; continue
        if s.startswith("## "):
            out.append("\n========== " + strip_md(s[3:]) + " =========="); i += 1; continue
        if s.startswith("### "):
            out.append("\n----- " + strip_md(s[4:]) + " -----"); i += 1; continue
        if s.startswith(">"):  # algorithm block lines
            out.append(strip_md(s.lstrip("> ").rstrip())); i += 1; continue
        if s.startswith("```"):
            i += 1
            out.append("[--- code/algorithm: put in a framed box in Word ---]")
            while i < len(lines) and not lines[i].strip().startswith("```"):
                out.append(lines[i]); i += 1
            out.append("[--- end ---]"); i += 1; continue
        if s.startswith("|"):  # table -> tab separated
            out.append("[--- TABLE: paste then Convert Text to Table, separator=Tab ---]")
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if not re.match(r"^\|[\s:|-]+\|?$", row):
                    cells = [strip_md(c.strip()) for c in row.strip("|").split("|")]
                    out.append("\t".join(cells))
                i += 1
            out.append("[--- end table ---]"); continue
        out.append(strip_md(s)); i += 1

    Path(OUT).write_text("\n".join(out))
    print("saved", OUT, "(", len(out), "lines )")


if __name__ == "__main__":
    main()
