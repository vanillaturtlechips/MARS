"""Fill the KROS Word template with paper_ko.md content.

Reuses the sample's styles/page setup (hstyle1=heading, hstyle0=body,
hstyle3=abstract/keywords). Replaces the title block, drops the sample body,
and emits our sections + tables + figures + references.

    python3 docs/build_docx.py
Output: /home/user/Downloads/MARS_JKROS_draft.docx
Final font/column/equation polish is done by hand in Word.
"""
import re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt

SAMPLE = "/home/user/Downloads/KROS_Revised_Sample_수정.docx"
OUT = "/home/user/Downloads/MARS_JKROS_draft.docx"
KO = Path(__file__).parent / "paper_ko.md"
FIGDIR = Path(__file__).parent.parent / "eval" / "figs"

KO_TITLE = "물류 로봇 fleet을 위한 LLM 감독 에이전트의 결정론적 검증: 다중 모델 연구"
EN_TITLE = "Deterministic Validation of LLM Supervisory Agents for Warehouse Robot Fleets: A Multi-Model Study"
KO_AUTHOR = "이 명 일†"
EN_AUTHOR = "Myong-Il Lee†"
KEYWORDS = "Keywords: LLM agents, Retrieval-augmented generation, Deterministic validation, Warehouse robot fleet, AI safety"
AFFIL = "†Corresponding author: Student, Dept. of Cyber Security, Korea Polytechnic University Gangseo Campus, Seoul, Korea (2220110150@office.kopo.ac.kr)"
FIGS = [
    ("fig_mm_rag.png", "Fig. 1. Diagnosis cause accuracy by model, RAG on vs. off (test, n=100)."),
    ("fig_mm_safety.png", "Fig. 2. Confident-wrong (unsafe) rate by model, RAG on vs. off."),
    ("fig_mm_intent.png", "Fig. 3. Intent defense-in-depth by model over 15 unsafe intents."),
    ("fig_gaming.png", "Fig. 4. Honest vs. acceptance-incentivized agent: the confidence hold collapses while confident-wrong stays near zero."),
]


def strip_md(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = t.replace("`", "")
    return t


def get_abstract():
    s = KO.read_text()
    ab = s.split("## Abstract\n", 1)[1].split("\n**Keywords")[0].strip()
    return " ".join(l.strip() for l in ab.split("\n") if l.strip())


def del_para(p):
    p._element.getparent().remove(p._element)


def main():
    doc = Document(SAMPLE)
    body_style = "hstyle0"; head_style = "hstyle1"; abs_style = "hstyle3"

    # --- 1. rewrite title block by matching the sample's placeholder text ---
    def setp(par, text):
        for r in list(par.runs):
            r.text = ""
        if par.runs:
            par.runs[0].text = text
        else:
            par.add_run(text)
    ab_text = "Abstract  " + get_abstract()
    for p in doc.paragraphs[:14]:
        t = p.text.strip()
        if "로봇학회논문지 논문 작성 방법" in t:        setp(p, KO_TITLE)
        elif t.startswith("Preparation of Papers"):     setp(p, EN_TITLE)
        elif "홍 길 동" in t:                            setp(p, KO_AUTHOR)
        elif t.startswith("Gildong Hong"):              setp(p, EN_AUTHOR)
        elif t.startswith("Abstract"):                  setp(p, ab_text)
        elif t.startswith("Keywords"):                  setp(p, KEYWORDS)
        elif t.startswith("※"):                         setp(p, "")
        elif t.startswith("1. Principal"):              setp(p, AFFIL)
        elif t.startswith("2. Manager"):                setp(p, "")
        elif t.startswith("†Associate"):                setp(p, "")

    # --- 2. delete sample body: everything from first "1. " heading to end ---
    start = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("1.") and p.style.name == head_style:
            start = i; break
    if start is not None:
        for p in list(doc.paragraphs)[start:]:
            del_para(p)
    for t in list(doc.tables):       # drop sample table
        t._element.getparent().remove(t._element)

    # --- 3. parse paper_ko body (from "## 1. 서론") and emit ---
    lines = KO.read_text().splitlines()
    bi = next(i for i, l in enumerate(lines) if l.startswith("## 1. 서론"))
    ei = next(i for i, l in enumerate(lines) if l.startswith("## Figures"))
    body = lines[bi:ei]

    i = 0
    while i < len(body):
        ln = body[i]
        s = ln.strip()
        if not s:
            i += 1; continue
        if s.startswith("## "):
            doc.add_paragraph(strip_md(s[3:]), style=head_style); i += 1; continue
        if s.startswith("### "):
            doc.add_paragraph(strip_md(s[4:]), style=head_style); i += 1; continue
        if s.startswith("```"):       # code block -> monospace
            i += 1; buf = []
            while i < len(body) and not body[i].strip().startswith("```"):
                buf.append(body[i]); i += 1
            i += 1
            par = doc.add_paragraph(style=body_style)
            run = par.add_run("\n".join(buf)); run.font.name = "Courier New"; run.font.size = Pt(8)
            continue
        if s.startswith("|"):         # markdown table
            tbl_lines = []
            while i < len(body) and body[i].strip().startswith("|"):
                tbl_lines.append(body[i].strip()); i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in tbl_lines
                    if not re.match(r"^\|[\s:|-]+\|?$", r)]
            if rows:
                t = doc.add_table(rows=len(rows), cols=len(rows[0]))
                t.style = "Table Grid"
                for r, row in enumerate(rows):
                    for c, val in enumerate(row):
                        if c < len(t.rows[r].cells):
                            t.rows[r].cells[c].text = strip_md(val)
            continue
        # caption line (Table N / **Table)
        doc.add_paragraph(strip_md(s), style=body_style); i += 1

    # --- 4. figures ---
    doc.add_paragraph("Figures", style=head_style)
    for fn, cap in FIGS:
        fp = FIGDIR / fn
        if fp.exists():
            doc.add_picture(str(fp), width=Inches(3.0))
            doc.add_paragraph(cap, style=body_style)

    # --- 5. references ---
    refs_start = next(i for i, l in enumerate(lines) if l.strip() == "## References")
    doc.add_paragraph("References", style=head_style)
    for l in lines[refs_start+1:]:
        s = l.strip()
        if s.startswith("["):
            doc.add_paragraph(strip_md(s), style=body_style)

    doc.save(OUT)
    print("saved", OUT)
    print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))


if __name__ == "__main__":
    main()
