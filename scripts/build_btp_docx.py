#!/usr/bin/env python3
"""Build COMPASS_BTP_REPORT.docx aligned with COMPASS_BTP_REPORT.tex.

- A4, 1\" margins; Times New Roman 12 pt body (matches LaTeX geometry + Normal style).
- Title page: optional header logos from ``figures/btp_template_assets`` (PNG placeholders).
- Front matter order matches the .tex file: Title → Declaration → Certificate → Abstract →
  Table of Contents → List of Figures → List of Tables → List of Abbreviations → body.
- Main text embeds figure PNGs from figures/ (same paths as \\includegraphics in the snippet).
- Supporting Information uses Appendix A, B, … labels like LaTeX \\appendix.

Requires: python-docx, pylatexenc. Run from repository root:
  python scripts/build_btp_docx.py
"""
from __future__ import annotations

import re
import string
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Inches, Pt
from pylatexenc.latex2text import LatexNodes2Text

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT_DOCX = PAPER / "COMPASS_BTP_REPORT.docx"
FIGURES = ROOT / "figures"
TEMPLATE_LOGOS = FIGURES / "btp_template_assets"

CONVERTER = LatexNodes2Text()

STRUCT_PAT = re.compile(
    r"(\\(?:chapter|section|subsection|paragraph)\*?(?:\[[^\]]*\])?\{[^}]*\})",
    re.DOTALL,
)
FIGURE_ENV_PAT = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.DOTALL)
GRAPHICS_PAT = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def extract_braced_command(tex: str, cmd: str) -> str | None:
    """First ``\\cmd{...}`` argument with balanced braces."""
    needle = "\\" + cmd + "{"
    i = tex.find(needle)
    if i < 0:
        return None
    i += len(needle)
    depth = 1
    start = i
    j = i
    while j < len(tex) and depth:
        if tex[j] == "{":
            depth += 1
        elif tex[j] == "}":
            depth -= 1
            if depth == 0:
                return tex[start:j]
        j += 1
    return None


def page_setup_a4_margins(doc: Document) -> None:
    for sec in doc.sections:
        sec.page_height = Cm(29.7)
        sec.page_width = Cm(21.0)
        sec.top_margin = Inches(1)
        sec.bottom_margin = Inches(1)
        sec.left_margin = Inches(1)
        sec.right_margin = Inches(1)


def set_times_body(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        try:
            st = doc.styles[name]
            st.font.name = "Times New Roman"
            if name == "Heading 1":
                st.font.size = Pt(14)
                st.font.bold = True
            elif name == "Heading 2":
                st.font.size = Pt(13)
                st.font.bold = True
            else:
                st.font.size = Pt(12)
                st.font.bold = True
        except KeyError:
            pass


def strip_tex_environments(tex: str, skip_figures: bool = False) -> str:
    envs = (
        "figure",
        "table",
        "equation",
        "align",
        "split",
        "gather",
        "enumerate",
        "itemize",
        "tabular",
        "tikzpicture",
    )
    for e in envs:
        if skip_figures and e == "figure":
            continue
        tex = re.sub(
            rf"\\begin\{{{re.escape(e)}\}}.*?\\end\{{{re.escape(e)}\}}",
            "\n\n",
            tex,
            flags=re.DOTALL,
        )
    tex = re.sub(
        r"\\begin\{longtable\}.*?\\end\{longtable\}",
        "\n[Table S — see PDF for full sweep metrics]\n",
        tex,
        flags=re.DOTALL,
    )
    return tex


def latex_plain(tex: str, *, strip_figures: bool = False) -> str:
    tex = strip_tex_environments(tex, skip_figures=strip_figures)
    try:
        return CONVERTER.latex_to_text(tex)
    except Exception:
        return tex


def parse_heading_cmd(cmd: str) -> tuple[int, str] | None:
    m = re.match(
        r"\\(chapter|section|subsection|paragraph)\*?(?:\[[^\]]*\])?\{([^}]*)\}",
        cmd.strip(),
    )
    if not m:
        return None
    kind, title = m.group(1), m.group(2).strip()
    level = {"chapter": 1, "section": 2, "subsection": 3, "paragraph": 4}[kind]
    return level, title


def add_center_title_line(doc: Document, text: str, *, bold: bool = False, size_pt: float = 12) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size_pt)
    r.font.name = "Times New Roman"


def add_title_page(doc: Document) -> None:
    """Match COMPASS_BTP_REPORT.tex titlepage + optional IIT/template logos (Word template order)."""
    p0 = doc.add_paragraph()
    p0.paragraph_format.space_before = Pt(36)  # ~1.2 cm top spacing
    p0.paragraph_format.space_after = Pt(6)

    logo_paths = [TEMPLATE_LOGOS / f"image{i}.png" for i in range(1, 5)]
    if all(p.exists() for p in logo_paths):
        tbl = doc.add_table(rows=1, cols=4)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, lp in enumerate(logo_paths):
            cell = tbl.rows[0].cells[i]
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cp.add_run()
            run.add_picture(str(lp), width=Inches(1.28))
        doc.add_paragraph()
    else:
        doc.add_paragraph()

    # \Large\bfseries title (~14–16 pt; template uses emphasis)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(
        "COMPASS: Retrospective Benchmarking of Transferred Slow-Coordinate Priors "
        "for Adaptive Peptide Sampling"
    )
    r.bold = True
    r.font.size = Pt(14)
    r.font.name = "Times New Roman"
    t.paragraph_format.space_after = Pt(14)

    add_center_title_line(doc, "A PROJECT REPORT", bold=False, size_pt=13)
    doc.add_paragraph()
    add_center_title_line(
        doc,
        "Submitted as part of BBD452 BTech Major Project (BB1)",
        bold=False,
        size_pt=12,
    )

    # \vfill analogue — vertical space before author block
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(72)
    sp.paragraph_format.space_after = Pt(12)

    sb = doc.add_paragraph()
    sb.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = sb.add_run("Submitted by: ")
    r1.bold = True
    r1.font.name = "Times New Roman"
    r2 = sb.add_run("Anmol Goel    (Entry No. 2022BB11000)")
    r2.font.name = "Times New Roman"

    gd = doc.add_paragraph()
    gd.alignment = WD_ALIGN_PARAGRAPH.CENTER
    g1 = gd.add_run("Guided by: ")
    g1.bold = True
    g1.font.name = "Times New Roman"
    gd.add_run("Prof. Ishaan Gupta").font.name = "Times New Roman"

    doc.add_paragraph()
    dept = doc.add_paragraph()
    dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d1 = dept.add_run("DEPARTMENT OF BIOCHEMICAL ENGINEERING AND BIOTECHNOLOGY")
    d1.bold = True
    d1.font.name = "Times New Roman"
    dept.add_run().add_break()
    d2 = dept.add_run("INDIAN INSTITUTE OF TECHNOLOGY DELHI")
    d2.bold = True
    d2.font.name = "Times New Roman"

    doc.add_paragraph()
    add_center_title_line(doc, "May 2, 2026", bold=False, size_pt=12)
    doc.add_page_break()


def add_declaration(doc: Document) -> None:
    """Match enumerate in COMPASS_BTP_REPORT.tex."""
    h = doc.add_heading("Declaration", level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT

    intro = doc.add_paragraph("I certify that")
    intro.paragraph_format.space_after = Pt(6)

    items = [
        "The work contained in this report is original and has been done by me under the guidance of my supervisor.",
        "I have followed the guidelines provided by the Department in preparing the report.",
        "I have conformed to the norms and guidelines given in the Honor Code of Conduct of the Institute.",
        "Whenever I have used materials (data, theoretical analysis, figures, and text) from other sources, "
        "I have given due credit to them by citing them in the text of the report and giving their details in the references.",
    ]
    for it in items:
        try:
            p = doc.add_paragraph(it, style="List Number")
        except KeyError:
            p = doc.add_paragraph(it, style="List Paragraph")
        for run in p.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)

    doc.add_paragraph()
    doc.add_paragraph()
    sig = doc.add_paragraph()
    sig.add_run("Signed by:").bold = True
    name = doc.add_paragraph("Anmol Goel")
    name.paragraph_format.space_before = Pt(36)

    doc.add_page_break()


def add_certificate(doc: Document) -> None:
    h = doc.add_heading("Certificate", level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.add_paragraph(
        'It is certified that the work contained in this report titled "COMPASS: Retrospective '
        'Benchmarking of Transferred Slow-Coordinate Priors for Adaptive Peptide Sampling" is the '
        "original work done by Anmol Goel and has been carried out under my supervision."
    )
    doc.add_paragraph()
    doc.add_paragraph()
    s = doc.add_paragraph()
    s.add_run("Signed by:").bold = True
    prof = doc.add_paragraph()
    prof.add_run("Prof. Ishaan Gupta")
    prof.paragraph_format.space_before = Pt(36)

    doc.add_paragraph("Department of Biochemical Engineering and Biotechnology")
    doc.add_paragraph("Indian Institute of Technology Delhi")
    doc.add_paragraph()
    d = doc.add_paragraph()
    d.add_run("Date: ").bold = True
    d.add_run("May 2, 2026")
    doc.add_page_break()


def _extract_figure_caption(fig_tex: str) -> str:
    cap = extract_braced_command(fig_tex, "caption")
    if cap is None:
        return ""
    return latex_plain(cap).strip()


def collect_list_of_figures(tex: str) -> list[tuple[str, str]]:
    """(filename, caption text) in document order."""
    out: list[tuple[str, str]] = []
    for m in FIGURE_ENV_PAT.finditer(tex):
        block = m.group()
        gm = GRAPHICS_PAT.search(block)
        cap = _extract_figure_caption(block)
        if gm:
            out.append((gm.group(1).strip(), cap))
    return out


def collect_list_of_tables(tex: str) -> list[str]:
    caps: list[str] = []
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, re.DOTALL):
        cap = extract_braced_command(m.group(), "caption")
        if cap:
            caps.append(latex_plain(cap).strip())
    return caps


def build_toc_entries() -> list[str]:
    """Approximate PDF front matter + numbered chapters (matches single-file PDF)."""
    lines = [
        "Declaration",
        "Certificate",
        "Abstract",
        "Table of Contents",
        "List of Figures",
        "List of Tables",
        "List of Abbreviations",
    ]
    snip = (PAPER / "COMPASS_BTP_CHAPTERS_SNIPPET.tex").read_text(encoding="utf-8")
    main_ch = re.findall(r"\\chapter\{([^}]+)\}", snip)
    for i, t in enumerate(main_ch, start=1):
        lines.append(f"Chapter {i}  {t}")
    lines.append("Supporting Information")
    si = (PAPER / "latex_fragments" / "COMPASS_BTP_SI_BODY.tex").read_text(encoding="utf-8")
    si_ch = re.findall(r"\\chapter\{([^}]+)\}", si)
    for j, t in enumerate(si_ch, start=0):
        letter = string.ascii_uppercase[j]
        lines.append(f"Appendix {letter}  {t}")
    lines.extend(["Acknowledgements", "References"])
    return lines


def add_table_of_contents(doc: Document) -> None:
    doc.add_heading("Table of Contents", level=1)
    note = doc.add_paragraph(
        "Page numbers correspond to COMPASS_BTP_REPORT.pdf when printed from LaTeX; "
        "update pagination here if needed after editing."
    )
    note.runs[0].italic = True
    doc.add_paragraph()
    for entry in build_toc_entries():
        try:
            doc.add_paragraph(entry, style="List Bullet")
        except KeyError:
            doc.add_paragraph(entry)
    doc.add_page_break()


def add_list_of_figures(doc: Document, tex: str) -> None:
    doc.add_heading("List of Figures", level=1)
    figs = collect_list_of_figures(tex)
    for i, (_, cap) in enumerate(figs, start=1):
        doc.add_paragraph(f"Figure {i}. {cap}")
    doc.add_page_break()


def add_list_of_tables(doc: Document, tex: str) -> None:
    doc.add_heading("List of Tables", level=1)
    tabs = collect_list_of_tables(tex)
    for i, cap in enumerate(tabs, start=1):
        doc.add_paragraph(f"Table {i}. {cap}")
    doc.add_page_break()


def add_abbreviations_table(doc: Document) -> None:
    doc.add_heading("List of Abbreviations", level=1)
    rows = [
        ("COMPASS", "Composable Prior-Aware Sampling Strategy (retrospective replay and MSM emulator in this work)"),
        ("JSD", "Jensen–Shannon divergence"),
        ("MD", "Molecular dynamics"),
        ("MSM", "Markov state model"),
        ("SPIB", "State predictive information bottleneck"),
        ("TICA", "Time-lagged independent component analysis"),
    ]
    tbl = doc.add_table(rows=len(rows), cols=2)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, (abbr, desc) in enumerate(rows):
        tbl.rows[i].cells[0].text = abbr
        tbl.rows[i].cells[1].text = desc
        for c in tbl.rows[i].cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(12)
    doc.add_page_break()


def add_picture_with_caption(doc: Document, rel_path: str, caption: str, fig_num: list[int]) -> None:
    """rel_path from \\includegraphics (figures/…)."""
    path = FIGURES / rel_path.split("/")[-1]
    if not path.is_file():
        path = ROOT / rel_path
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_img.add_run()
    if path.is_file():
        run.add_picture(str(path), width=Inches(6.0))
    else:
        p_img.add_run(f"[Missing image file: {rel_path}]")

    fig_num[0] += 1
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cap.add_run(f"Figure {fig_num[0]}. {caption}")
    cr.italic = True
    cr.font.size = Pt(10)
    cap.paragraph_format.space_after = Pt(12)


def process_body_with_figures(
    doc: Document,
    tex: str,
    *,
    chapter_formatter,
    fig_counter: list[int],
) -> None:
    parts = STRUCT_PAT.split(tex)
    preamble = parts[0].strip()

    def flush_text_plain(sub: str) -> None:
        sub = strip_tex_environments(sub, skip_figures=True)
        plain = latex_plain(sub, strip_figures=True).strip()
        if plain:
            for para in re.split(r"\n\s*\n+", plain):
                if para.strip():
                    doc.add_paragraph(para.strip())

    def add_figure_block(doc: Document, fig_tex: str, fc: list[int]) -> None:
        gm = GRAPHICS_PAT.search(fig_tex)
        cap = _extract_figure_caption(fig_tex)
        if gm:
            add_picture_with_caption(doc, gm.group(1).strip(), cap, fc)
        elif cap:
            doc.add_paragraph(f"[Figure] {cap}")

    if preamble:
        pos = 0
        for fm in FIGURE_ENV_PAT.finditer(preamble):
            if fm.start() > pos:
                flush_text_plain(preamble[pos : fm.start()])
            add_figure_block(doc, fm.group(), fig_counter)
            pos = fm.end()
        if pos < len(preamble):
            flush_text_plain(preamble[pos:])

    i = 1
    ch_main = [0]
    ch_app = [0]
    while i < len(parts):
        cmd = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2
        parsed = parse_heading_cmd(cmd)
        if not parsed:
            continue
        level, title = parsed
        title_plain = latex_plain(title).strip()
        if level == 1:
            if chapter_formatter[0] == "main":
                ch_main[0] += 1
                display = f"Chapter {ch_main[0]}  {title_plain}"
            else:
                ch_app[0] += 1
                letter = string.ascii_uppercase[ch_app[0] - 1]
                display = f"Appendix {letter}  {title_plain}"
            p = doc.add_heading(display, level=1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif level == 2:
            doc.add_heading(title_plain, level=2)
        elif level == 3:
            doc.add_heading(title_plain, level=3)
        else:
            pr = doc.add_paragraph()
            pr.add_run(title_plain).bold = True

        pos = 0
        for fm in FIGURE_ENV_PAT.finditer(body):
            if fm.start() > pos:
                sub = body[pos : fm.start()]
                sub_p = strip_tex_environments(sub, skip_figures=True)
                pl = latex_plain(sub_p, strip_figures=True).strip()
                if pl:
                    for para in re.split(r"\n\s*\n+", pl):
                        if para.strip():
                            doc.add_paragraph(para.strip())
            add_figure_block(doc, fm.group(), fig_counter)
            pos = fm.end()
        if pos < len(body):
            sub = body[pos:]
            sub = strip_tex_environments(sub, skip_figures=True)
            pl = latex_plain(sub, strip_figures=True).strip()
            if pl:
                for para in re.split(r"\n\s*\n+", pl):
                    if para.strip():
                        doc.add_paragraph(para.strip())


def safe_abstract_from_file() -> str:
    raw = (PAPER / "COMPASS_BTP_REPORT.tex").read_text(encoding="utf-8")
    if "\\chapter*{Abstract}" not in raw:
        return ""
    chunk = raw.split("\\chapter*{Abstract}", 1)[1]
    chunk = chunk.split("\\cleardoublepage", 1)[0]
    chunk = re.sub(r"\\phantomsection\s*", "", chunk)
    chunk = re.sub(r"\\addcontentsline\{[^\}]+\}\{[^\}]+\}\{[^\}]+\}\s*", "", chunk)
    chunk = chunk.replace("\\noindent", "").strip()
    return chunk


def add_abstract_fixed(doc: Document) -> None:
    h = doc.add_heading("Abstract", level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    chunk = safe_abstract_from_file()
    if not chunk:
        raise RuntimeError("Could not parse Abstract from COMPASS_BTP_REPORT.tex")
    doc.add_paragraph(latex_plain(chunk).strip())
    doc.add_page_break()


def main() -> None:
    doc = Document()
    page_setup_a4_margins(doc)
    set_times_body(doc)

    add_title_page(doc)
    add_declaration(doc)
    add_certificate(doc)
    add_abstract_fixed(doc)

    chapters_tex = (PAPER / "COMPASS_BTP_CHAPTERS_SNIPPET.tex").read_text(encoding="utf-8")
    add_table_of_contents(doc)
    add_list_of_figures(doc, chapters_tex)
    add_list_of_tables(doc, chapters_tex)
    add_abbreviations_table(doc)

    fig_n = [0]
    process_body_with_figures(
        doc,
        chapters_tex,
        chapter_formatter=["main"],
        fig_counter=fig_n,
    )

    doc.add_page_break()
    doc.add_heading("Supporting Information", level=1).alignment = WD_ALIGN_PARAGRAPH.CENTER

    si_tex = (PAPER / "latex_fragments" / "COMPASS_BTP_SI_BODY.tex").read_text(encoding="utf-8")
    process_body_with_figures(
        doc,
        si_tex,
        chapter_formatter=["app"],
        fig_counter=fig_n,
    )

    doc.add_page_break()
    doc.add_heading("Acknowledgements", level=1)
    doc.add_paragraph(
        "The authors thank colleagues and mentors for discussions on adaptive sampling and molecular "
        "simulation. A.G. designed the COMPASS framework, implemented the retrospective replay and "
        "MSM-emulator benchmark code, ran the experiments, performed the analyses, and drafted "
        "the manuscript. Prof. Ishaan Gupta provided guidance, reviewed outputs, and suggested "
        "direction. The authors thank the Department of Biochemical Engineering and Biotechnology "
        "for providing the necessary knowledge and opportunity to undertake this work."
    )

    doc.add_page_break()
    doc.add_heading("References", level=1)
    doc.add_paragraph(
        "The numbered reference list is identical to COMPASS_BTP_REPORT.pdf (natbib + "
        "COMPASS_JCTC_REFERENCES.bib). Open the PDF for citation formatting, or paste the "
        "bibliography from the PDF / .bbl file."
    )

    doc.save(OUT_DOCX)
    print(f"Wrote {OUT_DOCX}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
