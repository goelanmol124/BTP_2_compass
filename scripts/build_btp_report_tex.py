"""One-off helper: extract and reshape COMPASS_JCTC_MANUSCRIPT.tex body for BTP report."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SRC = PAPER / "COMPASS_JCTC_MANUSCRIPT.tex"
OUT = PAPER / "COMPASS_BTP_CHAPTERS_SNIPPET.tex"


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    m_doc = re.search(r"\\begin\{document\}", text)
    m_end = re.search(r"\\end\{document\}", text)
    assert m_doc and m_end
    body = text[m_doc.end() : m_end.start()]

    body = re.sub(
        r"\\begin\{tocentry\}.*?\\end\{tocentry\}",
        "",
        body,
        flags=re.DOTALL,
    )

    # Drop ACS abstract block (repeated on title front matter)
    body = re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}\s*",
        "",
        body,
        flags=re.DOTALL,
    )

    # Remove trailing sections handled elsewhere
    body = re.sub(
        r"\\section\{Associated Content\}.*?(?=\\section\{Author Information\}|\\section\{Acknowledgments\}|\\section\*\{Abbreviations\}|\\bibliography)",
        "",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"\\section\{Author Information\}.*?(?=\\section\{Acknowledgments\}|\\section\*\{Abbreviations\}|\\bibliography)",
        "",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"\\section\{Acknowledgments\}.*?(?=\\section\*\{Abbreviations\}|\\bibliography)",
        "",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"\\section\*\{Abbreviations\}.*?\\bibliography",
        "",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"\\bibliography\{COMPASS_JCTC_REFERENCES\}\s*",
        "",
        body,
    )
    body = body.replace("{COMPASS_JCTC_REFERENCES}", "").strip()
    # Drop manuscript-only underfull settings
    body = re.sub(
        r"(?ms)^% Two-column layout:.*?\n\\hbadness=10000\n\\vbadness=10000\n\n",
        "",
        body,
    ).strip()

    # Promote main sections to chapters
    body = body.replace("\\section{Introduction}", "\\chapter{Introduction}")
    body = body.replace("\\section{Theory and Background}", "\\chapter{Theory and Background}")
    body = body.replace("\\section{Methodology}", "\\chapter{Methodology}")

    # Merge Results + Discussion into one numbered chapter "Analysis" (BTP style)
    body = body.replace("\\section{Results}", "\\chapter{Results and Discussion}\\label{chap:results}")
    body = body.replace("\\section{Discussion}", "\\section{Discussion}")

    body = body.replace("\\section{Conclusions}", "\\chapter{Conclusions}")

    # Promote heading levels (deepest first): subsubsection -> subsection -> section
    body = re.sub(r"\\subsubsection\{", r"\\subsectionXXXX{", body)
    body = re.sub(r"\\subsection\{", r"\\section{", body)
    body = body.replace("\\subsectionXXXX{", "\\subsection{")

    disc_subs = [
        "A Coherent Interpretation",
        "Implications for Adaptive MD",
        "Why Random Is a Strong Baseline",
        "Design Principle: Control Prior Trust",
        "Scope and Limitations",
    ]
    for title in disc_subs:
        body = body.replace(
            "\\section{%s}" % title,
            "\\subsection{%s}" % title,
        )

    body = body.replace(
        "\\chapter{Theory and Background}",
        "\\chapter{Theory and Background}\\label{chap:theory}",
    )
    body = body.replace(
        "\\chapter{Methodology}",
        "\\chapter{Methodology}\\label{chap:methods}",
    )
    body = body.replace(
        "\\chapter{Conclusions}",
        "\\chapter{Conclusions}\\label{chap:conclusions}",
    )
    body = body.replace(
        "The remainder of this report introduces the theoretical framework (Section~2), describes the experimental protocol (Section~3), reports the main retrospective results (Section~4), discusses interpretation and scope (Section~5), and concludes (Section~6).",
        r"The remainder of this report introduces the theoretical framework (Chapter~\ref{chap:theory}), describes the experimental protocol (Chapter~\ref{chap:methods}), reports the main retrospective results and discussion (Chapter~\ref{chap:results}), and concludes (Chapter~\ref{chap:conclusions}).",
    )

    # Single-column floats
    body = body.replace("\\begin{figure*}", "\\begin{figure}")
    body = body.replace("\\end{figure*}", "\\end{figure}")
    body = body.replace("\\begin{table*}", "\\begin{table}")
    body = body.replace("\\end{table*}", "\\end{table}")
    body = re.sub(r"\\includegraphics\[width=0\.92\\linewidth\]", r"\\includegraphics[width=0.95\\textwidth]", body)

    OUT.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(body)} chars)")


if __name__ == "__main__":
    main()
