# Paper workspace (`paper/`)

All **LaTeX sources**, the shared **bibliography**, and **`latex_fragments/`** (including the generated SI sweep table) live here. **Figures** remain in the repo-root **`figures/`** directory; each main file sets `\graphicspath{{../figures/}}` so `\includegraphics{…}` resolves correctly **when you compile from this directory**.

## Files

| File | Role |
|------|------|
| `COMPASS_BTP_REPORT.tex` | IIT Delhi BTP report master (inputs snippet + SI fragment + natbib) |
| `COMPASS_BTP_CHAPTERS_SNIPPET.tex` | Main-body chapters for the BTP report |
| `COMPASS_JCTC_MANUSCRIPT.tex` | ACS/JCTC-style two-column manuscript (achemso) |
| `COMPASS_JCTC_MANUSCRIPT_ONLINE_TARGET.tex` | Alternate “online target” manuscript variant |
| `COMPASS_JCTC_SUPPORTING_INFORMATION.tex` | Stand-alone SI PDF (also mirrored into BTP appendix via fragment) |
| `COMPASS_JCTC_REFERENCES.bib` | Shared bibliography |
| `latex_fragments/` | `COMPASS_BTP_SI_BODY.tex`, generated `si_sweep_table.tex` |

PDFs built by TeX are normally written **next to these `.tex` files** (`COMPASS_BTP_REPORT.pdf`, etc.).

## Builds

From **repository root**:

- **JCTC main + SI:** `pwsh -File scripts/compile_btp_report.ps1`  
  (runs `scripts/gen_si_sweep_table.py`, then Tectonic or `pdflatex`/`bibtex` **with cwd = `paper/`**).

From **`paper/`**:

- **BTP report:** e.g. `tectonic --reruns 8 COMPASS_BTP_REPORT.tex` or `pdflatex` + `bibtex` + `pdflatex`×2.

## `COMPASS_BTP_CHAPTERS_SNIPPET.tex` vs `COMPASS_JCTC_MANUSCRIPT.tex`

The snippet has been **edited for the BTP structure** (for example a dedicated Literature Survey chapter and BTP cross-references). It is **not** guaranteed to match a mechanical extract from `COMPASS_JCTC_MANUSCRIPT.tex`.

`scripts/build_btp_report_tex.py` regenerates the snippet **from** the JCTC manuscript and will **overwrite** local edits—run it only when you **intend** to re-sync from JCTC.

## ACS auxiliary bibliography

If you use achemdso/achemso, the engine may create **`acs-*.bib`** next to the manuscript. Those files are **regenerated** and are gitignored under `paper/acs-*.bib`.
