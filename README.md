# BTP_2_Calm — COMPASS peptide MD / adaptive sampling

BTP / thesis-scale project: **COMPASS** (Conformation-Map Optimized Mutant Prioritized Adaptive Sequential Sampling) for A10→A8Q2 peptide benchmarks, OpenMM/CHARMM36m, retrospective replay, MSM-style diagnostics, and **COMPASS v2** causal replay.

## Quick orientation

| Topic | Where |
|--------|--------|
| **Current code + paths** | [`docs/CURRENT_PIPELINE.md`](docs/CURRENT_PIPELINE.md) |
| Doc index | [`docs/README.md`](docs/README.md) |
| LaTeX / PDFs | [`paper/README.md`](paper/README.md) |
| IITD BTP/MTP requirements PDF | [`docs/assets/IITD_BTP_MTP_report_details.pdf`](docs/assets/IITD_BTP_MTP_report_details.pdf) |
| Archived outputs & old markdown | `archive/` (ignored by git — keep locally) |
| External benchmark downloads | `data/external_benchmarks/` (ignored by git — leave as-is) |

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

OpenMM / CUDA notes: [`docs/OPENMM_INSTALL.md`](docs/OPENMM_INSTALL.md). Full optional stack: `requirements-optional.txt`.

## Main workflows

- **Ground-truth MD:** `python -u -m src.simulate --system both --phase all` (writes under `outputs/btp_equil/<A10|A8Q2>/`; see `docs/EXPERIMENT_PROTOCOL.md`). Post-production analysis: `Analysis_FES_Transitions.ipynb`
- **COMPASS v2:** see [`docs/COMPASS_V2_REPRODUCTION.md`](docs/COMPASS_V2_REPRODUCTION.md); drivers include `scripts/compass_v2_bench.py`
- **JCTC manuscript build** (from repo root): `pwsh -File scripts/compile_btp_report.ps1`
- **BTP report PDF:** `cd paper` then run Tectonic or `pdflatex`/`bibtex` on `COMPASS_BTP_REPORT.tex` (see `paper/README.md`)

Figures for LaTeX live in **`figures/`** at repo root; TeX sources live in **`paper/`**.

## Git hygiene

- `outputs/`, `archive/`, `data/external_benchmarks/`, `.venv/`, and LaTeX aux files under `paper/` are listed in `.gitignore` — commit **code + docs + `paper/*.tex`/`*.bib`**, not large local trees.

## Repository layout (short)

- `src/` — simulation/analysis/COMPASS modules (`src/compass/v2/` is the headline benchmark layer)
- `scripts/` — experiment runners, plotting, RunPod helpers, `gen_si_sweep_table.py`, `compile_btp_report.ps1`
- `paper/` — BTP + JCTC LaTeX, bibliography, `latex_fragments/`
- `docs/` — protocol, findings, reproduction notes
- `figures/` — PNG assets referenced from `paper/`
