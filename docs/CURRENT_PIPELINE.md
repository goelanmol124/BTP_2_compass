# Current pipeline (trust this overview first)

This file reflects the **Python modules and entrypoints that exist in `src/` and `scripts/` as of the repo cleanup**. Older markdown narratives under `archive/docs_deprecated/` or swept artifacts under `archive/outputs_legacy/` may describe superseded layouts.

## Molecular dynamics and analysis (physics backbone)

- **MD driver:** `python -u -m src.simulate` (`src/simulate.py`, constants in `src/config.py`). **Analysis notebook:** `Analysis_FES_Transitions.ipynb` (expects production DCD + `*_npt_final.pdb` paths you configure in that notebook).
- **Protocol:** `docs/EXPERIMENT_PROTOCOL.md`.
- **Primary outputs (still under `outputs/`):**
  - `outputs/btp_equil/` — equilibration / production tops and trajectories.
  - `outputs/phase5_a10/`, `outputs/phase5_a8q2/` — phase-5 summaries referenced by `docs/FINDINGS.md`.
  - `outputs/full_md_analysis/`, `outputs/deep_analysis/` — longer analyses tied to adaptive seeding / MD comparisons.
  - `outputs/transfer_a8q2/`, `outputs/tica_artifacts_a10/`, `outputs/artifacts/` — transfer pipeline and featurization caches.

## COMPASS v2 (headline adaptive replay benchmark)

- **Modules:** `src/compass/v2/` (`bench.py`, `policy.py`, `equivalence.py`, `evaluate.py`, `validate_equivalence.py`, …).
- **CLI wrappers:** `scripts/compass_v2_bench.py`, `scripts/compass_v2_validate_equivalence.py`, `scripts/compare_v2_policy_baselines.py`, `scripts/coverage_acceleration_benchmark.py`.
- **Outputs:** `outputs/v2/` (including oracle equivalence caches under `outputs/v2/oracle/` when built).
- **Reproduction notes:** `docs/COMPASS_V2_REPRODUCTION.md`, `docs/EQUIVALENCE_STRATEGY.md`.

## Retrospective replay substrate (still imported by v2)

- **`src/compass/replay_enhanced.py`** — enhanced oracle replay (chunk scoring, schedules, SPIB/TICA maps, plots).
- **`src/compass/replay_online.py`**, **`src/compass/msm.py`** — MSM-flavoured emulator path and diagnostics (also used by plotting / comparison scripts).
- **`src/compass/replay.py`** — legacy lite replay; **deprecated** as a headline benchmark (DeprecationWarning at CLI). `scripts/compass_experiments.py` still shells out to it for the original eleven-run sweep layout.

## Sweep drivers (write new data under `outputs/…` when rerun)

- **`scripts/compass_experiments_enhanced.py`** — enhanced retrospective sweep (historical default root `outputs/compass_sweep2/`).
- **`scripts/compass_experiments_online.py`** — online-style sweep (`outputs/compass_sweep_online/` style roots).
- **`scripts/compass_experiments.py`** — legacy `-m src.compass.replay` experiments under `outputs/compass/`.

Completed sweep trees from earlier work now live under **`archive/outputs_legacy/`** (see `docs/OUTPUTS_ARCHIVE_LAYOUT.md`).

## Manuscript / BTP LaTeX

All TeX, bibliography, and generated fragments live under **`paper/`** (figures remain in repo-root **`figures/`**). See **`paper/README.md`**.

- **BTP report:** `paper/COMPASS_BTP_REPORT.tex`, body `paper/COMPASS_BTP_CHAPTERS_SNIPPET.tex`, `paper/latex_fragments/` → PDF `paper/COMPASS_BTP_REPORT.pdf`.
- **JCTC draft:** `paper/COMPASS_JCTC_MANUSCRIPT.tex`, SI `paper/COMPASS_JCTC_SUPPORTING_INFORMATION.tex` — build via `scripts/compile_btp_report.ps1` (runs from repo root; compiles inside `paper/`).
- **Online-target variant:** `paper/COMPASS_JCTC_MANUSCRIPT_ONLINE_TARGET.tex` (+ PDF alongside).
- **SI sweep table generator:** `scripts/gen_si_sweep_table.py` writes `paper/latex_fragments/si_sweep_table.tex` from archived `archive/outputs_legacy/compass_sweep2/sweep/sweep2_results.csv`.

## SPIB

- **Library code:** `src/spib_lib.py`, `src/spib_features.py` (do **not** archive — imports depend on them).
- **Trained models / analyses** from earlier runs were moved to **`archive/outputs_legacy/spib_*`**. New training runs can repopulate `outputs/spib_*` using `scripts/train_spib_a10.py` and related ratchet scripts.
