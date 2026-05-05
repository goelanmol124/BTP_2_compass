# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**btp_2_compass** implements **COMPASS** (Conformation-Map Optimized Mutant Prioritized Adaptive Sequential Sampling) — a framework for accelerating molecular dynamics (MD) simulations of peptide helix-coil transitions. The scientific goal: test whether a TICA collective variable (CV) trained on deca-alanine (A10) transfers to a glutamine-substituted variant (A8Q2), and whether it accelerates free energy surface convergence vs. random seeding.

Two peptide systems at 330 K:
- **A10:** Ac-Ala₁₀-NHMe (CV training source)
- **A8Q2:** Ac-Ala₈Gln₂-NHMe (Gln at positions 3 & 8, transfer target)

Start with **`docs/CURRENT_PIPELINE.md`** for code navigation and `docs/EXPERIMENT_PROTOCOL.md` for full scientific protocol.

## Common Commands

```bash
# Full MD pipeline (structure → equil → 500 ns production)
python -u -m src.simulate --system both --phase all

# Ground-truth analysis (FES, helix%, kinetics)
# Run Analysis_FES_Transitions.ipynb OR:
python -m src.analysis

# Train TICA on A10
python -m src.tica_train --output-dir outputs/tica_artifacts_a10

# Assess CV transfer to A8Q2
python -m src.tica_transfer --source-model outputs/tica_artifacts_a10 \
  --target-system A8Q2 --output-dir outputs/transfer_a8q2

# COMPASS v2 equivalence validation (run once, picks best method)
python scripts/compass_v2_validate_equivalence.py --methods M3,M5,M7,M8 --anchors 200

# Build equivalence index
python scripts/compass_v2_bench.py --build-eq-only --eq-method M3 --eq-k 40

# Run COMPASS v2 benchmark
python scripts/compass_v2_bench.py \
  --map-kind tica2d --schedule two_phase \
  --policy compass --eq-method M3 --eq-k 40 \
  --output-dir outputs/v2/benchmark_run --n-steps 80

# Quick smoke tests
python scripts/seed_pipeline_smoke.py
python scripts/qc_iter1_snapshot.py

# Build thesis/manuscript PDF (PowerShell)
pwsh -File scripts/compile_btp_report.ps1
```

## Architecture

### Module Map

```
src/
├── config.py             — Single source of truth for ALL simulation parameters
├── simulate.py           — Phases 1–4: structure prep → solvation → equil → 500 ns prod MD
├── analysis.py           — Phase 5: FES, helix%, N_HB, RMSD, DSSP analysis
├── tica_train.py         — Phase 6: TICA eigenvector training on A10
├── tica_transfer.py      — Phase 7: CV transfer fidelity to A8Q2
├── tica_featurizers.py   — 40-D sin/cos backbone dihedral features
├── adaptive_seeding.py   — Phase 8: adaptive MD with COMPASS seeding
├── spib_lib.py           — SPIB neural network (PyTorch)
├── spib_features.py      — 85-D SPIB features (45 Cα 1/r + 40 sin/cos)
├── metrics.py            — FES, helix% metrics
└── utils.py              — Platform selection, DOF calc, trajectory I/O

src/compass/
├── online_state.py       — ObservedTargetState, TrajectorySegments dataclasses
├── msm.py                — MSM-style Markov state model
├── replay_enhanced.py    — Enhanced oracle replay (chunk scoring, ATLAS, plots)
├── replay_online.py      — Online replay (MSM + score computation)
└── v2/
    ├── bench.py          — COMPASS v2 main: causal archive playback
    ├── policy.py         — Policies: compass, uniform_prior, random, min_count, farthest, chrono
    ├── equivalence.py    — Equivalence methods M1–M8 (default: M3 torsion space)
    ├── equivalence_index.py — k-NN index build/load
    ├── evaluate.py       — Milestone tracking, effective sample size
    └── validate_equivalence.py — Oracle equivalence validation
```

### Data Flow

1. `simulate.py` → DCDs + topologies in `outputs/btp_equil/{A10|A8Q2}/`
2. Featurization → sin/cos backbone dihedrals, RMSD, N_HB
3. `tica_train.py` → eigenvectors in `outputs/tica_artifacts_a10/`
4. `equivalence_index.py` → k-NN archive index for causal replay
5. `bench.py` → causal forward simulation from archive seeds
6. `policy.py` → scores next seed to pick
7. `evaluate.py` → FES convergence metrics, acceleration factor

### Key Conventions

- **`src/config.py` is the single source of truth** — all 330 K, CHARMM36m, timestep (2 fs), cutoff (1.2 nm), and duration constants live here. Never hardcode simulation parameters elsewhere.
- **Deferred imports:** Heavy optional deps (openmm, mdtraj, pymongo) use `try/except` guards or `TYPE_CHECKING` — allows importing without a full GPU stack.
- **Dataclass state objects:** `ReplayConfig` is immutable and passed through the full bench → policy → scoring pipeline.
- **All scripts have `main(argv=None)`** for both programmatic and CLI use.
- **Repo-root-aware paths:** Scripts use `Path(__file__).resolve().parents[N]` for sys.path; never depend on CWD.
- **Equivalence default is M3** (torsion space k-NN); see `docs/EQUIVALENCE_STRATEGY.md` for rationale.
- **Burn-in:** First 25 ns (250 frames @ 100 ps/frame) stripped before analysis — hardcoded in `analysis.py` and `tica_train.py`.
- **Helix detection:** Multi-criteria — N_HB ≥ 3, RMSD < 0.20 nm vs. ideal helix, DSSP ≥ 50% 'H'.

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
# For OpenMM + CUDA: see docs/OPENMM_INSTALL.md
pip install 'openmm[cuda12]'
```

Optional deps (MongoDB, PDBFixer, Jupyter): `pip install -r requirements-optional.txt`

**Environment variables:**
- `MONGO_URI` — MongoDB connection for distributed RunPod coordination (optional)
- `CUDA_VISIBLE_DEVICES` — GPU selection

## Output Directory Structure

```
outputs/
├── btp_equil/{A10,A8Q2}/       — DCDs, topologies, logs from simulate.py
├── tica_artifacts_a10/          — tica_mean.npy, tica_eigenvectors.npy, etc.
├── transfer_a8q2/               — CV transfer metrics
├── spib_a10_model/              — SPIB weights, scaler
├── phase5_{A10,A8Q2}/           — FES plots, kinetics CSVs
└── v2/
    ├── oracle/                  — equivalence indices, equivalence_choice.json
    └── {experiment_name}/       — benchmark results CSVs, plots
```

## Distributed Execution (RunPod)

- `scripts/pod2_orchestrate.py` — MongoDB-backed job coordination
- `scripts/runpod_full_md_analysis.py` — remote analysis on GPU pods
- `scripts/sync_runpod_missing_outputs.py` — fetch results

## Paper/Report

LaTeX source in `paper/`:
- `COMPASS_BTP_REPORT.tex` — IIT Delhi BTP report
- `COMPASS_JCTC_MANUSCRIPT.tex` — journal manuscript

Build: `pwsh -File scripts/compile_btp_report.ps1`

Figures: `scripts/make_manuscript_figures.py`
