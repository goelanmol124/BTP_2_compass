# COMPASS — adaptive sampling & CV transfer for helix–coil peptides

Research codebase for studying **slow helix↔coil dynamics** in short peptides using **OpenMM** MD, **TICA** collective variables, and **COMPASS** (Conformation-Map Optimized Mutant Prioritized Adaptive Sequential Sampling) — a retrospective and **v2 causal replay** framework for prioritizing simulations when you already have ground-truth trajectories.

Developed as a **Bachelor’s Thesis Project (BTP)** project at IIT Delhi; LaTeX for the institute report and a JCTC-style manuscript lives under [`paper/`](paper/).

---

## What this repository is about

Two capped peptides at **330 K** (parameter details live in `src/config.py`):

| System | Sequence | Role |
|--------|----------|------|
| **A10** | Ac-Ala₁₀-NHMe | Source system for TICA / CV training |
| **A8Q2** | Ac-Ala₈Gln₂-NHMe (Gln at positions 3 & 8) | Transfer target — same length, two polar substitutions |

**Central question:** Does a **TICA CV trained on A10** still describe the helix–coil reaction coordinate on **A8Q2**, and can **COMPASS-style seeding** improve exploration or metric convergence compared to naive baselines when evaluated against long reference MD?

The code walks through: build equilibration and production trajectories → featurize (backbone dihedrals, etc.) → train/assess transfer → run COMPASS / COMPASS v2 benchmarks and metrics. The full staged protocol (phases, timings, analysis choices) is in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

---

## Main components (how the code is organized)

1. **MD pipeline** — `src/simulate.py` drives structure prep through equilibration and production; shared constants are in **`src/config.py`** (avoid duplicating force field, timestep, temperature, duration elsewhere).
2. **Analysis** — FES, helix content, DSSP/H-bond summaries: `src/analysis.py` and the notebook **`Analysis_FES_Transitions.ipynb`**.
3. **TICA** — Train on A10 (`src/tica_train.py`), evaluate transfer to A8Q2 (`src/tica_transfer.py`); featurizers in `src/tica_featurizers.py`.
4. **Adaptive / COMPASS** — `src/adaptive_seeding.py` for adaptive MD experiments; **`src/compass/v2/`** holds the headline **COMPASS v2** stack (causal archive replay, policies, equivalence methods, evaluation).
5. **Optional scale-out** — RunPod / Mongo coordination scripts under **`scripts/`** (e.g. `pod2_orchestrate.py`) for remote GPU runs.

Module-level map and trust-first overview: **[`docs/CURRENT_PIPELINE.md`](docs/CURRENT_PIPELINE.md)**.

---

## Requirements & setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

- **CUDA OpenMM:** see [`docs/OPENMM_INSTALL.md`](docs/OPENMM_INSTALL.md) (e.g. `pip install 'openmm[cuda12]'` when appropriate).
- **Notebooks / PDB prep / Mongo:** optional packages in **`requirements-optional.txt`**.

Environment variable **`MONGO_URI`** is optional; it enables distributed coordination helpers when set.

---

## Quick commands

**Full MD backbone (both systems, all phases)**

```bash
python -u -m src.simulate --system both --phase all
```

Artifacts default under **`outputs/btp_equil/{A10|A8Q2}/`** (topology, DCD, logs).

**Ground-truth style analysis**

```bash
python -m src.analysis
```

Or configure paths in **`Analysis_FES_Transitions.ipynb`**.

**TICA on A10, then transfer check to A8Q2**

```bash
python -m src.tica_train --output-dir outputs/tica_artifacts_a10
python -m src.tica_transfer --source-model outputs/tica_artifacts_a10 \
  --target-system A8Q2 --output-dir outputs/transfer_a8q2
```

**COMPASS v2 — validate equivalence methods and run a benchmark**

```bash
python scripts/compass_v2_validate_equivalence.py --methods M3,M5,M7,M8 --anchors 200
python scripts/compass_v2_bench.py --build-eq-only --eq-method M3 --eq-k 40
python scripts/compass_v2_bench.py \
  --map-kind tica2d --schedule two_phase \
  --policy compass --eq-method M3 --eq-k 40 \
  --output-dir outputs/v2/benchmark_run --n-steps 80
```

Reproduction-focused notes: [`docs/COMPASS_V2_REPRODUCTION.md`](docs/COMPASS_V2_REPRODUCTION.md); equivalence rationale: [`docs/EQUIVALENCE_STRATEGY.md`](docs/EQUIVALENCE_STRATEGY.md).

**Build PDFs from LaTeX (PowerShell from repo root)**

```powershell
pwsh -File scripts/compile_btp_report.ps1
```

Figure generation and `paper/` layout: [`paper/README.md`](paper/README.md).

---

## Repository layout

| Path | Purpose |
|------|---------|
| **`src/`** | Simulation (`simulate.py`), analysis, TICA, adaptive seeding, `compass/` (v2 replay, policies, evaluation) |
| **`scripts/`** | CLIs for benchmarks, sweeps, plotting, RunPod helpers, manuscript compile |
| **`docs/`** | Protocol, findings, equivalence/reproduction, install notes ([`docs/README.md`](docs/README.md) indexes topics) |
| **`paper/`** | BTP report + JCTC manuscript TeX/Bib |
| **`figures/`** | Graphics referenced from LaTeX |

Large or machine-local trees (`outputs/`, `.venv/`, LaTeX build garbage, archives) are **gitignored** — commit source, configs, and small docs rather than trajectory outputs.

---

## Documentation map (start here)

| If you want to… | Read |
|-----------------|------|
| Understand the staged experiment end-to-end | [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) |
| See which scripts/modules are authoritative today | [`docs/CURRENT_PIPELINE.md`](docs/CURRENT_PIPELINE.md) |
| Reproduce COMPASS v2 numbers | [`docs/COMPASS_V2_REPRODUCTION.md`](docs/COMPASS_V2_REPRODUCTION.md) |
| Install OpenMM / optional stack | [`docs/OPENMM_INSTALL.md`](docs/OPENMM_INSTALL.md) |
| Index of remaining docs | [`docs/README.md`](docs/README.md) |

Older narratives and bulky outputs may live under **`archive/`** (typically not committed).

---

## Contributing & reuse

Issues and PRs welcome for clarity, reproducibility, and portability. Because simulation parameters are centralized in **`src/config.py`**, propose changes there rather than scattering magic numbers.

If you use this work academically, cite the thesis/manuscript once published and point to this repository for the exact code revision.
