# COMPASS v2 — designed experiment matrix (Blocks A–E)

This is the **headline** COMPASS v2 story: matched compute, matched candidate pools, and strong unguided baselines (`compass_v2_bench`). Legacy oracle / MSM tooling remains **Supporting Information**, not primary evidence.

## Runner

```bash
# Windows PowerShell (repo root)
$env:PYTHONPATH = "src"

# Full plan (default: blocks A–E + optional cold-start row). Uses --random-repeats 36.
python scripts/compass_v2_experiment_suite.py

# Smoke test (tiny steps/repeats)
python scripts/compass_v2_experiment_suite.py --quick

# Skip long Block E or SPIB Block B as needed
python scripts/compass_v2_experiment_suite.py --skip-block-e
python scripts/compass_v2_experiment_suite.py --skip-block-b
```

Artifacts land under `outputs/v2/plan/` with a **`manifest.json`** listing every subprocess command. If **`outputs/spib_a10_analysis/spib_a10_frame_assignments.npz`** is missing, Block **B** is **skipped automatically** (unless you pass `--require-spib`, which fails loudly).

### Optional cold-start sensitivity

```bash
python scripts/compass_v2_experiment_suite.py --blocks cold --quick
```

Adds `outputs/v2/plan/block_a/compass_tica2d_cs4500` (canonical alternate cold / seed from the plan notes).

## Block definitions

| Block | Intent | Policies / factors | Outputs |
|-------|--------|----------------------|---------|
| **A** | Adaptive scoring vs fair random + prior ablation + chronological | `compass`, **`random`**, `uniform_prior`, `chrono`; TICA2D; `--run-random-summary` on stochastic rows; **M3**; shared cold start | `outputs/v2/plan/block_a/*` |
| **B** | Same budget, **map_kind** geometry (no cross-map JSD ranking) | `compass` **SPIB1D** vs Block A TICA2D | `outputs/v2/plan/block_b/compass_spib1d` |
| **C** | Chunk horizon at fixed sampled budget | `chunk_frames=20`, `n_steps=40` vs baseline `10×80` (scaled in `--quick`) | `outputs/v2/plan/block_c/compass_chunk20` |
| **D** | Equivalence → candidate pool stress | `compass`, **M7** (override with `--eq-alt M5`) vs M3 baseline elsewhere | `outputs/v2/plan/block_d/compass_M7` |
| **E** | Longer budget if short runs are ambiguous | TICA2D `compass`, longer `n_steps` | `outputs/v2/plan/block_e/compass_long` |

**Fairness:** same `two_phase` / `information_gain` / `balanced` defaults, same pool construction, random baseline = **Monte Carlo** repeats from `outputs/v2/oracle` equivalence index.

## Reporting

After runs exist:

```bash
python scripts/compass_v2_reporting.py
```

Writes **`outputs/v2/plan/report/`**:

- `final_budget_by_run.csv` — per-run final A-layer + B-layer metrics and random final means  
- `ttt_lattice.csv` — time-to-threshold / speedup vs random (censoring preserved)  
- `fig2_speedup_block_a_compass.png` — example speedup bars  
- `fig3_tica_vs_spib_effect_bars.png` — **only if** Block B completed (effect-style deltas; no absolute cross-map comparison)

**Tier 1 Fig 1 (learning curves):** each bench directory already contains **`compass_replay_curves.png`** (map + observable panels) from `plot_curves` in `src/compass/replay_enhanced.py`.

### Physical post hoc trace (optional)

```bash
python scripts/compass_v2_physical_traces.py --run-dir outputs/v2/plan/block_a/compass_tica2d
```

Produces mean **`a8q2_nhb`** over observed frames vs `sampled_ns` (evaluation-only).

## Prerequisites

- **`outputs/v2/oracle/`** — equivalence choice + `eq_M3_k{K}.npz` (suite calls `--build-eq-only` for M3 / `--eq-alt` when files are missing).  
- **SPIB Block B:** `outputs/spib_a10_analysis/spib_a10_frame_assignments.npz` (+ SPIB model dir as in `build_spib_atlas`).

## Relation to the thesis / report

Use this matrix for the **Selected Experiments** section; park large exploratory sweeps and MSM-primary narratives under **SI**, per the first-principles plan.
