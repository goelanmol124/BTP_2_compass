<!-- Verify CSV paths under `outputs/` before citing numbers. Legacy sweep trees were moved to `archive/outputs_legacy/`. Current pipeline overview: `docs/CURRENT_PIPELINE.md`. -->


# Experimental Findings: Adaptive Seeding and TICA Transferability for Helix–Coil Transitions

**System:** Ac-Ala₁₀-NHMe (A10) and Ac-Ala₈Gln₂-NHMe (A8Q2), explicit solvent, OpenMM/CHARMM36m, Langevin dynamics at 300 K.

**Data sources:** All numbers in this document are extracted from machine-readable output artifacts (`outputs/full_md_analysis/`, `outputs/phase5_*/`, `outputs/transfer_a8q2/`, `outputs/deep_analysis/`). Each claim is annotated with the source file and field.

---

## 1. Effect of Gln Substitution on Helix Stability

### Claim

Replacing Ala₉–Ala₁₀ with Gln₉–Gln₁₀ destabilises the α-helix, raising the folding free energy by 0.17–0.19 kcal/mol and reducing the equilibrium helix fraction.

### Evidence

| Property | A10 extended | A10 helix | A8Q2 extended | A8Q2 helix |
|---|---|---|---|---|
| f_helix | 0.053 | 0.066 | 0.041 | 0.050 |
| ΔG_fold (kcal/mol) | +1.89 | +1.74 | +2.06 | +1.93 |

*Source: `phase5_a10/A10/A10_{extended,helix}_kinetics.csv` and `phase5_a8q2/A8Q2/A8Q2_{extended,helix}_kinetics.csv`, fields `f_helix` and `dG_fold_kcal`.*

ΔΔG (A8Q2 − A10) = +0.17 kcal/mol from extended starts, +0.19 kcal/mol from helix starts. The mean number of backbone hydrogen bonds is also lower: 0.343 for A8Q2 vs 0.394 for A10 (13% reduction).

*Source: `artifacts/a10_nhb.npy` mean = 0.394; `artifacts/a8q2_nhb.npy` mean = 0.343.*

### Strength

**Strong.** Consistent across two independent starting conformations, two independent observables (f_helix, NHB), and agrees with empirical helix propensity scales (Gln < Ala).

### Impact

Provides a quantitative benchmark (ΔΔG ~ 0.18 kcal/mol per two-residue substitution) for validating force-field accuracy in capturing sequence-dependent helix propensities of short capped peptides.

---

## 2. Asymmetric Mutation Effect on Folding vs Unfolding Kinetics

### Claim

The Gln substitution increases both folding and unfolding rates, but the effect is larger on unfolding, producing a net decrease in helix population. The magnitude of the folding-rate acceleration is start-state dependent.

### Evidence

| Rate | A10 ext | A10 helix | A8Q2 ext | A8Q2 helix |
|---|---|---|---|---|
| k_fold (ns⁻¹) | 0.082 | 0.109 | 0.222 | 0.113 |
| k_unfold (ns⁻¹) | 1.349 | 1.406 | 1.939 | 2.008 |
| n_fold events | 34 | 44 | 38 | 48 |

*Source: same kinetics CSVs, fields `k_f_ns_inv`, `k_u_ns_inv`, `n_fold`.*

From the extended start-state: k_fold increases 2.7-fold (0.082 → 0.222), k_unfold increases 1.44-fold (1.349 → 1.939). From the helix start-state: k_fold is essentially unchanged (0.109 → 0.113, 1.04-fold), while k_unfold still increases 1.43-fold (1.406 → 2.008).

The autocorrelation slow time-scale drops from 8.6–9.6 ns (A10) to 3.4–6.1 ns (A8Q2).

*Source: `acf_tau_slow_ns` in the same CSVs.*

### Strength

**Moderate.** Unfolding-rate increase is consistent across start-states (1.43–1.44x). Folding-rate acceleration from extended state (2.7x) is large but not reproduced from helix-start (1.04x), suggesting it may reflect barrier shape sensitivity to initial conditions rather than an intrinsic kinetic effect. Statistics are limited: 34–48 folding events per trajectory.

### Impact

Suggests the mutation lowers the helix-state well depth (faster escape) more than it lowers the folding barrier. The start-state dependence of k_fold indicates residual non-equilibrium effects at 475 ns per start-state and motivates longer production runs or enhanced sampling.

---

## 3. Random Seeding Achieves the Best Overall FES Convergence

### Claim

Among three adaptive-seeding strategies tested on A8Q2 (random, native-TICA, transferred-TICA), random seeding reaches the lowest JSD divergence from the ground-truth FES and full coverage of the 2D RMSD×NHB histogram.

### Evidence

Each strategy ran 6 seeds × 25 ns × 4 iterations = 600 ns total. Ground truth: pooled equilibrium f_helix = 0.048. Noise floor JSD² = 0.00855 (JSD between two independent equilibrium halves).

| Iter | ns | random JSD | native JSD | transferred JSD |
|---|---|---|---|---|
| 1 | 150 | 0.0256 | 0.0341 | 0.0238 |
| 2 | 300 | 0.0137 | 0.0234 | 0.0126 |
| 3 | 450 | 0.0086 | 0.0223 | 0.0093 |
| 4 | 600 | **0.0058** | 0.0196 | 0.0071 |

*Source: `full_md_analysis/full_metrics_long.csv`, field `jsd_vs_gt`.*

| Iter | random coverage | native coverage | transferred coverage |
|---|---|---|---|
| 3 | **1.000** | 0.908 | 0.882 |
| 4 | **1.000** | 0.921 | 0.947 |

*Source: same CSV, field `coverage_vs_gt`.*

Random reaches the noise floor at iter 3 (JSD = 0.0086 ≈ noise floor 0.00855) and stays below it at iter 4 (0.0058). Transferred reaches the noise floor at iter 4 (0.0071). Native never does (0.0196 at iter 4, 2.3× above noise floor).

The overall composite score (lower = better) at 600 ns: random 0.020, transferred 0.063, native 0.116.

*Source: same CSV, field `composite_lower_better`.*

### Strength

**Strong.** Monotonic improvement across all four iterations. JSD and coverage both favour random at convergence. Composite advantage is 3× over transferred and 6× over native.

### Impact

For this peptide system at the 150–600 ns adaptive-sampling budget, simple random seeding from the production trajectory outperforms CV-guided seeding for FES reconstruction. This is a negative result for the transferability hypothesis but a useful practical finding: random seeding provides a strong, low-effort baseline.

---

## 4. Transferred Seeding Gives the Most Accurate Helix Fraction Estimate

### Claim

Despite losing on JSD and coverage, transferred seeding produces the most accurate estimate of the equilibrium helix fraction at convergence.

### Evidence

| Iter | random f_est / error | native f_est / error | transferred f_est / error |
|---|---|---|---|
| 1 | 0.053 / 0.005 | 0.025 / 0.023 | 0.033 / 0.015 |
| 2 | 0.077 / 0.029 | 0.060 / 0.012 | **0.050 / 0.002** |
| 3 | 0.064 / 0.016 | **0.053 / 0.005** | 0.042 / 0.006 |
| 4 | 0.062 / 0.014 | 0.065 / 0.017 | **0.044 / 0.004** |

*Source: `full_md_analysis/full_metrics_long.csv`, fields `f_helix_estimate` and `f_helix_error_vs_gt`. Ground truth = 0.048.*

At 600 ns, transferred estimates f_helix = 0.044 (error 0.004, or 7.6% relative), vs random 0.062 (error 0.014, 28%) and native 0.065 (error 0.017, 36%).

Transferred wins 2 of 4 iterations (iter 2 and 4). Random wins iter 1. Native wins iter 3.

### Strength

**Moderate.** Transferred is best at convergence and its error is 3.7× smaller than random's. But the margin at individual iterations is small (0.002–0.006), and the win-count (2/4) is not dominant.

### Impact

This is the key finding supporting partial transfer utility: even when the transferred coordinate fails geometric/kinetic diagnostics, it biases the seed distribution toward a more balanced helix–coil ratio, avoiding the systematic oversampling of coil states that inflates f_helix estimates in random and native strategies. For thermodynamic observables (helix fraction), a "broken but directionally correct" CV still helps.

---

## 5. TICA Transfer from A10 to A8Q2 Fails Geometrically but Retains Helix-Content Signal

### Claim

The A10-trained TICA model does not transfer to A8Q2 by formal geometric and kinetic criteria, but the projected TIC-1 coordinate still tracks helix content on A8Q2 with |r| = 0.79.

### Evidence

| Transfer metric | Value | Pass threshold | Result |
|---|---|---|---|
| D₀ (subspace non-orthogonality) | 0.984 | < 0.7 | **FAIL** |
| D_τ (kinetic preservation) | 0.648 | ∈ [0.3, 0.7) | **MARGINAL** |
| D_KM (TIC-1 recoverability) | 14.04 | < 0.6 | **FAIL** |
| \|Pearson r\| (TIC-1 vs NHB) | 0.786 | ≥ 0.75 | **PASS** |

*Source: `transfer_a8q2/transfer_summary.json`, fields `D0`, `D_tau`, `D_KM`, `pearson_abs_r`.*

Overall verdict: 1 of 4 metrics pass.

Despite this formal failure, transferred seeding has the lowest integrated JSD (AUC = 5.61 ns·nats²) compared to random (5.70) and native (10.89).

*Source: `full_md_analysis/acceleration_summary.json`, field `auc_jsd_vs_ns`.*

The transferred/native gain ratio is 1.011–1.013 across iterations 2–4, meaning transferred consistently beats native by 1.1–1.3%.

*Source: `full_md_analysis/acceleration_summary.json`, `protocol_target_check[*].ratio_transferred_over_native`.*

### Strength

**Strong.** Four independent transfer metrics, and a head-to-head AUC comparison across three strategies and four iterations. The Pearson correlation has p ≈ 0 (computed on 9500 frames).

### Impact

This is the central scientific result. A TICA model can fail three out of four formal transferability tests and still provide seeding utility comparable to random — and superior to native retraining. This suggests that the bar for "useful transfer" is lower than the bar for "geometrically faithful transfer". For practical adaptive sampling, even a rough directional CV from a related peptide provides value, while expensive native TICA retraining on A8Q2 data actually *hurts* FES convergence (AUC 10.89 vs 5.61–5.70), presumably because it over-concentrates seeds near the folding barrier at the expense of coil-state coverage.

---

## 6. Native-TICA Seeding Generates More Transitions but Worse FES Coverage

### Claim

Native seeding produces the highest helix–coil transition rate but the worst FES convergence, revealing a trade-off between local barrier-crossing activity and global configurational coverage.

### Evidence

| Iter | random trans/ns | native trans/ns | transferred trans/ns |
|---|---|---|---|
| 1 | 0.147 | 0.187 | 0.200 |
| 2 | 0.180 | 0.180 | 0.213 |
| 3 | 0.182 | 0.204 | 0.182 |
| 4 | 0.153 | **0.260** | 0.157 |

*Source: `full_md_analysis/full_metrics_long.csv`, field `transitions_nhb_rate_per_ns`.*

At 600 ns, native has 0.26 transitions/ns — 1.7× higher than random (0.15) or transferred (0.16). Yet native has the highest JSD (0.0196) and lowest coverage (0.921).

### Strength

**Strong.** The transition rate advantage is large (70% more than random at iter 4) and the FES disadvantage is equally clear (JSD 3.4× worse than random).

### Impact

This quantifies the "barrier-crossing trap": a CV that is too well-tuned to the folding coordinate concentrates seeds near the transition state, generating many short fold/unfold events but systematically under-exploring the broader coil basin. For FES reconstruction, diversity of starting points (random seeding) matters more than transition-state proximity (native seeding).

---

## 7. A Single Dominant Slow Mode Governs Both Peptides

### Claim

TICA analysis of A10 reveals a clear spectral gap: at lag time = 10 frames (1 ns), only TIC-1 retains a high singular value, while TIC-2 through TIC-5 decay sharply. The TIC-1 direction is robust to lag-time choice.

### Evidence

| Component | Singular value (lag=1) | Singular value (lag=10) |
|---|---|---|
| TIC-1 | 0.936 | 0.720 |
| TIC-2 | 0.849 | 0.401 |
| TIC-3 | 0.823 | 0.364 |
| TIC-4 | 0.808 | 0.318 |
| TIC-5 | 0.801 | 0.290 |

*Source: `btp_equil/A10/tica_validation_run/a10_tica_singular_values.npy` (lag=1) and `btp_equil/A10/tica_validation_run_v2/a10_tica_singular_values.npy` (lag=10).*

At lag=1 all five TICs have similar singular values (0.80–0.94, no clear gap). At lag=10, TIC-1 (0.72) separates from TIC-2 (0.40), a ratio of 1.8:1. TIC-2 through TIC-5 cluster together (0.29–0.40).

The cosine similarity between TIC-1 vectors at lag=1 and lag=10 is 0.880.

*Source: computed from `a10_tica_singular_vectors_left.npy` in both directories.*

### Strength

**Moderate.** The spectral gap at lag=10 is clear (1.8:1 ratio). TIC-1 directional robustness (cos=0.88) confirms both lag times identify the same slow process. However, this has only been computed for A10; an A8Q2-native TICA decomposition is needed to confirm the same single-mode picture on the mutant.

### Impact

Justifies the use of a single CV (TIC-1) for adaptive seeding. If A8Q2 also shows a single dominant slow mode, the practical question reduces to whether that mode's *direction* can be imported from A10 — which Finding 5 shows is partially possible.

---

## Summary Table

| # | Finding | Signal strength | Key number |
|---|---|---|---|
| 1 | Gln substitution destabilises helix by ΔΔG ~ 0.18 kcal/mol | Strong | ΔΔG = 0.17–0.19 kcal/mol |
| 2 | Mutation accelerates unfolding 1.4× consistently; folding acceleration is start-state dependent | Moderate | k_u ratio = 1.43–1.44 |
| 3 | Random seeding achieves best FES convergence (lowest JSD, full coverage) | Strong | JSD = 0.0058 at 600 ns |
| 4 | Transferred seeding gives most accurate f_helix estimate | Moderate | Error = 0.004 at 600 ns |
| 5 | TICA transfer fails 3/4 geometric tests but retains |r| = 0.79 with helix content | Strong | AUC = 5.61 (transferred) vs 10.89 (native) |
| 6 | Native seeding maximises transitions (0.26/ns) but minimises coverage (0.921) | Strong | Transition rate 1.7× higher, JSD 3.4× worse |
| 7 | Single dominant slow mode in A10 (spectral gap 1.8:1 at lag=10 ns) | Moderate | cos(TIC-1, lag1 vs lag10) = 0.88 |

---

## 8. COMPASS-Lite replay: oracle baseline caveat

**Claim.** Any narrative or table that reports COMPASS beating “random” using the enhanced replay sweep (`scripts/compass_experiments_enhanced.py` / `outputs/compass_sweep2/`) must state explicitly that **both** policies drew candidate chunks from the **full** cached A8Q2 trajectory—an oracle shared action set.

**Implication.** Those margins are **not** valid as-is for “online adaptive MD on a fresh mutant,” where only **already-simulated** frames may be re-seeded. The corrected offline protocol is **online-faithful replay** (`src/compass/replay_online.py`, `scripts/compass_experiments_online.py`): observed-seed pools, optional deterministic-trajectory diagnostics, and a segment-aware MSM stochastic emulator for primary offline validation.

**Strength.** Procedural (definition of the baseline), not a new MD measurement.

**Impact.** Headline COMPASS-vs-random numbers from the oracle sweep should be relabeled as retrospective-oracle diagnostics until replaced by online-faithful sweep outputs.

---

## 9. MSM six-baseline benchmark (segment-aware emulator)

**Claim.** Under the **same** MSM transition matrix, lag, atlas bins, and lag-weighted **sampled_ns** budget, **B6 (COMPASS-full)** is compared to **B0** (single long walk sliced per step), **B1** (independent replicas round-robin from initial seeds), **B2** (uniform adaptive restarts over observed microstates), **B3** (min-count adaptive), and **B4** (farthest-point adaptive). This isolates **prior value** (B6 vs B3), **generic adaptivity** (B3 vs B1), and a **Markov ceiling** (B0 mixes fast on a model fit to the full trajectory—see caveats below).

**Artifacts (50 repeats × 5 presets × default six policies, 2026-05-02 run).**

- Runner: [`scripts/compare_msm_three_policies.py`](scripts/compare_msm_three_policies.py) — `--output-dir outputs/msm_benchmark` (optional `--with-b5`, `--save-hists`).
- Long-form per-step table: `outputs/msm_benchmark/per_step_all.parquet` (or `.csv` if parquet unavailable).
- Final-step summary: `outputs/msm_benchmark/benchmark_final_step_summary.csv`, `outputs/msm_benchmark/three_policy_summary.csv`.
- References (GT hist, noise floor bootstrap): `outputs/msm_benchmark/references/<preset>/references.{json,npz}`.
- Figures: `outputs/msm_benchmark/figures/fig1_convergence_grid.png` … `fig6_pair_delta_curves.png` (from [`scripts/plot_msm_benchmark.py`](scripts/plot_msm_benchmark.py)).

**Representative final-ns means (SPIB1D preset, 160 ns).**

| Policy | JSD | \|Δf_helix\| | \|Δf_open\| |
|--------|-----|-------------|------------|
| B0 | 0.063 | 0.025 | 0.031 |
| B1 | 0.320 | 0.053 | 0.088 |
| B2 | 0.117 | 0.084 | 0.106 |
| B3 | 0.213 | 0.215 | 0.264 |
| B4 | 0.319 | 0.247 | 0.286 |
| B6 | **0.077** | **0.037** | **0.045** |

**Representative final-ns means (TICA2D 200-step / 400 ns preset).**

| Policy | JSD | \|Δf_helix\| |
|--------|-----|-------------|
| B0 | **0.334** | 0.024 |
| B1 | 0.555 | 0.045 |
| B6 | **0.316** | **0.019** |

**Caveats.**

- MSM mixing is optimistic relative to real MD restart cost; B0 is a **diagnostic ceiling**, not a fair “one long GPU run” competitor when slow modes are under-resolved in 2D bins.
- `sampled_ns` is lag-weighted histogram mass × 0.1 ns/frame spacing, not wall-clock.
- B4 (pure geometry) can severely underperform on TICA2D maps when “farthest” bins are empty for helix/open observables.

**Strength.** Moderate: many repeats and aligned budgets; conclusions are **emulator-level**, not new production MD.

**Impact.** Supports the decomposition “prior + structured score” vs **count-based** and **uniform** adaptive restarts under online-faithful Markov assumptions; pair-difference plots (**fig6**) highlight B6−B3 (prior) vs B3−B1 (adaptivity).

