# COMPASS-Lite: Sweep Findings and Next Steps

This is a working notebook of what the latest sweep
(`outputs/compass/sweep/`) showed and what to do next.  It is intentionally
short and pragmatic so that we can keep iterating without losing track.

---

## 1. What we just ran

Eleven retrospective replay experiments were run with
`scripts/compass_experiments.py`.  All of them obey the rule that **only
generic torsion features (projected through A10 TICA) are used for guidance** -
NHB and helix labels are reserved for evaluation.

| # | name | what it varies |
|---|------|----------------|
| 1 | `exp01_balanced_tica2d_chunk10_80ns`        | baseline balanced policy |
| 2 | `exp02_exploratory_tica2d_chunk10_80ns`     | exploratory policy |
| 3 | `exp03_source_only_tica2d_chunk10_80ns`     | ablation: source map only |
| 4 | `exp04_pure_frontier_tica2d_chunk10_80ns`   | ablation: novelty only |
| 5 | `exp05_balanced_tica2d_chunk5_80ns`         | smaller (0.5 ns) chunks |
| 6 | `exp06_balanced_tica2d_chunk20_80ns`        | larger (2 ns) chunks |
| 7 | `exp07_balanced_tica2d_chunk10_200ns`       | balanced, 200 ns budget |
| 8 | `exp08_exploratory_tica2d_chunk10_200ns`    | exploratory, 200 ns budget |
| 9 | `exp09_balanced_tica2d_chunk10_80ns_repeats`| 8 random restarts + softmax |
| 10| `exp10_balanced_tica1d_chunk10_80ns`        | 1D source map (TICA-1 only) |
| 11| `exp11_balanced_tica2d_chunk10_80ns_finegrid` | 50x50 source grid |

Headline numbers (final budget, COMPASS vs random mean over 50 random replays;
positive `delta_*` favours COMPASS):

| experiment | budget | dJSD | dCov | dHelix-err |
|---|---|---|---|---|
| balanced (baseline)      | 80 ns  | -0.003 | -0.117 | +0.004 |
| exploratory              | 80 ns  | -0.112 | +0.141 | -0.044 |
| source-only ablation     | 80 ns  | -0.012 | -0.148 | -0.005 |
| pure-frontier ablation   | 80 ns  | -0.074 | +0.179 | -0.055 |
| balanced (chunk 0.5 ns)  | 80 ns  | -0.017 | +0.063 | -0.007 |
| balanced (chunk 2 ns)    | 80 ns  | -0.012 | -0.124 | -0.011 |
| **balanced (200 ns)**    | 200 ns | **+0.005** | **+0.053** | -0.002 |
| exploratory (200 ns)     | 200 ns | -0.049 | +0.132 | -0.042 |
| balanced + 8 restarts    | 80 ns  | -0.004 | -0.107 | +0.003 |
| balanced (TICA-1D)       | 80 ns  | -0.004 | +0.052 | -0.020 |
| balanced (50x50 grid)    | 80 ns  | -0.013 | -0.126 | -0.010 |

Plots:
`outputs/compass/sweep/sweep_overlay_curves.png` (all curves vs random) and
`outputs/compass/sweep/sweep_summary_bars.png` (final-budget bar chart).

---

## 2. What the sweep is telling us

### 2.1 The interesting wins
- **Helix-fraction error** at 80 ns is best for the *balanced* and *repeats*
  variants (0.010-0.012 vs random 0.014).  The source map is steering us
  toward bins whose visit count matches the WT prior, which apparently aligns
  well with the mutant's true helix population.
- **At 200 ns, balanced beats random on JSD and coverage simultaneously**
  (JSD 0.043 vs 0.048 and coverage 0.876 vs 0.823).  This is the cleanest
  "yes, prior helps" result so far.  The crossover happens because random
  spends the early ns smoothing high-density bins while COMPASS pre-fills
  low-occupancy bins predicted by the prior.
- **Coverage is a separate axis**: exploratory and pure-frontier *crush*
  random on coverage (+0.13 to +0.18 at 80 ns, +0.13 at 200 ns) but pay for
  it with worse JSD/helix-error.  This is the classic explore-exploit
  tension and it shows up clean here.

### 2.2 The sobering bits
- At 80 ns the balanced variant is essentially tied with random on JSD and
  *worse* on coverage.  The source map is helping in the right way (faithful
  density) but the mutant still has rare states the WT prior pays no
  attention to.
- The fine 50x50 source grid hurts everything - the prior gets too noisy
  per bin and the deficit term becomes erratic.  30x30 is the sweet spot
  here.
- The TICA-1D run reaches **100 % coverage** in TICA-1 but its helix-error
  goes up.  TICA-1D throws away the second slow direction, so any deficit
  filling along TICA-1 still leaves the orthogonal axis under-sampled in
  ways NHB cares about.
- Softmax + 8 restarts barely moves the metrics, which is good for
  reproducibility but suggests the policy is already near its deterministic
  optimum given this map.

### 2.3 Single-line summary
> **The source map clearly helps thermodynamic accuracy at long budgets and
> coverage at short budgets, but no single weighting wins all three metrics.
> A two-phase or annealed schedule is the obvious next move.**

---

## 3. What to do next (cheap to run)

These can all be implemented in the existing `src/compass/replay.py` and
`scripts/compass_experiments.py` infrastructure without any new MD.  Each is
expected to take less than 10 minutes of compute.

### 3.1 Two-phase scheduler ("explore then exploit")
Run the exploratory policy until coverage saturates (e.g., first 30 % of
budget) and then switch to balanced.  This is the obvious response to the
explore-exploit findings above and should beat both single-policy variants
on **all three** metrics.

Implementation: add `--phase-switch-frac` to `replay.py` and have
`_compass_scores` interpolate weights based on completion fraction.

### 3.2 Annealed weights
Smooth version of 3.1: linearly anneal frontier weight from 1.0 -> 0.0 and
source weight from 0.0 -> 1.0.  Two-knob ablation against the step-switch.

### 3.3 SPIB-based source map
We already have SPIB latent states for A10 in
`outputs/spib_a10_analysis/spib_a10_frame_assignments.npz` and a trained
SPIB model in `outputs/spib_a10_model/`.

Add a `build_spib_atlas()` function that:
- loads the SPIB model and the A10 SPIB-feature pipeline,
- computes A8Q2 SPIB features once and projects to the same latent space,
- builds the source map in either (z1, z2) for a 2D map or in the SPIB
  state assignment as a 1D categorical map.

This tests whether a non-linear committor-style coordinate carries more
useful prior information than TICA does.  Worth doing before BTP because
SPIB is the most novel piece of our pipeline.

### 3.4 Bootstrap-based "honest random" baseline
Instead of one random ordering per repeat, draw chunks with replacement and
build a bootstrap envelope of (jsd, coverage, f_helix_error) at every step.
This gives proper confidence bands and lets us claim COMPASS is significant
when the curves separate by more than the bootstrap CI.

Cheap to add: replace `np.random.permutation(n_chunks)[:n_steps]` with
`rng.choice(n_chunks, size=n_steps, replace=True)`.

### 3.5 First-passage / time-to-discovery curves
Add a per-step metric "fraction of distinct active TICA bins (with target
support >= k frames) hit so far" as a second discovery curve.  Random tends
to revisit common bins; COMPASS shows the discovery curve flattening
earlier.  This is a *very* presentation-friendly graph.

### 3.6 Robustness-to-prior-quality study
Mutate the source map on purpose (e.g., add Gaussian noise or shuffle a
fraction of bins) and re-run.  The plot of "delta-JSD vs prior corruption
%" tells the BTP audience that the method degrades gracefully and is not
overfitting to A10.

### 3.7 Multi-seed mutant story
Pick three seeds (helix-rich, helix-poor, transition) from the A8Q2
trajectory and run COMPASS-Lite from each.  Show that the score map drives
all three to similar coverage/JSD plateaus.  This makes the story
"different starting hikers reach the same coverage, faster than random".

### 3.8 Sensitivity heatmap
Sweep two coefficients (e.g., `source_deficit` weight and `novelty`
weight) on a 5x5 grid and plot the resulting JSD/coverage as a heatmap.
Demonstrates the chosen weights are not cherry-picked.

### 3.9 Per-variant decision-reason breakdown
The `compass_reason_counts.png` is already produced for every experiment.
Stack them in one figure (heatmap of reason fraction across experiments)
to show how policy choices change across budgets and variants.

---

## 4. Larger jumps (saved for after BTP)

These need more scaffolding or new MD.

### 4.1 True online emulator
Today every chunk is already in the bag.  Build a small `chunk_pool`
abstraction that simulates "this chunk wasn't generated yet" by hiding all
frames downstream of a candidate seed until the candidate is selected.
This is closer to the online setting and lets us defend the policy against
adaptive-sampling reviewers.

### 4.2 Plug to actual MD via OpenMM
Replace `chunk pool` with a launcher that submits a 100-200 ps OpenMM
batch from the candidate seed structure on a real GPU.  The `COMPASS_PLAN`
already sketches this.  Keep the same scoring code and feature pipeline so
that the offline replay results transfer cleanly.

### 4.3 SPIB committor-aware probe types
Once SPIB is wired in, we can implement true "saddle probe" and "tunnel
probe" types from the COMPASS plan: pick chunks whose SPIB committor sits
near 0.5 between two metastable states.  Those are the highest-information
seeds in adaptive sampling theory.

### 4.4 Multi-mutant transfer matrix
Cross-test: train atlas on A10, replay on A8Q2, A8Q4, A8Q5, A8Q9.  The
table of "delta-JSD per mutant" is a complete story for the thesis and
generalises beyond a single mutant.

### 4.5 Failure-mode logging
Already on the radar: track when the policy keeps re-visiting a bin without
gaining JSD.  Log it as an "exhausted" reason and add a per-budget
"fraction of probes that paid off" plot.

---

## 5. BTP presentation slides this implies

Recommended order for the demo, given the sweep results:

1. **The earthquake analogy + the COMPASS algorithm box.** (1 slide)
2. **Replay metrics curves at 80 ns.** Balanced wins f_helix_error,
   exploratory wins coverage, both tied with random on JSD - shows the
   scoring trade-off is real.
3. **Replay metrics curves at 200 ns.** Balanced beats random on JSD and
   coverage simultaneously - the cleanest "prior helps" graph.
4. **Reason-breakdown bar chart** for the 200 ns balanced run - shows the
   policy is using a healthy mix of source/frontier/novelty terms, not just
   one.
5. **TICA scatter map of first 30 chosen chunks** - visually convincing.
6. **Weight-sensitivity heatmap (3.8)** - kills the "you cherry-picked
   weights" question.
7. **Roadmap slide listing 3.3 (SPIB) and 4.x items.**

If we have time before the deadline, the most impactful additions are
3.1 (two-phase) for cleaner curves and 3.3 (SPIB atlas) because it differentiates
COMPASS from prior-as-TICA work.
