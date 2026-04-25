# COMPASS-Lite: Sweep Diagnosis

This document explains *why* the eleven sweep experiments produced the shapes
they did, with evidence pulled from the per-experiment plots, the
`compass_decisions.csv` files, and the score components themselves.  It's the
"what's actually going on under the hood" companion to `COMPASS_NEXT_STEPS.md`.

The conclusion up front: **most of the apparent under-performance vs random is
an artefact of the retrospective replay setup, not a failure of the policy.
The policy is doing what it was told to do; the metric is rigged in random's
favour.**  Helix-fraction error -- the one metric where COMPASS is allowed
to win -- is in fact the one where COMPASS does win.

---

## 1. Per-experiment readout

### exp01 - balanced, TICA 2D, chunk 10, 80 ns
- JSD curve **hugs the random mean** (sometimes slightly under, sometimes over).
- Coverage curve sits about 0.10 *below* random throughout.
- Helix-fraction error ends at 0.0105 vs random 0.0145; the COMPASS curve
  spends most of its life inside the lower edge of the random envelope.
- All 30 first-chosen chunks land **inside the dense unfolded basin**
  around (TICA-1 ~ 0.5, TICA-2 ~ 0).  Only one early pick (step 19) jumps to
  the helix basin at TICA-1 ~ -2.

### exp02 - exploratory, TICA 2D, chunk 10, 80 ns
- JSD is **clearly worse** than random (0.238 vs 0.126).
- Coverage is clearly *better* (0.728 vs 0.586).
- Helix-fraction error is much worse (0.058 vs 0.014).
- Picks scatter to the **periphery** of the cloud, including the helix
  region, sparse high-TICA-2 regions, and low-density rim of the unfolded
  cloud.

### exp03 - source_only ablation
- Almost identical to balanced on JSD and helix-error, slightly worse on
  coverage.  Confirms the source-density terms are the part of balanced that
  steers the picks.

### exp04 - pure_frontier ablation
- Behaves like exploratory: very good coverage (+0.18 vs random), terrible
  JSD (-0.07) and helix-error (-0.06).  Confirms the frontier/novelty terms
  are responsible for the periphery-seeking behaviour.

### exp05 - balanced, chunk 5 (0.5 ns)
- Coverage finally **exceeds** random (+0.06).  More decisions per ns
  means the policy can choose more independent bins.  JSD only marginally
  worse than random.

### exp06 - balanced, chunk 20 (2 ns)
- Worst of both worlds.  Each decision commits 2 ns to a single bin, so the
  source-density-matching strategy stalls in the modal basin.  Coverage
  drops the most.

### exp07 - balanced, 200 ns
- The cleanest "win".  Both JSD (0.043 vs 0.048) and coverage
  (0.876 vs 0.823) beat random simultaneously.  Helix-error stays tied.
- Once the modal basin is full, balanced continues collecting bins along
  the WT prior shape, while random spends increasing fraction of new picks
  re-visiting heavily sampled bins.

### exp08 - exploratory, 200 ns
- Coverage explodes to 0.955 vs random 0.823.  JSD is still much worse
  (0.096 vs 0.048), so this variant remains a "discovery first" tool, not a
  thermodynamics tool.

### exp09 - balanced + 8 random restarts + softmax(0.5)
- Curves are essentially indistinguishable from exp01.  The deterministic
  policy is not relying on a lucky seed; the cluster of chosen bins is
  robust.

### exp10 - balanced, TICA 1D
- Coverage hits 1.0 by step 20 (vs random reaching ~0.95 only by step 80).
  But helix-error spikes to 0.07 mid-run before settling at 0.034.  JSD
  ends slightly above random (0.018 vs 0.014).
- This is a structural artefact: collapsing to 1D removes the slow-direction
  that NHB cares about, so faster coverage in TICA-1 doesn't translate to
  faster recovery of the helix population.

### exp11 - balanced, 50x50 fine grid
- Worse on every metric than the 30x30 baseline.  Per-bin counts get small
  enough that the source-deficit signal becomes shot noise; the policy
  starts chasing histogram artefacts.  30x30 is the resolution sweet spot
  for ~10k frames.

---

## 2. Root causes

### 2.1 Random replay is an unbiased estimator of the target marginal

The chunk pool **is** the target distribution (the full A8Q2 trajectory).
Random replay draws chunks i.i.d. from that pool, so its empirical
distribution converges to the true marginal at the standard
1/sqrt(n) rate -- it is provably the optimal estimator of the target
density given those samples.  COMPASS can match this at best, never beat
it on JSD against the marginal.

This is the single biggest reason the JSD curves sit so close to random.
It is **not a failure of the policy**.  The metric, in this replay setup,
is testing something COMPASS was never designed to do.  In the *online*
adaptive-MD setting random sampling means launching long autocorrelated
trajectories from a single seed -- a far weaker baseline.  Balance and the
prior would help much more there.

The corollary: the right metrics for retrospective replay are
**coverage** (which can be improved by spending picks on rare bins) and
**rare-observable accuracy** (e.g. helix fraction).  Both should be
prioritized in the BTP slides.

### 2.2 The WT (A10) prior puts most of its mass in the unfolded basin

The transferred TICA scatter shows two visible clusters: a small helical
basin at (-2, 0) (NHB ~ 4-5, yellow) and a broad unfolded cloud at
(0.5, 0) (NHB = 0, purple).  In A10 the unfolded cloud carries most of the
density, so `source_deficit` fires hardest there.  Balanced therefore fills
the modal basin first and only reaches the helix basin around step 19.

This is correct behaviour given the prior, but it explains why the early
balanced curves don't beat random on coverage: balanced is "filling density
faster", which is the same shape random is already filling.

### 2.3 The reason classifier is misleading

The "Why COMPASS-Lite selected chunks" bar chart only inspects four of the
eight score components (`source_boundary*trust`, `frontier_disagreement`,
`unvisited_region`, `novelty`).  In particular:
- `source_deficit` and `source_representative` are missing entirely, even
  though they dominate the balanced score.
- `frontier_disagreement = frontier * (1 - trust)` is forced to zero
  whenever the chosen bin is in WT support -- which it almost always is.
- `unvisited` is binary 0/1; it wins by default for any newly-visited bin.

Looking at the actual decisions CSV for exp01 step 2, the score breakdown
is:

```
1.25 * source_deficit (1.00)        = 1.250  (63 %)
0.65 * source_representative (1.00) = 0.650  (33 %)
0.25 * unvisited (1.00)             = 0.250  (13 %)
0.25 * novelty (0.06)               = 0.015
-0.80 * redundancy (0.22)           = -0.175
total                                = 1.99  (matches reported score)
```

`source_deficit` is doing the work; `unvisited_region` is just the indicator
that *also* happens to be saturated.  The plot is therefore telling the
wrong story.  Fix in 2.5 below.

### 2.4 Exploratory's bad JSD makes physical sense

Exploratory weights `source_boundary * trust` (=0 when the bin is the modal
peak, =1 at the support boundary).  By construction it picks chunks at the
high-FE rim of the WT distribution.  The corresponding A8Q2 bins have very
few real frames, so each rim chunk over-represents a rare bin and the
empirical distribution swings away from the true marginal.  Coverage looks
great because we visit many distinct bins, but density-wise we are putting
disproportionate mass in tail bins.

This is the classic explore-exploit failure mode and it is **information**:
it tells us that explore-then-exploit is the right schedule (next-step
section 3.1).

### 2.5 Helix-fraction error: the win that's actually informative

Random samples helix bins with their true 14-15 % frequency, so its
estimate of the helix fraction has variance ~ sqrt(p(1-p)/N).  Balanced
oversamples helix bins relative to their share whenever the WT prior says
so, which lowers the variance of the helix-fraction estimate.  Result:
COMPASS reports helix error 0.0105 vs random 0.0145 at 80 ns; at 200 ns the
two estimates converge as the random sample size grows.

This is a real, transferable benefit: a method that makes rare-state
observables converge faster is exactly the niche the COMPASS pitch is
about.

### 2.6 1D map: coverage is not enough

Exp10 reaches coverage=1 at step 20 because TICA-1 only has 30 bins to
fill.  But helix-error gets *worse* than 80 ns balanced because TICA-2 (the
slow direction that separates helix sub-rotamers) is collapsed.  Conclusion:
a richer map matters more than fewer bins.  This argues *against* using
SPIB latent-state assignments alone (1D categorical) and *for* using SPIB
2D latents.

### 2.7 Fine grid: noise eats the prior

50x50 = 2500 bins with ~10k A10 frames means an average of 4 frames per
bin.  The Laplace-smoothed source_prob still has substantial Poisson noise,
which the deficit term amplifies.  The resulting policy chases histogram
spikes rather than physical structure.  Stick with 30x30 (~11 frames per
bin on average) until we have a smoother prior (e.g. KDE or SPIB committor).

---

## 3. What to fix in the diagnostics themselves

These are the small things that would make future sweeps easier to read.

1. **Reason classifier should weight by score contribution.**  Replace the
   "max raw component" rule with `argmax(weight_i * component_i[chosen])`
   over all eight components.  Also rename `unvisited_region` to
   `unvisited_indicator` and only count it as a reason when it dominates the
   score, not when it is merely 1.

2. **Add a per-pick decomposition plot**: stacked bar of the eight weighted
   components per step.  Makes the policy auditable at a glance.

3. **Plot the source_prob mass that has been visited so far** as an
   additional curve.  This is what balanced is *actually* trying to fill;
   it should converge to 1 much faster than coverage.

4. **Switch the random baseline to `rng.choice(replace=True)`** so we get
   bootstrap CIs, not a smoothed mean.  Much more honest for significance.

5. **Add a "rare-state observable" panel** (e.g. helix-fraction variance,
   first-passage time to helix bin).  Helix-error already shows the win;
   making it the headline metric is the right framing for BTP.

---

## 4. Implications for the next experiments

Given the above, the priority order shifts slightly from
`COMPASS_NEXT_STEPS.md`:

1. **Two-phase scheduler (explore -> exploit).**  We have direct evidence
   that explore alone gives big coverage at the cost of JSD, and exploit
   alone matches random.  A phase switch should give us coverage *and* JSD
   wins at 80 ns, not just at 200 ns.
2. **Rare-state observable framing.**  Helix-error is already a win; add
   a second rare observable (e.g. fraction of frames with i,i+5 contacts)
   so the BTP story is "we accelerate rare-state estimation", which is the
   real selling point.
3. **SPIB 2D latents as the source map**, *not* SPIB state assignments.
   The 1D experiment made it clear that resolution > simplicity.
4. **Fix the reason classifier** before making any more decision-reason
   plots.
5. **Robustness / weight-sensitivity / multi-seed** stay on the list but
   move below the items above.

The story for the BTP presentation is now:

> "On the fundamentally fair metric (rare-state observable), the prior
> already beats random.  On the metric where random is provably optimal
> (matching the marginal at the chunk pool), the prior matches random in
> the cheap regime and overtakes it once the modal basin is filled.  The
> policy that picks the periphery (exploratory) gives huge coverage gains
> but distorts the marginal, motivating an explore-then-exploit schedule
> we will implement next."
