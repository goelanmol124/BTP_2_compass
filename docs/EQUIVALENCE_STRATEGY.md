# Equivalence strategy (COMPASS v2)

## What was chosen

**Default equivalence method: M3** — archive neighborhoods in **backbone torsion (dihedral) space**, with a precomputed k-NN index (`outputs/v2/oracle/eq_M3_k*.npz`).

The bench driver uses **`--eq-method auto`**, which reads **`outputs/v2/oracle/equivalence_choice.json`** and takes `equivalence_choice.winner_method` (expected to be `M3` after validation).

## Why M3

We compared M3, M5 (target-TICA), and M7 (MSM macrostate) with an **offline oracle check**: for many random anchor frames, find k nearest archive frames under each method and measure how well those neighbors predict **near-identical forward trajectory windows** — lower median **composite** score is better.

The composite combines (with equal weights when NHB traces exist):

1. Jensen–Shannon divergence of **feature** statistics over the forward window vs each neighbor’s forward window.  
2. JSD of **native hydrogen-bond path** summaries over that window (`a8q2_nhb.npy` when available).

**Outcome:** **M3 won consistently** under several settings intended to stress stability:

| Setting | Observation |
|---------|--------------|
| Shorter vs longer forward window (`chunk_frames` 10 vs 20) | M3 favored; longer windows give a more informative NHB term |
| Multiple RNG seeds aggregated | Same ranking |
| `--same-segment-only-gate` | Same ranking |
| Combined: chunk 20 + aggregated seeds + gate | Clear gap: **M3** best mean composite vs M7 and M5 |

**Interpretation:** For this oracle, **dihedral equivalence** aligns better with “similar short-term physics” in the latent sense we score than **target-plane TICA** (M5) or **coarse MSM labels** (M7). M5/M7 encode useful coarse structure but can lump or split states in ways that are **too coarse or task-aligned** relative to archive–trajectory equivalence for seeding.

## M3 similarity thresholds (calibration)

Semantic tiers are defined first on **forward-window composite JSD** (same construction as `validate_equivalence`: median feature JSD + optional NHB JSD), then **M3 Euclidean distance** is summarized **conditional** on those tiers. This keeps “close / almost / not similar” tied to **predictive** agreement on the oracle archive, not raw distance alone.

### Chosen thresholds for the current archive (what we use and why)

This section records the **actual decision** for the A8Q2-style oracle that ships in this repo. All numbers below come from the committed artifact **`outputs/v2/oracle/m3_similarity_thresholds.json`** (regenerate with `calibrate_m3_thresholds` if features or trajectories change).

#### 1) Equivalence *method* (M3 vs M5/M7)

- **Choice:** **M3** (torsional / `a8q2_features.npy` k-NN) as the default equivalence geometry.
- **Metric used:** `validate_equivalence` ranks methods by the **median forward-window composite** over held anchors (lower is better):  
  `0.5 × median_feature_JSD + 0.5 × median_NHB_path_JSD` when `a8q2_nhb.npy` is present, else feature-only.
- **Where it is stored:** `outputs/v2/oracle/equivalence_choice.json` → `equivalence_choice.winner_method` (used by bench `--eq-method auto`).

#### 2) Similarity *tiers* (close / almost / not similar) — **composite is authoritative**

Calibration samples many **(anchor, neighbor column)** pairs from the **M3** index, scores each pair with the **same** forward-window composite as validation, then defines tiers from **pooled** composite quantiles.

| Setting in canonical run | Value |
|--------------------------|--------|
| `chunk_frames` | 20 |
| `eq_k` / neighbor columns used | 40 / ranks **1–20** (matches a wide neighbor cloud like the bench) |
| Anchors | 750 |
| RNG seed | 42 |
| Tier quantiles | `q_close = 0.25`, `q_far = 0.75` (by construction: ~25% close, ~50% almost, ~25% not_similar) |
| Composite weights | `feat_weight = nhb_weight = 0.5` |

**Operational definitions (canonical):**

- **Close:** forward composite \(\le \tau_{\mathrm{close}} \approx\) **0.0708** (JSON: `recommended_operational.composite_similar_to_close_upper`).
- **Almost similar:** \(\tau_{\mathrm{close}} <\) composite \(< \tau_{\mathrm{far}}\).
- **Not similar:** composite \(\ge \tau_{\mathrm{far}} \approx\) **0.1533** (JSON: `recommended_operational.composite_not_similar_lower`).

**Bootstrap stability (80 resamples over anchors, same script):** \(\tau_{\mathrm{close}}\) in about **[0.0701, 0.0718]** and \(\tau_{\mathrm{far}}\) in about **[0.1466, 0.1583]** — see `bootstrap` in the same JSON. These are the intervals we treat as “measurement noise” on the tier boundaries, not separate scientific claims.

#### 3) M3 Euclidean distance — **reported, not used as a hard gate** (for this system)

The JSON still lists **geometry** summaries (per-tier distance percentiles, a **p90-within-close** distance of order **2.95** in sin/cos feature space, etc.). We **do not** treat a single global distance cutoff as a reliable “similarity dial” here.

**Metrics that drove that decision** (all in `diagnostics` in the JSON):

| Diagnostic | Canonical value (order of magnitude) | Interpretation |
|------------|----------------------------------------|----------------|
| `mean_gap_far_minus_close` | ~0.03 | Mean M3 distance for *not_similar* minus *close* is **tiny** compared to the spread of distances. |
| `histogram_overlap_coeff_approx` | ~0.90 | KDE-style overlap of distance distributions for **close** vs **not_similar** is **large**. |
| `ks_statistic_distance_close_vs_far` | ~0.08 | Two-sample KS sees a **small** geometry shift (statistically significant at huge *n*, but not useful for a clean mask). |
| `rho_comp_vs_distance` | ~0.05 | Spearman correlation between composite and M3 distance is **weak** on this pooled set. |
| `distance_separation_actionable_for_hard_cap` | **0** | Internal rule: geometry is **not** strong enough to justify an automatic bench mask from distance alone. |
| `composite_nhb_js_median_over_pairs` | ~0 | NHB channel often **degenerate** on this run; composite behaves like **feature JSD** in practice. |

So: **label pairs using composite \(\tau\)**; use M3 distance plots only for intuition (see `plots_written` paths in the JSON).

**Bench `--eq-mask-m3-by-calibration`:** the loader only supplies a distance cap when calibration marks separation **actionable**; for the canonical file it returns **no cap**, so **masking stays off** unless you change the system and re-calibrate.

### Run calibration

From the repo root (defaults: `outputs/v2/oracle`, `chunk_frames=20`, `eq_k=40`):

```bash
python -m src.compass.v2.calibrate_m3_thresholds ^
  --repo-root . ^
  --oracle-cache outputs/v2/oracle ^
  --eq-k 40 ^
  --chunk-frames 20 ^
  --anchors 400 ^
  --neighbor-ranks 1-20 ^
  --q-close 0.25 --q-far 0.75 ^
  --bootstrap 100 ^
  --plots-dir outputs/v2/oracle/m3_calibration_explore ^
  --plots-tag myrun
```

Writes **`outputs/v2/oracle/m3_similarity_thresholds.json`** with:

- **`tiers.forward_composite_jsd`**: `tau_close_inclusive_upper`, `tau_far_inclusive_lower` (quantiles of the pooled composite over sampled pairs).
- **`tiers.m3_distance_euclidean.per_tier_percentiles`**: p10/p50/p90 of neighbor distance within each composite tier.
- **`recommended_hard_split_quantile_method`**: `d_close_cap_at_percentile` (within *close*-tier distances, high percentile) and `d_not_similar_floor_at_percentile` (within *not_similar* tier, low percentile), plus Youden-style `d_youden_close_star` when separable.
- **`diagnostics.overlap_note_*`**: when the distance cap and floor **cross**, Euclidean bands are **ambiguous** (tiers overlap heavily in distance)—rely on composite JSD for semantics.
- Optional **`bootstrap`** block: 2.5–97.5% bands on stated cutoffs.

**Caveats:** Cutoffs are **dataset- and settings-specific** (anchors, chunk length, gates, `neighbor-ranks`). Re-run after changing `a8q2_features.npy` / trajectories / index `k`.

More exploratory runs (e.g. restricting neighbor ranks to diagnose geometry) are summarized in **`outputs/v2/oracle/README_CALIBRATION.txt`**; qualitative overlap is visible in the PNG paths listed under **`plots_written`** in the JSON.

### Optional: mask bench candidates by calibrated distance (M3 only)

Default behavior is unchanged (full k-NN). To attempt a distance mask when the calibration run deems geometry actionable:

```bash
python -m src.compass.v2.bench --eq-method M3 --eq-k 40 ^
  --eq-mask-m3-by-calibration ^
  --eq-m3-calibration-json outputs/v2/oracle/m3_similarity_thresholds.json
```

If `--eq-m3-calibration-json` is omitted, **`<oracle-cache>/m3_similarity_thresholds.json`** is used. `summary.json` includes **`m3_distance_calibration`** describing whether the mask was applied and the cap used.

## Rank-based behavior (unchanged default)

With **no mask**, neighbors are still chosen as the **\(k\) smallest** M3 distances: the calibration JSON is for **labeling** and **optional filtering**, not for replacing k-NN ranks.

## Compute / budget angle

Equivalence is **nearest-neighbor lookup** at bench time — cheap versus MD. Calibration is offline (pairwise forward JSD over sampled anchors × ranks); increase `--anchors` or tighten `--neighbor-ranks` for smoother percentiles.

Example **policy sweep** (COMPASS vs chronological single-chain vs random MC baselines): [`scripts/compare_v2_policy_baselines.py`](scripts/compare_v2_policy_baselines.py), summarized under [`outputs/v2/exp_policy_ablation_RESULTS.txt`](outputs/v2/exp_policy_ablation_RESULTS.txt).

## Alternatives still in code

Explicit **`--eq-method M1|M3|M5|M7|M8`** overrides auto. **M1** (RMSD) and **M8** (hybrid) are available where trajectories/features exist to build their indices.

## Operational note

After changing trajectories or archive, re-run:

`python -m src.compass.v2.validate_equivalence ...`

so `equivalence_choice.json` stays consistent with `--eq-method auto`. Use `python -m src.compass.v2.calibrate_m3_thresholds ...` to refresh distance/JSD tiers and `m3_similarity_thresholds.json` when you rely on `--eq-mask-m3-by-calibration`.
