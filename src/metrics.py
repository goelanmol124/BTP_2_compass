"""
metrics.py — Acceleration metrics and final analysis for the btp_2_compass experiment.

This module quantifies how effectively each adaptive seeding strategy (Random,
Transferred TICA, Native TICA) converges the A8Q2 free energy surface relative
to the 500 ns × 2 replica ground truth.

All distance/divergence metrics compare 2D FES histograms on the masked domain
Ω = {bins (i,j) where ground-truth count ≥ FES_MIN_COUNT}, which excludes
statistically unreliable bins at the periphery of the sampled landscape.

Metric overview
---------------
jsd          : Jensen-Shannon Divergence in bits [0, 1]. Primary convergence metric.
coverage     : Fraction of Ω bins visited. Grows 0→1 across iterations.
entropy      : Normalized Shannon entropy of the sampled distribution on Ω.
f_helix_err  : |f_helix_seeded − f_helix_ground_truth|. Most interpretable scalar.
noise_floor  : JSD between the two ground-truth replicas. Best achievable JSD.
accel_factor : ns_random_to_converge / ns_transferred_to_converge.

References
----------
- Tiberti et al. PLOS CB 2015 (JSD methodology, ensemble comparison)
- Moffett & Shukla arXiv:1710.00443 (D₀, D_τ, D_KM transfer metrics)
- Hruska et al. J. Chem. Phys. 2018 (adaptive sampling strategy comparison)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon

from src.config import (
    FES_MIN_COUNT,
    FES_NBINS,
    N_HB_MAX,
    NHB_THRESHOLD_FRAC,
    STRATEGY_COLORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

_DPI = 300
_STRATEGY_ORDER = ["Random", "Transferred TICA", "Native TICA"]


# ─────────────────────────────────────────────────────────────────────────────
# CORE DIVERGENCE / COVERAGE / ENTROPY
# ─────────────────────────────────────────────────────────────────────────────


def compute_jsd(
    hist_ref: np.ndarray,
    hist_sample: np.ndarray,
    min_count: int = FES_MIN_COUNT,
) -> float:
    """Jensen-Shannon Divergence between sampled and reference 2D FES histograms.

    Both arrays are raw count arrays of the same shape (typically 25 × 25).

    Masked domain
    -------------
    Only bins where ``hist_ref >= min_count`` are included.  Bins with fewer
    than *min_count* ground-truth counts are too noisy to serve as reliable
    reference points, so they are excluded from both p and q before computing
    the divergence.

    Computation
    -----------
    On the masked domain the reference distribution is::

        p = hist_ref[mask] / sum(hist_ref[mask])

    and the sample distribution is::

        q = hist_sample[mask] / sum(hist_sample[mask])   (if sum > 0)
        q = zeros                                         (if sum == 0)

    ``scipy.jensenshannon`` returns √JSD (the Jensen-Shannon *distance*), so
    the result is squared to recover JSD in bits.

    Edge case
    ---------
    If the sample has zero counts on the entire masked domain (e.g. the very
    first iteration seeds starting far from the reference basins), JSD is
    defined as 1.0 (maximum divergence) rather than NaN.

    Parameters
    ----------
    hist_ref : np.ndarray
        Raw count array from the ground-truth simulation.  Shape (N, N).
    hist_sample : np.ndarray
        Raw count array from the seeded simulation.  Same shape as hist_ref.
    min_count : int, optional
        Minimum ground-truth count for a bin to enter the masked domain.
        Default: FES_MIN_COUNT (3).

    Returns
    -------
    float
        JSD in bits, in [0, 1].  0 = identical distributions; 1 = completely
        disjoint.

    Notes
    -----
    JSD is preferred over KL divergence because KL → ∞ when the sample has
    zero counts in bins where the reference is non-zero.  JSD uses the mixture
    M = (p + q)/2 as the pivot, keeping the value finite even with zero sample
    counts on parts of the masked domain.
    """
    p = hist_ref.flatten().astype(float)
    q = hist_sample.flatten().astype(float)

    mask = p >= min_count
    if not mask.any():
        warnings.warn(
            "compute_jsd: no reference bins satisfy min_count threshold — "
            "returning JSD = 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0

    p_m = p[mask]
    q_m = q[mask]

    p_m = p_m / p_m.sum()

    q_sum = q_m.sum()
    if q_sum == 0.0:
        return 1.0
    q_m = q_m / q_sum

    # jensenshannon returns sqrt(JSD); square to get JSD in bits.
    return float(jensenshannon(p_m, q_m, base=2) ** 2)


def compute_coverage(
    hist_ref: np.ndarray,
    hist_sample: np.ndarray,
    min_count: int = FES_MIN_COUNT,
) -> float:
    """Fraction of reference bins visited by the sample.

    Counts only bins in the masked domain (ground-truth count ≥ min_count) and
    reports the fraction of those that have at least one count in the sample.

    Interpretation
    --------------
    Coverage grows from ~10 % in iteration 1 (starting near a single basin)
    toward ~90–100 % as the seeding experiment covers the full landscape.  A
    transferred-TICA strategy that correctly identifies the slow CV should
    achieve higher coverage at lower cumulative simulation time than random
    seeding.

    Parameters
    ----------
    hist_ref : np.ndarray
        Raw count array from the ground-truth simulation.  Shape (N, N).
    hist_sample : np.ndarray
        Raw count array from the seeded simulation.  Same shape as hist_ref.
    min_count : int, optional
        Minimum ground-truth count for a bin to enter the masked domain.

    Returns
    -------
    float
        Coverage in [0, 1].
    """
    valid = hist_ref >= min_count
    if not valid.any():
        warnings.warn(
            "compute_coverage: no reference bins satisfy min_count threshold — "
            "returning coverage = 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    return float((hist_sample[valid] > 0).mean())


def compute_entropy(
    hist_sample: np.ndarray,
    hist_ref: np.ndarray,
    min_count: int = FES_MIN_COUNT,
) -> float:
    """Normalized Shannon entropy of the sampled distribution on the masked domain.

    Computes the entropy of the sample distribution restricted to the masked
    domain Ω and normalizes it by the maximum possible entropy log₂(|Ω|) so
    the result lies in [0, 1].

    Definition
    ----------
    Let q_i = count_sample[i] / sum(count_sample on Ω) for bins i ∈ Ω.
    Then::

        H(q) = −Σ_{i ∈ Ω, q_i > 0} q_i · log₂(q_i)
        H_norm = H(q) / log₂(|Ω|)

    Interpretation
    --------------
    H_norm = 1 : the sample visits all reference bins equally (perfectly
                 uniform, maximally unbiased sampling).
    H_norm < 1 : the sample is concentrated in a subset of the landscape —
                 indicating undersampled regions remain.

    Comparing seeded entropy to ground-truth entropy tracks whether the
    strategy replicates the full thermal distribution, not just the location
    of the major basin.

    Parameters
    ----------
    hist_sample : np.ndarray
        Raw count array from the seeded simulation.  Shape (N, N).
    hist_ref : np.ndarray
        Raw count array from the ground-truth simulation.  Same shape.
    min_count : int, optional
        Minimum ground-truth count for a bin to enter the masked domain.

    Returns
    -------
    float
        Normalized entropy in [0, 1].  Returns 0.0 if the sample is empty on
        the masked domain.
    """
    mask = hist_ref.flatten() >= min_count
    n_omega = int(mask.sum())

    if n_omega == 0:
        warnings.warn(
            "compute_entropy: no reference bins satisfy min_count threshold — "
            "returning entropy = 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0

    q = hist_sample.flatten().astype(float)[mask]
    q_sum = q.sum()
    if q_sum == 0.0:
        return 0.0

    q = q / q_sum
    nonzero = q > 0.0
    h = -float(np.sum(q[nonzero] * np.log2(q[nonzero])))

    h_max = np.log2(n_omega)
    if h_max == 0.0:
        return 1.0
    return float(np.clip(h / h_max, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# HELIX FRACTION
# ─────────────────────────────────────────────────────────────────────────────


def compute_f_helix_error(
    nhb_sample: np.ndarray,
    f_helix_groundtruth: float,
    nhb_max: int = N_HB_MAX,
    nhb_threshold_frac: float = NHB_THRESHOLD_FRAC,
) -> float:
    """Absolute error in helix fraction relative to the ground truth.

    Helix fraction is defined as the fraction of frames in which the number of
    α-helical backbone H-bonds (nhb) meets or exceeds the threshold::

        nhb_threshold = nhb_max × nhb_threshold_frac
                      = 6 × 0.5 = 3   (for a 10-residue peptide)

    Error
    -----
    ``|f_helix_seeded − f_helix_groundtruth|``

    Convergence target: < 0.02 (2 percentage points) by iteration 3.

    Interpretation
    --------------
    This is the most physically interpretable convergence metric.
    ``f_helix_groundtruth`` is approximately 10–20 % for A8Q2 at 330 K (lower
    than A10 because Gln is less helicogenic than Ala).  When the adaptive
    seeding strategy correctly samples the helix basin in proportion to its
    Boltzmann weight, the seeded f_helix approaches the ground-truth value.

    Parameters
    ----------
    nhb_sample : np.ndarray
        1-D integer array of per-frame H-bond counts from the seeded
        trajectories.  Shape (n_frames,).
    f_helix_groundtruth : float
        Ground-truth helix fraction from the 500 ns + 500 ns reference
        simulation.  Dimensionless, in [0, 1].
    nhb_max : int, optional
        Maximum possible α-helical H-bonds for the peptide.  Default: N_HB_MAX
        (6 for a 10-residue peptide).
    nhb_threshold_frac : float, optional
        Fraction of nhb_max that defines the helix state.  Default:
        NHB_THRESHOLD_FRAC (0.5 → threshold = 3).

    Returns
    -------
    float
        Absolute helix fraction error in [0, 1].  Units: dimensionless
        (fraction, not percentage).
    """
    threshold = nhb_max * nhb_threshold_frac
    f_helix_seeded = float((nhb_sample >= threshold).mean())
    return abs(f_helix_seeded - f_helix_groundtruth)


# ─────────────────────────────────────────────────────────────────────────────
# NOISE FLOOR
# ─────────────────────────────────────────────────────────────────────────────


def compute_noise_floor(
    hist_replica1: np.ndarray,
    hist_replica2: np.ndarray,
    min_count: int = FES_MIN_COUNT,
) -> float:
    """JSD between the two independent ground-truth replicas.

    The two A8Q2 replicas (helix-start vs. extended-start, each 500 ns) are
    statistically independent with the same total simulation length.  Their JSD
    quantifies the irreducible sampling noise at this level of effort — the best
    achievable divergence with 500 ns of unbiased MD.

    Usage
    -----
    The seeding JSD must drop *below* the noise floor before claiming that the
    seeded distribution is "equivalent to ground truth."  A JSD value above the
    noise floor means the seeding experiment still has systematic coverage gaps
    larger than the intrinsic statistical fluctuations of the reference.

    Parameters
    ----------
    hist_replica1 : np.ndarray
        Raw count histogram from the helix-start A8Q2 replica.  Shape (N, N).
    hist_replica2 : np.ndarray
        Raw count histogram from the extended-start A8Q2 replica.  Same shape.
    min_count : int, optional
        Minimum count in the *combined* reference for a bin to enter the masked
        domain.  Uses the sum of both replicas as the reference mask to be
        consistent with the ground-truth mask used in compute_jsd.

    Returns
    -------
    float
        Noise-floor JSD in bits [0, 1].

    Notes
    -----
    The reference mask is defined on the pooled histogram
    (replica1 + replica2) so that the same masked domain Ω is used for the
    noise-floor JSD and the seeding JSD comparisons.
    """
    pooled = hist_replica1 + hist_replica2
    return compute_jsd(pooled, hist_replica1, min_count=min_count)


# ─────────────────────────────────────────────────────────────────────────────
# ITERATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────


def build_iteration_summary(
    results_by_strategy: dict[str, list[dict[str, Any]]],
    noise_floor: float,
) -> pd.DataFrame:
    """Build a tidy DataFrame summarising all metrics across strategies and iterations.

    Parameters
    ----------
    results_by_strategy : dict
        Keys are strategy names (e.g. 'Random', 'Transferred TICA',
        'Native TICA').  Values are lists of per-iteration result dicts, each
        containing at minimum::

            {
                "iteration":       int,         # 1-indexed
                "cumulative_ns":   float,        # ns of simulation so far
                "jsd":             float,        # JSD vs ground truth
                "coverage":        float,        # fraction of Ω visited
                "entropy":         float,        # normalized entropy
                "f_helix_error":   float,        # |f_helix_seed − f_gt|
            }

    noise_floor : float
        JSD noise floor (from compute_noise_floor).  Used to populate the
        ``below_noise_floor`` boolean column.

    Returns
    -------
    pd.DataFrame
        Columns: strategy, iteration, cumulative_ns, jsd, coverage, entropy,
        f_helix_error, below_noise_floor.
        Sorted by strategy (preserving _STRATEGY_ORDER where possible) then by
        iteration ascending.
    """
    rows = []
    for strategy, iter_list in results_by_strategy.items():
        for res in iter_list:
            rows.append(
                {
                    "strategy":         strategy,
                    "iteration":        int(res["iteration"]),
                    "cumulative_ns":    float(res["cumulative_ns"]),
                    "jsd":              float(res["jsd"]),
                    "coverage":         float(res["coverage"]),
                    "entropy":          float(res["entropy"]),
                    "f_helix_error":    float(res["f_helix_error"]),
                    "below_noise_floor": bool(res["jsd"] < noise_floor),
                }
            )

    df = pd.DataFrame(rows)

    # Preserve canonical strategy order where available.
    known = [s for s in _STRATEGY_ORDER if s in df["strategy"].values]
    unknown = [s for s in df["strategy"].unique() if s not in known]
    order = known + unknown
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    df = df.sort_values(["strategy", "iteration"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE RATE (EXPONENTIAL FIT)
# ─────────────────────────────────────────────────────────────────────────────


def _exp_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """Exponential decay model: A · exp(−t / τ) + C."""
    return A * np.exp(-t / tau) + C


def compute_convergence_rate(
    jsd_values: np.ndarray,
    cumulative_ns: np.ndarray,
) -> dict[str, float]:
    """Fit an exponential decay model to JSD vs. cumulative simulation time.

    Model
    -----
    ``JSD(t) = A · exp(−t / τ) + C``

    where:
    - A     : amplitude (initial excess JSD above asymptote)
    - τ     : characteristic convergence time-scale in ns
    - C     : asymptotic JSD (the irreducible residual)

    Derived quantities
    ------------------
    - half_life_ns : τ · ln(2), the time for JSD to drop halfway from its
                     initial value A+C to the asymptote C.
    - r_squared    : coefficient of determination for the fitted curve.

    Parameters
    ----------
    jsd_values : np.ndarray
        1-D array of JSD values, one per iteration.  Shape (n_iter,).
    cumulative_ns : np.ndarray
        1-D array of cumulative simulation times in ns.  Same shape as
        jsd_values.  Must be strictly increasing.

    Returns
    -------
    dict
        Keys: 'A' (float), 'tau_ns' (float), 'C' (float),
        'half_life_ns' (float), 'r_squared' (float).
        On fit failure all values are np.nan and a RuntimeWarning is issued.

    Notes
    -----
    Bounds constrain A ≥ 0, τ ∈ (0, 1e4] ns, C ∈ [0, 1] so that the fit
    remains physically interpretable.  Initial parameter guess: A = JSD[0],
    τ = cumulative_ns[-1] / 2, C = JSD[-1].
    """
    jsd_values = np.asarray(jsd_values, dtype=float)
    cumulative_ns = np.asarray(cumulative_ns, dtype=float)

    if len(jsd_values) < 3:
        warnings.warn(
            "compute_convergence_rate: need at least 3 data points for an "
            "exponential fit — returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return dict(A=np.nan, tau_ns=np.nan, C=np.nan,
                    half_life_ns=np.nan, r_squared=np.nan)

    p0 = [
        max(jsd_values[0], 1e-6),
        max(cumulative_ns[-1] / 2.0, 1.0),
        max(jsd_values[-1], 0.0),
    ]
    bounds = (
        [0.0, 1e-3, 0.0],
        [1.0, 1e4,  1.0],
    )

    try:
        popt, _ = curve_fit(
            _exp_decay, cumulative_ns, jsd_values,
            p0=p0, bounds=bounds, maxfev=10_000,
        )
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"compute_convergence_rate: curve_fit failed ({exc}) — returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return dict(A=np.nan, tau_ns=np.nan, C=np.nan,
                    half_life_ns=np.nan, r_squared=np.nan)

    A, tau, C = popt
    y_pred = _exp_decay(cumulative_ns, A, tau, C)
    ss_res = float(np.sum((jsd_values - y_pred) ** 2))
    ss_tot = float(np.sum((jsd_values - jsd_values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return dict(
        A=float(A),
        tau_ns=float(tau),
        C=float(C),
        half_life_ns=float(tau * np.log(2)),
        r_squared=float(r2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ACCELERATION FACTOR
# ─────────────────────────────────────────────────────────────────────────────


def compute_acceleration_factor(
    jsd_transferred: np.ndarray,
    jsd_random: np.ndarray,
    cumulative_ns: np.ndarray,
    noise_floor: float,
) -> float:
    """Estimate the speed-up of the transferred-TICA strategy over random seeding.

    Method
    ------
    Find the cumulative simulation time at which each strategy's JSD first
    drops at or below the noise floor::

        t_transferred = min { cumulative_ns[i] : jsd_transferred[i] <= noise_floor }
        t_random      = min { cumulative_ns[i] : jsd_random[i]      <= noise_floor }
        acceleration  = t_random / t_transferred

    Linear interpolation is used to estimate the crossing time between
    discrete iteration points.

    Edge cases
    ----------
    - If transferred never reaches the noise floor: returns np.inf, signalling
      'did not converge' (caller should report this as such, not as a number).
    - If random never reaches the noise floor: returns np.inf as well (neither
      strategy converged; the acceleration factor is undefined).
    - If both never converge: returns np.nan.

    Parameters
    ----------
    jsd_transferred : np.ndarray
        JSD values for the transferred-TICA strategy, shape (n_iter,).
    jsd_random : np.ndarray
        JSD values for the random strategy, shape (n_iter,).
    cumulative_ns : np.ndarray
        Shared cumulative simulation time in ns, shape (n_iter,).
    noise_floor : float
        Target JSD threshold (from compute_noise_floor).

    Returns
    -------
    float
        Acceleration factor (dimensionless).  Values > 1 mean transferred
        converges faster; values < 1 mean random is faster.  np.inf means one
        strategy did not converge; np.nan means both did not converge.
    """
    jsd_transferred = np.asarray(jsd_transferred, dtype=float)
    jsd_random = np.asarray(jsd_random, dtype=float)
    cumulative_ns = np.asarray(cumulative_ns, dtype=float)

    def _crossing_ns(jsd: np.ndarray) -> float:
        """Return cumulative_ns at first JSD ≤ noise_floor via interpolation."""
        idx = np.where(jsd <= noise_floor)[0]
        if len(idx) == 0:
            return np.inf
        i = idx[0]
        if i == 0:
            return float(cumulative_ns[0])
        # Linear interpolation between point i-1 and i.
        t0, t1 = cumulative_ns[i - 1], cumulative_ns[i]
        j0, j1 = jsd[i - 1], jsd[i]
        if j1 == j0:
            return float(t0)
        frac = (noise_floor - j0) / (j1 - j0)
        return float(t0 + frac * (t1 - t0))

    t_transferred = _crossing_ns(jsd_transferred)
    t_random = _crossing_ns(jsd_random)

    if np.isinf(t_transferred) and np.isinf(t_random):
        return np.nan
    if np.isinf(t_transferred):
        return np.inf
    if np.isinf(t_random):
        # Random did not converge but transferred did — infinite acceleration.
        return np.inf
    return float(t_random / t_transferred)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def plot_convergence_curves(
    results_by_strategy: dict[str, list[dict[str, Any]]],
    noise_floor: float,
    output_path: Path,
) -> None:
    """JSD vs. cumulative simulation time for all three seeding strategies.

    Produces a line plot with one curve per strategy and a horizontal dashed
    line indicating the noise floor (best achievable JSD with 500 ns of
    ground-truth sampling).  The seeding JSD must cross below this line to
    claim ground-truth equivalence.

    Parameters
    ----------
    results_by_strategy : dict
        Same format as accepted by build_iteration_summary.
    noise_floor : float
        JSD noise floor from compute_noise_floor.
    output_path : Path
        Output file path (.png recommended).  Parent directory is created if
        it does not exist.

    Output
    ------
    PNG at *output_path*, 300 DPI, tight layout.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for strategy in _STRATEGY_ORDER:
        if strategy not in results_by_strategy:
            continue
        iters = sorted(results_by_strategy[strategy], key=lambda d: d["iteration"])
        ns_vals = [d["cumulative_ns"] for d in iters]
        jsd_vals = [d["jsd"] for d in iters]
        color = STRATEGY_COLORS.get(strategy, None)
        ax.plot(ns_vals, jsd_vals, marker="o", label=strategy, color=color, linewidth=2)

    ax.axhline(
        noise_floor,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Noise floor ({noise_floor:.3f})",
    )

    ax.set_xlabel("Cumulative simulation time (ns)", fontsize=11)
    ax.set_ylabel("Jensen-Shannon Divergence (bits)", fontsize=11)
    ax.set_title("FES Convergence: JSD vs. Cumulative Sampling", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def plot_coverage_curves(
    results_by_strategy: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Coverage (fraction of reference bins visited) vs. cumulative ns.

    Tracks how quickly each seeding strategy explores the full ground-truth
    landscape.  Higher coverage at lower cumulative time indicates more
    efficient seeding.

    Parameters
    ----------
    results_by_strategy : dict
        Same format as accepted by build_iteration_summary.
    output_path : Path
        Output file path (.png recommended).

    Output
    ------
    PNG at *output_path*, 300 DPI, tight layout.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for strategy in _STRATEGY_ORDER:
        if strategy not in results_by_strategy:
            continue
        iters = sorted(results_by_strategy[strategy], key=lambda d: d["iteration"])
        ns_vals = [d["cumulative_ns"] for d in iters]
        cov_vals = [d["coverage"] for d in iters]
        color = STRATEGY_COLORS.get(strategy, None)
        ax.plot(ns_vals, cov_vals, marker="s", label=strategy, color=color, linewidth=2)

    ax.set_xlabel("Cumulative simulation time (ns)", fontsize=11)
    ax.set_ylabel("Coverage (fraction of reference bins visited)", fontsize=11)
    ax.set_title("Exploration Coverage vs. Cumulative Sampling", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def plot_transferability_scores(
    d0: float,
    d_tau: float,
    d_km: float,
    pearson_r: float,
    output_path: Path,
) -> None:
    """Bar chart of TICA transferability scores D₀, D_τ, D_KM.

    The three distance metrics from Moffett & Shukla (arXiv:1710.00443)
    quantify how well the A10 TICA eigenvectors transfer to A8Q2:

    - D₀   : eigenvalue distance (slow-mode importance transfer)
    - D_τ  : implied timescale distance
    - D_KM : Koopman matrix distance

    All three are in [0, 1]; lower is better.  Pearson r between the
    transferred TIC-1 projection and N_HB on A8Q2 trajectories is displayed
    as a text annotation (higher is better; r ≥ 0.85 is a clean success).

    Reference thresholds
    --------------------
    - D < 0.3  : good transfer
    - D = 0.3–0.5 : partial transfer
    - D > 0.5  : poor transfer

    Parameters
    ----------
    d0, d_tau, d_km : float
        Transfer distance scores in [0, 1].
    pearson_r : float
        Pearson correlation between transferred TIC-1 and N_HB on A8Q2.
    output_path : Path
        Output file path (.png recommended).

    Output
    ------
    PNG at *output_path*, 300 DPI, tight layout.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["D₀", "D_τ", "D_KM"]
    values = [d0, d_tau, d_km]
    bar_colors = ["#4878d0", "#ee854a", "#6acc65"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=bar_colors, width=0.5, edgecolor="black",
                  linewidth=0.8)

    # Reference threshold lines.
    ax.axhline(0.3, color="green", linestyle="--", linewidth=1.2,
               label="Good transfer threshold (0.3)")
    ax.axhline(0.5, color="orange", linestyle="--", linewidth=1.2,
               label="Poor transfer threshold (0.5)")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.text(
        0.97, 0.97,
        f"Pearson r = {pearson_r:.3f}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Transfer distance score", fontsize=11)
    ax.set_title("TICA Transferability: A10 → A8Q2", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def plot_transfer_fes_1d(
    tic1_a8q2: np.ndarray,
    nhb_a8q2: np.ndarray,
    output_path: Path,
    tic1_a10: np.ndarray | None = None,
    nhb_a10: np.ndarray | None = None,
    helix_nhb_threshold: int = 3,
    n_bins: int = 50,
    kT_kcal: float = 0.656,
    # Legacy kwargs silently accepted so old call-sites don't break
    **_kwargs,
) -> None:
    """Two-panel 1D FES comparison: A10 (left) and A8Q2 (right).

    Each curve is continuously colored by the local helix fraction in each bin
    (orange/warm = helix-rich, blue/cool = coil-rich), replacing the previous
    arbitrary half-space shading and boundary line.

    Parameters
    ----------
    tic1_a8q2  : (n_frames,) transferred TIC-1 values on A8Q2
    nhb_a8q2   : (n_frames,) H-bond counts for A8Q2
    output_path: path to save figure
    tic1_a10   : (n_frames,) native A10 TIC-1 (left panel); if None, only one panel
    nhb_a10    : (n_frames,) H-bond counts for A10
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("RdBu_r")  # blue=coil (low f_helix), red=helix (high f_helix)
    norm = Normalize(vmin=0.0, vmax=1.0)

    def _fes_colored(tic1: np.ndarray, nhb: np.ndarray, ax: "plt.Axes", n: int = n_bins) -> None:
        """Plot FES as a LineCollection colored by per-bin helix fraction."""
        counts, edges = np.histogram(tic1, bins=n)
        centers = 0.5 * (edges[:-1] + edges[1:])
        prob = counts.astype(float) / counts.sum()
        with np.errstate(divide="ignore"):
            fes = -kT_kcal * np.log(np.where(prob > 0, prob, np.nan))
        fes -= np.nanmin(fes)

        # Per-bin helix fraction
        is_helix = nhb >= helix_nhb_threshold
        f_helix_bin = np.array([
            float(is_helix[(tic1 >= edges[i]) & (tic1 < edges[i + 1])].mean())
            if np.any((tic1 >= edges[i]) & (tic1 < edges[i + 1]))
            else 0.0
            for i in range(n)
        ])

        # Build LineCollection: one segment per adjacent pair of valid points
        valid = ~np.isnan(fes)
        xs = centers[valid]
        ys = fes[valid]
        fh = f_helix_bin[valid]

        if len(xs) < 2:
            ax.plot(xs, ys, color="gray", lw=2)
            return

        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_fh = 0.5 * (fh[:-1] + fh[1:])  # midpoint color for each segment

        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=3, zorder=3)
        lc.set_array(seg_fh)
        ax.add_collection(lc)
        ax.set_xlim(xs.min() - 0.05 * (xs.max() - xs.min()),
                    xs.max() + 0.05 * (xs.max() - xs.min()))
        ax.set_ylim(-0.1, np.nanmax(ys) * 1.12)

        # Annotate helix/coil wells
        helix_center = float(xs[fh == fh.max()][0]) if fh.max() > 0 else xs[0]
        coil_center = float(xs[fh == fh.min()][-1]) if fh.min() < 1 else xs[-1]
        ymax = float(np.nanmax(ys))
        ax.text(helix_center, ymax * 0.97, "Helix", ha="center", va="top",
                fontsize=8.5, color=cmap(0.95), fontweight="bold")
        ax.text(coil_center, ymax * 0.97, "Coil", ha="center", va="top",
                fontsize=8.5, color=cmap(0.05), fontweight="bold")
        return lc

    has_a10 = tic1_a10 is not None and nhb_a10 is not None
    ncols = 2 if has_a10 else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2), sharey=False,
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    if has_a10:
        _fes_colored(tic1_a10.ravel(), nhb_a10, axes[0])
        axes[0].set_xlabel("TIC-1 (A10 native)")
        axes[0].set_ylabel("Free Energy (kcal/mol)")
        axes[0].set_title("A10 (Ac-Ala$_{10}$-NHMe)")

    _fes_colored(tic1_a8q2.ravel(), nhb_a8q2, axes[-1])
    axes[-1].set_xlabel("TIC-1 (transferred A10→A8Q2)")
    if not has_a10:
        axes[-1].set_ylabel("Free Energy (kcal/mol)")
    axes[-1].set_title("A8Q2 (Ac-Ala$_8$Gln$_2$-NHMe)")

    # Shared colorbar on the right
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist() if ncols > 1 else axes[0],
                        fraction=0.025, pad=0.02, aspect=20)
    cbar.set_label("Local helix fraction $f_\\mathrm{helix}$", fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle("1D Free Energy Profile: Helix/Coil Content Along TIC-1", fontsize=11)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: ENTROPY AND F_HELIX CURVE PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def _plot_generic_curve(
    results_by_strategy: dict[str, list[dict[str, Any]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    hline: float | None = None,
    hline_label: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generic line-plot helper for any per-iteration scalar metric."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for strategy in _STRATEGY_ORDER:
        if strategy not in results_by_strategy:
            continue
        iters = sorted(results_by_strategy[strategy], key=lambda d: d["iteration"])
        ns_vals = [d["cumulative_ns"] for d in iters]
        metric_vals = [d[metric_key] for d in iters]
        color = STRATEGY_COLORS.get(strategy, None)
        ax.plot(ns_vals, metric_vals, marker="^", label=strategy, color=color,
                linewidth=2)

    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.2,
                   label=hline_label or f"Reference ({hline:.3f})")

    ax.set_xlabel("Cumulative simulation time (ns)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FULL FIGURE SET
# ─────────────────────────────────────────────────────────────────────────────


def generate_final_figures(results_dir: Path, output_dir: Path) -> None:
    """Load JSON results and generate the complete figure set for the thesis.

    Expected files in *results_dir*
    --------------------------------
    - ``jsd_curves_random.json``
    - ``jsd_curves_transferred.json``
    - ``jsd_curves_native.json``
    - ``noise_floor.json``           (keys: 'noise_floor', 'hist_replica1',
                                       'hist_replica2')
    - ``transferability_scores.json`` (keys: 'd0', 'd_tau', 'd_km',
                                        'pearson_r')
    - ``fes_a10.npy``                 (2D FES PMF array, shape FES_NBINS×FES_NBINS)
    - ``fes_a8q2.npy``                (same)
    - ``helix_fraction_per_residue_a10.npy``   (shape (n_residues,))
    - ``helix_fraction_per_residue_a8q2.npy``  (same)

    Figures produced in *output_dir*
    ---------------------------------
    1. ``fig_jsd_convergence.png``         — JSD vs. cumulative ns
    2. ``fig_coverage_curves.png``         — Coverage vs. cumulative ns
    3. ``fig_entropy_curves.png``          — Entropy vs. cumulative ns
    4. ``fig_f_helix_error_curves.png``    — f_helix error vs. cumulative ns
    5. ``fig_transferability_scores.png``  — D₀, D_τ, D_KM bar chart
    6. ``fig_fes_comparison.png``          — 2D FES side-by-side (A10 vs A8Q2)
    7. ``fig_helix_fraction_per_residue.png`` — Per-residue helix fraction overlay

    Parameters
    ----------
    results_dir : Path
        Directory containing all JSON/npy result files.
    output_dir : Path
        Directory in which figures are written.  Created if absent.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load_json(name: str) -> dict:
        p = results_dir / name
        if not p.exists():
            warnings.warn(f"generate_final_figures: {name} not found — skipping.", UserWarning)
            return {}
        with open(p) as f:
            return json.load(f)

    def _load_npy(name: str) -> np.ndarray | None:
        p = results_dir / name
        if not p.exists():
            warnings.warn(f"generate_final_figures: {name} not found — skipping.", UserWarning)
            return None
        return np.load(p)

    # ── Load strategy results ──────────────────────────────────────────────
    strategy_map = {
        "Random":           "jsd_curves_random.json",
        "Transferred TICA": "jsd_curves_transferred.json",
        "Native TICA":      "jsd_curves_native.json",
    }
    results_by_strategy: dict[str, list[dict]] = {}
    for strategy, fname in strategy_map.items():
        data = _load_json(fname)
        if data:
            # Accept either a list directly or wrapped under an 'iterations' key.
            if isinstance(data, list):
                results_by_strategy[strategy] = data
            elif "iterations" in data:
                results_by_strategy[strategy] = data["iterations"]

    noise_data = _load_json("noise_floor.json")
    noise_floor = float(noise_data.get("noise_floor", np.nan))

    ts_data = _load_json("transferability_scores.json")

    # ── Figure 1: JSD convergence ──────────────────────────────────────────
    if results_by_strategy and not np.isnan(noise_floor):
        plot_convergence_curves(
            results_by_strategy,
            noise_floor,
            output_dir / "fig_jsd_convergence.png",
        )

    # ── Figure 2: Coverage curves ──────────────────────────────────────────
    if results_by_strategy:
        plot_coverage_curves(
            results_by_strategy,
            output_dir / "fig_coverage_curves.png",
        )

    # ── Figure 3: Entropy curves ───────────────────────────────────────────
    if results_by_strategy:
        _plot_generic_curve(
            results_by_strategy,
            metric_key="entropy",
            ylabel="Normalized entropy H(q) / log₂(|Ω|)",
            title="Distribution Entropy vs. Cumulative Sampling",
            output_path=output_dir / "fig_entropy_curves.png",
            ylim=(0.0, 1.05),
        )

    # ── Figure 4: f_helix error curves ────────────────────────────────────
    if results_by_strategy:
        _plot_generic_curve(
            results_by_strategy,
            metric_key="f_helix_error",
            ylabel="|f_helix_seeded − f_helix_ground_truth|",
            title="Helix Fraction Error vs. Cumulative Sampling",
            output_path=output_dir / "fig_f_helix_error_curves.png",
            hline=0.02,
            hline_label="Convergence target (0.02)",
            ylim=(0.0, None),
        )

    # ── Figure 5: Transferability scores ──────────────────────────────────
    if ts_data:
        plot_transferability_scores(
            d0=float(ts_data.get("D0", ts_data.get("d0", np.nan))),
            d_tau=float(ts_data.get("D_tau", ts_data.get("d_tau", np.nan))),
            d_km=float(ts_data.get("D_KM", ts_data.get("d_km", np.nan))),
            pearson_r=float(ts_data.get("pearson_r", np.nan)),
            output_path=output_dir / "fig_transferability_scores.png",
        )

    # ── Figure 6: 2D FES comparison ───────────────────────────────────────
    fes_a10 = _load_npy("fes_a10.npy")
    fes_a8q2 = _load_npy("fes_a8q2.npy")
    if fes_a10 is not None and fes_a8q2 is not None:
        _plot_fes_comparison(fes_a10, fes_a8q2, output_dir / "fig_fes_comparison.png")

    # ── Figure 7: Per-residue helix fraction ──────────────────────────────
    hf_a10 = _load_npy("helix_fraction_per_residue_a10.npy")
    hf_a8q2 = _load_npy("helix_fraction_per_residue_a8q2.npy")
    if hf_a10 is not None and hf_a8q2 is not None:
        _plot_helix_fraction_per_residue(
            hf_a10, hf_a8q2, output_dir / "fig_helix_fraction_per_residue.png"
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _plot_fes_comparison(
    fes_a10: np.ndarray,
    fes_a8q2: np.ndarray,
    output_path: Path,
) -> None:
    """Side-by-side 2D PMF comparison: A10 vs. A8Q2.

    Both PMFs are converted from raw count arrays to kcal/mol via::

        G(i,j) = −kB·T · ln(P(i,j))

    with the minimum set to 0.  Bins with zero probability are shown as white
    (NaN masking).

    Parameters
    ----------
    fes_a10, fes_a8q2 : np.ndarray
        2D raw count arrays, shape (FES_NBINS, FES_NBINS).
    output_path : Path
        Output file path.
    """
    from src.config import TEMPERATURE_K

    kbt_kcal = 1.987e-3 * TEMPERATURE_K  # kB in kcal/mol/K

    def _counts_to_pmf(counts: np.ndarray) -> np.ndarray:
        counts = counts.astype(float)
        prob = counts / counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmf = -kbt_kcal * np.log(np.where(prob > 0, prob, np.nan))
        pmf -= np.nanmin(pmf)
        return pmf

    pmf_a10 = _counts_to_pmf(fes_a10)
    pmf_a8q2 = _counts_to_pmf(fes_a8q2)

    vmax = max(np.nanmax(pmf_a10), np.nanmax(pmf_a8q2))
    vmax = min(vmax, 6.0)  # cap at 6 kcal/mol for visual clarity

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    titles = ["A10 (Ac-Ala₁₀-NHMe)", "A8Q2 (Ac-Ala₈Gln₂-NHMe)"]

    for ax, pmf, title in zip(axes, [pmf_a10, pmf_a8q2], titles):
        im = ax.imshow(
            pmf.T,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("RMSD bin", fontsize=10)
        ax.set_ylabel("N_HB bin", fontsize=10)
        fig.colorbar(im, ax=ax, label="PMF (kcal/mol)")

    fig.suptitle("2D Free Energy Surface Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def _plot_helix_fraction_per_residue(
    hf_a10: np.ndarray,
    hf_a8q2: np.ndarray,
    output_path: Path,
) -> None:
    """Per-residue helix fraction overlay for A10 and A8Q2.

    Highlights the reduction in helix content at Gln positions (residues 3
    and 8 in A8Q2).  Uses a grouped bar chart with error bars if provided as
    shape (n_residues, 2) or a simple line plot for 1-D inputs.

    Parameters
    ----------
    hf_a10, hf_a8q2 : np.ndarray
        Per-residue helix fractions, shape (n_residues,).
    output_path : Path
        Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_res = len(hf_a10)
    residues = np.arange(1, n_res + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(residues, hf_a10,  marker="o", color="#1f77b4", linewidth=2,
            label="A10")
    ax.plot(residues, hf_a8q2, marker="s", color="#ff7f0e", linewidth=2,
            label="A8Q2")

    # Mark Gln positions in A8Q2 (3 and 8, 1-indexed).
    for gln_pos in [3, 8]:
        if 1 <= gln_pos <= n_res:
            ax.axvline(gln_pos, color="gray", linestyle=":", linewidth=1.0,
                       label="Gln position" if gln_pos == 3 else None)

    ax.set_xlabel("Residue index", fontsize=11)
    ax.set_ylabel("Helix fraction", fontsize=11)
    ax.set_title("Per-Residue Helix Fraction: A10 vs. A8Q2", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(residues)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# POST-SEEDING: ground-truth split hists + adaptive_seeding_results.json → Phase 9
# ─────────────────────────────────────────────────────────────────────────────


def ground_truth_fes_histograms_split(
    dcd_helix: str | Path,
    top_helix: str | Path,
    dcd_extended: str | Path,
    top_extended: str | Path,
    nbins: int = FES_NBINS,
) -> tuple[np.ndarray, np.ndarray]:
    """2D histograms (sin φ₁ × cos φ₁) per replica — same grid as adaptive seeding GT.

    Used with :func:`compute_noise_floor` for Phase 9 noise-floor JSON.
    """
    import mdtraj as md

    from src.utils import featurize

    hists: list[np.ndarray] = []
    for dcd_path, top_pdb in (
        (str(dcd_helix), str(top_helix)),
        (str(dcd_extended), str(top_extended)),
    ):
        traj_full = md.load_dcd(dcd_path, top=top_pdb)
        traj_prot = traj_full.atom_slice(traj_full.topology.select("protein"))
        feats = featurize(traj_prot)
        x = feats[:, 0]
        y = feats[:, 10]
        hist, _, _ = np.histogram2d(
            x, y,
            bins=nbins,
            range=[[-1.05, 1.05], [-1.05, 1.05]],
        )
        hists.append(hist)
    return hists[0], hists[1]


def iteration_records_from_adaptive_seeding_json(
    path: str | Path,
    *,
    seed_ns: float = 30.0,
    n_seeds: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Load ``adaptive_seeding_results.json`` into plot-ready strategy lists.

    Skips iterations with ``jsd is None`` (e.g. iteration 1 summaries).
    ``cumulative_ns`` = iteration × n_seeds × seed_ns (protocol budget accounting).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    key_map = {
        "random": "Random",
        "transferred": "Transferred TICA",
        "native": "Native TICA",
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for raw_strat, iters_map in data.items():
        disp = key_map.get(raw_strat)
        if disp is None:
            continue
        rows: list[dict[str, Any]] = []
        for it_key in sorted(iters_map.keys(), key=lambda x: int(x)):
            summ = iters_map[it_key]
            jsd = summ.get("jsd")
            if jsd is None:
                continue
            itn = int(it_key)
            cum = itn * n_seeds * seed_ns
            cov = summ.get("coverage")
            ent = summ.get("entropy")
            fhe = summ.get("f_helix_error")
            rows.append(
                {
                    "iteration": itn,
                    "cumulative_ns": float(cum),
                    "jsd": float(jsd),
                    "coverage": float(cov) if cov is not None else 0.0,
                    "entropy": float(ent) if ent is not None else 0.0,
                    "f_helix_error": float(fhe) if fhe is not None else float("nan"),
                }
            )
        if rows:
            out[disp] = rows
    return out


def write_phase9_strategy_jsons(
    adaptive_json: str | Path,
    output_dir: str | Path,
    *,
    seed_ns: float = 30.0,
    n_seeds: int = 8,
) -> None:
    """Write ``jsd_curves_{random,transferred,native}.json`` for :func:`generate_final_figures`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_strat = iteration_records_from_adaptive_seeding_json(
        adaptive_json, seed_ns=seed_ns, n_seeds=n_seeds,
    )
    fname_map = {
        "Random": "jsd_curves_random.json",
        "Transferred TICA": "jsd_curves_transferred.json",
        "Native TICA": "jsd_curves_native.json",
    }
    for strat, fname in fname_map.items():
        if strat not in by_strat:
            continue
        with open(output_dir / fname, "w", encoding="utf-8") as fh:
            json.dump(by_strat[strat], fh, indent=2)


def write_noise_floor_json(
    hist_helix: np.ndarray,
    hist_extended: np.ndarray,
    output_path: str | Path,
    min_count: int = FES_MIN_COUNT,
) -> float:
    """Write ``noise_floor.json`` and return the scalar noise floor."""
    nf = compute_noise_floor(hist_helix, hist_extended, min_count=min_count)
    output_path = Path(output_path)
    payload = {
        "noise_floor": float(nf),
        "description": (
            "JSD between helix-start and extended-start A8Q2 replicas "
            "(masked domain from pooled counts)."
        ),
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return float(nf)


def write_transferability_scores_from_transfer_summary(
    transfer_summary_json: str | Path,
    output_path: str | Path,
) -> None:
    """Map Phase 7 ``transfer_summary.json`` keys for :func:`generate_final_figures`."""
    with open(transfer_summary_json, encoding="utf-8") as fh:
        s = json.load(fh)
    out = {
        "d0": float(s.get("D0", s.get("d0", float("nan")))),
        "d_tau": float(s.get("D_tau", s.get("d_tau", float("nan")))),
        "d_km": float(s.get("D_KM", s.get("d_km", float("nan")))),
        "pearson_r": float(s.get("pearson_r", float("nan"))),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


def run_phase9_bundle_for_seeding_run(
    *,
    adaptive_results_json: Path,
    sim_dir: Path,
    prefix: str,
    transfer_summary_json: Path | None,
    bundle_dir: Path,
    seed_ns: float = 30.0,
    n_seeds: int = 8,
) -> None:
    """Assemble JSON inputs and call :func:`generate_final_figures` into ``bundle_dir/figures``."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    dcd_h = sim_dir / f"{prefix}_helix_prod.dcd"
    top_h = sim_dir / f"{prefix}_helix_prod_top.pdb"
    dcd_e = sim_dir / f"{prefix}_extended_prod.dcd"
    top_e = sim_dir / f"{prefix}_extended_prod_top.pdb"
    h_h, h_e = ground_truth_fes_histograms_split(dcd_h, top_h, dcd_e, top_e)
    write_noise_floor_json(h_h, h_e, bundle_dir / "noise_floor.json")
    write_phase9_strategy_jsons(
        adaptive_results_json,
        bundle_dir,
        seed_ns=seed_ns,
        n_seeds=n_seeds,
    )
    if transfer_summary_json is not None and transfer_summary_json.is_file():
        write_transferability_scores_from_transfer_summary(
            transfer_summary_json,
            bundle_dir / "transferability_scores.json",
        )
    generate_final_figures(bundle_dir, bundle_dir / "figures")
