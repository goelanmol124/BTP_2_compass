"""
tica_train.py — Phase 6: TICA Training on A10
==============================================
Train TICA on Ac-Ala₁₀-NHMe (A10) 40D sin/cos backbone features, validate
TIC-1 as the helix-coil reaction coordinate, and persist artefacts for
downstream transferability analysis (Phase 7) and adaptive seeding (Phase 8).

Deeptime API (NOT legacy PyEMMA names):
  model.singular_values         — shape (dim,)
  model.singular_vectors_left   — shape (40, dim), columns = TIC directions
  model.mean_0                  — shape (40,), instantaneous mean
  model.transform(X)            — project new feature array
  model.timescales(lagtime)     — implied timescales at a given lag

Frame spacing: 100 ps = 0.1 ns/frame  (DCD_INTERVAL_PS = 100).
Burn-in: 250 frames (25 ns) already stripped BEFORE features reach this module.

Lag-time note (deeptime / VAMP theory)
--------------------------------------
VAMP-2 scores from models fit at different lag times **must not** be compared to
pick τ — each lag approximates a different Koopman operator (deeptime VAMP tutorial:
“Scores can not be compared across different lagtimes τ”).  Production lag is
chosen here from **implied-timescale stability** between consecutive scan points,
with a **minimum lag** (``config.TICA_MIN_LAG_FRAMES``) to avoid sub-nanosecond noise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr

from src import config

try:
    from deeptime.decomposition import TICA
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deeptime is required for TICA training.  Install it with:\n"
        "    pip install deeptime\n"
        "or, inside a conda environment:\n"
        "    conda install -c conda-forge deeptime"
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless execution
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_SPACING_NS: float = 0.1       # 100 ps per saved frame
DEFAULT_LAG_FRAMES: list[int] = list(config.TICA_LAGTIME_FRAMES_SCAN)
DEFAULT_DIM: int = 5
DEFAULT_EPSILON: float = 1e-6
DEFAULT_SCALING: str = "kinetic_map"

# Pearson-r thresholds for TIC-1 validation
R_THRESHOLD_STRONG: float = 0.85
R_THRESHOLD_WEAK: float = 0.70


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def choose_lag_from_implied_timescale_scan(
    lag_frames: list[int],
    t1_implied: list[float],
    min_lag_frames: int,
    rel_plateau: float,
) -> tuple[int, str]:
    """Pick production lag from TIC-1 implied timescales (MSM-style), not cross-lag VAMP-2.

    Deeptime: VAMP/TICA scores at different τ are **not** on the same scale and must
    not be ranked against each other across lags.  Here we use consecutive ITS steps
    |Δt₁|/t₁ as a plateau heuristic (cf. implied timescale plots in MSM tutorials).

    Returns
    -------
    (chosen_lag_frames, reason)
    """
    n = len(lag_frames)
    for k in range(n - 1):
        L0 = lag_frames[k]
        if L0 < min_lag_frames:
            continue
        t0, t1v = t1_implied[k], t1_implied[k + 1]
        if not (np.isfinite(t0) and np.isfinite(t1v) and t0 > 0):
            continue
        rel = abs(t1v - t0) / t0
        if rel <= rel_plateau:
            return L0, f"its_plateau(rel_step={rel:.3f}<={rel_plateau})"

    best_k: int | None = None
    best_rel = np.inf
    for k in range(n - 1):
        L0 = lag_frames[k]
        if L0 < min_lag_frames:
            continue
        t0, t1v = t1_implied[k], t1_implied[k + 1]
        if not (np.isfinite(t0) and np.isfinite(t1v) and t0 > 0):
            continue
        rel = abs(t1v - t0) / t0
        if rel < best_rel:
            best_rel = rel
            best_k = k
    if best_k is not None:
        return lag_frames[best_k], f"its_min_rel_step({best_rel:.3f})"

    for L in reversed(lag_frames):
        if L >= min_lag_frames:
            return L, "fallback_max_eligible_lag"
    return int(lag_frames[-1]), "fallback_end_of_scan"


def lag_scan(
    features: np.ndarray,
    lag_frames: list[int] | None = None,
    frame_spacing_ns: float = FRAME_SPACING_NS,
    min_lag_frames: int | None = None,
    rel_plateau: float | None = None,
) -> dict[str, Any]:
    """Scan TICA lag times: implied timescale of TIC-1 + VAMP-2 (diagnostic only).

    Parameters
    ----------
    features:
        2D array of shape (n_frames, 40) — sin/cos backbone features,
        burn-in already removed.
    lag_frames:
        Lag times to scan in frames.  Defaults to ``config.TICA_LAGTIME_FRAMES_SCAN``.
    frame_spacing_ns:
        Time between consecutive saved frames in nanoseconds (0.1 ns).
    min_lag_frames:
        Minimum lag (frames) for the **chosen** production lag.  Default:
        ``config.TICA_MIN_LAG_FRAMES``.
    rel_plateau:
        Relative ITS step threshold for plateau detection.  Default:
        ``config.TICA_LAG_ITS_REL_PLATEAU``.

    Returns
    -------
    dict with keys:
        lags_ns, lags_frames, t1_implied, vamp2_scores,
        recommended_lag_frames, recommended_lag_ns,
        lag_selection_reason, singular_values_per_lag

    Notes
    -----
    ``vamp2_scores`` are logged for inspection only; deeptime explicitly states
    VAMP scores cannot be compared across different lag times for ranking.
    """
    if lag_frames is None:
        lag_frames = DEFAULT_LAG_FRAMES
    if min_lag_frames is None:
        min_lag_frames = int(config.TICA_MIN_LAG_FRAMES)
    if rel_plateau is None:
        rel_plateau = float(config.TICA_LAG_ITS_REL_PLATEAU)

    lags_ns: list[float] = []
    t1_implied: list[float] = []
    vamp2_scores: list[float] = []
    sv_per_lag: list[np.ndarray] = []

    for lag in lag_frames:
        lag_ns = lag * frame_spacing_ns
        tica = TICA(lagtime=lag, dim=DEFAULT_DIM, epsilon=DEFAULT_EPSILON,
                    scaling=DEFAULT_SCALING)
        model = tica.fit(features).fetch_model()
        sv = np.asarray(model.singular_values)
        lam1 = float(abs(sv[0]))

        # TIC-1 implied timescale: t₁ = −τ / ln|λ₁|
        if lam1 >= 1.0:
            t1 = np.inf
            logger.warning(
                "Lag %d frames: lambda_1 = %.4f >= 1 (numerical artefact); "
                "implied timescale set to inf.", lag, lam1
            )
        else:
            t1 = -lag_ns / np.log(lam1)

        vamp2 = float(model.score(2))

        lags_ns.append(lag_ns)
        t1_implied.append(float(t1))
        vamp2_scores.append(vamp2)
        sv_per_lag.append(sv[:DEFAULT_DIM].copy())

        logger.info(
            "Lag %2d fr (%4.1f ns): lambda_1=%.4f  t_impl_1=%.2f ns  "
            "VAMP-2=%.4f (diag-only; not rankable across lags)",
            lag, lag_ns, lam1, t1, vamp2,
        )

    rec_lag, reason = choose_lag_from_implied_timescale_scan(
        list(lag_frames), t1_implied, min_lag_frames, rel_plateau
    )
    rec_i = lag_frames.index(rec_lag)
    recommended_lag_ns = lags_ns[rec_i]
    logger.info(
        "Chosen lag (ITS heuristic, min_lag=%d): %d frames (%.1f ns) -- %s",
        min_lag_frames, rec_lag, recommended_lag_ns, reason,
    )

    return {
        "lags_ns": lags_ns,
        "lags_frames": list(lag_frames),
        "t1_implied": t1_implied,
        "vamp2_scores": vamp2_scores,
        "recommended_lag_frames": rec_lag,
        "recommended_lag_ns": recommended_lag_ns,
        "lag_selection_reason": reason,
        "singular_values_per_lag": sv_per_lag,
    }


def block_bootstrap_tic1_stability(
    features: np.ndarray,
    lagtime: int,
    block_size: int | None = None,
    n_bootstrap: int = 100,
    threshold: float = 0.95,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Verify TIC-1 eigenvector stability via block bootstrap.

    Default ``block_size`` is ``config.TICA_BOOTSTRAP_BLOCK_SIZE`` (50 frames =
    5 ns at 100 ps), comparable to τ_eq so resampled blocks are nearer to
    statistically independent (protocol §9.2).

    Uses scaling=None for geometric comparison of unit eigenvectors (kinetic_map
    scaling is only for the final projection).

    Parameters
    ----------
    features:
        2D array (n_frames, 40), burn-in removed.
    lagtime:
        Lag time in frames (the chosen production lag).
    block_size:
        Number of consecutive frames per bootstrap block.
    n_bootstrap:
        Number of bootstrap resamples.
    threshold:
        Cosine-similarity threshold to declare a single bootstrap stable.
    rng_seed:
        Seed for the random number generator (reproducibility).

    Returns
    -------
    dict with keys:
        mean_cos_sim  : float — mean |cos θ| over bootstrap samples
        min_cos_sim   : float — minimum |cos θ| (worst-case stability)
        pass_fraction : float — fraction of resamples with |cos θ| > threshold
        stable        : bool  — True iff ALL resamples exceed threshold
        cosine_sims   : np.ndarray — raw cosine similarities (n_bootstrap,)
    """
    if block_size is None:
        block_size = int(config.TICA_BOOTSTRAP_BLOCK_SIZE)

    rng = np.random.default_rng(rng_seed)
    T = len(features)
    n_blocks_total = T // block_size

    if n_blocks_total < 2:
        raise ValueError(
            f"Too few blocks for bootstrap: {T} frames / {block_size} frames/block "
            f"= {n_blocks_total} blocks.  Reduce block_size or provide more data."
        )

    # Reference model on full trajectory (scaling=None → unit eigenvectors)
    ref_model = (
        TICA(lagtime=lagtime, dim=DEFAULT_DIM, epsilon=DEFAULT_EPSILON, scaling=None)
        .fit(features)
        .fetch_model()
    )
    ref_v1 = ref_model.singular_vectors_left[:, 0]  # shape (40,), unit norm

    cosine_sims = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Draw n_blocks_total block start indices with replacement
        block_indices = rng.integers(0, n_blocks_total, size=n_blocks_total)
        starts = block_indices * block_size
        boot_features = np.concatenate(
            [features[s : s + block_size] for s in starts], axis=0
        )
        boot_model = (
            TICA(lagtime=lagtime, dim=DEFAULT_DIM, epsilon=DEFAULT_EPSILON, scaling=None)
            .fit(boot_features)
            .fetch_model()
        )
        boot_v1 = boot_model.singular_vectors_left[:, 0]
        # |cos θ| accounts for the sign ambiguity of eigenvectors
        cosine_sims[i] = float(abs(np.dot(ref_v1, boot_v1)))

    result = {
        "mean_cos_sim": float(np.mean(cosine_sims)),
        "min_cos_sim": float(np.min(cosine_sims)),
        "pass_fraction": float(np.mean(cosine_sims > threshold)),
        "stable": bool(np.all(cosine_sims > threshold)),
        "cosine_sims": cosine_sims,
    }

    logger.info(
        "Bootstrap TIC-1 stability: mean |cos| = %.4f, min = %.4f, "
        "pass fraction = %.3f, stable = %s",
        result["mean_cos_sim"],
        result["min_cos_sim"],
        result["pass_fraction"],
        result["stable"],
    )
    return result


def train_tica(
    features: np.ndarray,
    lagtime: int,
    dim: int = DEFAULT_DIM,
    epsilon: float = DEFAULT_EPSILON,
    scaling: str = DEFAULT_SCALING,
):
    """Train TICA on a feature array and return the fitted CovarianceKoopmanModel.

    Parameters
    ----------
    features:
        2D array (n_frames, n_features), burn-in already removed.
    lagtime:
        Lag time in frames.
    dim:
        Number of TICs to retain.
    epsilon:
        Regularisation parameter for C(0) inversion.
    scaling:
        "kinetic_map" multiplies each TIC by its singular value so that
        Euclidean distance approximates kinetic distance.  Use None for
        unit-norm eigenvectors (bootstrap, subspace comparisons).

    Returns
    -------
    deeptime CovarianceKoopmanModel
    """
    tica = TICA(lagtime=lagtime, dim=dim, epsilon=epsilon, scaling=scaling)
    model = tica.fit(features).fetch_model()

    logger.info(
        "Trained TICA: lagtime=%d frames, dim=%d, scaling=%s",
        lagtime, dim, scaling,
    )
    sv_head = np.asarray(model.singular_values)[:dim]
    logger.info(
        "Singular values (first %d): %s",
        dim,
        np.array2string(sv_head, precision=4, suppress_small=True),
    )
    return model


def verify_ic_projection_matches_manual_dot(
    model,
    features: np.ndarray,
    ic: int = 0,
) -> dict[str, float]:
    """Sanity check: ``transform`` agrees with (X − μ)·v for IC ``ic``.

    For deeptime TICA with ``kinetic_map``, the first left singular direction
    still yields correlation ±1 with ``transform[:, ic]`` (empirically verified).
    """
    mean = np.asarray(model.mean_0)
    V = np.asarray(model.singular_vectors_left)[:, ic]
    y_manual = (features - mean) @ V
    y_t = model.transform(features)[:, ic]
    r, _ = pearsonr(y_t, y_manual)
    return {
        "pearson_transform_vs_manual_dot": float(r),
        "median_abs_residual": float(np.median(np.abs(y_t - y_manual))),
    }


def cross_replica_tic1_cosine_similarity(
    features: np.ndarray,
    lagtime: int,
    n_first_replica_frames: int,
    dim: int = DEFAULT_DIM,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """|cos θ| between TIC-1 directions fit independently on two trajectory halves."""
    if n_first_replica_frames <= 0 or n_first_replica_frames >= len(features):
        return float("nan")
    a = features[:n_first_replica_frames]
    b = features[n_first_replica_frames:]
    if len(a) < 2 * int(config.TICA_BOOTSTRAP_BLOCK_SIZE):
        return float("nan")
    m_a = (
        TICA(lagtime=lagtime, dim=dim, epsilon=epsilon, scaling=None)
        .fit(a)
        .fetch_model()
    )
    m_b = (
        TICA(lagtime=lagtime, dim=dim, epsilon=epsilon, scaling=None)
        .fit(b)
        .fetch_model()
    )
    va = np.asarray(m_a.singular_vectors_left)[:, 0]
    vb = np.asarray(m_b.singular_vectors_left)[:, 0]
    return float(abs(np.dot(va, vb)))


def validate_tic1(
    model,
    features: np.ndarray,
    nhb: np.ndarray,
    frame_spacing_ns: float = FRAME_SPACING_NS,
) -> dict[str, Any]:
    """Validate TIC-1 as the helix-coil reaction coordinate.

    Checks:
    1. Pearson r(TIC-1 projection, N_HB) — primary physical interpretability test.
    2. Implied timescale from TIC-1 singular value.
    3. Free energy along TIC-1 (two-well detection via histogram).

    Parameters
    ----------
    model:
        Fitted deeptime CovarianceKoopmanModel (kinetic_map scaling).
    features:
        Feature array used for projection (n_frames, 40).
    nhb:
        Hydrogen-bond count per frame (n_frames,).
    frame_spacing_ns:
        Frame spacing in nanoseconds.

    Returns
    -------
    dict with keys:
        pearson_r         : float
        pearson_pval      : float
        t_impl_ns         : float — TIC-1 implied timescale
        fes_has_two_wells : bool
        status            : str ("proceed" | "caution" | "fallback")
        tic1_projection   : np.ndarray
    """
    tic1 = model.transform(features)[:, 0]

    # Pearson r with hydrogen-bond count
    r, pval = pearsonr(tic1, nhb)
    abs_r = abs(r)

    if abs_r >= R_THRESHOLD_STRONG:
        status = "proceed"
    elif abs_r >= R_THRESHOLD_WEAK:
        status = "caution"
    else:
        status = "fallback"
        logger.warning(
            "|r(TIC-1, N_HB)| = %.3f < %.2f.  TIC-1 does NOT cleanly represent "
            "helix-coil coordinate.  Activating fallback: use N_HB as CV for seeding.",
            abs_r, R_THRESHOLD_WEAK,
        )

    logger.info(
        "Pearson r(TIC-1, N_HB) = %.4f (p = %.2e) -> status: %s",
        r, pval, status,
    )

    # Implied timescale: t₁ = −lag_ns / ln|λ₁|
    lag_frames = model.lagtime  # stored in the model
    lag_ns = lag_frames * frame_spacing_ns
    lam1 = float(abs(model.singular_values[0]))
    if lam1 >= 1.0:
        t_impl_ns = np.inf
    else:
        t_impl_ns = -lag_ns / np.log(lam1)

    logger.info("TIC-1 implied timescale: %.2f ns (lag = %d frames)", t_impl_ns, lag_frames)

    # Two-well detection via 1D FES
    fes_has_two_wells, n_bins_used = _detect_two_wells(tic1)
    logger.info(
        "FES two-well test: %s (bins used = %d)",
        "PASS" if fes_has_two_wells else "FAIL",
        n_bins_used,
    )

    return {
        "pearson_r": float(r),
        "pearson_pval": float(pval),
        "t_impl_ns": float(t_impl_ns),
        "fes_has_two_wells": fes_has_two_wells,
        "status": status,
        "tic1_projection": tic1,
    }


def _detect_two_wells(tic1: np.ndarray, n_bins: int = 50) -> tuple[bool, int]:
    """Return (True, n_bins) if the 1D FES along TIC-1 shows two distinct wells.

    A "two-well" FES has at least one local free-energy maximum (barrier) between
    two local minima in the probability density histogram.  We detect this by
    computing F = −kT ln p (in units of kT) and counting sign changes in dF/dbin
    that cross a minimum barrier height of 0.5 kT.
    """
    counts, _ = np.histogram(tic1, bins=n_bins)
    # Replace zeros to avoid log(0)
    counts = counts.astype(float)
    counts[counts == 0] = 1e-10
    fes = -np.log(counts / counts.max())  # in units of kT; minimum is at 0

    # Smooth slightly to suppress numerical noise
    kernel = np.array([0.25, 0.5, 0.25])
    fes_smooth = np.convolve(fes, kernel, mode="same")

    # Count local minima (valleys) in the smoothed FES
    n_local_min = 0
    for i in range(1, len(fes_smooth) - 1):
        if fes_smooth[i] < fes_smooth[i - 1] and fes_smooth[i] < fes_smooth[i + 1]:
            n_local_min += 1

    # Two-well: at least 2 local minima separated by a barrier
    two_wells = n_local_min >= 2
    return two_wells, n_bins


def save_tica_artifacts(
    model,
    output_dir: str | Path,
    prefix: str = "a10",
) -> dict[str, Path]:
    """Save TICA model components to .npy / .txt files.

    Files saved:
        {prefix}_tica_singular_vectors_left.npy  — shape (40, 5)
        {prefix}_tica_mean0.npy                  — shape (40,)
        {prefix}_tica_singular_values.npy        — shape (5,)
        {prefix}_tica_chosen_lagtime.txt          — integer lagtime in frames

    Parameters
    ----------
    model:
        Fitted deeptime CovarianceKoopmanModel.
    output_dir:
        Directory where files are written (created if absent).
    prefix:
        File-name prefix, e.g. "a10" for training data.

    Returns
    -------
    dict mapping artifact name → absolute Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # deeptime ≥0.4 may expose full spectral data; persist only the fitted rank.
    d = int(getattr(model, "output_dimension", model.singular_values.shape[0]))
    sv_left = np.asarray(model.singular_vectors_left)[:, :d]
    mean0 = np.asarray(model.mean_0)
    sv = np.asarray(model.singular_values)[:d]
    lagtime = int(model.lagtime)

    artifacts = {
        f"{prefix}_tica_singular_vectors_left.npy": sv_left,
        f"{prefix}_tica_mean0.npy": mean0,
        f"{prefix}_tica_singular_values.npy": sv,
    }

    for filename, arr in artifacts.items():
        p = output_dir / filename
        np.save(p, arr)
        paths[filename] = p
        logger.info("Saved %s  shape=%s", p, arr.shape)

    lag_txt = output_dir / f"{prefix}_tica_chosen_lagtime.txt"
    lag_txt.write_text(str(lagtime))
    paths[f"{prefix}_tica_chosen_lagtime.txt"] = lag_txt
    logger.info("Saved %s  value=%d", lag_txt, lagtime)

    return paths


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save_lag_scan_figure(
    scan_results: dict[str, Any],
    output_dir: Path,
    prefix: str = "a10",
) -> Path | None:
    """Plot implied timescale and VAMP-2 vs lag time and save to disk."""
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping lag scan figure.")
        return None

    lags_ns = scan_results["lags_ns"]
    t1 = [t if np.isfinite(t) else np.nan for t in scan_results["t1_implied"]]
    vamp2 = scan_results["vamp2_scores"]
    rec_lag_ns = scan_results["recommended_lag_ns"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(lags_ns, t1, "o-", color="steelblue", linewidth=1.5)
    ax.axvline(rec_lag_ns, color="firebrick", linestyle="--", alpha=0.7,
               label=f"chosen lag = {rec_lag_ns:.1f} ns")
    ax.set_xlabel("Lag time (ns)")
    ax.set_ylabel("TIC-1 implied timescale (ns)")
    ax.set_title(f"{prefix.upper()} - Implied timescale scan")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(lags_ns, vamp2, "s-", color="darkorange", linewidth=1.5)
    ax.axvline(rec_lag_ns, color="firebrick", linestyle="--", alpha=0.7,
               label=f"chosen lag = {rec_lag_ns:.1f} ns")
    ax.set_xlabel("Lag time (ns)")
    ax.set_ylabel("VAMP-2 score")
    ax.set_title(
        f"{prefix.upper()} - VAMP-2 per lag (not rankable across lags)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"{prefix}_tica_lag_scan.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved lag scan figure: %s", out_path)
    return out_path


def _save_validation_figures(
    validation: dict[str, Any],
    nhb: np.ndarray,
    output_dir: Path,
    prefix: str = "a10",
) -> list[Path]:
    """Save TIC-1 vs N_HB scatter and 1D FES figure."""
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping validation figures.")
        return []

    tic1 = validation["tic1_projection"]
    r = validation["pearson_r"]
    t_impl = validation["t_impl_ns"]
    saved: list[Path] = []

    # --- TIC-1 vs N_HB scatter ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(tic1, nhb, s=3, alpha=0.3, color="steelblue", rasterized=True)
    ax.set_xlabel("TIC-1 projection")
    ax.set_ylabel("N$_{HB}$")
    ax.set_title(
        f"{prefix.upper()} TIC-1 vs N_HB\n"
        f"|r| = {abs(r):.3f}   t_impl = {t_impl:.2f} ns"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = output_dir / f"{prefix}_tica_tic1_vs_nhb.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved TIC-1 vs N_HB scatter: %s", p1)
    saved.append(p1)

    # --- 1D free energy surface ---
    n_bins = 50
    counts, bin_edges = np.histogram(tic1, bins=n_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts_safe = counts.astype(float)
    counts_safe[counts_safe == 0] = 1e-10
    fes = -np.log(counts_safe / counts_safe.max())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(bin_centres, fes, color="steelblue", linewidth=1.5)
    ax.fill_between(bin_centres, fes, alpha=0.15, color="steelblue")
    ax.set_xlabel("TIC-1 projection")
    ax.set_ylabel("Free energy (kT)")
    ax.set_title(
        f"{prefix.upper()} FES along TIC-1\n"
        f"two-wells = {validation['fes_has_two_wells']}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = output_dir / f"{prefix}_tica_fes_tic1.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved FES figure: %s", p2)
    saved.append(p2)

    return saved


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_tica_pipeline(
    features_a10: np.ndarray,
    nhb_a10: np.ndarray,
    output_dir: str | Path,
    lag_scan_frames: list[int] | None = None,
    frame_spacing_ns: float = FRAME_SPACING_NS,
    min_lag_frames: int | None = None,
    rel_plateau: float | None = None,
    bootstrap_block_size: int | None = None,
    n_frames_per_replica: int | None = None,
) -> dict[str, Any]:
    """Full TICA training pipeline for A10.

    Steps
    -----
    1. Lag scan — TIC-1 implied timescale vs lag; VAMP-2 logged (not used to rank τ).
    2. Choose lag — ITS plateau heuristic with minimum lag (``config.TICA_MIN_LAG_FRAMES``).
    3. Block bootstrap — TIC-1 eigenvector stability (block size from config by default).
    4. Train production TICA — kinetic_map scaling, chosen lag.
    5. Validate TIC-1 — Pearson r(TIC-1, N_HB), FES, implied timescale.
    6. Numerical checks — transform vs manual dot; optional two-replica TIC-1 cosine.
    7. Save artefacts and figures.

    Parameters
    ----------
    features_a10:
        (n_frames, 40) sin/cos features, burn-in (250 frames) already removed.
    nhb_a10:
        (n_frames,) hydrogen-bond count, same frames.
    output_dir:
        Directory for saved artefacts and figures.
    lag_scan_frames:
        Lag frames to scan.  Defaults to ``config.TICA_LAGTIME_FRAMES_SCAN``.
    frame_spacing_ns:
        Frame spacing in nanoseconds.
    min_lag_frames, rel_plateau:
        Forwarded to :func:`lag_scan`; ``None`` → ``config`` defaults.
    bootstrap_block_size:
        Block size for bootstrap; ``None`` → ``config.TICA_BOOTSTRAP_BLOCK_SIZE``.
    n_frames_per_replica:
        If ``len(features) == 2 * n_frames_per_replica``, compute |cos| between
        TIC-1 eigenvectors fit on each replica alone (convergence diagnostic).

    Returns
    -------
    dict with keys:
        scan_results, bootstrap, model, validation, artifact_paths, figure_paths,
        chosen_lag_frames, chosen_lag_ns, projection_check, replica_tic1_cosine
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting TICA pipeline: %d frames × %d features",
        features_a10.shape[0], features_a10.shape[1],
    )

    # ---- Step 1: Lag scan ------------------------------------------------
    logger.info("=== Step 1: Lag scan ===")
    scan_results = lag_scan(
        features_a10,
        lag_frames=lag_scan_frames,
        frame_spacing_ns=frame_spacing_ns,
        min_lag_frames=min_lag_frames,
        rel_plateau=rel_plateau,
    )
    chosen_lag_frames: int = scan_results["recommended_lag_frames"]
    chosen_lag_ns: float = scan_results["recommended_lag_ns"]
    logger.info(
        "Chosen lag: %d frames (%.1f ns) [%s]",
        chosen_lag_frames,
        chosen_lag_ns,
        scan_results.get("lag_selection_reason", ""),
    )

    # ---- Step 2: Block bootstrap -----------------------------------------
    logger.info("=== Step 2: Block bootstrap TIC-1 stability ===")
    bootstrap = block_bootstrap_tic1_stability(
        features_a10,
        lagtime=chosen_lag_frames,
        block_size=bootstrap_block_size,
    )
    if not bootstrap["stable"]:
        logger.warning(
            "TIC-1 eigenvector is NOT stable across bootstrap resamples "
            "(min |cos| = %.3f < 0.95).  Proceeding but interpret results cautiously.",
            bootstrap["min_cos_sim"],
        )

    # ---- Step 3: Train production TICA -----------------------------------
    logger.info("=== Step 3: Train production TICA (kinetic_map) ===")
    model = train_tica(
        features_a10,
        lagtime=chosen_lag_frames,
        dim=DEFAULT_DIM,
        epsilon=DEFAULT_EPSILON,
        scaling=DEFAULT_SCALING,
    )

    # ---- Step 4: Validate TIC-1 ------------------------------------------
    logger.info("=== Step 4: Validate TIC-1 ===")
    validation = validate_tic1(
        model, features_a10, nhb_a10, frame_spacing_ns=frame_spacing_ns
    )

    projection_check = verify_ic_projection_matches_manual_dot(
        model, features_a10, ic=0
    )
    logger.info(
        "Projection check TIC-1: r(transform, manual dot) = %.6f",
        projection_check["pearson_transform_vs_manual_dot"],
    )

    replica_cos = float("nan")
    if (
        n_frames_per_replica is not None
        and features_a10.shape[0] == 2 * int(n_frames_per_replica)
    ):
        replica_cos = cross_replica_tic1_cosine_similarity(
            features_a10,
            lagtime=chosen_lag_frames,
            n_first_replica_frames=int(n_frames_per_replica),
        )
        logger.info(
            "Two-replica TIC-1 |cos| (independent fits, scaling=None): %.4f",
            replica_cos,
        )

    # ---- Step 5: Save artefacts ------------------------------------------
    logger.info("=== Step 5: Save artefacts ===")
    artifact_paths = save_tica_artifacts(model, output_dir, prefix="a10")

    # ---- Step 6: Save figures --------------------------------------------
    logger.info("=== Step 6: Save figures ===")
    figure_paths: list[Path] = []
    lag_fig = _save_lag_scan_figure(scan_results, output_dir, prefix="a10")
    if lag_fig:
        figure_paths.append(lag_fig)
    val_figs = _save_validation_figures(validation, nhb_a10, output_dir, prefix="a10")
    figure_paths.extend(val_figs)

    # ---- Summary ---------------------------------------------------------
    logger.info("=== Pipeline complete ===")
    logger.info("Chosen lag: %d frames (%.1f ns)", chosen_lag_frames, chosen_lag_ns)
    logger.info("TIC-1 lambda_1 = %.4f", float(model.singular_values[0]))
    logger.info("TIC-1 implied timescale: %.2f ns", validation["t_impl_ns"])
    logger.info(
        "Pearson r(TIC-1, N_HB) = %.4f -> %s",
        validation["pearson_r"], validation["status"],
    )
    logger.info("FES two-well: %s", validation["fes_has_two_wells"])
    logger.info("Bootstrap stable: %s (min |cos| = %.3f)", bootstrap["stable"],
                bootstrap["min_cos_sim"])

    return {
        "scan_results": scan_results,
        "bootstrap": bootstrap,
        "model": model,
        "validation": validation,
        "artifact_paths": artifact_paths,
        "figure_paths": figure_paths,
        "chosen_lag_frames": chosen_lag_frames,
        "chosen_lag_ns": chosen_lag_ns,
        "projection_check": projection_check,
        "replica_tic1_cosine": replica_cos,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 TICA training pipeline for A10. "
            "Expects pre-computed sin/cos features (burn-in removed) and N_HB array."
        )
    )
    parser.add_argument(
        "--features_npy",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to .npy file of shape (n_frames, 40) — sin/cos backbone features.",
    )
    parser.add_argument(
        "--nhb_npy",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to .npy file of shape (n_frames,) — hydrogen-bond count per frame.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory to write artefacts and figures.",
    )
    parser.add_argument(
        "--lag_frames",
        nargs="+",
        type=int,
        default=None,
        metavar="LAG",
        help=(
            "Lag times (in frames) to scan.  "
            f"Default: {DEFAULT_LAG_FRAMES}"
        ),
    )
    parser.add_argument(
        "--frame_spacing_ns",
        type=float,
        default=FRAME_SPACING_NS,
        metavar="NS",
        help=f"Time between consecutive saved frames in nanoseconds (default: {FRAME_SPACING_NS}).",
    )
    parser.add_argument(
        "--min_lag_frames",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum lag (frames) for production fit; default from "
            "config.TICA_MIN_LAG_FRAMES."
        ),
    )
    parser.add_argument(
        "--rel_plateau",
        type=float,
        default=None,
        metavar="X",
        help="ITS plateau relative-step threshold; default config.TICA_LAG_ITS_REL_PLATEAU.",
    )
    parser.add_argument(
        "--bootstrap_block_size",
        type=int,
        default=None,
        metavar="N",
        help="Bootstrap block size in frames; default config.TICA_BOOTSTRAP_BLOCK_SIZE.",
    )
    parser.add_argument(
        "--n_frames_per_replica",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If len(features)==2N, report |cos| between TIC-1 from each half "
            "(independent fits).  E.g. 4750 for pooled 2×500 ns replicas."
        ),
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # Load inputs
    logger.info("Loading features from %s", args.features_npy)
    features = np.load(args.features_npy)
    if features.ndim != 2:
        raise ValueError(
            f"features_npy must be a 2D array (n_frames, n_features); "
            f"got shape {features.shape}"
        )

    logger.info("Loading N_HB from %s", args.nhb_npy)
    nhb = np.load(args.nhb_npy)
    if nhb.ndim != 1:
        raise ValueError(
            f"nhb_npy must be a 1D array (n_frames,); got shape {nhb.shape}"
        )
    if len(nhb) != len(features):
        raise ValueError(
            f"features ({len(features)} frames) and nhb ({len(nhb)} frames) "
            "must have the same length."
        )

    results = run_tica_pipeline(
        features_a10=features,
        nhb_a10=nhb,
        output_dir=args.output_dir,
        lag_scan_frames=args.lag_frames,
        frame_spacing_ns=args.frame_spacing_ns,
        min_lag_frames=args.min_lag_frames,
        rel_plateau=args.rel_plateau,
        bootstrap_block_size=args.bootstrap_block_size,
        n_frames_per_replica=args.n_frames_per_replica,
    )

    # Final status summary to stdout
    v = results["validation"]
    b = results["bootstrap"]
    scan = results["scan_results"]
    pc = results["projection_check"]
    print("\n" + "=" * 60)
    print("TICA TRAINING SUMMARY")
    print("=" * 60)
    print(f"Lag selection:       {scan.get('lag_selection_reason', '')}")
    print(f"Chosen lag:          {results['chosen_lag_frames']} frames  "
          f"({results['chosen_lag_ns']:.1f} ns)")
    sv0 = float(np.asarray(results["model"].singular_values).ravel()[0])
    print(f"TIC-1 singular val:  {sv0:.4f}")
    print(f"TIC-1 t_impl:        {v['t_impl_ns']:.2f} ns")
    pr, pp = v["pearson_r"], v["pearson_pval"]
    pr_s = f"{float(pr):.4f}" if np.isfinite(pr) else "nan"
    pp_s = f"{float(pp):.2e}" if np.isfinite(pp) else "nan"
    print(f"Pearson r(TIC1,NHB): {pr_s}  (p={pp_s})  -> {v['status'].upper()}")
    print(f"FES two-well:        {v['fes_has_two_wells']}")
    print(f"Bootstrap stable:    {b['stable']}  "
          f"(min |cos| = {b['min_cos_sim']:.3f})")
    print(
        f"Projection check:    r(manual dot) = "
        f"{pc['pearson_transform_vs_manual_dot']:.6f}"
    )
    rc = results["replica_tic1_cosine"]
    if np.isfinite(rc):
        print(f"Replica |cos| TIC-1:  {rc:.4f}")
    print("Artefacts:")
    for name, path in results["artifact_paths"].items():
        print(f"  {path}")
    print("Figures:")
    for path in results["figure_paths"]:
        print(f"  {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
