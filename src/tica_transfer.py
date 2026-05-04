"""
tica_transfer.py — TICA collective variable transferability assessment.

Implements the Moffett & Shukla (arXiv:1710.00443) transfer metrics and the
full transferability pipeline from docs/EXPERIMENT_PROTOCOL.md § 10 (Phase 7).

Systems
-------
  Donor   : Ac-Ala₁₀-NHMe (A10) — TICA trained on 40-D sin/cos torsion features
  Acceptor: Ac-Ala₈Gln₂-NHMe (A8Q2) — same feature dimensionality (10 residues)

Transfer logic
--------------
  The A10 TICA eigenvector was learned in the coordinate frame centred at
  mean_a10 = model_a10.mean_0.  To ask "where does A8Q2 fall on the A10
  helix-coil coordinate?" we must keep that reference frame → subtract
  mean_a10, NOT mean_a8q2, when projecting acceptor features.

Metric summary
--------------
  D₀   — decorrelation score: how well A10 TICs remain mutually uncorrelated
          on the A8Q2 covariance structure.  D₀ ≈ 0 → geometrically compatible.
  D_τ  — kinetic structure match: whether A10 TICs preserve A8Q2 slow-mode
          structure at lag τ.  D_τ ≈ 0 → kinetically equivalent.
  D_KM — subspace error: whether A8Q2 TIC-1 is reconstructable from the top-K
          A10 TICs.  D_{21} is the key metric for seeding validity.
  r    — Pearson correlation of transferred TIC-1 with N_HB (physical check).

Usage (CLI)
-----------
  python -m src.tica_transfer \\
      --features_a10 data/a10_features.npy \\
      --features_a8q2 data/a8q2_features.npy \\
      --nhb_a8q2 data/a8q2_nhb.npy \\
      --tica_artifacts_dir artifacts/ \\
      --lagtime 4 \\
      --output_dir results/transfer/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

try:
    from deeptime.decomposition import TICA
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deeptime is required: pip install deeptime"
    ) from exc

from src.config import TEMPERATURE_K

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Protocol constants (from docs/EXPERIMENT_PROTOCOL.md § 9)
# ─────────────────────────────────────────────────────────────────────────────

TICA_DIM: int = 5
"""Number of TICA components kept when training/validating the A10 model."""

TICA_EPSILON: float = 1e-6
"""
Tikhonov regularisation for the TICA generalised eigenvalue problem.
Prevents numerical issues when features are near-collinear.
"""

TICA_LAG_SCAN_FRAMES: list[int] = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
"""
Lag-time scan grid in trajectory frames (100 ps / frame = 0.1 ns).
Corresponds to [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] ns.
At 330 K a useful Markovian lag often falls near ~10–40 frames (~1–4 ns);
the scan includes short lags for fast motions and long lags for implied-timescale plateaus.
"""

FRAME_SPACING_NS: float = 0.1
"""Trajectory frame spacing in nanoseconds (DCD saved every 100 ps)."""

# Pearson |r| thresholds (Protocol § 10.2)
R_STRONG: float = 0.75
"""Above this, transferred TIC-1 still tracks helix content → transfer succeeds."""

R_WEAK: float = 0.60
"""Below this, the transferred coordinate has lost physical meaning on A8Q2."""

# ─────────────────────────────────────────────────────────────────────────────
# Core transfer function
# ─────────────────────────────────────────────────────────────────────────────


def project_transferred(
    features_acceptor: np.ndarray,
    mean_donor: np.ndarray,
    eigenvectors_donor: np.ndarray,
) -> np.ndarray:
    """Project acceptor features onto the donor TICA coordinate system.

    The donor TICA model was fitted in the frame centred at *mean_donor*
    (``model_a10.mean_0``).  To preserve that reference frame when asking
    "where does acceptor trajectory fall on the donor coordinate?" we subtract
    the donor mean — NOT the acceptor mean.  Using the acceptor mean would
    shift the origin and yield offsets that are hard to interpret physically.

    Parameters
    ----------
    features_acceptor:
        Raw (un-centred) sin/cos torsion features for the acceptor system.
        Shape ``(n_frames, n_features)``.
    mean_donor:
        Instantaneous mean vector from the donor TICA fit,
        ``model_a10.mean_0``.  Shape ``(n_features,)``.
    eigenvectors_donor:
        Left singular vectors from the donor TICA fit,
        ``model_a10.singular_vectors_left``.
        Shape ``(n_features, n_components)``.
        Pass the full matrix; the caller selects how many columns to use via
        slicing before or after this call.

    Returns
    -------
    np.ndarray
        Projected coordinates, shape ``(n_frames, n_components)`` where
        ``n_components = eigenvectors_donor.shape[1]``.
        Column 0 is the transferred TIC-1.
    """
    features_acceptor = np.asarray(features_acceptor, dtype=float)
    mean_donor = np.asarray(mean_donor, dtype=float)
    eigenvectors_donor = np.asarray(eigenvectors_donor, dtype=float)

    X_centered = features_acceptor - mean_donor
    Z = X_centered @ eigenvectors_donor
    return Z


# ─────────────────────────────────────────────────────────────────────────────
# Moffett & Shukla metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_d0(
    features_acceptor_centered: np.ndarray,
    eigenvectors_donor: np.ndarray,
    K: int = 2,
) -> float:
    """D₀ — decorrelation score (Moffett & Shukla arXiv:1710.00443, Eq. 3).

    Measures how well the top-K donor TICs remain mutually decorrelated on the
    acceptor system's covariance structure.  If D₀ ≈ 0 the TICs are still
    orthonormal on A8Q2's C(0) → coordinate system is geometrically compatible.
    If D₀ >> 0, the mutation changed the torsion-correlation structure so that
    the donor eigenvectors are no longer orthogonal on the acceptor.

    Derivation
    ----------
    Let V_K = eigenvectors_donor[:, :K], the (n_features × K) matrix of the
    top-K donor TIC directions.  On the donor, by construction
    V_K^T C₀_A V_K = I_K.  On the acceptor with centred covariance C₀_B:

        C_V = V_K^T C₀_B V_K

    D₀ = ‖C_V − I_K‖_F

    Parameters
    ----------
    features_acceptor_centered:
        Acceptor features **already centred** by the donor mean.
        Shape ``(n_frames, n_features)``.
    eigenvectors_donor:
        Left singular vectors of the donor TICA model.
        Shape ``(n_features, ≥K)``.
    K:
        Number of donor TICs to include (default 2, Protocol uses K=2).

    Returns
    -------
    float
        D₀ ≥ 0.  Smaller → more geometrically compatible.
    """
    X = np.asarray(features_acceptor_centered, dtype=float)
    V = np.asarray(eigenvectors_donor, dtype=float)[:, :K]

    n = len(X)
    C0_B = (X.T @ X) / n
    C_V = V.T @ C0_B @ V
    D0 = float(np.linalg.norm(C_V - np.eye(K), "fro"))
    return D0


def compute_dtau(
    features_acceptor_centered: np.ndarray,
    eigenvectors_donor: np.ndarray,
    native_singular_values_acceptor: np.ndarray,
    lag: int,
    K: int = 2,
) -> float:
    """D_τ — kinetic structure match (Moffett & Shukla arXiv:1710.00443, Eq. 4).

    Measures whether the top-K donor TICs preserve the slow-mode kinetic
    structure of the acceptor system at lag time *lag*.  If D_τ ≈ 0, the
    donor TIC basis diagonalises the acceptor lagged covariance with eigenvalues
    matching the native acceptor singular values → kinetically equivalent.

    Derivation
    ----------
    Symmetrised lagged covariance of the acceptor (centred) features:

        C_τ_B = ½ [ X[:-lag]^T X[lag:] + (X[:-lag]^T X[lag:])^T ] / (N − lag)

    Projected lagged covariance in the donor subspace:

        C_V_τ = V_K^T C_τ_B V_K

    Reference diagonal (native A8Q2 dynamics):

        Λ_B = diag(λ₁_B, …, λ_K_B)  where λᵢ_B = native A8Q2 singular values

    D_τ = ‖C_V_τ − Λ_B‖_F

    Parameters
    ----------
    features_acceptor_centered:
        Centred acceptor features (donor mean subtracted).
        Shape ``(n_frames, n_features)``.
    eigenvectors_donor:
        Left singular vectors of the donor TICA model.
        Shape ``(n_features, ≥K)``.
    native_singular_values_acceptor:
        Singular values from TICA trained **natively on the acceptor**.
        Only the first K values are used.  Shape ``(≥K,)``.
    lag:
        Lag in frames (same units as the TICA lagtime).
    K:
        Number of donor TICs to include (default 2).

    Returns
    -------
    float
        D_τ ≥ 0.  Smaller → better kinetic structure preservation.
    """
    X = np.asarray(features_acceptor_centered, dtype=float)
    V = np.asarray(eigenvectors_donor, dtype=float)[:, :K]
    lambdas = np.asarray(native_singular_values_acceptor, dtype=float)[:K]

    n = len(X)
    C_tau_B = (X[:-lag].T @ X[lag:]) / (n - lag)
    C_tau_B = 0.5 * (C_tau_B + C_tau_B.T)

    C_V_tau = V.T @ C_tau_B @ V
    Lambda_B = np.diag(lambdas)
    D_tau = float(np.linalg.norm(C_V_tau - Lambda_B, "fro"))
    return D_tau


def compute_dkm(
    features_acceptor: np.ndarray,
    eigenvectors_donor: np.ndarray,
    lagtime: int,
    K: int = 2,
    M: int = 1,
    epsilon: float = TICA_EPSILON,
) -> tuple[float, Any]:
    """D_{KM} — subspace projection error (Moffett & Shukla arXiv:1710.00443, Eq. 5).

    Asks: can the top-M native acceptor TICs be reconstructed from the
    top-K donor TICs?  D_{21} (K=2, M=1) is the most practically important
    metric for seeding: if D_{21} ≈ 0, directing exploration in the A10 TIC-1
    direction is equivalent to directing it in the A8Q2 TIC-1 direction.

    Algorithm
    ---------
    1. Fit TICA(lagtime, dim=M, scaling=None, epsilon=epsilon) natively on
       the acceptor features.  ``scaling=None`` gives raw (un-scaled)
       eigenvectors suitable for geometric subspace comparison.
    2. Let U_M = native_model.singular_vectors_left, shape (n_features, M).
    3. Let V_K = eigenvectors_donor[:, :K],           shape (n_features, K).
    4. Find X = pinv(V_K) @ U_M   (least-squares coefficients).
    5. D_KM = ‖V_K @ X − U_M‖_F  (residual after projection).

    Parameters
    ----------
    features_acceptor:
        Raw acceptor features (not centred — TICA internally centres).
        Shape ``(n_frames, n_features)``.
    eigenvectors_donor:
        Left singular vectors of the donor TICA model.
        Shape ``(n_features, ≥K)``.
    lagtime:
        Lag in frames for the native acceptor TICA fit.
    K:
        Dimension of the donor subspace (default 2).
    M:
        Number of native acceptor TICs to recover (default 1).
    epsilon:
        Tikhonov regularisation for the native TICA fit.

    Returns
    -------
    tuple[float, CovarianceKoopmanModel]
        D_KM ≥ 0 (smaller → better subspace alignment) and the fitted
        native acceptor model (useful for downstream FES or validation).
    """
    # Native acceptor TICA with scaling=None for raw geometric eigenvectors
    native_tica = TICA(lagtime=lagtime, dim=M, epsilon=epsilon, scaling=None)
    native_model = native_tica.fit(features_acceptor).fetch_model()
    U_M = native_model.singular_vectors_left  # (n_features, M)

    V_K = np.asarray(eigenvectors_donor, dtype=float)[:, :K]

    X_coeff = np.linalg.pinv(V_K) @ U_M
    D_KM = float(np.linalg.norm(V_K @ X_coeff - U_M, "fro"))
    return D_KM, native_model


# ─────────────────────────────────────────────────────────────────────────────
# Physical interpretability: Pearson r
# ─────────────────────────────────────────────────────────────────────────────


def compute_transfer_pearson(
    features_acceptor: np.ndarray,
    mean_donor: np.ndarray,
    eigenvectors_donor: np.ndarray,
    nhb_acceptor: np.ndarray,
) -> dict[str, Any]:
    """Pearson r between transferred TIC-1 and N_HB of the acceptor system.

    Physical interpretability check from Protocol § 10.2.
    If |r| > 0.75 the transferred TIC-1 still tracks helix content → transfer
    succeeds physically.  If |r| < 0.60 the transferred coordinate has lost
    physical meaning on A8Q2 for helix-coil.

    The sign of r is not fixed a priori (TICs can point in either direction);
    the absolute value is the relevant quantity.

    Parameters
    ----------
    features_acceptor:
        Raw (un-centred) acceptor sin/cos torsion features.
        Shape ``(n_frames, n_features)``.
    mean_donor:
        Donor TICA mean (``model_a10.mean_0``), shape ``(n_features,)``.
    eigenvectors_donor:
        Donor left singular vectors, shape ``(n_features, n_components)``.
        Column 0 is used as TIC-1.
    nhb_acceptor:
        Per-frame backbone hydrogen-bond count for the acceptor.
        Shape ``(n_frames,)``.

    Returns
    -------
    dict with keys:
        r            – Pearson correlation coefficient (signed).
        pval         – Two-tailed p-value.
        abs_r        – Absolute value of r.
        interpretation – Human-readable verdict string.
    """
    Z = project_transferred(features_acceptor, mean_donor, eigenvectors_donor)
    tic1 = Z[:, 0]
    nhb = np.asarray(nhb_acceptor, dtype=float)

    r, pval = pearsonr(tic1, nhb)
    abs_r = abs(r)

    if abs_r >= R_STRONG:
        interpretation = (
            f"|r| = {abs_r:.3f} ≥ {R_STRONG} → TRANSFER SUCCEEDS: "
            "transferred TIC-1 still tracks A8Q2 helix content."
        )
    elif abs_r >= R_WEAK:
        interpretation = (
            f"|r| = {abs_r:.3f} in [{R_WEAK}, {R_STRONG}) → MARGINAL: "
            "some correlation with helix content; proceed with caution."
        )
    else:
        interpretation = (
            f"|r| = {abs_r:.3f} < {R_WEAK} → TRANSFER FAILS: "
            "transferred coordinate has lost physical meaning for helix-coil on A8Q2."
        )

    return {"r": float(r), "pval": float(pval), "abs_r": float(abs_r),
            "interpretation": interpretation}


# ─────────────────────────────────────────────────────────────────────────────
# FES comparison
# ─────────────────────────────────────────────────────────────────────────────


def _compute_1d_fes(
    coordinate: np.ndarray,
    temperature_K: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 1D free energy surface from a 1D coordinate array.

    Uses F(x) = −k_B T ln P(x), with P(x) normalised over the histogram.
    The reference (minimum) is set to zero.  Bins with zero counts are
    assigned NaN.

    Parameters
    ----------
    coordinate:
        1D projected coordinate.  Shape ``(n_frames,)``.
    temperature_K:
        Temperature in Kelvin used for the k_B T prefactor.
    nbins:
        Number of histogram bins.

    Returns
    -------
    bin_centers : np.ndarray, shape (nbins,)
    fes_kcal    : np.ndarray, shape (nbins,)  — free energy in kcal/mol.
    """
    kBT_kcal = 1.987e-3 * temperature_K  # kcal/mol

    counts, bin_edges = np.histogram(coordinate, bins=nbins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    with np.errstate(divide="ignore"):
        fes = -kBT_kcal * np.where(counts > 0, np.log(counts.astype(float)), np.nan)

    fes_shifted = fes - np.nanmin(fes)
    return bin_centers, fes_shifted


def compute_transfer_fes(
    features_acceptor: np.ndarray,
    mean_donor: np.ndarray,
    eigenvectors_donor: np.ndarray,
    lagtime: int,
    temperature_K: float = TEMPERATURE_K,
    nbins: int = 50,
) -> dict[str, Any]:
    """1D FES along transferred TIC-1 and native A8Q2 TIC-1 for overlay comparison.

    Generates two free energy profiles on the acceptor trajectory:
      (a) FES along the transferred (donor) TIC-1.
      (b) FES along the natively trained A8Q2 TIC-1.

    If the transfer succeeds, both profiles should display a two-well structure
    (coil and helix basins).  A single-well native FES with two wells in the
    transferred FES — or vice versa — would indicate a failure mode.

    Note: the native A8Q2 TICA here uses ``scaling=None`` (geometric eigenvectors)
    for a fair geometric comparison with the transferred coordinate.

    Parameters
    ----------
    features_acceptor:
        Raw acceptor features.  Shape ``(n_frames, n_features)``.
    mean_donor:
        Donor TICA mean.  Shape ``(n_features,)``.
    eigenvectors_donor:
        Donor left singular vectors.  Shape ``(n_features, n_components)``.
    lagtime:
        Lag in frames for the native A8Q2 TICA.
    temperature_K:
        Temperature for FES calculation.  Defaults to ``config.TEMPERATURE_K``.
    nbins:
        Number of bins for the 1D FES histogram.

    Returns
    -------
    dict with keys:
        transferred_bins      – bin centres for the transferred FES.
        transferred_fes       – FES values (kcal/mol) along transferred TIC-1.
        native_bins           – bin centres for the native FES.
        native_fes            – FES values (kcal/mol) along native A8Q2 TIC-1.
        tic1_transferred      – raw projected values (for scatter / correlation).
        tic1_native           – raw native projected values.
    """
    # Transferred TIC-1
    Z_transferred = project_transferred(features_acceptor, mean_donor, eigenvectors_donor)
    tic1_transferred = Z_transferred[:, 0]

    # Native A8Q2 TIC-1 (scaling=None for geometric eigenvectors)
    native_tica = TICA(lagtime=lagtime, dim=1, epsilon=TICA_EPSILON, scaling=None)
    native_model = native_tica.fit(features_acceptor).fetch_model()
    tic1_native = native_model.transform(features_acceptor)[:, 0]

    t_bins, t_fes = _compute_1d_fes(tic1_transferred, temperature_K, nbins)
    n_bins, n_fes = _compute_1d_fes(tic1_native, temperature_K, nbins)

    return {
        "transferred_bins": t_bins,
        "transferred_fes": t_fes,
        "native_bins": n_bins,
        "native_fes": n_fes,
        "tic1_transferred": tic1_transferred,
        "tic1_native": tic1_native,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interpretation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _interpret_d0(d0: float) -> str:
    """Human-readable verdict for D₀."""
    if d0 < 0.3:
        return (
            f"D₀ = {d0:.4f} < 0.3 → GOOD: A10 TICs remain mutually decorrelated "
            "on A8Q2; coordinate system is geometrically compatible."
        )
    elif d0 < 0.7:
        return (
            f"D₀ = {d0:.4f} in [0.3, 0.7) → MARGINAL: moderate mixing of TICs "
            "on A8Q2 covariance structure."
        )
    else:
        return (
            f"D₀ = {d0:.4f} ≥ 0.7 → POOR: A10 TICs are substantially non-orthogonal "
            "on A8Q2; mutation changed the torsion-correlation structure."
        )


def _interpret_dtau(d_tau: float) -> str:
    """Human-readable verdict for D_τ."""
    if d_tau < 0.3:
        return (
            f"D_τ = {d_tau:.4f} < 0.3 → GOOD: A10 TICs preserve the slow kinetic "
            "structure of A8Q2."
        )
    elif d_tau < 0.7:
        return (
            f"D_τ = {d_tau:.4f} in [0.3, 0.7) → MARGINAL: partial kinetic-structure "
            "preservation."
        )
    else:
        return (
            f"D_τ = {d_tau:.4f} ≥ 0.7 → POOR: A10 TIC basis does not represent "
            "the slow dynamics of A8Q2."
        )


def _interpret_dkm(d_km: float) -> str:
    """Human-readable verdict for D_{KM}."""
    if d_km < 0.3:
        return (
            f"D_KM = {d_km:.4f} < 0.3 → GOOD: A8Q2 TIC-1 is well-represented "
            "within the top-2 A10 TIC subspace; transferred seeding is valid."
        )
    elif d_km < 0.6:
        return (
            f"D_KM = {d_km:.4f} in [0.3, 0.6) → MARGINAL: partial subspace overlap; "
            "seeding will be biased but may still outperform random."
        )
    else:
        return (
            f"D_KM = {d_km:.4f} ≥ 0.6 → POOR: A8Q2 TIC-1 is not recoverable "
            "from top-2 A10 TICs; transferred seeding unlikely to outperform random."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _plot_fes_comparison(
    fes_data: dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Save the FES overlay figure (transferred TIC-1 vs native A8Q2 TIC-1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(fes_data["transferred_bins"], fes_data["transferred_fes"],
            color="tab:blue", lw=2, label="Transferred A10 TIC-1")
    ax.set_xlabel("Transferred TIC-1")
    ax.set_ylabel("Free energy (kcal/mol)")
    ax.set_title("FES — Transferred TIC-1 on A8Q2")
    ax.legend()
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(fes_data["native_bins"], fes_data["native_fes"],
            color="tab:orange", lw=2, label="Native A8Q2 TIC-1")
    ax.set_xlabel("Native A8Q2 TIC-1")
    ax.set_ylabel("Free energy (kcal/mol)")
    ax.set_title("FES — Native A8Q2 TIC-1")
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.suptitle("TICA Transfer: 1D FES comparison (A8Q2 trajectory)", fontsize=12)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved FES comparison figure → %s", output_path)


def _plot_pearson_scatter(
    tic1_transferred: np.ndarray,
    nhb_a8q2: np.ndarray,
    pearson_result: dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Save scatter plot of transferred TIC-1 vs N_HB with Pearson r annotation."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(tic1_transferred, nhb_a8q2, s=2, alpha=0.3, color="tab:blue", rasterized=True)
    ax.set_xlabel("Transferred TIC-1 (A10 → A8Q2)")
    ax.set_ylabel("N_HB (A8Q2)")
    r_val = pearson_result["r"]
    p_val = pearson_result["pval"]
    ax.set_title(
        f"Transferred TIC-1 vs N_HB\n"
        f"Pearson r = {r_val:.3f}  (p = {p_val:.2e})"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved Pearson scatter figure → %s", output_path)


def _plot_metric_bar(
    d0: float, d_tau: float, d_km: float, pearson_r: float,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Save bar chart of all three Moffett-Shukla metrics and |r| (Protocol § 12.6)."""
    labels = ["D₀", "D_τ", "D_{KM}", "|r|"]
    # For |r| we plot (1 − |r|) so that 0 = perfect everywhere
    values_raw = [d0, d_tau, d_km, abs(pearson_r)]
    values_display = [d0, d_tau, d_km, 1.0 - abs(pearson_r)]
    colors = []
    for v in values_raw[:3]:
        colors.append("tab:green" if v < 0.3 else ("tab:orange" if v < 0.7 else "tab:red"))
    abs_r = abs(pearson_r)
    colors.append(
        "tab:green" if abs_r >= R_STRONG else ("tab:orange" if abs_r >= R_WEAK else "tab:red")
    )

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bars = ax.bar(labels, values_display, color=colors, edgecolor="black", linewidth=0.8)
    for bar, raw in zip(bars, values_raw):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{raw:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Metric value\n(D₀, D_τ, D_KM) or (1 − |r|) for Pearson")
    ax.set_title("TICA Transferability Scores (A10 → A8Q2)")
    ax.set_ylim(0, max(values_display) * 1.25 + 0.05)
    ax.axhline(0.3, color="grey", lw=1, ls="--", label="0.3 threshold")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved transferability bar chart → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_transfer_assessment(
    features_a10: np.ndarray,
    features_a8q2: np.ndarray,
    nhb_a8q2: np.ndarray,
    model_a10,
    lagtime_frames: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Full TICA transferability assessment pipeline (Protocol § 10).

    Runs all four metrics (D₀, D_τ, D_KM, Pearson r), generates all figures,
    and writes a JSON summary.  Corresponds to the analysis that runs on Pod 1
    once ``a8q2_features.npy`` has been downloaded from MongoDB.

    Parameters
    ----------
    features_a10:
        Sin/cos torsion features for the donor A10 system, shape
        ``(n_frames_a10, 40)``.  Used only to obtain ``model_a10`` context
        if the caller passes a pre-fit model; not re-fitted here.
    features_a8q2:
        Sin/cos torsion features for the acceptor A8Q2 system, shape
        ``(n_frames_a8q2, 40)``.
    nhb_a8q2:
        Per-frame backbone H-bond count for A8Q2, shape ``(n_frames_a8q2,)``.
    model_a10:
        Pre-fitted deeptime ``CovarianceKoopmanModel`` (from
        ``TICA(...).fit(features_a10).fetch_model()``).  Must expose:
        ``singular_vectors_left``, ``singular_values``, ``mean_0``.
    lagtime_frames:
        Lag time in frames to use for the native A8Q2 TICA in D_KM and FES.
    output_dir:
        Directory where figures and JSON summary are written.
        Created if it does not exist.

    Returns
    -------
    dict with keys:
        D0               – float, decorrelation score.
        D_tau            – float, kinetic structure match.
        D_KM             – float, subspace error (K=2, M=1).
        pearson_r        – float, signed Pearson r.
        pearson_abs_r    – float, |r|.
        pearson_pval     – float, p-value.
        interpretation_d0    – str.
        interpretation_dtau  – str.
        interpretation_dkm   – str.
        interpretation_r     – str.
        overall_verdict  – str, one-sentence synthesis.
        figure_fes       – Path, saved FES overlay figure.
        figure_pearson   – Path, saved Pearson scatter figure.
        figure_bars      – Path, saved bar chart figure.
        json_summary     – Path, saved JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract donor model attributes ────────────────────────────────────────
    mean_donor: np.ndarray = np.asarray(model_a10.mean_0, dtype=float)
    eigenvectors_donor: np.ndarray = np.asarray(
        model_a10.singular_vectors_left, dtype=float
    )  # shape (40, n_components)
    singular_values_donor: np.ndarray = np.asarray(
        model_a10.singular_values, dtype=float
    )

    # ── Centre A8Q2 features using the donor mean ─────────────────────────────
    features_a8q2 = np.asarray(features_a8q2, dtype=float)
    X_centered = features_a8q2 - mean_donor

    # ── Native A8Q2 TICA (for D_τ and D_KM) ──────────────────────────────────
    # dim=2 to supply two singular values for D_τ (K=2)
    logger.info("Fitting native A8Q2 TICA (dim=2, lagtime=%d) for D_τ …", lagtime_frames)
    native_tica_2 = TICA(lagtime=lagtime_frames, dim=2, epsilon=TICA_EPSILON, scaling=None)
    native_model_2 = native_tica_2.fit(features_a8q2).fetch_model()
    native_sv_a8q2 = native_model_2.singular_values  # shape (2,)

    # ── D₀ ────────────────────────────────────────────────────────────────────
    logger.info("Computing D₀ …")
    d0 = compute_d0(X_centered, eigenvectors_donor, K=2)
    interp_d0 = _interpret_d0(d0)
    logger.info(interp_d0)

    # ── D_τ ───────────────────────────────────────────────────────────────────
    logger.info("Computing D_τ (lag=%d frames) …", lagtime_frames)
    d_tau = compute_dtau(
        X_centered,
        eigenvectors_donor,
        native_singular_values_acceptor=native_sv_a8q2,
        lag=lagtime_frames,
        K=2,
    )
    interp_dtau = _interpret_dtau(d_tau)
    logger.info(interp_dtau)

    # ── D_KM ──────────────────────────────────────────────────────────────────
    logger.info("Computing D_KM (K=2, M=1, lagtime=%d frames) …", lagtime_frames)
    d_km, _ = compute_dkm(
        features_a8q2,
        eigenvectors_donor,
        lagtime=lagtime_frames,
        K=2,
        M=1,
        epsilon=TICA_EPSILON,
    )
    interp_dkm = _interpret_dkm(d_km)
    logger.info(interp_dkm)

    # ── Pearson r ─────────────────────────────────────────────────────────────
    logger.info("Computing Pearson r(TIC-1_transferred, N_HB) …")
    pearson_result = compute_transfer_pearson(
        features_a8q2, mean_donor, eigenvectors_donor, nhb_a8q2
    )
    logger.info(pearson_result["interpretation"])

    # ── FES comparison ────────────────────────────────────────────────────────
    logger.info("Computing 1D FES comparison …")
    fes_data = compute_transfer_fes(
        features_a8q2, mean_donor, eigenvectors_donor,
        lagtime=lagtime_frames,
        temperature_K=TEMPERATURE_K,
        nbins=50,
    )

    # ── Overall verdict ───────────────────────────────────────────────────────
    n_good = sum([
        d0 < 0.3,
        d_tau < 0.3,
        d_km < 0.3,
        pearson_result["abs_r"] >= R_STRONG,
    ])
    if n_good == 4:
        overall = (
            "TRANSFER SUCCEEDS: all four metrics pass (D₀, D_τ, D_KM < 0.3; |r| ≥ 0.75). "
            "The A10 TICA CV is valid for A8Q2 enhanced sampling."
        )
    elif n_good >= 2:
        overall = (
            f"TRANSFER PARTIAL: {n_good}/4 metrics pass. "
            "Interpret results with caution; transferred CV may still be useful but is imperfect."
        )
    else:
        overall = (
            f"TRANSFER FAILS: only {n_good}/4 metrics pass. "
            "The A10 TICA CV does not transfer to A8Q2; use native retraining for seeding."
        )
    logger.info("Overall verdict: %s", overall)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_fes_path = output_dir / "transfer_fes_comparison.png"
    fig_pearson_path = output_dir / "transfer_pearson_scatter.png"
    fig_bars_path = output_dir / "transfer_metric_bars.png"

    _plot_fes_comparison(fes_data, fig_fes_path)
    _plot_pearson_scatter(
        fes_data["tic1_transferred"], np.asarray(nhb_a8q2, dtype=float),
        pearson_result, fig_pearson_path,
    )
    _plot_metric_bar(d0, d_tau, d_km, pearson_result["r"], fig_bars_path)

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "D0": d0,
        "D_tau": d_tau,
        "D_KM": d_km,
        "pearson_r": pearson_result["r"],
        "pearson_abs_r": pearson_result["abs_r"],
        "pearson_pval": pearson_result["pval"],
        "lagtime_frames": lagtime_frames,
        "lagtime_ns": lagtime_frames * FRAME_SPACING_NS,
        "K": 2,
        "M": 1,
        "n_frames_a8q2": int(len(features_a8q2)),
        "n_frames_a10": int(len(features_a10)),
        "interpretation_d0": interp_d0,
        "interpretation_dtau": interp_dtau,
        "interpretation_dkm": interp_dkm,
        "interpretation_r": pearson_result["interpretation"],
        "overall_verdict": overall,
    }

    json_path = output_dir / "transfer_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved JSON summary → %s", json_path)

    return {
        **summary,
        "figure_fes": fig_fes_path,
        "figure_pearson": fig_pearson_path,
        "figure_bars": fig_bars_path,
        "json_summary": json_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "TICA CV transferability assessment: A10 → A8Q2 "
            "(Protocol § 10, Moffett & Shukla metrics)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--features_a10", required=True, type=Path,
        metavar="NPY",
        help="Path to a10_features.npy  (n_frames × 40 sin/cos torsion matrix).",
    )
    p.add_argument(
        "--features_a8q2", required=True, type=Path,
        metavar="NPY",
        help="Path to a8q2_features.npy (n_frames × 40 sin/cos torsion matrix).",
    )
    p.add_argument(
        "--nhb_a8q2", required=True, type=Path,
        metavar="NPY",
        help="Path to a8q2_nhb.npy (n_frames, per-frame backbone H-bond count).",
    )
    p.add_argument(
        "--tica_artifacts_dir", required=True, type=Path,
        metavar="DIR",
        help=(
            "Directory containing the pre-fitted A10 TICA artifacts: "
            "a10_tica_singular_vectors_left.npy, a10_tica_mean0.npy, "
            "a10_tica_singular_values.npy.  These are saved by train_tica.py."
        ),
    )
    p.add_argument(
        "--lagtime", required=True, type=int,
        metavar="INT",
        help="Chosen TICA lagtime in frames (from lag-scan, typically 2–4).",
    )
    p.add_argument(
        "--output_dir", default="results/transfer", type=Path,
        metavar="DIR",
        help="Directory to write figures and JSON summary.",
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def _load_a10_model_from_artifacts(artifacts_dir: Path):
    """Reconstruct a minimal model-like namespace from saved .npy artifact files.

    ``train_tica.py`` saves three arrays (Protocol § 9.6):
      - a10_tica_singular_vectors_left.npy  — shape (40, n_components)
      - a10_tica_mean0.npy                  — shape (40,)
      - a10_tica_singular_values.npy        — shape (n_components,)

    Returns a ``SimpleNamespace`` with attributes matching the deeptime
    ``CovarianceKoopmanModel`` interface used throughout this module.
    """
    from types import SimpleNamespace

    svl_path = artifacts_dir / "a10_tica_singular_vectors_left.npy"
    mean_path = artifacts_dir / "a10_tica_mean0.npy"
    sv_path   = artifacts_dir / "a10_tica_singular_values.npy"

    for p in (svl_path, mean_path, sv_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing TICA artifact: {p}\n"
                "Ensure train_tica.py has been run and artifacts saved to "
                f"{artifacts_dir}."
            )

    return SimpleNamespace(
        singular_vectors_left=np.load(svl_path),
        mean_0=np.load(mean_path),
        singular_values=np.load(sv_path),
    )


def main() -> None:
    """CLI entry point for the transferability assessment pipeline."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading feature arrays …")
    features_a10  = np.load(args.features_a10)
    features_a8q2 = np.load(args.features_a8q2)
    nhb_a8q2      = np.load(args.nhb_a8q2)

    logger.info(
        "Feature shapes — A10: %s, A8Q2: %s, N_HB: %s",
        features_a10.shape, features_a8q2.shape, nhb_a8q2.shape,
    )

    logger.info("Loading A10 TICA artifacts from %s …", args.tica_artifacts_dir)
    model_a10 = _load_a10_model_from_artifacts(args.tica_artifacts_dir)

    logger.info(
        "Eigenvectors shape: %s, mean_0 shape: %s, singular_values: %s",
        model_a10.singular_vectors_left.shape,
        model_a10.mean_0.shape,
        model_a10.singular_values,
    )

    results = run_transfer_assessment(
        features_a10=features_a10,
        features_a8q2=features_a8q2,
        nhb_a8q2=nhb_a8q2,
        model_a10=model_a10,
        lagtime_frames=args.lagtime,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("TICA TRANSFERABILITY ASSESSMENT COMPLETE")
    print("=" * 60)
    print(f"  D₀         = {results['D0']:.4f}   {results['interpretation_d0'].split('→')[1].strip()}")
    print(f"  D_τ        = {results['D_tau']:.4f}   {results['interpretation_dtau'].split('→')[1].strip()}")
    print(f"  D_KM       = {results['D_KM']:.4f}   {results['interpretation_dkm'].split('→')[1].strip()}")
    print(f"  Pearson |r|= {results['pearson_abs_r']:.4f}   {results['interpretation_r'].split('→')[1].strip()}")
    print()
    print(f"  OVERALL: {results['overall_verdict']}")
    print()
    print(f"  Figures  → {args.output_dir}")
    print(f"  Summary  → {results['json_summary']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
