"""Generate all experiment figures from the run_experiment_suite outputs.

Figures produced
----------------
fig1_convergence_base.png      Main convergence: all 5 policies, base preset
fig2_seed_trajectory.png       Seed restart positions over time (P3 vs P5)
fig3_compass_strategy.png      COMPASS score components over time
fig4_chunk_ablation.png        f_helix_error vs chunk size at fixed budgets
fig5_cv_ablation.png           f_helix_error vs CV type at 100 ns
fig6_noise_ablation.png        f_helix_error vs prior noise level
fig7_source_target_coverage.png Source vs target coverage correlation
fig8_acceleration.png          Speedup bar chart

Usage
-----
    python scripts/make_experiment_figures.py [--repo-root .] [--out-dir outputs/v2/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

# Publication style
plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "grid.linewidth": 0.55,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "figure.dpi": 120,
})

# Okabe-Ito colorblind-safe palette mapped to policies
POLICY_COLORS = {
    "chrono":               "#999999",  # grey
    "independent_replicas": "#E69F00",  # orange
    "random":               "#000000",  # black
    "min_count":            "#56B4E9",  # sky blue
    "uniform_prior":        "#CC79A7",  # pink
    "compass":              "#0072B2",  # blue
}
POLICY_LABELS = {
    "chrono":               "P1 Single Walk",
    "independent_replicas": "P2 Indep. Replicas",
    "random":               "P3 Uniform Random",
    "min_count":            "P4 Min-Count",
    "uniform_prior":        "P4b Uniform Prior",
    "compass":              "P5 COMPASS",
}

EXP_ROOT = Path("outputs/v2/experiments")


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _find_runs(repo: Path, policy: str, cv_type: str, chunk_ns: float, noise: str) -> list[Path]:
    chunk_str = f"{chunk_ns:.1f}ns".replace(".", "_")
    # Apply caching rules from run_experiment_suite.py
    if policy in ("chrono", "independent_replicas", "random"):
        cv_type = "tica2d"
        noise = "none"
    elif policy == "min_count":
        noise = "none"
    base = repo / EXP_ROOT / f"p_{policy}" / f"cv_{cv_type}" / f"chunk_{chunk_str}" / f"noise_{noise}"
    if not base.is_dir():
        return []
    return sorted([d for d in base.iterdir() if d.is_dir() and (d / "v2_metrics.csv").is_file()])


def _load_repeats(run_dirs: list[Path]) -> pd.DataFrame:
    parts = []
    for i, d in enumerate(run_dirs):
        df = pd.read_csv(d / "v2_metrics.csv")
        df["repeat"] = i
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _columnwise_mean_std(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std over repeats; all-NaN columns stay NaN (no empty-slice warnings)."""
    if mat.ndim != 2:
        raise ValueError("expected 2D array (n_repeats × n_x)")
    n_x = mat.shape[1]
    mean = np.full(n_x, np.nan, dtype=float)
    std = np.full(n_x, np.nan, dtype=float)
    for j in range(n_x):
        col = mat[:, j]
        good = col[np.isfinite(col)]
        if good.size == 0:
            continue
        mean[j] = float(np.mean(good))
        std[j] = 0.0 if good.size == 1 else float(np.std(good, ddof=0))
    return mean, std


def _mean_std(df: pd.DataFrame, x_col: str, y_col: str, x_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate each repeat at x_grid points, then compute mean ± std."""
    if df.empty or y_col not in df.columns:
        empty = np.full(10 if x_grid is None else len(x_grid), np.nan)
        xs = x_grid if x_grid is not None else np.linspace(0, 1, 10)
        return xs, empty, empty

    if x_grid is None:
        x_grid = np.linspace(df[x_col].min(), df[x_col].max(), 100)

    interp_vals = []
    for rep_id, grp in df.groupby("repeat"):
        grp = grp.sort_values(x_col)
        v = np.interp(x_grid, grp[x_col].values, grp[y_col].values,
                      left=np.nan, right=np.nan)
        interp_vals.append(v)

    if not interp_vals:
        nan_vec = np.full(len(x_grid), np.nan, dtype=float)
        return x_grid, nan_vec, nan_vec

    mat = np.asarray(interp_vals, dtype=float)
    mean, std = _columnwise_mean_std(mat)
    return x_grid, mean, std


def _budget_value(df: pd.DataFrame, y_col: str, budget_ns: float) -> tuple[float, float]:
    """Return mean ± std of y_col at a fixed budget (closest step)."""
    if df.empty or y_col not in df.columns:
        return np.nan, np.nan
    vals = []
    for _, grp in df.groupby("repeat"):
        grp = grp.sort_values("sampled_ns")
        idx = (grp["sampled_ns"] - budget_ns).abs().idxmin()
        vals.append(float(grp.loc[idx, y_col]))
    if not vals:
        return np.nan, np.nan
    arr = np.asarray(vals, dtype=float)
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return np.nan, np.nan
    m = float(np.mean(good))
    if good.size == 1:
        return m, 0.0
    return m, float(np.std(good, ddof=0))


# ---------------------------------------------------------------------------
# Figure 1: Main convergence curves (base preset)
# ---------------------------------------------------------------------------

def fig1_convergence_base(repo: Path, out_dir: Path) -> None:
    policies = list(POLICY_COLORS.keys())
    metrics = [
        ("f_helix_error_reweighted", "Helix fraction error (reweighted)", True),
        ("jsd", "JSD vs reference", True),
        ("coverage", "Coverage (target)", False),
        ("n_transitions", "Cumulative transitions", False),
    ]
    # Use fallback metric if reweighted not available
    fallback = {"f_helix_error_reweighted": "f_helix_error"}

    x_grid = np.linspace(0, 300, 200)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, ylabel, lower_better) in zip(axes, metrics):
        for policy in policies:
            dirs = _find_runs(repo, policy, "tica2d", 2.0, "none")
            if not dirs:
                continue
            df = _load_repeats(dirs)
            eff_col = col if col in df.columns else fallback.get(col, col)
            if eff_col not in df.columns:
                continue
            xs, mean, std = _mean_std(df, "sampled_ns", eff_col, x_grid)
            color = POLICY_COLORS[policy]
            label = POLICY_LABELS[policy]
            ax.plot(xs, mean, color=color, label=label, lw=1.5)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_ylabel(ylabel)
        if lower_better:
            ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("Budget (ns)")
    axes[0].set_title("Base preset: TICA-2D, 2 ns chunks, no noise")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_convergence_base.png")
    plt.close(fig)
    print("  fig1_convergence_base.png")


# ---------------------------------------------------------------------------
# Figure 2: Seed trajectory visualization
# ---------------------------------------------------------------------------

def fig2_seed_trajectory(repo: Path, out_dir: Path) -> None:
    ref_dir = repo / "outputs" / "v2" / "reference"
    tica2d_path = ref_dir / "a8q2_tica2d.npy"
    nhb_path = repo / "outputs" / "artifacts" / "a8q2_nhb.npy"
    if not tica2d_path.is_file() or not nhb_path.is_file():
        print("  fig2: reference TICA-2D or NHB not found, skipping")
        return

    tica2d = np.load(tica2d_path)
    nhb = np.load(nhb_path)
    is_helix = nhb >= 3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, policy in zip(axes, ["random", "compass"]):
        # Background: all frames colored helix/coil
        ax.scatter(tica2d[~is_helix, 0], tica2d[~is_helix, 1],
                   c="#56B4E9", alpha=0.06, s=1, rasterized=True, label="Coil")
        ax.scatter(tica2d[is_helix, 0], tica2d[is_helix, 1],
                   c="#E69F00", alpha=0.06, s=1, rasterized=True, label="Helix")

        # Load one representative repeat's seed decisions
        dirs = _find_runs(repo, policy, "tica2d", 2.0, "none")
        if not dirs:
            ax.set_title(f"{POLICY_LABELS.get(policy, policy)} — no data")
            continue

        dec_path = dirs[0] / "seed_decisions.csv"
        if not dec_path.is_file():
            ax.set_title(f"{POLICY_LABELS.get(policy, policy)} — no decisions")
            continue

        dec = pd.read_csv(dec_path)
        if "seed_frame" not in dec.columns:
            ax.set_title(f"{POLICY_LABELS.get(policy, policy)} — no seed_frame column")
            continue

        frames = dec["seed_frame"].values.astype(int)
        n = len(frames)
        colors = plt.cm.RdYlGn(np.linspace(1, 0, n))  # green→red over time

        ax.scatter(tica2d[frames, 0], tica2d[frames, 1],
                   c=colors, s=20, zorder=5, edgecolors="k", linewidths=0.3)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                    norm=plt.Normalize(vmin=0, vmax=n * 0.1 * 2))  # ns
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Seed time (ns)")

        ax.set_xlabel("TIC-1")
        ax.set_ylabel("TIC-2")
        ax.set_title(f"{POLICY_LABELS.get(policy, policy)}")
        # Legend proxy
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#56B4E9", markersize=8, label="Coil"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00", markersize=8, label="Helix"),
        ], loc="best", fontsize=8)

    fig.suptitle("Seed restart positions over time (green=early, red=late)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_seed_trajectory.png")
    plt.close(fig)
    print("  fig2_seed_trajectory.png")


# ---------------------------------------------------------------------------
# Figure 4: Chunk size ablation
# ---------------------------------------------------------------------------

def fig4_chunk_ablation(repo: Path, out_dir: Path) -> None:
    chunk_ns_list = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    budgets = [50.0, 100.0, 200.0]
    y_col = "f_helix_error_reweighted"
    fallback_col = "f_helix_error"

    fig, axes = plt.subplots(1, len(budgets), figsize=(4 * len(budgets), 4), sharey=True)
    for ax, budget in zip(axes, budgets):
        for policy in ["random", "compass"]:
            means, stds = [], []
            for chunk_ns in chunk_ns_list:
                dirs = _find_runs(repo, policy, "tica2d", chunk_ns, "none")
                if not dirs:
                    means.append(np.nan)
                    stds.append(np.nan)
                    continue
                df = _load_repeats(dirs)
                eff_col = y_col if y_col in df.columns else fallback_col
                m, s = _budget_value(df, eff_col, budget)
                means.append(m)
                stds.append(s)
            ax.errorbar(chunk_ns_list, means, yerr=stds,
                       label=POLICY_LABELS.get(policy, policy),
                       color=POLICY_COLORS[policy], marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_title(f"Budget = {budget:.0f} ns")
        ax.set_xlabel("Chunk size (ns)")
    axes[0].set_ylabel("Helix fraction error")
    axes[-1].legend()
    fig.suptitle("Ablation B: Chunk size (TICA-2D, no noise)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_chunk_ablation.png")
    plt.close(fig)
    print("  fig4_chunk_ablation.png")


# ---------------------------------------------------------------------------
# Figure 5: CV type ablation
# ---------------------------------------------------------------------------

def fig5_cv_ablation(repo: Path, out_dir: Path) -> None:
    cv_types = ["tica1d", "tica2d", "spib1d", "spib2d"]
    cv_labels = {"tica1d": "TICA-1D", "tica2d": "TICA-2D", "spib1d": "SPIB-1D", "spib2d": "SPIB-2D"}
    budget = 100.0
    y_col = "f_helix_error_reweighted"
    fallback_col = "f_helix_error"

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(cv_types))
    width = 0.35
    for i, policy in enumerate(["min_count", "compass"]):
        means, stds = [], []
        for cv in cv_types:
            dirs = _find_runs(repo, policy, cv, 2.0, "none")
            if not dirs:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            df = _load_repeats(dirs)
            eff_col = y_col if y_col in df.columns else fallback_col
            m, s = _budget_value(df, eff_col, budget)
            means.append(m)
            stds.append(s)
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=POLICY_LABELS.get(policy, policy),
               color=POLICY_COLORS[policy], capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([cv_labels[c] for c in cv_types])
    ax.set_ylabel("Helix fraction error")
    ax.set_title(f"Ablation A: CV type (chunk=2 ns, noise=none, budget={budget:.0f} ns)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_cv_ablation.png")
    plt.close(fig)
    print("  fig5_cv_ablation.png")


# ---------------------------------------------------------------------------
# Figure 6: Prior noise ablation
# ---------------------------------------------------------------------------

def fig6_noise_ablation(repo: Path, out_dir: Path) -> None:
    noise_levels = ["none", "low", "med", "high", "uniform"]
    noise_labels = {"none": "None", "low": "Low (σ=0.1)", "med": "Med (σ=0.3)",
                    "high": "High (σ=1.0)", "uniform": "Uniform"}
    budget = 100.0
    y_col = "f_helix_error_reweighted"
    fallback_col = "f_helix_error"

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(noise_levels))
    width = 0.35
    for i, policy in enumerate(["random", "compass"]):
        means, stds = [], []
        for noise in noise_levels:
            dirs = _find_runs(repo, policy, "tica2d", 2.0, noise)
            if not dirs:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            df = _load_repeats(dirs)
            eff_col = y_col if y_col in df.columns else fallback_col
            m, s = _budget_value(df, eff_col, budget)
            means.append(m)
            stds.append(s)
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=POLICY_LABELS.get(policy, policy),
               color=POLICY_COLORS[policy], capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([noise_labels[n] for n in noise_levels])
    ax.set_xlabel("Prior noise level")
    ax.set_ylabel("Helix fraction error")
    ax.set_title(f"Ablation C: Prior noise (TICA-2D, chunk=2 ns, budget={budget:.0f} ns)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_noise_ablation.png")
    plt.close(fig)
    print("  fig6_noise_ablation.png")


# ---------------------------------------------------------------------------
# Figure 7: Source ↔ Target coverage correlation
# ---------------------------------------------------------------------------

def fig7_coverage_correlation(repo: Path, out_dir: Path) -> None:
    policies = list(POLICY_COLORS.keys())
    budget_snapshots = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]

    fig, ax = plt.subplots(figsize=(6, 5))
    all_src, all_tgt = [], []
    for policy in policies:
        dirs = _find_runs(repo, policy, "tica2d", 2.0, "none")
        if not dirs:
            continue
        src_pts, tgt_pts = [], []
        for d in dirs:
            df = pd.read_csv(d / "v2_metrics.csv")
            if "coverage_source" not in df.columns or "coverage" not in df.columns:
                continue
            for b in budget_snapshots:
                idx = (df["sampled_ns"] - b).abs().idxmin()
                src_pts.append(float(df.loc[idx, "coverage_source"]))
                tgt_pts.append(float(df.loc[idx, "coverage"]))
        if src_pts:
            ax.scatter(src_pts, tgt_pts, s=20, alpha=0.5,
                      color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
            all_src.extend(src_pts)
            all_tgt.extend(tgt_pts)

    if len(all_src) > 3:
        r, pval = pearsonr(all_src, all_tgt)
        ax.set_title(f"Source ↔ Target coverage  (Pearson r={r:.3f}, p={pval:.3g})")
    else:
        ax.set_title("Source ↔ Target coverage")

    ax.set_xlabel("Source coverage (A10 latent bins)")
    ax.set_ylabel("Target coverage (A8Q2 latent bins)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_source_target_coverage.png")
    plt.close(fig)
    print("  fig7_source_target_coverage.png")


# ---------------------------------------------------------------------------
# Figure 8: Acceleration bar chart
# ---------------------------------------------------------------------------

def fig8_acceleration(repo: Path, out_dir: Path) -> None:
    """Speedup = median_budget(random to reach threshold) / median_budget(policy)."""
    thresholds = [
        ("f_helix_error_reweighted", "<=", 0.02, "f_helix_err≤2%"),
        ("coverage", ">=", 0.70, "coverage≥70%"),
    ]
    policies_show = ["independent_replicas", "min_count", "compass"]

    def _first_crossing(df: pd.DataFrame, col: str, op: str, thr: float) -> float | None:
        if col not in df.columns:
            col_alt = {"f_helix_error_reweighted": "f_helix_error"}.get(col, None)
            if col_alt and col_alt in df.columns:
                col = col_alt
            else:
                return None
        if op == "<=":
            hit = df[df[col] <= thr]
        else:
            hit = df[df[col] >= thr]
        if hit.empty:
            return None
        return float(hit["sampled_ns"].iloc[0])

    def _median_crossing(run_dirs: list[Path], col: str, op: str, thr: float) -> float:
        vals = []
        for d in run_dirs:
            df = pd.read_csv(d / "v2_metrics.csv").sort_values("sampled_ns")
            v = _first_crossing(df, col, op, thr)
            if v is not None:
                vals.append(v)
        if not vals:
            return np.nan
        return float(np.median(vals))

    random_dirs = _find_runs(repo, "random", "tica2d", 2.0, "none")

    n_thr = len(thresholds)
    n_pol = len(policies_show)
    fig, axes = plt.subplots(1, n_thr, figsize=(5 * n_thr, 4), sharey=False)
    if n_thr == 1:
        axes = [axes]

    for ax, (col, op, thr, label) in zip(axes, thresholds):
        random_med = _median_crossing(random_dirs, col, op, thr)
        speedups, errors = [], []
        for policy in policies_show:
            dirs = _find_runs(repo, policy, "tica2d", 2.0, "none")
            pol_med = _median_crossing(dirs, col, op, thr)
            if np.isnan(random_med) or np.isnan(pol_med) or pol_med <= 0:
                speedups.append(np.nan)
                errors.append(0)
            else:
                speedups.append(random_med / pol_med)
                # Bootstrap CI on speedup (simplified: use IQR)
                crossings = []
                for d in dirs:
                    df = pd.read_csv(d / "v2_metrics.csv").sort_values("sampled_ns")
                    v = _first_crossing(df, col, op, thr)
                    crossings.append(v)
                crossings = [v for v in crossings if v is not None]
                if len(crossings) >= 4:
                    q25, q75 = np.percentile(crossings, [25, 75])
                    err = 0.5 * (random_med / max(q25, 1e-6) - random_med / max(q75, 1e-6))
                    errors.append(abs(err))
                else:
                    errors.append(0)

        x = np.arange(n_pol)
        colors = [POLICY_COLORS[p] for p in policies_show]
        bars = ax.bar(x, speedups, color=colors, alpha=0.85, capsize=4)
        for bar_i, (b, e) in enumerate(zip(bars, errors)):
            if e > 0:
                ax.errorbar(b.get_x() + b.get_width() / 2, b.get_height(),
                            yerr=e, fmt="none", color="black", capsize=4)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="No gain (=random)")
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies_show], rotation=15, ha="right")
        ax.set_title(f"Speedup: {label}")
        ax.set_ylabel("Speedup (× faster than random)")
        ax.legend(fontsize=8)

    fig.suptitle("Acceleration factor (median budget to threshold)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_acceleration.png")
    plt.close(fig)
    print("  fig8_acceleration.png")


# ---------------------------------------------------------------------------
# Figure 9: Updated transfer diagnostics
# ---------------------------------------------------------------------------

def fig9_transfer_diagnostics(repo: Path, out_dir: Path) -> None:
    """1D FES comparison (A10 native vs A8Q2 transferred) with helix-fraction colored curves."""
    ref_dir = repo / "outputs" / "v2" / "reference"
    artifacts = repo / "outputs" / "artifacts"
    tica_dir = repo / "outputs" / "tica_artifacts_a10"

    # Required paths
    nhb_a8q2_path = artifacts / "a8q2_nhb.npy"
    tica1d_a8q2_path = ref_dir / "a8q2_tica1d.npy"
    if not nhb_a8q2_path.is_file() or not tica1d_a8q2_path.is_file():
        print("  fig9: A8Q2 NHB or transferred TIC-1 data missing, skipping")
        return

    nhb_a8q2 = np.load(nhb_a8q2_path)
    tic1_a8q2 = np.load(tica1d_a8q2_path).ravel()

    # Compute A10 native TIC-1 from features + eigenvectors
    tic1_a10 = None
    nhb_a10 = None
    feat_a10_path = artifacts / "a10_features.npy"
    mean_path = tica_dir / "a10_tica_mean0.npy"
    vecs_path = tica_dir / "a10_tica_singular_vectors_left.npy"
    nhb_a10_path = artifacts / "a10_nhb.npy"
    if feat_a10_path.is_file() and mean_path.is_file() and vecs_path.is_file():
        feats = np.load(feat_a10_path).astype(float)
        mean = np.load(mean_path)
        vecs = np.load(vecs_path)          # shape (40, n_components)
        tic1_a10 = (feats - mean) @ vecs[:, 0]
        if nhb_a10_path.is_file():
            nhb_a10 = np.load(nhb_a10_path)
    else:
        print("  fig9: A10 TICA artifacts not found — single-panel mode")

    from src.metrics import plot_transfer_fes_1d

    # Panel A: two-panel FES colored by local helix fraction
    plot_transfer_fes_1d(
        tic1_a8q2=tic1_a8q2,
        nhb_a8q2=nhb_a8q2,
        output_path=out_dir / "fig9a_transfer_fes_1d.png",
        tic1_a10=tic1_a10,
        nhb_a10=nhb_a10,
    )
    print("  fig9a_transfer_fes_1d.png")

    # Panel B: transferred TIC-1 vs NHB scatter (A8Q2 only)
    is_helix = nhb_a8q2 >= 3
    from scipy.stats import pearsonr as _pearsonr
    r, _ = _pearsonr(tic1_a8q2, nhb_a8q2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(tic1_a8q2[~is_helix], nhb_a8q2[~is_helix],
               s=1, alpha=0.05, color="#1f77b4", rasterized=True, label="Coil")
    ax.scatter(tic1_a8q2[is_helix], nhb_a8q2[is_helix],
               s=1, alpha=0.05, color="#E69F00", rasterized=True, label="Helix")
    ax.axhline(3, color="gray", linestyle="--", lw=1.0, label="$N_{\\mathrm{HB}}=3$")
    ax.set_title(f"Transferred TIC-1 vs $N_{{\\mathrm{{HB}}}}$ on A8Q2  ($r={r:.3f}$)")
    ax.set_xlabel("Transferred TIC-1 (A10→A8Q2)")
    ax.set_ylabel("$N$ hydrogen bonds")
    ax.legend(fontsize=8, markerscale=6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9b_transfer_tic1_nhb.png", dpi=300)
    plt.close(fig)
    print("  fig9b_transfer_tic1_nhb.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out-dir", type=Path, default=Path("outputs/v2/figures"))
    p.add_argument("--figures", type=str, default="all",
                   help="Comma-separated list: 1,2,4,5,6,7,8 or 'all'")
    args = p.parse_args(argv)

    repo = args.repo_root.resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = set()
    if args.figures.strip().lower() == "all":
        requested = {1, 2, 4, 5, 6, 7, 8, 9}
    else:
        requested = {int(x.strip()) for x in args.figures.split(",") if x.strip()}

    print(f"Generating figures -> {out_dir}")
    fns = {
        1: fig1_convergence_base,
        2: fig2_seed_trajectory,
        4: fig4_chunk_ablation,
        5: fig5_cv_ablation,
        6: fig6_noise_ablation,
        7: fig7_coverage_correlation,
        8: fig8_acceleration,
        9: fig9_transfer_diagnostics,
    }
    for num, fn in fns.items():
        if num in requested:
            fn(repo, out_dir)

    print(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
