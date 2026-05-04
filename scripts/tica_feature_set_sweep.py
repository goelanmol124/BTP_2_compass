#!/usr/bin/env python3
"""Multi–feature-set TICA sweep: transfer metrics, lag sensitivity, bootstrap, split-replica.

Loads pooled A10 and A8Q2 production trajectories (extended + helix), builds several
protein-agnostic featurizations, fits TICA on the donor, evaluates Moffett–Shukla
transfer metrics and Pearson r vs N_HB on the acceptor, and writes CSVs + figures to
``outputs/tica_feature_sweep/`` (or ``--out-dir``).

Example::

    python scripts/tica_feature_set_sweep.py \\
        --sim-dir-a10 outputs/btp_equil/A10 --prefix-a10 A10 \\
        --sim-dir-a8q2 outputs/btp_equil/A8Q2 --prefix-a8q2 A8Q2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path(root: Path) -> None:
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def pool_trajectory_features(
    sim_dir: Path,
    prefix: str,
    burnin_frames: int,
    feat_name: str,
    featurizers: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Load extended + helix replicas, strip burn-in, concatenate frames."""

    from src.analysis import compute_nhb, load_trajectory

    feat_fn = featurizers[feat_name]
    feats_list: list[np.ndarray] = []
    nhb_list: list[np.ndarray] = []
    for lab in ("extended", "helix"):
        stem = f"{prefix}_{lab}"
        dcd = sim_dir / f"{stem}_prod.dcd"
        top = sim_dir / f"{stem}_prod_top.pdb"
        if not dcd.is_file() or not top.is_file():
            raise FileNotFoundError(f"Missing {dcd} or {top}")
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        if traj.n_frames <= burnin_frames:
            raise RuntimeError(f"{dcd}: need > {burnin_frames} frames, got {traj.n_frames}")
        t = traj[burnin_frames:]
        feats_list.append(feat_fn(t).astype(np.float32))
        nhb_list.append(compute_nhb(t).astype(np.float64))
    return np.vstack(feats_list), np.concatenate(nhb_list, axis=0)


def tica_dim_for(n_features: int, requested: int) -> int:
    return int(max(1, min(requested, n_features)))


def fit_tica(
    X: np.ndarray,
    lag: int,
    dim: int,
    epsilon: float,
    scaling: str | None,
):
    from deeptime.decomposition import TICA

    d = tica_dim_for(X.shape[1], dim)
    return TICA(lagtime=lag, dim=d, epsilon=epsilon, scaling=scaling).fit(X).fetch_model()


def block_bootstrap_tic1(
    features: np.ndarray,
    lagtime: int,
    block_size: int,
    n_bootstrap: int,
    epsilon: float,
    dim: int,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    from deeptime.decomposition import TICA

    d = tica_dim_for(features.shape[1], dim)
    T = len(features)
    n_blocks = T // block_size
    if n_blocks < 2:
        return {"mean_cos_sim": float("nan"), "min_cos_sim": float("nan"), "stable": False}

    ref = TICA(lagtime=lagtime, dim=d, epsilon=epsilon, scaling=None).fit(features).fetch_model()
    ref_v = np.asarray(ref.singular_vectors_left)[:, 0]
    ref_v = ref_v / max(np.linalg.norm(ref_v), 1e-15)

    sims = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n_blocks, size=n_blocks) * block_size
        parts = [features[s : s + block_size] for s in starts]
        boot_x = np.concatenate(parts, axis=0)
        m = TICA(lagtime=lagtime, dim=d, epsilon=epsilon, scaling=None).fit(boot_x).fetch_model()
        v = np.asarray(m.singular_vectors_left)[:, 0]
        v = v / max(np.linalg.norm(v), 1e-15)
        sims[b] = float(abs(np.dot(ref_v, v)))
    return {
        "mean_cos_sim": float(np.mean(sims)),
        "min_cos_sim": float(np.min(sims)),
        "stable": bool(np.all(sims > 0.95)),
    }


def split_replica_cosine(
    features: np.ndarray,
    lag: int,
    epsilon: float,
    dim: int,
) -> float:
    from deeptime.decomposition import TICA

    d = tica_dim_for(features.shape[1], dim)
    n = len(features) // 2
    a = features[:n]
    b = features[n:]
    if len(a) < 2 * d or len(b) < 2 * d:
        return float("nan")
    ma = TICA(lagtime=lag, dim=d, epsilon=epsilon, scaling=None).fit(a).fetch_model()
    mb = TICA(lagtime=lag, dim=d, epsilon=epsilon, scaling=None).fit(b).fetch_model()
    va = np.asarray(ma.singular_vectors_left)[:, 0]
    vb = np.asarray(mb.singular_vectors_left)[:, 0]
    va = va / max(np.linalg.norm(va), 1e-15)
    vb = vb / max(np.linalg.norm(vb), 1e-15)
    return float(abs(np.dot(va, vb)))


def transfer_metrics_at_lag(
    X_a10: np.ndarray,
    X_a8: np.ndarray,
    nhb_a8: np.ndarray,
    lag: int,
    epsilon: float,
    tica_dim: int,
) -> dict[str, float]:
    from scipy.stats import pearsonr

    from src.tica_transfer import compute_d0, compute_dkm, compute_dtau

    model = fit_tica(X_a10, lag, tica_dim, epsilon, "kinetic_map")
    mean_d = np.asarray(model.mean_0, dtype=float)
    V = np.asarray(model.singular_vectors_left, dtype=float)
    sv_d = np.asarray(model.singular_values, dtype=float)

    Xc = X_a8.astype(float) - mean_d

    from deeptime.decomposition import TICA

    k_eval = int(min(2, V.shape[1]))
    native2 = TICA(lagtime=lag, dim=k_eval, epsilon=epsilon, scaling=None).fit(X_a8).fetch_model()
    native_sv = np.asarray(native2.singular_values, dtype=float).reshape(-1)[:k_eval]

    D0 = compute_d0(Xc, V, K=k_eval)
    Dtau = compute_dtau(
        Xc, V, native_singular_values_acceptor=native_sv, lag=lag, K=k_eval
    )
    d_km, _ = compute_dkm(
        X_a8, V, lagtime=lag, K=k_eval, M=1, epsilon=epsilon
    )
    z1 = (X_a8 - mean_d) @ V[:, 0]
    r, _ = pearsonr(z1, nhb_a8)
    if not np.isfinite(r):
        r = 0.0
    lam1 = float(abs(sv_d[0])) if sv_d.size else float("nan")
    frame_ns = 0.1
    if lam1 < 1.0 and np.isfinite(lam1) and lam1 > 0:
        t_impl = -lag * frame_ns / np.log(lam1)
    else:
        t_impl = float("inf")
    return {
        "D0": float(D0),
        "D_tau": float(Dtau),
        "D_KM": float(d_km),
        "pearson_r": float(r),
        "pearson_abs_r": float(abs(r)),
        "lambda1_a10": lam1,
        "t_impl_ns": float(t_impl) if np.isfinite(t_impl) else float("nan"),
    }


def lag_scan_its(
    X: np.ndarray,
    lag_list: list[int],
    frame_spacing_ns: float,
    min_lag: int,
    rel_plateau: float,
    epsilon: float,
    tica_dim: int,
) -> tuple[int, str, list[dict[str, float]]]:
    from src.tica_train import choose_lag_from_implied_timescale_scan

    t1_list: list[float] = []
    records: list[dict[str, float]] = []
    d = tica_dim_for(X.shape[1], tica_dim)
    for lag in lag_list:
        m = fit_tica(X, lag, tica_dim, epsilon, "kinetic_map")
        sv = np.asarray(m.singular_values, dtype=float)
        lam1 = float(abs(sv[0])) if sv.size else float("nan")
        lag_ns = lag * frame_spacing_ns
        if lam1 >= 1.0 or not np.isfinite(lam1) or lam1 <= 0:
            t1 = float("inf")
        else:
            t1 = -lag_ns / np.log(lam1)
        t1_list.append(t1)
        records.append(
            {
                "lag_frames": lag,
                "lag_ns": lag_ns,
                "lambda1": lam1,
                "t_impl_ns": t1 if np.isfinite(t1) else float("nan"),
                "dim_fit": d,
            }
        )

    rec_lag, reason = choose_lag_from_implied_timescale_scan(
        list(lag_list), t1_list, min_lag, rel_plateau
    )
    return rec_lag, reason, records


def plot_metrics_vs_lag(df_rows: list[dict], out_path: Path) -> None:
    """Rows must include feature_set, lag_frames, D0, D_tau, D_KM, pearson_abs_r."""

    by_fs: dict[str, list[dict]] = {}
    for row in df_rows:
        by_fs.setdefault(row["feature_set"], []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    metrics = [
        ("D0", "D₀"),
        ("D_tau", "D_τ"),
        ("D_KM", "D_KM"),
        ("pearson_abs_r", "|Pearson r| (TIC-1 vs NHB)"),
    ]
    for ax, (key, ylab) in zip(axes.ravel(), metrics):
        for fs, rows in sorted(by_fs.items()):
            rows = sorted(rows, key=lambda r: r["lag_frames"])
            x = [r["lag_frames"] * 0.1 for r in rows]
            y = [r[key] for r in rows]
            ax.plot(x, y, "o-", label=fs, linewidth=1.5, markersize=3)
        ax.set_xlabel("Lag τ (ns)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        if key == "pearson_abs_r":
            ax.axhline(0.75, color="0.5", ls="--", linewidth=0.8)
    axes.ravel()[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Transfer metrics vs TICA lag (donor re-fit per lag)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bootstrap_bars(rows: list[dict], out_path: Path) -> None:
    labels = [r["feature_set"] for r in rows]
    means = [r["bootstrap_mean_cos"] for r in rows]
    mins = [r["bootstrap_min_cos"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(x - 0.2, means, 0.4, label="mean |cos|")
    ax.bar(x + 0.2, mins, 0.4, label="min |cos|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(0.95, color="crimson", ls="--", label="0.95 threshold")
    ax.set_ylabel("Bootstrap TIC-1 stability")
    ax.legend()
    ax.set_title("Block bootstrap TIC-1 cosine (A10, scaling=None)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_split_replica(rows: list[dict], out_path: Path) -> None:
    labels = [r["feature_set"] for r in rows]
    vals = [r["split_replica_cos"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(labels, vals, color="steelblue")
    ax.axhline(0.95, color="crimson", ls="--")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("|cos(TIC-1 first half, TIC-1 second half)|")
    ax.set_title("Split-replica TIC-1 agreement (A10)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_lambda1_vs_lag(df_rows: list[dict], out_path: Path) -> None:
    by_fs: dict[str, list[dict]] = {}
    for row in df_rows:
        if row.get("lambda1_a10") is None:
            continue
        by_fs.setdefault(row["feature_set"], []).append(row)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for fs, rows in sorted(by_fs.items()):
        rows = sorted(rows, key=lambda r: r["lag_frames"])
        ax.plot(
            [r["lag_frames"] * 0.1 for r in rows],
            [r["lambda1_a10"] for r in rows],
            "o-",
            label=fs,
        )
    ax.set_xlabel("Lag τ (ns)")
    ax.set_ylabel("|λ₁| (TIC-1, donor A10, kinetic_map)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Donor TIC-1 singular value vs lag")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def pool_donor_zscore_torsions_plus_ca(
    sim_dir_a10: Path,
    prefix_a10: str,
    sim_dir_a8: Path,
    prefix_a8: str,
    burnin_frames: int,
    ca_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate z-scored torsions and z-scored CA features; μ and σ from **A10 only**.

    Both systems are transformed with the same donor statistics so the acceptor
    lives in the donor-defined standardized frame (target-specific means are not used).
    """

    from src.analysis import compute_featurization, compute_nhb, load_trajectory

    def pool_tor_ca(sim_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_list: list[np.ndarray] = []
        c_list: list[np.ndarray] = []
        n_list: list[np.ndarray] = []
        for lab in ("extended", "helix"):
            stem = f"{prefix}_{lab}"
            dcd = sim_dir / f"{stem}_prod.dcd"
            top = sim_dir / f"{stem}_prod_top.pdb"
            if not dcd.is_file() or not top.is_file():
                raise FileNotFoundError(f"Missing {dcd} or {top}")
            traj = load_trajectory(str(dcd), str(top), protein_only=True)
            if traj.n_frames <= burnin_frames:
                raise RuntimeError(f"{dcd}: need > {burnin_frames} frames, got {traj.n_frames}")
            t = traj[burnin_frames:]
            t_list.append(compute_featurization(t))
            c_list.append(np.asarray(ca_fn(t), dtype=np.float64))
            n_list.append(compute_nhb(t).astype(np.float64))
        return np.vstack(t_list), np.vstack(c_list), np.concatenate(n_list, axis=0)

    T10, C10, _ = pool_tor_ca(sim_dir_a10, prefix_a10)
    T8, C8, N8 = pool_tor_ca(sim_dir_a8, prefix_a8)
    if T10.shape[1] != T8.shape[1] or C10.shape[1] != C8.shape[1]:
        raise RuntimeError("Internal pool shape mismatch")

    eps = 1e-8
    mu_t, sig_t = T10.mean(axis=0), T10.std(axis=0)
    mu_c, sig_c = C10.mean(axis=0), C10.std(axis=0)
    sig_t = np.maximum(sig_t, eps)
    sig_c = np.maximum(sig_c, eps)

    Z10 = np.hstack([(T10 - mu_t) / sig_t, (C10 - mu_c) / sig_c]).astype(np.float32)
    Z8 = np.hstack([(T8 - mu_t) / sig_t, (C8 - mu_c) / sig_c]).astype(np.float32)
    return Z10, Z8, N8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument(
        "--sim-dir-a10",
        type=Path,
        default=Path("outputs/btp_equil/A10"),
        help="Directory with A10_*_prod.dcd and *_prod_top.pdb",
    )
    parser.add_argument("--prefix-a10", default="A10")
    parser.add_argument(
        "--sim-dir-a8q2",
        type=Path,
        default=Path("outputs/btp_equil/A8Q2"),
        help="Directory with A8Q2_*_prod.dcd",
    )
    parser.add_argument("--prefix-a8q2", default="A8Q2")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tica_feature_sweep"))
    parser.add_argument("--burnin-frames", type=int, default=-1)
    parser.add_argument("--tica-dim", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=[
            "torsions_sc",
            "ca_dist_lower",
            "inv_r_ca",
            "backbone_angle_cos",
            "torsions_plus_ca",
        ],
        help=(
            "Names from src.tica_featurizers.FEATURIZERS, plus special pools: "
            "torsions_ca_zscore_donor, torsions_longrange_zscore_donor"
        ),
    )
    parser.add_argument(
        "--lag-grid",
        type=int,
        nargs="*",
        default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50],
        help="Lag times in frames (100 ps each)",
    )
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    _ensure_sys_path(root)
    import os

    os.chdir(root)

    from src import config
    from src.analysis import BURNIN_FRAMES
    from src.tica_featurizers import (
        FEATURIZERS,
        SPECIAL_POOL_FEATURE_SETS,
        ca_distances_longrange,
        ca_distances_lower_triangle,
    )

    burnin = args.burnin_frames if args.burnin_frames >= 0 else int(BURNIN_FRAMES)
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_a10 = (root / args.sim_dir_a10).resolve()
    sim_a8 = (root / args.sim_dir_a8q2).resolve()

    allowed = set(FEATURIZERS.keys()) | set(SPECIAL_POOL_FEATURE_SETS)
    # --- Load pooled features per set ---
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in args.feature_sets:
        if name not in allowed:
            logger.error("Unknown feature set %r; allowed: %s", name, sorted(allowed))
            return 1
        logger.info("Pooling features: %s", name)
        try:
            if name == "torsions_ca_zscore_donor":
                x10, x8, n8 = pool_donor_zscore_torsions_plus_ca(
                    sim_a10, args.prefix_a10, sim_a8, args.prefix_a8q2, burnin, ca_distances_lower_triangle
                )
            elif name == "torsions_longrange_zscore_donor":
                x10, x8, n8 = pool_donor_zscore_torsions_plus_ca(
                    sim_a10,
                    args.prefix_a10,
                    sim_a8,
                    args.prefix_a8q2,
                    burnin,
                    lambda traj: ca_distances_longrange(traj, 4),
                )
            else:
                x10, _ = pool_trajectory_features(sim_a10, args.prefix_a10, burnin, name, FEATURIZERS)
                x8, n8 = pool_trajectory_features(sim_a8, args.prefix_a8q2, burnin, name, FEATURIZERS)
        except FileNotFoundError as e:
            logger.error("%s", e)
            logger.error(
                "Provide directories with extended+helix production DCDs or sync RunPod outputs."
            )
            return 1
        if x10.shape[1] != x8.shape[1]:
            raise RuntimeError(f"Feature dim mismatch {name}: A10 {x10.shape} vs A8Q2 {x8.shape}")
        data_cache[name] = (x10, x8, n8)
        logger.info("  A10 %s  A8Q2 %s", x10.shape, x8.shape)

    long_rows: list[dict] = []
    chosen_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    lag_choice_rows: list[dict] = []

    rng = np.random.default_rng(42)
    lag_list = list(args.lag_grid)

    for name, (X10, X8, nhb8) in data_cache.items():
        # ITS-based recommended lag (same grid as sweep)
        rec_lag, reason, scan_recs = lag_scan_its(
            X10,
            lag_list,
            0.1,
            int(config.TICA_MIN_LAG_FRAMES),
            float(config.TICA_LAG_ITS_REL_PLATEAU),
            args.epsilon,
            args.tica_dim,
        )
        lag_choice_rows.append(
            {
                "feature_set": name,
                "chosen_lag_frames": rec_lag,
                "chosen_lag_ns": rec_lag * 0.1,
                "reason": reason,
                "n_features": X10.shape[1],
                "n_frames_a10": X10.shape[0],
                "n_frames_a8q2": X8.shape[0],
            }
        )
        scan_fieldnames = ["feature_set"] + list(scan_recs[0].keys())
        write_csv(
            out_dir / f"lag_scan_{name}.csv",
            [{**{"feature_set": name}, **r} for r in scan_recs],
            scan_fieldnames,
        )

        for lag in lag_list:
            m = transfer_metrics_at_lag(X10, X8, nhb8, lag, args.epsilon, args.tica_dim)
            row = {
                "feature_set": name,
                "lag_frames": lag,
                "lag_ns": lag * 0.1,
                **m,
            }
            long_rows.append(row)

        m_chosen = transfer_metrics_at_lag(X10, X8, nhb8, rec_lag, args.epsilon, args.tica_dim)
        boot = block_bootstrap_tic1(
            X10,
            rec_lag,
            int(config.TICA_BOOTSTRAP_BLOCK_SIZE),
            int(config.TICA_N_BOOTSTRAP),
            args.epsilon,
            args.tica_dim,
            rng,
        )
        bootstrap_rows.append(
            {
                "feature_set": name,
                "chosen_lag_frames": rec_lag,
                **boot,
                "split_replica_cos": split_replica_cosine(X10, rec_lag, args.epsilon, args.tica_dim),
                **{f"metric_{k}": v for k, v in m_chosen.items()},
            }
        )
        chosen_rows.append({"feature_set": name, "chosen_lag_frames": rec_lag, **m_chosen})

    write_csv(
        out_dir / "metrics_vs_lag_long.csv",
        long_rows,
        [
            "feature_set",
            "lag_frames",
            "lag_ns",
            "D0",
            "D_tau",
            "D_KM",
            "pearson_r",
            "pearson_abs_r",
            "lambda1_a10",
            "t_impl_ns",
        ],
    )
    write_csv(
        out_dir / "summary_at_chosen_lag.csv",
        chosen_rows,
        list(chosen_rows[0].keys()) if chosen_rows else [],
    )
    write_csv(
        out_dir / "lag_choice.csv",
        lag_choice_rows,
        list(lag_choice_rows[0].keys()) if lag_choice_rows else [],
    )
    fn_boot = [
        "feature_set",
        "chosen_lag_frames",
        "bootstrap_mean_cos",
        "bootstrap_min_cos",
        "bootstrap_stable",
        "split_replica_cos",
        "metric_D0",
        "metric_D_tau",
        "metric_D_KM",
        "metric_pearson_r",
        "metric_pearson_abs_r",
        "metric_lambda1_a10",
        "metric_t_impl_ns",
    ]
    br = []
    for r in bootstrap_rows:
        br.append(
            {
                "feature_set": r["feature_set"],
                "chosen_lag_frames": r["chosen_lag_frames"],
                "bootstrap_mean_cos": r["mean_cos_sim"],
                "bootstrap_min_cos": r["min_cos_sim"],
                "bootstrap_stable": r["stable"],
                "split_replica_cos": r["split_replica_cos"],
                "metric_D0": r["metric_D0"],
                "metric_D_tau": r["metric_D_tau"],
                "metric_D_KM": r["metric_D_KM"],
                "metric_pearson_r": r["metric_pearson_r"],
                "metric_pearson_abs_r": r["metric_pearson_abs_r"],
                "metric_lambda1_a10": r["metric_lambda1_a10"],
                "metric_t_impl_ns": r["metric_t_impl_ns"],
            }
        )
    write_csv(out_dir / "bootstrap_and_replica.csv", br, fn_boot)

    plot_metrics_vs_lag(long_rows, out_dir / "fig_metrics_vs_lag.png")
    plot_lambda1_vs_lag(long_rows, out_dir / "fig_lambda1_vs_lag.png")
    plot_bootstrap_bars(br, out_dir / "fig_bootstrap_stability.png")
    plot_split_replica(br, out_dir / "fig_split_replica.png")

    # Summary figure: radar-style bar at chosen lag — Pearson + inverted D metrics
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    xs = np.arange(len(chosen_rows))
    w = 0.2
    for i, key in enumerate(["pearson_abs_r", "D0", "D_tau", "D_KM"]):
        vals = [r[key] for r in chosen_rows]
        if key != "pearson_abs_r":
            vals = [min(v, 3.0) / 3.0 for v in vals]
        ax.bar(xs + (i - 1.5) * w, vals, w, label=key)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["feature_set"] for r in chosen_rows], rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.set_title("At ITS-chosen lag: |r| raw; D-metrics scaled by min(v,3)/3")
    fig.savefig(out_dir / "fig_summary_chosen_lag.png", dpi=200)
    plt.close(fig)

    narrative = {
        "experiment": (
            "Compared five transferable featurizations on the same pooled MD for A10 and A8Q2. "
            "For each feature set we ran an implied-timescale lag scan on A10, picked a recommended lag, "
            "then recomputed donor TICA at each lag on the sweep grid and evaluated D₀, D_τ, D_KM and "
            "Pearson(transferred TIC-1, NHB | A8Q2). Block bootstrap and split-replica stability "
            "were computed on A10 at the chosen lag."
        ),
        "outputs": [str(p.name) for p in sorted(out_dir.glob("*"))],
    }
    with open(out_dir / "experiment_notes.json", "w", encoding="utf-8") as f:
        json.dump(narrative, f, indent=2)

    logger.info("Wrote results under %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
