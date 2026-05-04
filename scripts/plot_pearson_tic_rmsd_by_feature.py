#!/usr/bin/env python3
"""Bar comparison: Pearson r(transferred TIC-1/2, RMSD) across TICA feature sets.

Computes the same quantities as ``plot_transferred_tic_vs_coil_rmsd.py`` but for
every feature set used in ``tica_feature_set_sweep.py``, then saves a grouped
bar chart and CSV.

RMSD axes (both use the pooled A8Q2 trajectory, extended | helix):

1. **Coil** — Cα RMSD (nm) to the first post–burn-in frame of the extended replica.
2. **Helix** — backbone RMSD (nm) vs ideal helix ``{prefix}_helix.pdb`` in the
   simulation directory (same definition as Phase-5 FES).

Example::

    python scripts/plot_pearson_tic_rmsd_by_feature.py \\
        --out-dir outputs/tica_rmsd_correlation
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
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FEATURE_SETS = [
    "torsions_sc",
    "ca_dist_lower",
    "inv_r_ca",
    "backbone_angle_cos",
    "torsions_plus_ca",
    "torsions_ca_zscore_donor",
    "torsions_longrange_zscore_donor",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path(root: Path) -> None:
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def pool_a8q2_trajs_extended_then_helix(
    sim_dir: Path,
    prefix: str,
    burnin_frames: int,
):
    from src.analysis import load_trajectory

    parts = []
    for lab in ("extended", "helix"):
        stem = f"{prefix}_{lab}"
        dcd = sim_dir / f"{stem}_prod.dcd"
        top = sim_dir / f"{stem}_prod_top.pdb"
        if not dcd.is_file() or not top.is_file():
            raise FileNotFoundError(f"Missing {dcd} or {top}")
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        if traj.n_frames <= burnin_frames:
            raise RuntimeError(f"{dcd}: need > {burnin_frames} frames")
        parts.append(traj[burnin_frames:])
    import mdtraj as md

    return md.join(parts)


def pool_trajectory_features(
    sim_dir: Path,
    prefix: str,
    burnin_frames: int,
    feat_name: str,
    featurizers: dict,
) -> tuple[np.ndarray, np.ndarray]:
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


def pool_donor_zscore_torsions_plus_ca(
    sim_dir_a10: Path,
    prefix_a10: str,
    sim_dir_a8: Path,
    prefix_a8: str,
    burnin_frames: int,
    ca_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def coil_reference_snapshot_from_extended(
    sim_dir: Path,
    prefix: str,
    burnin_frames: int,
):
    from src.analysis import load_trajectory

    stem = f"{prefix}_extended"
    dcd = sim_dir / f"{stem}_prod.dcd"
    top = sim_dir / f"{stem}_prod_top.pdb"
    traj = load_trajectory(str(dcd), str(top), protein_only=True)
    if traj.n_frames <= burnin_frames:
        raise RuntimeError(f"{dcd}: need > {burnin_frames} frames")
    return traj[burnin_frames : burnin_frames + 1]


def load_pooled_features(
    name: str,
    sim_a10: Path,
    prefix_a10: str,
    sim_a8: Path,
    prefix_a8: str,
    burnin: int,
    FEATURIZERS: dict,
    ca_distances_lower_triangle,
    ca_distances_longrange,
) -> tuple[np.ndarray, np.ndarray]:
    from src.tica_featurizers import SPECIAL_POOL_FEATURE_SETS

    if name in SPECIAL_POOL_FEATURE_SETS:
        if name == "torsions_ca_zscore_donor":
            x10, x8, _ = pool_donor_zscore_torsions_plus_ca(
                sim_a10, prefix_a10, sim_a8, prefix_a8, burnin, ca_distances_lower_triangle
            )
        else:
            x10, x8, _ = pool_donor_zscore_torsions_plus_ca(
                sim_a10,
                prefix_a10,
                sim_a8,
                prefix_a8,
                burnin,
                lambda traj: ca_distances_longrange(traj, 4),
            )
        return x10, x8
    x10, _ = pool_trajectory_features(sim_a10, prefix_a10, burnin, name, FEATURIZERS)
    x8, _ = pool_trajectory_features(sim_a8, prefix_a8, burnin, name, FEATURIZERS)
    return x10, x8


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    r, p = pearsonr(a, b)
    if not np.isfinite(r):
        return 0.0, 1.0
    return float(r), float(p)


def plot_grouped_bars(
    labels: list[str],
    r1_coil: list[float],
    r2_coil: list[float],
    r1_helix: list[float],
    r2_helix: list[float],
    out_path: Path,
) -> None:
    x = np.arange(len(labels))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.1), 4.8), constrained_layout=True)

    for ax, r1, r2, title in [
        (axes[0], r1_coil, r2_coil, "Cα RMSD vs coil snapshot (nm)"),
        (axes[1], r1_helix, r2_helix, "BB RMSD vs ideal helix (nm)"),
    ]:
        ax.bar(x - 1.5 * w, r1, w, label="TIC-1", color="tab:blue")
        ax.bar(x - 0.5 * w, r2, w, label="TIC-2", color="tab:orange")
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel("Pearson r")
        ax.set_title(title)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("A10→A8Q2: transferred TICA vs RMSD by feature set", fontsize=12)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=_repo_root())
    p.add_argument("--sim-dir-a10", type=Path, default=Path("outputs/btp_equil/A10"))
    p.add_argument("--prefix-a10", default="A10")
    p.add_argument("--sim-dir-a8q2", type=Path, default=Path("outputs/btp_equil/A8Q2"))
    p.add_argument("--prefix-a8q2", default="A8Q2")
    p.add_argument(
        "--helix-ref-pdb",
        type=Path,
        default=None,
        help="Ideal helix PDB for BB RMSD (default: {sim-dir}/{prefix}_helix.pdb)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/tica_rmsd_correlation"))
    p.add_argument("--burnin-frames", type=int, default=-1)
    p.add_argument("--feature-sets", nargs="*", default=DEFAULT_FEATURE_SETS)
    p.add_argument("--lag-frames", type=int, default=-1)
    p.add_argument("--tica-dim", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=1e-6)
    args = p.parse_args(argv)

    root = args.repo_root.resolve()
    _ensure_sys_path(root)
    import os

    os.chdir(root)

    from src import config
    from src.analysis import BURNIN_FRAMES, compute_backbone_rmsd_to_ideal_helix, compute_backbone_rmsd_to_snapshot
    from src.tica_featurizers import FEATURIZERS, ca_distances_longrange, ca_distances_lower_triangle
    from src.tica_transfer import project_transferred
    from deeptime.decomposition import TICA

    burnin = args.burnin_frames if args.burnin_frames >= 0 else int(BURNIN_FRAMES)
    lag = args.lag_frames if args.lag_frames >= 0 else int(config.TICA_MIN_LAG_FRAMES)

    sim_a10 = (root / args.sim_dir_a10).resolve()
    sim_a8 = (root / args.sim_dir_a8q2).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    helix_ref = args.helix_ref_pdb
    if helix_ref is None:
        helix_ref = sim_a8 / f"{args.prefix_a8q2}_helix.pdb"
    else:
        helix_ref = helix_ref.resolve()
    if not helix_ref.is_file():
        logger.error("Helix reference missing: %s", helix_ref)
        return 1

    ref_snap = coil_reference_snapshot_from_extended(sim_a8, args.prefix_a8q2, burnin)
    traj_pool = pool_a8q2_trajs_extended_then_helix(sim_a8, args.prefix_a8q2, burnin)
    rmsd_coil = np.asarray(compute_backbone_rmsd_to_snapshot(traj_pool, ref_snap), dtype=np.float64)
    rmsd_helix = np.asarray(
        compute_backbone_rmsd_to_ideal_helix(traj_pool, str(helix_ref)), dtype=np.float64
    )

    allowed = set(FEATURIZERS.keys()) | {"torsions_ca_zscore_donor", "torsions_longrange_zscore_donor"}
    rows: list[dict] = []
    labels_plot: list[str] = []
    r1c: list[float] = []
    r2c: list[float] = []
    r1h: list[float] = []
    r2h: list[float] = []

    for name in args.feature_sets:
        if name not in allowed:
            logger.warning("Skip unknown feature set %r", name)
            continue
        try:
            x10, x8 = load_pooled_features(
                name,
                sim_a10,
                args.prefix_a10,
                sim_a8,
                args.prefix_a8q2,
                burnin,
                FEATURIZERS,
                ca_distances_lower_triangle,
                ca_distances_longrange,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.warning("Skip %s: %s", name, e)
            continue

        if x8.shape[0] != rmsd_coil.shape[0]:
            logger.warning("Skip %s: frame mismatch", name)
            continue

        dfit = int(max(2, min(args.tica_dim, x10.shape[1])))
        model = TICA(lagtime=lag, dim=dfit, epsilon=args.epsilon, scaling="kinetic_map").fit(x10).fetch_model()
        mean_d = np.asarray(model.mean_0, dtype=float)
        V = np.asarray(model.singular_vectors_left, dtype=float)
        Z = project_transferred(x8, mean_d, V)
        t1, t2 = Z[:, 0], Z[:, 1]

        p1c, pv1c = _pearson_safe(t1, rmsd_coil)
        p2c, pv2c = _pearson_safe(t2, rmsd_coil)
        p1h, pv1h = _pearson_safe(t1, rmsd_helix)
        p2h, pv2h = _pearson_safe(t2, rmsd_helix)

        labels_plot.append(name)
        r1c.append(p1c)
        r2c.append(p2c)
        r1h.append(p1h)
        r2h.append(p2h)

        rows.append(
            {
                "feature_set": name,
                "lag_frames": lag,
                "pearson_tic1_rmsd_coil": p1c,
                "p_tic1_coil": pv1c,
                "pearson_tic2_rmsd_coil": p2c,
                "p_tic2_coil": pv2c,
                "pearson_tic1_rmsd_helix_bb": p1h,
                "p_tic1_helix": pv1h,
                "pearson_tic2_rmsd_helix_bb": p2h,
                "p_tic2_helix": pv2h,
                "n_frames": len(rmsd_coil),
            }
        )
        logger.info(
            "%s: r(TIC1,coil)=%.3f r(TIC2,coil)=%.3f | r(TIC1,helix)=%.3f r(TIC2,helix)=%.3f",
            name,
            p1c,
            p2c,
            p1h,
            p2h,
        )

    if not rows:
        logger.error("No feature sets succeeded.")
        return 1

    csv_path = out_dir / "pearson_tic_vs_rmsd_by_feature.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "lag_frames": lag,
        "lag_ns": lag * 0.1,
        "coil_rmsd": "Cα RMSD to first extended frame after burn-in",
        "helix_rmsd": "BB RMSD vs ideal helix (same as Phase-5)",
        "helix_ref_pdb": str(helix_ref),
        "rows": rows,
    }
    with open(out_dir / "pearson_tic_vs_rmsd_by_feature.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    fig_path = out_dir / "fig_pearson_tic_vs_rmsd_by_feature.png"
    plot_grouped_bars(labels_plot, r1c, r2c, r1h, r2h, fig_path)
    logger.info("Wrote %s and %s", csv_path, fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
