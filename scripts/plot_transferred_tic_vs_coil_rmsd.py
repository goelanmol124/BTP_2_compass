#!/usr/bin/env python3
"""Pearson r and scatter: transferred TIC-1 / TIC-2 vs Cα RMSD to a coil reference (A8Q2).

The coil reference is, by default, the first post–burn-in frame of the
**extended** production replica (disordered starting geometry).  Per-frame Cα
RMSD (after superposition onto that reference) measures deviation from that coil
snapshot — complementary to BB RMSD vs ideal helix used elsewhere.

Example::

    python scripts/plot_transferred_tic_vs_coil_rmsd.py \\
        --sim-dir-a10 outputs/btp_equil/A10 --prefix-a10 A10 \\
        --sim-dir-a8q2 outputs/btp_equil/A8Q2 --prefix-a8q2 A8Q2 \\
        --out-dir outputs/tica_rmsd_correlation
"""

from __future__ import annotations

import argparse
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
    """Return concatenated trajectory (extended | helix), same order as feature pooling."""

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
    """Load extended + helix replicas, strip burn-in, concatenate frames (same as sweep)."""

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


def coil_reference_snapshot_from_extended(
    sim_dir: Path,
    prefix: str,
    burnin_frames: int,
):
    """First frame after burn-in on the extended replica → fixed coil reference."""

    from src.analysis import load_trajectory

    stem = f"{prefix}_extended"
    dcd = sim_dir / f"{stem}_prod.dcd"
    top = sim_dir / f"{stem}_prod_top.pdb"
    traj = load_trajectory(str(dcd), str(top), protein_only=True)
    if traj.n_frames <= burnin_frames:
        raise RuntimeError(f"{dcd}: need > {burnin_frames} frames")
    return traj[burnin_frames : burnin_frames + 1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=_repo_root())
    p.add_argument("--sim-dir-a10", type=Path, default=Path("outputs/btp_equil/A10"))
    p.add_argument("--prefix-a10", default="A10")
    p.add_argument("--sim-dir-a8q2", type=Path, default=Path("outputs/btp_equil/A8Q2"))
    p.add_argument("--prefix-a8q2", default="A8Q2")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/tica_rmsd_correlation"))
    p.add_argument("--burnin-frames", type=int, default=-1)
    p.add_argument("--feature-set", default="torsions_sc", help="Name from src.tica_featurizers.FEATURIZERS")
    p.add_argument("--lag-frames", type=int, default=-1, help="TICA lag (default: config TICA_MIN_LAG_FRAMES)")
    p.add_argument("--tica-dim", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=1e-6)
    args = p.parse_args(argv)

    root = args.repo_root.resolve()
    _ensure_sys_path(root)
    import os

    os.chdir(root)

    from src import config
    from src.analysis import BURNIN_FRAMES, compute_backbone_rmsd_to_snapshot
    from src.tica_featurizers import FEATURIZERS
    from src.tica_transfer import project_transferred
    from deeptime.decomposition import TICA

    burnin = args.burnin_frames if args.burnin_frames >= 0 else int(BURNIN_FRAMES)
    lag = args.lag_frames if args.lag_frames >= 0 else int(config.TICA_MIN_LAG_FRAMES)

    sim_a10 = (root / args.sim_dir_a10).resolve()
    sim_a8 = (root / args.sim_dir_a8q2).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_set not in FEATURIZERS:
        logger.error("Unknown feature set %r", args.feature_set)
        return 1

    name = args.feature_set
    try:
        x10, _ = pool_trajectory_features(sim_a10, args.prefix_a10, burnin, name, FEATURIZERS)
        x8, _ = pool_trajectory_features(sim_a8, args.prefix_a8q2, burnin, name, FEATURIZERS)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    ref_snap = coil_reference_snapshot_from_extended(sim_a8, args.prefix_a8q2, burnin)
    traj_pool = pool_a8q2_trajs_extended_then_helix(sim_a8, args.prefix_a8q2, burnin)
    rmsd_coil = np.asarray(
        compute_backbone_rmsd_to_snapshot(traj_pool, ref_snap), dtype=np.float64
    )

    if rmsd_coil.shape[0] != x8.shape[0]:
        raise RuntimeError(
            f"Frame count mismatch: RMSD {rmsd_coil.shape[0]} vs features {x8.shape[0]}"
        )

    dfit = int(max(2, min(args.tica_dim, x10.shape[1])))
    model = TICA(lagtime=lag, dim=dfit, epsilon=args.epsilon, scaling="kinetic_map").fit(
        x10
    ).fetch_model()
    mean_d = np.asarray(model.mean_0, dtype=float)
    V = np.asarray(model.singular_vectors_left, dtype=float)
    Z = project_transferred(x8, mean_d, V)
    t1, t2 = Z[:, 0], Z[:, 1]

    r1, p1 = pearsonr(t1, rmsd_coil)
    r2, p2 = pearsonr(t2, rmsd_coil)
    if not np.isfinite(r1):
        r1, p1 = 0.0, 1.0
    if not np.isfinite(r2):
        r2, p2 = 0.0, 1.0

    summary = {
        "feature_set": name,
        "lag_frames": lag,
        "lag_ns": lag * 0.1,
        "tica_dim_fit": dfit,
        "n_frames_a8q2": int(len(rmsd_coil)),
        "coil_ref": "first post-burn-in frame of extended replica (Cα RMSD, nm)",
        "pearson_tic1_vs_rmsd_coil": float(r1),
        "p_value_tic1": float(p1),
        "pearson_tic2_vs_rmsd_coil": float(r2),
        "p_value_tic2": float(p2),
        "rmsd_coil_mean_nm": float(np.mean(rmsd_coil)),
        "rmsd_coil_std_nm": float(np.std(rmsd_coil)),
    }
    with open(out_dir / "pearson_tic_vs_coil_rmsd.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, y, tit, r, pv in [
        (axes[0], t1, "Transferred TIC-1", r1, p1),
        (axes[1], t2, "Transferred TIC-2", r2, p2),
    ]:
        ax.scatter(rmsd_coil, y, s=2, alpha=0.25, rasterized=True)
        ax.set_xlabel("Cα RMSD vs coil ref (nm)")
        ax.set_ylabel(tit)
        ax.set_title(f"{tit}\nPearson r = {r:.4f}  (p = {pv:.2e})")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"A10→A8Q2 transferred TICA vs coil RMSD ({name}, lag={lag} frames)",
        fontsize=11,
    )
    fig.savefig(out_dir / "fig_tic12_vs_coil_rmsd.png", dpi=200)
    plt.close(fig)

    logger.info("Wrote %s", out_dir / "fig_tic12_vs_coil_rmsd.png")
    logger.info("Pearson TIC-1 vs coil RMSD: r=%.4f", r1)
    logger.info("Pearson TIC-2 vs coil RMSD: r=%.4f", r2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
