#!/usr/bin/env python3
"""
2D FES (RMSD × N_HB) on the pooled ground-truth bin grid for:
  - 500 ns pooled ground truth (helix + extended replicas)
  - each adaptive-seeding strategy (random, native, transferred), all seeds pooled

Designed for RunPod / machines that have ``outputs/seeding_a8q2/**/*.dcd`` and
equilibrium tops/DCDs under ``outputs/btp_equil/A8Q2/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import (
    FES_CLIP_KCAL,
    KB_KCAL,
    TEMPERATURE_K,
    _plot_2d_fes,
    _save_fig,
    compute_backbone_rmsd_to_ideal_helix,
    compute_nhb,
    histogram_rmsd_nhb_fixed_edges,
    load_trajectory,
    pooled_ground_truth_fes_rmsd_nhb,
)
from src.config import N_ITERATIONS

STRATEGIES = ("random", "native", "transferred")


def _hist_to_fes_dict(
    hist: np.ndarray,
    edges_rmsd: np.ndarray,
    edges_nhb: np.ndarray,
    clip_kcal: float = FES_CLIP_KCAL,
    temperature_K: float = TEMPERATURE_K,
) -> dict:
    RT = KB_KCAL * temperature_K
    h = np.asarray(hist, dtype=float)
    m = float(h.max())
    if m <= 0.0:
        fes = np.full_like(h, np.nan)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            fes = np.where(h > 0.0, -RT * np.log(h / m), np.nan)
    fes = np.where(np.isnan(fes), np.nan, np.clip(fes, 0.0, clip_kcal))
    return {
        "fes": fes,
        "centers_rmsd": 0.5 * (edges_rmsd[:-1] + edges_rmsd[1:]),
        "centers_nhb": 0.5 * (edges_nhb[:-1] + edges_nhb[1:]),
    }


def _collect_strategy_frames(
    seeding_dir: Path,
    prefix: str,
    strategy: str,
    max_iteration: int,
    prod_top_pdb: Path,
    helix_ref_pdb: Path,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, int, int, list[str]]:
    dcds: list[Path] = []
    for it in range(1, max_iteration + 1):
        dcds.extend(
            sorted(
                (seeding_dir / f"iter{it:02d}" / strategy).glob(
                    f"{prefix}_{strategy}_iter{it:02d}_seed*.dcd"
                )
            )
        )
    rmsd_all: list[np.ndarray] = []
    nhb_all: list[np.ndarray] = []
    unreadable: list[str] = []
    for dcd in dcds:
        try:
            traj = load_trajectory(str(dcd), str(prod_top_pdb))
            traj = traj[::stride]
            rmsd_all.append(compute_backbone_rmsd_to_ideal_helix(traj, str(helix_ref_pdb)))
            nhb_all.append(compute_nhb(traj))
        except Exception:
            unreadable.append(str(dcd))
    if not rmsd_all:
        return np.array([], dtype=float), np.array([], dtype=float), 0, len(dcds), unreadable
    return (
        np.concatenate(rmsd_all),
        np.concatenate(nhb_all),
        int(sum(len(x) for x in rmsd_all)),
        len(dcds),
        unreadable,
    )


def _plot_combined(
    panels: list[tuple[dict, str]],
    output_path: Path,
) -> None:
    from src.analysis import FES_CLIP_KCAL as clip

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    levels = np.linspace(0, clip, 20)
    for ax, (fes_dict, title) in zip(axes_flat, panels):
        fes = fes_dict["fes"]
        cx = fes_dict["centers_rmsd"] * 10.0
        cy = fes_dict["centers_nhb"]
        xi, yi = np.meshgrid(cx, cy, indexing="xy")
        z = np.asarray(fes).T
        cf = ax.contourf(xi, yi, z, levels=levels, cmap="RdYlBu_r", extend="max")
        ax.contour(xi, yi, z, levels=levels, colors="k", linewidths=0.25, alpha=0.35)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("N_HB")
    for ax in axes[1, :]:
        ax.set_xlabel("BB RMSD vs ideal helix (Å)")
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(cf, cax=cax, label="ΔG (kcal/mol)")
    fig.suptitle("2D FES — same RMSD×N_HB grid as pooled 500 ns ground truth", fontsize=11, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    _save_fig(fig, str(output_path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--prefix", default="A8Q2")
    p.add_argument("--sim-dir", type=Path, default=Path("outputs/btp_equil/A8Q2"))
    p.add_argument("--seeding-dir", type=Path, default=Path("outputs/seeding_a8q2"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/fes_three_strategies"))
    p.add_argument("--max-iteration", type=int, default=N_ITERATIONS)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    sim_dir = (repo_root / args.sim_dir).resolve()
    seeding_dir = (repo_root / args.seeding_dir).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    prod_top = sim_dir / f"{args.prefix}_extended_prod_top.pdb"
    gt_dcd_h = sim_dir / f"{args.prefix}_helix_prod.dcd"
    gt_dcd_e = sim_dir / f"{args.prefix}_extended_prod.dcd"
    gt_top_h = sim_dir / f"{args.prefix}_helix_prod_top.pdb"
    gt_top_e = sim_dir / f"{args.prefix}_extended_prod_top.pdb"
    helix_ref = sim_dir / f"{args.prefix}_helix.pdb"

    gt = pooled_ground_truth_fes_rmsd_nhb(
        [(str(gt_dcd_h), str(gt_top_h)), (str(gt_dcd_e), str(gt_top_e))],
        str(helix_ref),
    )
    gt_hist = np.asarray(gt["hist"], dtype=float)
    edges_rmsd = np.asarray(gt["edges_rmsd"])
    edges_nhb = np.asarray(gt["edges_nhb"])

    meta: dict = {
        "prefix": args.prefix,
        "stride": args.stride,
        "max_iteration": args.max_iteration,
        "seeding_dir": str(seeding_dir),
        "sim_dir": str(sim_dir),
    }

    panels: list[tuple[dict, str]] = []

    gt_fes = _hist_to_fes_dict(gt_hist, edges_rmsd, edges_nhb)
    gt_path = out_dir / f"{args.prefix}_gt_fes2d_rmsd_nhb.png"
    _plot_2d_fes(gt_fes, "Ground truth (500 ns pooled)", str(gt_path))
    panels.append((gt_fes, "Ground truth (500 ns pooled)"))
    meta["ground_truth"] = {"n_bins_rmsd": int(gt_hist.shape[0]), "f_helix_pooled_gt": float(gt["f_helix"])}

    for strategy in STRATEGIES:
        rmsd, nhb, n_frames, n_traj, bad = _collect_strategy_frames(
            seeding_dir,
            args.prefix,
            strategy,
            args.max_iteration,
            prod_top,
            helix_ref,
            args.stride,
        )
        label = f"Seeding — {strategy} (pooled, iter 1–{args.max_iteration})"
        if n_frames == 0:
            meta[strategy] = {"status": "no_frames", "n_traj_files": n_traj, "unreadable": bad}
            continue
        hist = histogram_rmsd_nhb_fixed_edges(rmsd, nhb, edges_rmsd, edges_nhb)
        fd = _hist_to_fes_dict(hist, edges_rmsd, edges_nhb)
        out_png = out_dir / f"{args.prefix}_{strategy}_fes2d_rmsd_nhb.png"
        _plot_2d_fes(fd, label, str(out_png))
        panels.append((fd, f"{strategy}\n({n_frames} frames)"))
        np.save(out_dir / f"{args.prefix}_{strategy}_hist2d_gt_grid.npy", hist.astype(float))
        meta[strategy] = {
            "status": "ok",
            "n_frames": n_frames,
            "n_traj_files": n_traj,
            "unreadable_count": len(bad),
            "unreadable": bad[:20],
        }

    if len(panels) == 4:
        _plot_combined(panels, out_dir / f"{args.prefix}_fes2d_rmsd_nhb_combined_2x2.png")

    (out_dir / "fes_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote FES figures under {out_dir}")


if __name__ == "__main__":
    main()
