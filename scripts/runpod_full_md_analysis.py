#!/usr/bin/env python3
"""
Full-trajectory adaptive-seeding analysis (RunPod-focused).

This script computes Phase-7/8/9 style metrics directly from trajectory files
without relying on previously cached adaptive_seeding_results.json values.
It is intended for use on a machine that has all DCDs (e.g., RunPod).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adaptive_seeding import project_cv
from src.analysis import (
    compute_backbone_rmsd_to_ideal_helix,
    compute_featurization,
    compute_nhb,
    histogram_rmsd_nhb_fixed_edges,
    load_trajectory,
    pooled_ground_truth_fes_rmsd_nhb,
)
from src.config import (
    DCD_INTERVAL_PS,
    FES_MIN_COUNT,
    N_HB_MAX,
    N_ITERATIONS,
    N_SEEDS,
    NHB_THRESHOLD_FRAC,
    RMSD_COIL_NM,
    RMSD_HELIX_NM,
    SEED_NS,
)


STRATEGIES = ("random", "native", "transferred")


@dataclass
class TrajectoryBundle:
    rmsd: np.ndarray
    nhb: np.ndarray
    features: np.ndarray
    n_frames: int
    n_traj: int
    unreadable: list[str]


def _jsd_squared(hist_ref: np.ndarray, hist_sample: np.ndarray, min_count: int = FES_MIN_COUNT) -> float:
    p = hist_ref.flatten().astype(float)
    q = hist_sample.flatten().astype(float)
    mask = p >= min_count
    if not np.any(mask):
        return 1.0
    p = p[mask]
    q = q[mask]
    p = p / p.sum()
    q_sum = q.sum()
    if q_sum <= 0.0:
        return 1.0
    q = q / q_sum
    return float(jensenshannon(p, q, base=2) ** 2)


def _coverage(hist_ref: np.ndarray, hist_sample: np.ndarray, min_count: int = FES_MIN_COUNT) -> float:
    mask = hist_ref >= min_count
    if not np.any(mask):
        return 0.0
    return float(np.mean(hist_sample[mask] > 0.0))


def _entropy(hist_sample: np.ndarray, hist_ref: np.ndarray, min_count: int = FES_MIN_COUNT) -> float:
    mask = hist_ref.flatten() >= min_count
    n = int(mask.sum())
    if n == 0:
        return 0.0
    q = hist_sample.flatten().astype(float)[mask]
    s = q.sum()
    if s <= 0.0:
        return 0.0
    q = q / s
    nz = q > 0.0
    h = -float(np.sum(q[nz] * np.log2(q[nz])))
    return float(h / np.log2(n)) if n > 1 else 0.0


def _hysteretic_rmsd_state(rmsd_nm: np.ndarray, lo: float = RMSD_HELIX_NM, hi: float = RMSD_COIL_NM) -> np.ndarray:
    state = np.zeros(len(rmsd_nm), dtype=bool)  # True=helix, False=coil
    if len(rmsd_nm) == 0:
        return state
    state[0] = rmsd_nm[0] <= (lo + hi) / 2.0
    for i in range(1, len(rmsd_nm)):
        r = rmsd_nm[i]
        if r <= lo:
            state[i] = True
        elif r >= hi:
            state[i] = False
        else:
            state[i] = state[i - 1]
    return state


def _transitions_per_ns(state: np.ndarray, frame_spacing_ns: float) -> tuple[int, float]:
    if len(state) < 2:
        return 0, 0.0
    transitions = int(np.sum(state[1:] != state[:-1]))
    total_ns = len(state) * frame_spacing_ns
    rate = float(transitions / total_ns) if total_ns > 0 else 0.0
    return transitions, rate


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _collect_strategy_bundle(
    seeding_dir: Path,
    prefix: str,
    strategy: str,
    max_iteration: int,
    prod_top_pdb: Path,
    helix_ref_pdb: Path,
    stride: int,
) -> TrajectoryBundle:
    dcds: list[Path] = []
    for it in range(1, max_iteration + 1):
        dcds.extend(sorted((seeding_dir / f"iter{it:02d}" / strategy).glob(f"{prefix}_{strategy}_iter{it:02d}_seed*.dcd")))

    rmsd_all: list[np.ndarray] = []
    nhb_all: list[np.ndarray] = []
    feat_all: list[np.ndarray] = []
    unreadable: list[str] = []
    for dcd in dcds:
        try:
            traj = load_trajectory(str(dcd), str(prod_top_pdb))
            traj = traj[::stride]
            rmsd_all.append(compute_backbone_rmsd_to_ideal_helix(traj, str(helix_ref_pdb)))
            nhb_all.append(compute_nhb(traj))
            feat_all.append(compute_featurization(traj))
        except Exception:
            unreadable.append(str(dcd))
    if not rmsd_all:
        return TrajectoryBundle(np.array([], dtype=float), np.array([], dtype=float), np.empty((0, 40)), 0, len(dcds), unreadable)
    rmsd = np.concatenate(rmsd_all, axis=0)
    nhb = np.concatenate(nhb_all, axis=0)
    feats = np.concatenate(feat_all, axis=0)
    return TrajectoryBundle(rmsd, nhb, feats, int(len(rmsd)), len(dcds), unreadable)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--prefix", default="A8Q2")
    p.add_argument("--sim-dir", type=Path, default=Path("outputs/btp_equil/A8Q2"))
    p.add_argument("--seeding-dir", type=Path, default=Path("outputs/seeding_a8q2"))
    p.add_argument("--tica-dir", type=Path, default=Path("outputs/tica_artifacts_a10"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/full_md_analysis"))
    p.add_argument("--stride", type=int, default=1, help="Trajectory frame stride (1 = full frames).")
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    sim_dir = (repo_root / args.sim_dir).resolve()
    seeding_dir = (repo_root / args.seeding_dir).resolve()
    tica_dir = (repo_root / args.tica_dir).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prod_top = sim_dir / f"{args.prefix}_extended_prod_top.pdb"
    gt_dcd_h = sim_dir / f"{args.prefix}_helix_prod.dcd"
    gt_dcd_e = sim_dir / f"{args.prefix}_extended_prod.dcd"
    gt_top_h = sim_dir / f"{args.prefix}_helix_prod_top.pdb"
    gt_top_e = sim_dir / f"{args.prefix}_extended_prod_top.pdb"
    helix_ref = sim_dir / f"{args.prefix}_helix.pdb"

    gt = pooled_ground_truth_fes_rmsd_nhb([(str(gt_dcd_h), str(gt_top_h)), (str(gt_dcd_e), str(gt_top_e))], str(helix_ref))
    gt_hist = np.asarray(gt["hist"], dtype=float)
    edges_rmsd = np.asarray(gt["edges_rmsd"])
    edges_nhb = np.asarray(gt["edges_nhb"])
    gt_f_helix = float(gt["f_helix"])

    # Replica split noise floor (protocol §8.4, §12.2)
    tr_h = load_trajectory(str(gt_dcd_h), str(gt_top_h))[:: args.stride]
    tr_e = load_trajectory(str(gt_dcd_e), str(gt_top_e))[:: args.stride]
    h_h = histogram_rmsd_nhb_fixed_edges(compute_backbone_rmsd_to_ideal_helix(tr_h, str(helix_ref)), compute_nhb(tr_h), edges_rmsd, edges_nhb)
    h_e = histogram_rmsd_nhb_fixed_edges(compute_backbone_rmsd_to_ideal_helix(tr_e, str(helix_ref)), compute_nhb(tr_e), edges_rmsd, edges_nhb)
    noise_floor = _jsd_squared(np.asarray(h_h, dtype=float), np.asarray(h_e, dtype=float))

    mean_donor = np.load(str(tica_dir / "a10_tica_mean0.npy"))
    ev_donor = np.load(str(tica_dir / "a10_tica_singular_vectors_left.npy"))
    helix_threshold_nhb = int(np.ceil(N_HB_MAX * NHB_THRESHOLD_FRAC))
    frame_spacing_ns = float(DCD_INTERVAL_PS) * float(args.stride) / 1000.0

    rows: list[dict[str, Any]] = []
    for it in range(1, N_ITERATIONS + 1):
        cumulative_ns = float(it * N_SEEDS * SEED_NS)
        for strategy in STRATEGIES:
            b = _collect_strategy_bundle(
                seeding_dir=seeding_dir,
                prefix=args.prefix,
                strategy=strategy,
                max_iteration=it,
                prod_top_pdb=prod_top,
                helix_ref_pdb=helix_ref,
                stride=args.stride,
            )
            if b.n_frames == 0:
                rows.append({"iteration": it, "strategy": strategy, "status": "no_frames", "cumulative_ns": cumulative_ns})
                continue
            hist = histogram_rmsd_nhb_fixed_edges(b.rmsd, b.nhb, edges_rmsd, edges_nhb)
            nhb_state = b.nhb >= helix_threshold_nhb
            rmsd_state = _hysteretic_rmsd_state(b.rmsd)
            t_nhb, rate_nhb = _transitions_per_ns(nhb_state, frame_spacing_ns)
            t_rmsd, rate_rmsd = _transitions_per_ns(rmsd_state, frame_spacing_ns)
            tic1 = project_cv(b.features, strategy="transferred", mean_donor=mean_donor, eigenvectors_donor=ev_donor)
            rows.append(
                {
                    "iteration": it,
                    "strategy": strategy,
                    "status": "ok",
                    "cumulative_ns": cumulative_ns,
                    "n_traj": b.n_traj,
                    "n_frames": b.n_frames,
                    "unreadable_count": len(b.unreadable),
                    "jsd_vs_gt": _jsd_squared(gt_hist, hist),
                    "coverage_vs_gt": _coverage(gt_hist, hist),
                    "entropy_vs_gt": _entropy(hist, gt_hist),
                    "f_helix_estimate": float(np.mean(nhb_state)),
                    "f_helix_error_vs_gt": float(abs(np.mean(nhb_state) - gt_f_helix)),
                    "transitions_nhb_total": t_nhb,
                    "transitions_nhb_rate_per_ns": rate_nhb,
                    "transitions_rmsd_total": t_rmsd,
                    "transitions_rmsd_rate_per_ns": rate_rmsd,
                    "criterion_agreement_frac": float(np.mean(nhb_state == rmsd_state)),
                    "pearson_tic1_nhb": _safe_pearson(tic1.astype(float), b.nhb.astype(float)),
                    "pearson_tic1_rmsd": _safe_pearson(tic1.astype(float), b.rmsd.astype(float)),
                    "rmsd_mean_nm": float(np.mean(b.rmsd)),
                    "rmsd_std_nm": float(np.std(b.rmsd)),
                    "nhb_mean": float(np.mean(b.nhb)),
                    "nhb_std": float(np.std(b.nhb)),
                }
            )

    df = pd.DataFrame(rows).sort_values(["iteration", "strategy"]).reset_index(drop=True)
    ok = df[df["status"] == "ok"].copy()
    ok["composite_lower_better"] = ok["jsd_vs_gt"] + ok["f_helix_error_vs_gt"] + (1.0 - ok["coverage_vs_gt"])

    # Acceleration-oriented summaries
    accel: dict[str, Any] = {"noise_floor_jsd2": noise_floor, "thresholds": {"jsd2_0p05": 0.05}}
    for strategy in STRATEGIES:
        s = ok[ok["strategy"] == strategy].sort_values("iteration")
        if len(s) == 0:
            continue
        jsd = s["jsd_vs_gt"].to_numpy(dtype=float)
        ns = s["cumulative_ns"].to_numpy(dtype=float)
        auc = float(np.trapezoid(jsd, ns))
        t_noise = None
        t_005 = None
        for _, r in s.iterrows():
            if t_noise is None and float(r["jsd_vs_gt"]) <= noise_floor:
                t_noise = float(r["cumulative_ns"])
            if t_005 is None and float(r["jsd_vs_gt"]) <= 0.05:
                t_005 = float(r["cumulative_ns"])
        accel[strategy] = {"auc_jsd_vs_ns": auc, "ns_to_noise_floor": t_noise, "ns_to_jsd2_0p05": t_005}

    # protocol target: transferred should achieve >=50% of native JSD drop in same ns
    target_check: list[dict[str, Any]] = []
    for it in range(2, N_ITERATIONS + 1):
        s_i = ok[ok["iteration"] == it].set_index("strategy")
        if "native" in s_i.index and "transferred" in s_i.index:
            native_drop = float(s_i.loc["native", "jsd_vs_gt"])
            trans_drop = float(s_i.loc["transferred", "jsd_vs_gt"])
            # Lower JSD is better; compare absolute value from 1.0 reference for robustness.
            native_gain = 1.0 - native_drop
            trans_gain = 1.0 - trans_drop
            ratio = float(trans_gain / native_gain) if native_gain > 1e-12 else None
            target_check.append({"iteration": it, "native_gain": native_gain, "transferred_gain": trans_gain, "ratio_transferred_over_native": ratio})
    accel["protocol_target_check"] = target_check

    # Save tables
    ok.to_csv(out_dir / "full_metrics_long.csv", index=False)
    df.to_csv(out_dir / "full_metrics_all_rows.csv", index=False)
    (out_dir / "acceleration_summary.json").write_text(json.dumps(accel, indent=2), encoding="utf-8")
    meta = {
        "repo_root": str(repo_root),
        "sim_dir": str(sim_dir),
        "seeding_dir": str(seeding_dir),
        "stride": args.stride,
        "frame_spacing_ns": frame_spacing_ns,
        "helix_threshold_nhb": helix_threshold_nhb,
        "rmsd_hysteresis_nm": {"helix": RMSD_HELIX_NM, "coil": RMSD_COIL_NM},
        "ground_truth_f_helix": gt_f_helix,
        "noise_floor_jsd2": noise_floor,
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Plots
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"random": "#1f77b4", "native": "#2ca02c", "transferred": "#d62728"}
    curves = [
        ("jsd_vs_gt", "JSD^2 vs ground truth (lower better)"),
        ("coverage_vs_gt", "Coverage vs ground truth (higher better)"),
        ("f_helix_error_vs_gt", "|f_helix - f_gt| (lower better)"),
        ("transitions_nhb_rate_per_ns", "Helix-coil transition rate (/ns, N_HB criterion)"),
        ("criterion_agreement_frac", "RMSD vs N_HB helix-state agreement"),
        ("composite_lower_better", "Composite score (lower better)"),
    ]
    for col, title in curves:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for s in STRATEGIES:
            sdat = ok[ok["strategy"] == s]
            ax.plot(sdat["iteration"], sdat[col], marker="o", label=s, color=colors[s])
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_xticks(range(1, N_ITERATIONS + 1))
        if col == "jsd_vs_gt":
            ax.axhline(noise_floor, color="black", linestyle="--", linewidth=1, label=f"noise floor={noise_floor:.4f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"curve_{col}.png", dpi=220)
        plt.close(fig)

    print(f"Wrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()

