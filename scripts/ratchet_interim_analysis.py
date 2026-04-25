"""
Read-only interim analysis of the 50 ns ratchet run.
Reads the current ratchet_manifest.json and generates:
  1. Updated checkpoint plots via --plots-only
  2. Per-round coverage growth curve
  3. Per-round RMSD distribution evolution
  4. Per-round NHB distribution evolution
  5. Scatter of seed origins (which bins were selected)

Does NOT write any DCD or modify the manifest.

Usage:
  python scripts/ratchet_interim_analysis.py
  python scripts/ratchet_interim_analysis.py --output-dir outputs/ratchet_demo_50ns_run_new
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis import (
    compute_backbone_rmsd_to_ideal_helix,
    compute_nhb,
    histogram_rmsd_nhb_fixed_edges,
    load_trajectory,
    pooled_ground_truth_fes_rmsd_nhb,
)
from src.metrics import compute_coverage

A8Q2_DIR = _ROOT / "outputs" / "btp_equil" / "A8Q2"
PROD_TOP_PDB = A8Q2_DIR / "A8Q2_extended_prod_top.pdb"
GT_DCD_HELIX = A8Q2_DIR / "A8Q2_helix_prod.dcd"
GT_DCD_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod.dcd"
GT_TOP_HELIX = A8Q2_DIR / "A8Q2_helix_prod_top.pdb"
GT_TOP_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod_top.pdb"
HELIX_REF_PDB = A8Q2_DIR / "A8Q2_helix.pdb"

DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "ratchet_demo_50ns"
GT_DT_NS = 0.1


def main(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "ratchet_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    ordered = manifest["ordered_runs"]
    rounds_done = manifest["rounds_done"]
    total_ns = len(ordered) * 0.1
    print(f"Manifest: {rounds_done} rounds, {len(ordered)} DCDs, {total_ns:.1f} ns ratchet budget")

    print("Loading ground truth FES ...")
    gt_fes = pooled_ground_truth_fes_rmsd_nhb(
        [
            (str(GT_DCD_HELIX), str(GT_TOP_HELIX)),
            (str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED)),
        ],
        str(HELIX_REF_PDB),
    )
    edges_rmsd = np.asarray(gt_fes["edges_rmsd"])
    edges_nhb = np.asarray(gt_fes["edges_nhb"])
    hist_ref = np.asarray(gt_fes["hist"], dtype=float)

    # Load standard MD (full 50 ns available)
    print("Loading standard MD trajectory ...")
    traj_std = load_trajectory(str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED), protein_only=True)
    std_r_all = compute_backbone_rmsd_to_ideal_helix(traj_std, str(HELIX_REF_PDB))
    std_h_all = compute_nhb(traj_std)

    # Load all ratchet DCDs, keeping per-round breakdown
    print("Loading ratchet DCDs per round ...")
    per_round: dict[int, list[dict]] = {}
    for run in ordered:
        rnd = run["round"]
        per_round.setdefault(rnd, []).append(run)

    cum_rmsd: list[np.ndarray] = []
    cum_nhb: list[np.ndarray] = []
    round_nums: list[int] = []
    cum_ns_at_round: list[float] = []
    cov_ratchet_at_round: list[float] = []
    cov_standard_at_same_ns: list[float] = []
    rmsd_min_ratchet: list[float] = []
    nhb_max_ratchet: list[int] = []
    rmsd_min_standard: list[float] = []
    nhb_max_standard: list[int] = []

    cum_ns = 0.0
    for rnd in sorted(per_round.keys()):
        runs = per_round[rnd]
        for run in runs:
            dcd = run["path"]
            if not Path(dcd).is_file():
                continue
            try:
                traj = load_trajectory(dcd, str(PROD_TOP_PDB), protein_only=True)
            except Exception:
                continue
            r = compute_backbone_rmsd_to_ideal_helix(traj, str(HELIX_REF_PDB))
            h = compute_nhb(traj)
            cum_rmsd.append(r)
            cum_nhb.append(h)
            cum_ns += float(run["ns"])

        all_r = np.concatenate(cum_rmsd) if cum_rmsd else np.array([])
        all_h = np.concatenate(cum_nhb) if cum_nhb else np.array([])

        rat_hist = histogram_rmsd_nhb_fixed_edges(all_r, all_h, edges_rmsd, edges_nhb)
        cov_rat = float(compute_coverage(hist_ref, rat_hist, min_count=3))

        n_std = max(1, int(round(cum_ns / GT_DT_NS)))
        std_r_slice = std_r_all[:n_std]
        std_h_slice = std_h_all[:n_std]
        std_hist = histogram_rmsd_nhb_fixed_edges(std_r_slice, std_h_slice, edges_rmsd, edges_nhb)
        cov_std = float(compute_coverage(hist_ref, std_hist, min_count=3))

        round_nums.append(rnd)
        cum_ns_at_round.append(cum_ns)
        cov_ratchet_at_round.append(cov_rat)
        cov_standard_at_same_ns.append(cov_std)
        rmsd_min_ratchet.append(float(np.min(all_r)) if len(all_r) else float("nan"))
        nhb_max_ratchet.append(int(np.max(all_h)) if len(all_h) else 0)
        rmsd_min_standard.append(float(np.min(std_r_slice)))
        nhb_max_standard.append(int(np.max(std_h_slice)))

    # ---- Plot 1: Coverage growth curve ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum_ns_at_round, cov_ratchet_at_round, "o-", color="tab:red", ms=4, label="Ratchet (TIC1 least-counts)")
    ax.plot(cum_ns_at_round, cov_standard_at_same_ns, "s-", color="tab:blue", ms=4, label="Standard MD (same budget)")
    ax.set_xlabel("Cumulative simulation time (ns)", fontsize=12)
    ax.set_ylabel("Coverage (fraction of GT bins with count >= 3)", fontsize=12)
    ax.set_title("Coverage growth: Ratchet vs Standard MD", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "interim_coverage_curve.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output_dir / 'interim_coverage_curve.png'}")

    # ---- Plot 2: Min RMSD evolution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum_ns_at_round, rmsd_min_ratchet, "o-", color="tab:red", ms=4, label="Ratchet min RMSD")
    ax.plot(cum_ns_at_round, rmsd_min_standard, "s-", color="tab:blue", ms=4, label="Standard min RMSD")
    ax.set_xlabel("Cumulative simulation time (ns)", fontsize=12)
    ax.set_ylabel("Min backbone RMSD to helix ref (nm)", fontsize=12)
    ax.set_title("Exploration toward helical state: min RMSD over time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "interim_min_rmsd_curve.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output_dir / 'interim_min_rmsd_curve.png'}")

    # ---- Plot 3: Max NHB evolution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum_ns_at_round, nhb_max_ratchet, "o-", color="tab:red", ms=4, label="Ratchet max NHB")
    ax.plot(cum_ns_at_round, nhb_max_standard, "s-", color="tab:blue", ms=4, label="Standard max NHB")
    ax.set_xlabel("Cumulative simulation time (ns)", fontsize=12)
    ax.set_ylabel("Max N_HB observed", fontsize=12)
    ax.set_title("Hydrogen bond formation: max NHB over time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "interim_max_nhb_curve.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output_dir / 'interim_max_nhb_curve.png'}")

    # ---- Plot 4: RMSD distribution comparison at current budget ----
    all_r = np.concatenate(cum_rmsd) if cum_rmsd else np.array([])
    all_h = np.concatenate(cum_nhb) if cum_nhb else np.array([])
    n_std_final = max(1, int(round(cum_ns / GT_DT_NS)))
    std_r_final = std_r_all[:n_std_final]
    std_h_final = std_h_all[:n_std_final]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins_rmsd = np.linspace(0, 0.6, 40)
    axes[0].hist(std_r_final, bins=bins_rmsd, alpha=0.6, color="tab:blue", density=True, label=f"Standard ({n_std_final} fr)")
    axes[0].hist(all_r, bins=bins_rmsd, alpha=0.5, color="tab:red", density=True, label=f"Ratchet ({len(all_r)} fr)")
    axes[0].set_xlabel("Backbone RMSD to helix (nm)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title(f"RMSD distribution at {cum_ns:.1f} ns", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].axvline(0.15, ls="--", color="green", alpha=0.5, label="helix threshold")

    bins_nhb = np.arange(-0.5, 7.5, 1)
    axes[1].hist(std_h_final, bins=bins_nhb, alpha=0.6, color="tab:blue", density=True, label=f"Standard")
    axes[1].hist(all_h, bins=bins_nhb, alpha=0.5, color="tab:red", density=True, label=f"Ratchet")
    axes[1].set_xlabel("N_HB", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title(f"NHB distribution at {cum_ns:.1f} ns", fontsize=12)
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "interim_distributions.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {output_dir / 'interim_distributions.png'}")

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print(f"INTERIM ANALYSIS SUMMARY (ratchet budget = {cum_ns:.1f} ns, rounds = {rounds_done})")
    print("=" * 80)
    print(f"{'Metric':<35} {'Standard MD':<20} {'Ratchet':<20}")
    print("-" * 75)
    print(f"{'Frames':<35} {n_std_final:<20} {len(all_r):<20}")
    print(f"{'Coverage (GT bins, min_count=3)':<35} {cov_standard_at_same_ns[-1]:<20.4f} {cov_ratchet_at_round[-1]:<20.4f}")
    print(f"{'Min RMSD to helix (nm)':<35} {rmsd_min_standard[-1]:<20.4f} {rmsd_min_ratchet[-1]:<20.4f}")
    print(f"{'Max NHB':<35} {nhb_max_standard[-1]:<20} {nhb_max_ratchet[-1]:<20}")
    print(f"{'Mean RMSD (nm)':<35} {np.mean(std_r_final):<20.4f} {np.mean(all_r):<20.4f}")
    print(f"{'Std RMSD (nm)':<35} {np.std(std_r_final):<20.4f} {np.std(all_r):<20.4f}")
    print(f"{'Mean NHB':<35} {np.mean(std_h_final):<20.4f} {np.mean(all_h):<20.4f}")
    frac_nhb2_std = float((std_h_final >= 2).mean())
    frac_nhb2_rat = float((all_h >= 2).mean())
    print(f"{'Frac NHB >= 2':<35} {frac_nhb2_std:<20.4f} {frac_nhb2_rat:<20.4f}")
    frac_rmsd_low_std = float((std_r_final <= 0.25).mean())
    frac_rmsd_low_rat = float((all_r <= 0.25).mean())
    print(f"{'Frac RMSD <= 0.25 nm':<35} {frac_rmsd_low_std:<20.4f} {frac_rmsd_low_rat:<20.4f}")

    # Coverage speedup
    target_cov = cov_ratchet_at_round[-1]
    first_std_above = None
    for i, c in enumerate(cov_standard_at_same_ns):
        if c >= target_cov:
            first_std_above = cum_ns_at_round[i]
            break
    if first_std_above is not None:
        print(f"\n  Ratchet reached {target_cov:.3f} coverage at {cum_ns:.1f} ns.")
        print(f"  Standard MD reached the same coverage at {first_std_above:.1f} ns.")
        print(f"  Speedup factor: {first_std_above / cum_ns:.1f}x")
    else:
        print(f"\n  Ratchet coverage = {target_cov:.3f} at {cum_ns:.1f} ns.")
        print(f"  Standard MD has NOT yet reached this coverage (max = {max(cov_standard_at_same_ns):.3f} at {cum_ns:.1f} ns).")
        if max(cov_standard_at_same_ns) > 0:
            # Extrapolate: check full standard trajectory
            for ns_test in [20, 30, 40, 50, 75, 100]:
                n = min(len(std_r_all), max(1, int(round(ns_test / GT_DT_NS))))
                h = histogram_rmsd_nhb_fixed_edges(std_r_all[:n], std_h_all[:n], edges_rmsd, edges_nhb)
                c = float(compute_coverage(hist_ref, h, min_count=3))
                if c >= target_cov:
                    print(f"  Standard MD reaches {target_cov:.3f} coverage around {ns_test} ns -> speedup ~{ns_test/cum_ns:.1f}x")
                    break
            else:
                n_max = len(std_r_all)
                h_max = histogram_rmsd_nhb_fixed_edges(std_r_all, std_h_all, edges_rmsd, edges_nhb)
                c_max = float(compute_coverage(hist_ref, h_max, min_count=3))
                print(f"  Even with full standard trajectory ({n_max} frames, ~{n_max*GT_DT_NS:.0f} ns), coverage = {c_max:.3f}")
                if c_max < target_cov:
                    print(f"  Standard MD NEVER reaches ratchet coverage within the available trajectory!")

    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory with ratchet_manifest.json (default: {DEFAULT_OUTPUT_DIR.relative_to(_ROOT)}).",
    )
    args = ap.parse_args()
    main(output_dir=args.output_dir)
