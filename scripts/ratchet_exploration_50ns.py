"""
Ratchet exploration (50 ns total) vs standard MD (50 ns from extended GT slice).

- Each seed: 0.1 ns; 5 seeds per round; 100 rounds -> 50 ns ratchet budget.
- Round 1: five independent 0.1 ns runs from the generator structure.
- Later rounds: pool *all* trajectories so far; seed **only** from transferred
  TIC-1 (A10 donor) using equal-width bin least-counts (``select_seeds_least_counts``).
  RMSD and N_HB are **not** used for seeding — only for checkpoint plots / metrics.
- Standard arm: first 50 ns of existing extended production (100 ps/frame).

Writes manifest + plots at 1, 5, 10, 20, 30, 40, 50 ns under ``--output-dir``
(default: ``outputs/ratchet_demo_50ns``).

Run from repo root:
  set PYTHONPATH=.
  python scripts/ratchet_exploration_50ns.py --output-dir outputs/my_ratchet_run

Regenerate figures from an existing manifest (no OpenMM):
  python scripts/ratchet_exploration_50ns.py --plots-only --output-dir outputs/my_ratchet_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm as mm
import openmm.app as app
from matplotlib.colors import LogNorm
from openmm import XmlSerializer, unit

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.adaptive_seeding import project_cv, select_seeds_least_counts  # noqa: E402
from src.analysis import (  # noqa: E402
    compute_backbone_rmsd_to_ideal_helix,
    compute_featurization,
    compute_nhb,
    histogram_rmsd_nhb_fixed_edges,
    load_trajectory,
    pooled_ground_truth_fes_rmsd_nhb,
)
from src.metrics import compute_coverage  # noqa: E402
from src.utils import get_best_platform  # noqa: E402

# --- Paths ---
A8Q2_DIR = _ROOT / "outputs" / "btp_equil" / "A8Q2"
GENERATOR_PDB = A8Q2_DIR / "A8Q2_extended_npt_final.pdb"
SYSTEM_XML = A8Q2_DIR / "A8Q2_extended_npt_final.xml"
PROD_TOP_PDB = A8Q2_DIR / "A8Q2_extended_prod_top.pdb"
GT_DCD_HELIX = A8Q2_DIR / "A8Q2_helix_prod.dcd"
GT_DCD_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod.dcd"
GT_TOP_HELIX = A8Q2_DIR / "A8Q2_helix_prod_top.pdb"
GT_TOP_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod_top.pdb"
HELIX_REF_PDB = A8Q2_DIR / "A8Q2_helix.pdb"
TICA_DIR = _ROOT / "outputs" / "btp_equil" / "A10" / "tica_validation_run_v2"
TICA_EV = TICA_DIR / "a10_tica_singular_vectors_left.npy"
TICA_MEAN = TICA_DIR / "a10_tica_mean0.npy"

DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "ratchet_demo_50ns"

# --- MD ---
TEMPERATURE = 330.0 * unit.kelvin
FRICTION = 1.0 / unit.picosecond
TIMESTEP = 2.0 * unit.femtoseconds
TIMESTEP_FS = 2.0
STEPS_PER_PS = int(1000 / TIMESTEP_FS)

SEED_NS = 0.1
SEEDS_PER_ROUND = 5
TOTAL_NS = 50.0
ROUNDS = int(round(TOTAL_NS / (SEED_NS * SEEDS_PER_ROUND)))  # 100
CHECKPOINTS_NS = (1, 5, 10, 20, 30, 40, 50)

# 0.1 ns -> 50_000 steps; log every 5 ps -> 20 frames per seed
DCD_INTERVAL_PS = 5.0
SEED_STEPS = int(SEED_NS * 1000 * STEPS_PER_PS)
DCD_STEPS = int(DCD_INTERVAL_PS * STEPS_PER_PS)

# Ground truth saves every 100 ps -> 10 frames / ns
GT_DT_NS = 0.1

# DCDReporter creates the file immediately; treat tiny files as incomplete.
_MIN_DCD_BYTES = 4096


def _seed_dcd_complete(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= _MIN_DCD_BYTES
    except OSError:
        return False


def run_seed_simulation(
    seed_frame_coords,
    box_vectors,
    out_dcd: str,
    steps: int,
    platform,
    rng_seed: int,
) -> None:
    with open(SYSTEM_XML) as f:
        system = XmlSerializer.deserialize(f.read())
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
    integrator.setRandomNumberSeed(rng_seed)

    pdb_top = app.PDBFile(str(PROD_TOP_PDB))
    n_sys = system.getNumParticles()
    if pdb_top.topology.getNumAtoms() == n_sys:
        omm_top = pdb_top.topology
    else:
        traj_top = md.load_pdb(str(PROD_TOP_PDB))
        omm_top = traj_top.topology.to_openmm()

    sim = app.Simulation(omm_top, system, integrator, platform)
    if box_vectors is not None and len(box_vectors) == 3:
        sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setPositions(seed_frame_coords)
    sim.context.applyConstraints(1e-7)
    sim.context.setVelocitiesToTemperature(TEMPERATURE, rng_seed)
    sim.reporters.append(app.DCDReporter(out_dcd, DCD_STEPS))
    sim.step(steps)


def get_frame_positions(dcd_path: str, frame_idx: int):
    traj = md.load_frame(dcd_path, frame_idx, top=str(PROD_TOP_PDB))
    return traj.openmm_positions(0)


def _traj_feats_frame_map(dcd: Path) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Load one ratchet DCD (protein-only): featurization + frame map for TICA seeding."""
    traj_prot = load_trajectory(str(dcd), str(PROD_TOP_PDB), protein_only=True)
    feats = compute_featurization(traj_prot)
    resolved = str(dcd.resolve())
    fm = [(resolved, fi) for fi in range(len(traj_prot))]
    return feats, fm


def rebuild_cumulative_pool_from_paths(paths: list[Path]) -> tuple[list[np.ndarray], list[tuple[str, int]]]:
    """One-time rebuild when resuming a long experiment (all prior DCDs)."""
    fl: list[np.ndarray] = []
    fm: list[tuple[str, int]] = []
    for dcd in paths:
        if not dcd.is_file():
            continue
        f, fmi = _traj_feats_frame_map(dcd)
        fl.append(f)
        fm.extend(fmi)
    return fl, fm


def load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, data: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def collect_ratchet_xy(
    ordered_runs: list[dict],
    top_pdb: Path,
    helix_ref: Path,
    max_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all ratchet DCDs whose cumulative end time <= max_ns."""
    rmsd_all: list[np.ndarray] = []
    nhb_all: list[np.ndarray] = []
    t = 0.0
    for run in ordered_runs:
        t_next = t + float(run["ns"])
        if t_next > max_ns + 1e-9:
            break
        dcd = run["path"]
        if not Path(dcd).is_file():
            continue
        traj = load_trajectory(dcd, str(top_pdb), protein_only=True)
        rmsd_all.append(compute_backbone_rmsd_to_ideal_helix(traj, str(helix_ref)))
        nhb_all.append(compute_nhb(traj))
        t = t_next
    if not rmsd_all:
        return np.array([]), np.array([])
    return np.concatenate(rmsd_all), np.concatenate(nhb_all)


def standard_xy_first_ns(ns: float) -> tuple[np.ndarray, np.ndarray]:
    """First ns of extended GT at 100 ps spacing."""
    n_frames = max(1, int(round(ns / GT_DT_NS)))
    traj = load_trajectory(str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED), protein_only=True)
    traj = traj[:n_frames]
    r = compute_backbone_rmsd_to_ideal_helix(traj, str(HELIX_REF_PDB))
    h = compute_nhb(traj)
    return r, h


def plot_pair(
    gt_fes: dict,
    std_r: np.ndarray,
    std_h: np.ndarray,
    rat_r: np.ndarray,
    rat_h: np.ndarray,
    out_png: Path,
    title_left: str,
    title_right: str,
) -> None:
    hist = np.asarray(gt_fes["hist"], dtype=float)
    hist_plot = hist.copy()
    hist_plot[hist_plot == 0] = np.nan
    X, Y = np.meshgrid(gt_fes["edges_rmsd"], gt_fes["edges_nhb"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    vmax = float(np.nanmax(hist_plot))
    for ax in axes:
        ax.pcolormesh(
            X, Y, hist_plot.T, norm=LogNorm(vmin=1, vmax=max(vmax, 2.0)), cmap="Greys", alpha=0.55
        )
        ax.set_xlabel("Backbone RMSD (nm)", fontsize=11)
    axes[0].set_ylabel("N_HB", fontsize=11)
    axes[0].plot(std_r, std_h, "o", color="tab:blue", ms=3, alpha=0.45, label="Standard MD")
    axes[0].set_title(f"Standard MD — {title_left}", fontsize=12)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].plot(rat_r, rat_h, "o", color="tab:red", ms=2.5, alpha=0.35, label="Ratchet")
    axes[1].set_title(f"Ratchet (transferred TIC1 least-counts) — {title_right}", fontsize=12)
    axes[1].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def load_ground_truth_fes() -> dict:
    print("Loading ground truth FES …")
    return pooled_ground_truth_fes_rmsd_nhb(
        [
            (str(GT_DCD_HELIX), str(GT_TOP_HELIX)),
            (str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED)),
        ],
        str(HELIX_REF_PDB),
    )


def write_checkpoint_outputs(
    out_dir: Path,
    ordered: list[dict],
    gt_fes: dict,
    hist_ref: np.ndarray,
    edges_rmsd: np.ndarray,
    edges_nhb: np.ndarray,
) -> None:
    print("=== Checkpoint plots & metrics ===")
    metrics_out: dict[str, dict] = {}
    total_ratchet_ns = sum(float(r.get("ns", 0.0)) for r in ordered)

    for T in CHECKPOINTS_NS:
        std_r, std_h = standard_xy_first_ns(T)
        rat_budget_ns = min(float(T), total_ratchet_ns)
        rat_r, rat_h = collect_ratchet_xy(ordered, PROD_TOP_PDB, HELIX_REF_PDB, rat_budget_ns)

        std_hist = histogram_rmsd_nhb_fixed_edges(std_r, std_h, edges_rmsd, edges_nhb)
        rat_hist = histogram_rmsd_nhb_fixed_edges(rat_r, rat_h, edges_rmsd, edges_nhb)
        cov_std = float(compute_coverage(hist_ref, std_hist, min_count=3))
        cov_rat = float(compute_coverage(hist_ref, rat_hist, min_count=3))

        metrics_out[f"{int(T)}ns"] = {
            "standard_n_frames": int(len(std_r)),
            "ratchet_n_frames": int(len(rat_r)),
            "coverage_standard": cov_std,
            "coverage_ratchet": cov_rat,
            "standard_rmsd_min": float(np.min(std_r)) if len(std_r) else None,
            "ratchet_rmsd_min": float(np.min(rat_r)) if len(rat_r) else None,
            "standard_nhb_max": int(np.max(std_h)) if len(std_h) else None,
            "ratchet_nhb_max": int(np.max(rat_h)) if len(rat_h) else None,
        }

        title_left = f"first {int(T)} ns"
        title_right = (
            f"first {int(T)} ns (only {rat_budget_ns:.2f} ns generated)"
            if rat_budget_ns + 1e-6 < float(T)
            else f"first {int(T)} ns"
        )

        out_png = out_dir / f"ratchet_comparison_{int(T)}ns.png"
        plot_pair(
            gt_fes,
            std_r,
            std_h,
            rat_r,
            rat_h,
            out_png,
            title_left,
            title_right,
        )
        print(
            f"  T={T:g} ns: cov_std={cov_std:.3f} cov_rat={cov_rat:.3f} "
            f"frames std={len(std_r)} rat={len(rat_r)} "
            f"-> {out_png}"
        )

    metrics_path = out_dir / "ratchet_checkpoint_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Wrote metrics: {metrics_path}")


def main(out_dir: Path, plots_only: bool = False) -> None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "ratchet_manifest.json"

    gt_fes = load_ground_truth_fes()
    edges_rmsd = np.asarray(gt_fes["edges_rmsd"])
    edges_nhb = np.asarray(gt_fes["edges_nhb"])
    hist_ref = np.asarray(gt_fes["hist"], dtype=float)

    if plots_only:
        manifest = load_manifest(manifest_path)
        if not manifest or not manifest.get("ordered_runs"):
            raise SystemExit(f"No runs in {manifest_path}; nothing to plot.")
        write_checkpoint_outputs(out_dir, manifest["ordered_runs"], gt_fes, hist_ref, edges_rmsd, edges_nhb)
        return

    platform = get_best_platform()

    print("Loading TICA …")
    mean_donor = np.load(str(TICA_MEAN))
    eigenvectors_donor = np.load(str(TICA_EV))

    from src.simulate import _npt_final_topology_positions_box

    with open(SYSTEM_XML) as f:
        system = XmlSerializer.deserialize(f.read())
    _, pos0, box0 = _npt_final_topology_positions_box(GENERATOR_PDB, system.getNumParticles())

    manifest = load_manifest(manifest_path) or {"version": 1, "ordered_runs": [], "rounds_done": 0}

    pool_dcds: list[Path] = []

    # Rebuild pool from manifest if resuming
    for run in manifest.get("ordered_runs", []):
        pool_dcds.append(Path(run["path"]))

    start_round = int(manifest.get("rounds_done", 0)) + 1
    if start_round > 1:
        print(f"Resuming from round {start_round} ({len(pool_dcds)} seed DCDs on disk).")

    # Incremental pool: only append each round's new DCDs (avoid O(rounds²) reloads).
    cum_feats_list: list[np.ndarray] = []
    cum_frame_map: list[tuple[str, int]] = []
    if start_round > 1:
        print("Rebuilding pooled features from manifest (one-time on resume) …")
        cum_feats_list, cum_frame_map = rebuild_cumulative_pool_from_paths(pool_dcds)

    for r in range(start_round, ROUNDS + 1):
        print(f"=== Round {r}/{ROUNDS} (cumulative target {(r * SEEDS_PER_ROUND * SEED_NS):.1f} ns) ===")
        round_paths: list[Path] = []

        if r == 1:
            for s in range(SEEDS_PER_ROUND):
                out_dcd = out_dir / f"ratchet_r{r:03d}_s{s}.dcd"
                round_paths.append(out_dcd)
                if not _seed_dcd_complete(out_dcd):
                    if out_dcd.is_file():
                        out_dcd.unlink(missing_ok=True)
                    run_seed_simulation(
                        pos0, box0, str(out_dcd), SEED_STEPS, platform, rng_seed=10_000 * r + s
                    )
        else:
            if not cum_frame_map:
                raise RuntimeError("Round >=2 requires a non-empty pooled trajectory (manifest bug?).")
            pooled_f = np.concatenate(cum_feats_list, axis=0)
            tic1 = project_cv(
                pooled_f,
                strategy="transferred",
                mean_donor=mean_donor,
                eigenvectors_donor=eigenvectors_donor,
            )
            assert tic1 is not None

            # Seeding uses transferred TIC-1 only (same as ratchet_exploration_demo.py).
            seed_pool_idx = select_seeds_least_counts(
                tic1,
                n_seeds=SEEDS_PER_ROUND,
                n_bins=10,
                rng_seed=10_000 + r,
            )

            for s, pooled_idx in enumerate(seed_pool_idx):
                out_dcd = out_dir / f"ratchet_r{r:03d}_s{s}.dcd"
                round_paths.append(out_dcd)
                if not _seed_dcd_complete(out_dcd):
                    if out_dcd.is_file():
                        out_dcd.unlink(missing_ok=True)
                    src_dcd, src_frame = cum_frame_map[pooled_idx]
                    pos = get_frame_positions(src_dcd, src_frame)
                    run_seed_simulation(
                        pos, box0, str(out_dcd), SEED_STEPS, platform, rng_seed=10_000 * r + s
                    )

        for p in round_paths:
            pool_dcds.append(p)
            manifest["ordered_runs"].append(
                {"path": str(p.resolve()), "round": r, "ns": SEED_NS}
            )
            f_new, fm_new = _traj_feats_frame_map(p)
            cum_feats_list.append(f_new)
            cum_frame_map.extend(fm_new)
        manifest["rounds_done"] = r
        save_manifest(manifest_path, manifest)

    write_checkpoint_outputs(out_dir, manifest["ordered_runs"], gt_fes, hist_ref, edges_rmsd, edges_nhb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="50 ns ratchet demo vs standard MD (checkpoint plots).")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for DCDs, ratchet_manifest.json, and plots (default: {DEFAULT_OUTPUT_DIR}).",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate checkpoint PNGs + metrics from existing ratchet_manifest.json (no OpenMM).",
    )
    args = ap.parse_args()
    out_dir = (args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR).resolve()
    main(out_dir=out_dir, plots_only=args.plots_only)
