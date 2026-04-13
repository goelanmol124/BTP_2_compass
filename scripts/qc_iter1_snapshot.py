#!/usr/bin/env python3
"""
Lightweight scientific QC snapshot for adaptive seeding outputs (any iteration).

Reads completed seed trajectories under ``seeding_dir/iter{NN}/{strategy}/``, computes
quality indicators vs ground-truth 2D FES, and writes JSON. Safe to run while MD is
active (read-only). Use ``--iteration 1`` (default) or ``2``, ``3``, etc.

Ground-truth and seeded trajectories use the same atom selection as
:func:`src.analysis.load_trajectory` (``FEATURIZATION_ATOM_SELECTION``). FES
metrics use **RMSD vs ideal helix × N_HB** on the same bin edges as Phase 5
(:func:`src.analysis.pooled_ground_truth_fes_rmsd_nhb`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import (  # noqa: E402
    FEATURIZATION_ATOM_SELECTION,
    RMSD_HELIX_NM,
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
    NHB_THRESHOLD_FRAC,
    SEED_NS,
)
from scipy.spatial.distance import jensenshannon


def compute_jsd_squared(
    hist_ref: np.ndarray,
    hist_sample: np.ndarray,
    min_count: int = FES_MIN_COUNT,
) -> float:
    """Return Jensen–Shannon divergence **squared** (JSD²), base-2, on masked bins.

    SciPy's ``jensenshannon`` returns JSD; we square for consistency with
    ``src.metrics`` / protocol tables. Not a bug — interpret magnitudes as JSD².
    """
    p = hist_ref.flatten().astype(float)
    q = hist_sample.flatten().astype(float)
    mask = p >= min_count
    if not mask.any():
        return 1.0
    p_m = p[mask]
    q_m = q[mask]
    p_m = p_m / p_m.sum()
    q_sum = q_m.sum()
    if q_sum == 0.0:
        return 1.0
    q_m = q_m / q_sum
    return float(jensenshannon(p_m, q_m, base=2) ** 2)


def compute_coverage(
    hist_ref: np.ndarray,
    hist_sample: np.ndarray,
    min_count: int = FES_MIN_COUNT,
) -> float:
    valid = hist_ref >= min_count
    if not valid.any():
        return 0.0
    return float((hist_sample[valid] > 0).mean())


def compute_entropy(
    hist_sample: np.ndarray,
    hist_ref: np.ndarray,
    min_count: int = FES_MIN_COUNT,
) -> float:
    mask = hist_ref.flatten() >= min_count
    n_omega = int(mask.sum())
    if n_omega == 0:
        return 0.0
    q = hist_sample.flatten().astype(float)[mask]
    q_sum = q.sum()
    if q_sum == 0.0:
        return 0.0
    q = q / q_sum
    nonzero = q > 0.0
    h = -float(np.sum(q[nonzero] * np.log2(q[nonzero])))
    h_max = np.log2(n_omega)
    return float(h / h_max) if h_max > 0 else 1.0


def _lag1_autocorr(x: np.ndarray) -> float | None:
    """Lag-1 normalised autocorrelation; None if undefined."""
    z = np.asarray(x, dtype=np.float64)
    if z.size < 3:
        return None
    z = z - np.mean(z)
    v = float(np.var(z))
    if v < 1e-18:
        return None
    return float(np.mean(z[:-1] * z[1:]) / v)


def _max_true_run_length(mask: np.ndarray) -> int:
    """Longest consecutive run of True in a 1D boolean array."""
    if mask.size == 0:
        return 0
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0
    # Pad to detect boundaries
    padded = np.concatenate(([False], m, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return int(np.max(ends - starts)) if len(starts) else 0


def _helix_coil_transitions(is_helix: np.ndarray) -> int:
    """Count helix↔coil changes along consecutive frames (same stride as data)."""
    h = np.asarray(is_helix, dtype=bool)
    if h.size < 2:
        return 0
    return int(np.sum(h[1:] != h[:-1]))


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _json_safe(obj: object) -> object:
    """Recursively convert numpy scalars for ``json.dump`` (belt-and-suspenders)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (defaults to parent of scripts/).",
    )
    p.add_argument(
        "--seeding-dir",
        type=Path,
        default=Path("outputs/seeding_a8q2"),
        help="Adaptive seeding output directory (contains iter01/, iter02/, ...).",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=1,
        metavar="N",
        help="Which iteration folder when --iterations is not used (1 → iter01, …).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Run QC for multiple folders (e.g. --iterations 1 2 3). Overrides --iteration.",
    )
    p.add_argument(
        "--sim-dir",
        type=Path,
        default=Path("outputs/btp_equil/A8Q2"),
        help="Directory with production DCDs and *_prod_top.pdb for this system.",
    )
    p.add_argument(
        "--helix-ref",
        type=Path,
        default=None,
        help="Ideal helix PDB for RMSD (default: {sim-dir}/{prefix}_helix.pdb).",
    )
    p.add_argument("--prefix", default="A8Q2", help="System filename prefix (e.g. A8Q2).")
    p.add_argument(
        "--tica-dir",
        type=Path,
        default=Path("outputs/tica_artifacts_a10"),
        help=(
            "Directory with **donor** (A10) TICA .npy files for transfer diagnostics. "
            "Override for other layouts; use --skip-a10-tica-corr to omit correlation."
        ),
    )
    p.add_argument(
        "--tica-mean-npy",
        default="a10_tica_mean0.npy",
        help="Mean vector filename inside --tica-dir (A10 donor).",
    )
    p.add_argument(
        "--tica-ev-npy",
        default="a10_tica_singular_vectors_left.npy",
        help="Left singular vectors filename inside --tica-dir (A10 donor).",
    )
    p.add_argument(
        "--skip-a10-tica-corr",
        action="store_true",
        help="Do not load A10 TICA artifacts or compute TIC-1 vs N_HB Pearson.",
    )
    p.add_argument("--stride", type=int, default=5, help="Frame stride for lightweight QC.")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/qc_iter1_snapshot/qc_iter{N}_snapshot.json).",
    )
    args = p.parse_args()

    iterations = list(args.iterations) if args.iterations is not None else [args.iteration]
    for it in iterations:
        if it < 1:
            p.error("Each iteration index must be >= 1")

    repo_root = args.repo_root.resolve()
    seeding_dir = (repo_root / args.seeding_dir).resolve()
    sim_dir = (repo_root / args.sim_dir).resolve()
    tica_dir = (repo_root / args.tica_dir).resolve()

    prod_top = sim_dir / f"{args.prefix}_extended_prod_top.pdb"
    gt_dcd_h = sim_dir / f"{args.prefix}_helix_prod.dcd"
    gt_dcd_e = sim_dir / f"{args.prefix}_extended_prod.dcd"
    gt_top_h = sim_dir / f"{args.prefix}_helix_prod_top.pdb"
    gt_top_e = sim_dir / f"{args.prefix}_extended_prod_top.pdb"

    helix_ref = args.helix_ref if args.helix_ref is not None else sim_dir / f"{args.prefix}_helix.pdb"
    helix_ref = helix_ref.resolve()

    gt_bundle = pooled_ground_truth_fes_rmsd_nhb(
        [
            (str(gt_dcd_h), str(gt_top_h)),
            (str(gt_dcd_e), str(gt_top_e)),
        ],
        str(helix_ref),
    )
    gt_hist = np.asarray(gt_bundle["hist"], dtype=float)
    edges_rmsd = np.asarray(gt_bundle["edges_rmsd"])
    edges_nhb = np.asarray(gt_bundle["edges_nhb"])
    f_helix_gt = float(gt_bundle["f_helix"])

    mean_donor = None
    ev_donor = None
    if not args.skip_a10_tica_corr:
        mean_donor_path = tica_dir / args.tica_mean_npy
        ev_donor_path = tica_dir / args.tica_ev_npy
        mean_donor = np.load(mean_donor_path) if mean_donor_path.is_file() else None
        ev_donor = np.load(ev_donor_path) if ev_donor_path.is_file() else None

    helix_threshold = int(round(N_HB_MAX * NHB_THRESHOLD_FRAC))
    dt_frame_ns = DCD_INTERVAL_PS / 1000.0
    effective_dt_ns = dt_frame_ns * float(args.stride)

    for args_iteration in iterations:
        if args.output_json is None:
            if args_iteration == 1:
                out_json = (
                    repo_root / "outputs" / "qc_iter1_snapshot" / "qc_iter1_snapshot.json"
                )
            else:
                out_json = (
                    repo_root
                    / "outputs"
                    / "qc_iter1_snapshot"
                    / f"qc_iter{args_iteration:02d}_snapshot.json"
                )
        else:
            if len(iterations) > 1:
                p.error("--output-json cannot be used with --iterations (multiple outputs).")
            out_json = (repo_root / args.output_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)

        iter_tag = f"iter{args_iteration:02d}"

        report: dict[str, object] = {
            "qc_version": "iter_snapshot_v3",
            "iteration": int(args_iteration),
            "iter_subdir": iter_tag,
            "stride": int(args.stride),
            "dcd_interval_ps": float(DCD_INTERVAL_PS),
            "effective_frame_spacing_ps": float(DCD_INTERVAL_PS * args.stride),
            "effective_frame_spacing_ns": float(effective_dt_ns),
            "seed_ns_assumed": float(SEED_NS),
            "kinetics_note": (
                "Transition rates and run lengths use the same stride as FES metrics; "
                "they are lower bounds vs full 100 ps DCD resolution. "
                "Per-seed statistics avoid spurious crossings at trajectory concatenation. "
                "These are exploratory proxies for basin hopping, not Markov rates."
            ),
            "helix_threshold_nhb": helix_threshold,
            "rmsd_helix_threshold_nm": float(RMSD_HELIX_NM),
            "featurization_atom_selection": FEATURIZATION_ATOM_SELECTION,
            "fes_definition": "rmsd_x_nhb",
            "helix_ref_pdb": str(helix_ref),
            "ground_truth_f_helix": f_helix_gt,
            "metrics_note": (
                "2D histogram: RMSD (vs ideal helix) × N_HB, Phase 5 binning. "
                "jsd_vs_gt is JSD² (base 2) on bins where ground_truth count >= FES_MIN_COUNT."
            ),
            "a10_donor_tica_note": (
                "pearson_a10_donor_tic1_vs_nhb uses A10-trained mean and TIC-1 vector applied to "
                "A8Q2 features for every strategy (same projection); it is not native-TICA-specific."
            ),
            "strategies": {},
        }

        for strategy in ("random", "native", "transferred"):
            dcds = sorted(
                (seeding_dir / iter_tag / strategy).glob(
                    f"{args.prefix}_*_{iter_tag}_seed*.dcd"
                )
            )
            if not dcds:
                report["strategies"][strategy] = {
                    "status": "no_dcds_found",
                    "n_seeds": 0,
                    "glob": f"{args.prefix}_*_{iter_tag}_seed*.dcd",
                }
                continue

            feats_chunks: list[np.ndarray] = []
            nhb_chunks: list[np.ndarray] = []
            rmsd_chunks: list[np.ndarray] = []
            transitions_per_seed: list[int] = []
            lag1_nhb_per_seed: list[float] = []
            max_helix_run_frames_per_seed: list[int] = []
            total_frames_raw = 0
            total_frames_used = 0
            unreadable: list[str] = []

            for dcd in dcds:
                try:
                    traj = load_trajectory(str(dcd), str(prod_top), protein_only=True)
                    total_frames_raw += int(traj.n_frames)
                    traj = traj[:: args.stride]
                    total_frames_used += int(traj.n_frames)
                    nhb_i = compute_nhb(traj).astype(np.float64)
                    feats_i = compute_featurization(traj)
                    rmsd_i = compute_backbone_rmsd_to_ideal_helix(traj, str(helix_ref))
                    feats_chunks.append(feats_i)
                    nhb_chunks.append(nhb_i)
                    rmsd_chunks.append(rmsd_i)
                    is_h = nhb_i >= helix_threshold
                    transitions_per_seed.append(_helix_coil_transitions(is_h))
                    mxl = _max_true_run_length(is_h)
                    max_helix_run_frames_per_seed.append(mxl)
                    la = _lag1_autocorr(nhb_i)
                    if la is not None:
                        lag1_nhb_per_seed.append(la)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    unreadable.append(f"{dcd.name}: {exc}")

            if not feats_chunks:
                report["strategies"][strategy] = {
                    "status": "failed_to_load_any_dcd",
                    "n_seeds": len(dcds),
                    "unreadable": unreadable,
                }
                continue

            features = np.concatenate(feats_chunks, axis=0)
            nhb = np.concatenate(nhb_chunks, axis=0)
            rmsd = np.concatenate(rmsd_chunks, axis=0)

            hist = histogram_rmsd_nhb_fixed_edges(rmsd, nhb, edges_rmsd, edges_nhb)
            jsd = float(compute_jsd_squared(gt_hist, hist))
            cov = float(compute_coverage(gt_hist, hist))
            ent = float(compute_entropy(hist, gt_hist))
            f_helix = float((nhb >= helix_threshold).mean())
            f_helix_err = (
                abs(f_helix - f_helix_gt) if np.isfinite(f_helix_gt) else None
            )

            pearson_a10_donor_tic1_vs_nhb = None
            tic1_var = None
            if mean_donor is not None and ev_donor is not None:
                try:
                    tic1 = (features - mean_donor[np.newaxis, :]) @ ev_donor[:, 0]
                    tic1_var = float(np.var(np.asarray(tic1)))
                    pearson_a10_donor_tic1_vs_nhb = _safe_pearson(np.asarray(tic1), nhb)
                except Exception:
                    pearson_a10_donor_tic1_vs_nhb = None

            n_tr = int(np.sum(transitions_per_seed))
            n_seeds_ok = len(transitions_per_seed)
            trans_rate_mean = (
                float(np.mean(np.asarray(transitions_per_seed, dtype=float) / SEED_NS))
                if n_seeds_ok
                else 0.0
            )
            pooled_ns = float(n_seeds_ok * SEED_NS)
            trans_rate_pooled = float(n_tr / pooled_ns) if pooled_ns > 0 else 0.0

            max_helix_run_ns = (
                float(max(max_helix_run_frames_per_seed) * effective_dt_ns)
                if max_helix_run_frames_per_seed
                else 0.0
            )

            kinetics: dict[str, object] = {
                "helix_coil_transitions_total": n_tr,
                "helix_coil_transitions_per_seed_mean": float(
                    np.mean(transitions_per_seed)
                )
                if transitions_per_seed
                else 0.0,
                "helix_coil_transition_rate_per_ns_mean_over_seeds": trans_rate_mean,
                "helix_coil_transition_rate_per_ns_pooled": trans_rate_pooled,
                "mean_lag1_autocorr_nhb_over_seeds": float(np.mean(lag1_nhb_per_seed))
                if lag1_nhb_per_seed
                else None,
                "max_consecutive_helix_frames_strided_max_over_seeds": int(
                    max(max_helix_run_frames_per_seed)
                )
                if max_helix_run_frames_per_seed
                else 0,
                "max_consecutive_helix_ns_strided": max_helix_run_ns,
                "n_unique_nhb_values_sample": int(len(np.unique(nhb.astype(int)))),
            }

            exploration: dict[str, object] = {
                "rmsd_nm_mean": float(np.mean(rmsd)),
                "rmsd_nm_std": float(np.std(rmsd)),
                "frac_frames_rmsd_below_helix_threshold": float(
                    np.mean(rmsd < RMSD_HELIX_NM)
                ),
                "tic1_transferred_variance": tic1_var,
            }

            strat_report: dict[str, object] = {
                "status": "ok",
                "n_seeds": len(dcds),
                "frames_raw_total": total_frames_raw,
                "frames_used_total": total_frames_used,
                "jsd_vs_gt": jsd,
                "coverage_vs_gt": cov,
                "entropy_vs_gt": ent,
                "nhb_mean": float(np.mean(nhb)),
                "nhb_std": float(np.std(nhb)),
                "f_helix_estimate": f_helix,
                "f_helix_error_vs_gt": f_helix_err,
                "pearson_a10_donor_tic1_vs_nhb": pearson_a10_donor_tic1_vs_nhb,
                "kinetics": kinetics,
                "exploration": exploration,
                "unreadable": unreadable,
            }
            strat_report["corr_transferred_tic1_nhb"] = pearson_a10_donor_tic1_vs_nhb
            report["strategies"][strategy] = strat_report

        report["deprecated_field_aliases"] = {
            "strategies.*.corr_transferred_tic1_nhb": (
                "Duplicate of pearson_a10_donor_tic1_vs_nhb (A10 donor projection; not transferred-only)."
            ),
        }

        if args_iteration == 1:
            ver_path = seeding_dir / "iter01_verification.json"
            if ver_path.is_file():
                try:
                    report["iter01_verification"] = json.loads(
                        ver_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as exc:
                    report["iter01_verification"] = {
                        "status": "invalid_json",
                        "error": str(exc),
                    }
            else:
                report["iter01_verification"] = {
                    "status": "missing",
                    "path": str(ver_path),
                    "hint": "Written when adaptive seeding finishes iteration 1, or run scripts/validate_iter1_cross_strategy.py.",
                }

            iv = report.get("iter01_verification")
            if isinstance(iv, dict):
                if iv.get("passed") is True:
                    report["iter1_cross_strategy_trust"] = "verified"
                elif iv.get("passed") is False:
                    report["iter1_cross_strategy_trust"] = "do_not_trust"
                elif iv.get("skipped") and iv.get(
                    "reason"
                ) == "user_disabled_skip_iter1_consistency_check":
                    report["iter1_cross_strategy_trust"] = "check_disabled_by_user"
                elif iv.get("skipped"):
                    report["iter1_cross_strategy_trust"] = "not_applicable"
                elif iv.get("status") == "missing":
                    report["iter1_cross_strategy_trust"] = "unknown"
                else:
                    report["iter1_cross_strategy_trust"] = "unknown"
        else:
            report["iter01_verification"] = {
                "status": "not_applicable",
                "reason": "Initial-topology gate is defined for iteration 1 only; see --iteration 1 report or iter01_verification.json.",
            }

        with out_json.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(report), fh, indent=2)
        print(str(out_json))


if __name__ == "__main__":
    main()