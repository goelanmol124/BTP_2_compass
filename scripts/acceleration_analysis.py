"""
Comprehensive acceleration analysis: plain equilibrium MD vs adaptive seeding.

Builds equilibrium JSD/coverage/f_helix convergence curves by subsampling,
then directly compares with seeding convergence data.
"""
import sys, json, csv
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs"
RESULTS_DIR = ROOT / "outputs" / "acceleration_deep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FES_MIN_COUNT = 3
N_HB_MAX = 6
NHB_HELIX_FRAC = 0.5
HELIX_THRESHOLD = int(N_HB_MAX * NHB_HELIX_FRAC)  # 3
FES_NBINS = 25
FRAME_SPACING_NS = 0.1  # 100 ps per frame


def nhb_bin_edges():
    return np.arange(-0.5, N_HB_MAX + 1.5, 1.0)


def rmsd_bin_edges(rmsd, nbins=FES_NBINS):
    r = rmsd[np.isfinite(rmsd)]
    lo, hi = float(r.min()), float(r.max())
    if hi - lo < 1e-12:
        lo -= 0.5; hi += 0.5
    return np.linspace(lo, hi, nbins + 1)


def hist2d(rmsd, nhb, edges_r, edges_nh):
    mask = np.isfinite(rmsd) & np.isfinite(nhb)
    h, _, _ = np.histogram2d(rmsd[mask], nhb[mask], bins=[edges_r, edges_nh])
    return h


def jsd_squared(h_ref, h_sample, min_count=FES_MIN_COUNT):
    mask = h_ref >= min_count
    if mask.sum() == 0:
        return np.nan
    p = h_ref[mask].astype(float)
    q = h_sample[mask].astype(float)
    sp, sq = p.sum(), q.sum()
    if sp == 0:
        return np.nan
    p = p / sp
    if sq == 0:
        q = np.zeros_like(p)
    else:
        q = q / sq
    return float(jensenshannon(p, q, base=2) ** 2)


def coverage(h_ref, h_sample, min_count=FES_MIN_COUNT):
    mask = h_ref >= min_count
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(h_sample[mask] > 0))


def entropy_norm(h_ref, h_sample, min_count=FES_MIN_COUNT):
    mask = h_ref >= min_count
    n = int(mask.sum())
    if n <= 1:
        return np.nan
    q = h_sample[mask].astype(float)
    s = q.sum()
    if s == 0:
        return 0.0
    q = q / s
    q = q[q > 0]
    ent = -np.sum(q * np.log2(q))
    return float(ent / np.log2(n))


def f_helix(nhb):
    return float(np.mean(nhb >= HELIX_THRESHOLD))


def count_transitions(nhb, threshold=HELIX_THRESHOLD):
    state = (nhb >= threshold).astype(int)
    return int(np.sum(np.abs(np.diff(state))))


def lag1_autocorr(x):
    x = x - x.mean()
    c0 = np.dot(x, x)
    if c0 == 0:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / c0)


def effective_sample_size(x):
    """ESS from integrated autocorrelation time (Geyer's initial positive sequence)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return float(n)
    acf = np.correlate(x, x, mode='full')[n-1:]
    acf = acf / (var * n)
    tau = 0.0
    for k in range(1, n // 2):
        if acf[k] < 0:
            break
        tau += acf[k]
    tau = 1.0 + 2.0 * tau
    return float(n / tau)


# ── Load equilibrium data ────────────────────────────────────────────────────
print("Loading A8Q2 equilibrium trajectories...")
try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    print("mdtraj not available; will use pre-computed NHB only (no RMSD subsampling)")

if HAS_MDTRAJ:
    helix_ref = str(OUT / "btp_equil/A8Q2/A8Q2_helix.pdb")
    pairs = [
        (str(OUT / "btp_equil/A8Q2/A8Q2_extended_prod.dcd"),
         str(OUT / "btp_equil/A8Q2/A8Q2_extended_prod_top.pdb")),
        (str(OUT / "btp_equil/A8Q2/A8Q2_helix_prod.dcd"),
         str(OUT / "btp_equil/A8Q2/A8Q2_helix_prod_top.pdb")),
    ]

    from src.analysis import (load_trajectory, compute_backbone_rmsd_to_ideal_helix,
                              compute_nhb)

    all_rmsd_parts = []
    all_nhb_parts = []
    part_lengths = []

    for dcd, top in pairs:
        print(f"  Loading {Path(dcd).name} ...")
        traj = load_trajectory(dcd, top, protein_only=True)
        r = compute_backbone_rmsd_to_ideal_helix(traj, helix_ref)
        nh = compute_nhb(traj).astype(float)
        all_rmsd_parts.append(r)
        all_nhb_parts.append(nh)
        part_lengths.append(len(r))
        print(f"    {len(r)} frames, RMSD range [{r.min():.4f}, {r.max():.4f}] nm, "
              f"NHB range [{nh.min():.0f}, {nh.max():.0f}]")

    rmsd_all = np.concatenate(all_rmsd_parts)
    nhb_all = np.concatenate(all_nhb_parts)
    n_total = len(rmsd_all)
    total_ns = n_total * FRAME_SPACING_NS
    print(f"  Pooled: {n_total} frames = {total_ns:.1f} ns")

    # Build ground truth histogram
    edges_r = rmsd_bin_edges(rmsd_all)
    edges_nh = nhb_bin_edges()
    gt_hist = hist2d(rmsd_all, nhb_all, edges_r, edges_nh)
    gt_f_helix = f_helix(nhb_all)
    print(f"  GT f_helix = {gt_f_helix:.6f}")
    print(f"  GT hist shape = {gt_hist.shape}, total counts = {gt_hist.sum():.0f}")
    print(f"  Bins with >=3 counts: {(gt_hist >= FES_MIN_COUNT).sum()}")

    # ── Equilibrium convergence curves ────────────────────────────────────────
    print("\nBuilding equilibrium convergence curves...")

    # Strategy 1: subsample from beginning of pooled trajectory (sequential)
    # Strategy 2: subsample from each start-state independently
    # Strategy 3: interleave frames from both start-states

    time_points_ns = np.arange(25, total_ns + 1, 25)
    
    eq_curves = {"sequential": [], "extended_only": [], "helix_only": [], "interleaved": []}

    n_ext = part_lengths[0]
    n_hel = part_lengths[1]
    rmsd_ext, nhb_ext = all_rmsd_parts[0], all_nhb_parts[0]
    rmsd_hel, nhb_hel = all_rmsd_parts[1], all_nhb_parts[1]

    for ns in time_points_ns:
        n_frames = int(ns / FRAME_SPACING_NS)

        # Sequential (first n_frames of pooled)
        if n_frames <= n_total:
            h = hist2d(rmsd_all[:n_frames], nhb_all[:n_frames], edges_r, edges_nh)
            eq_curves["sequential"].append({
                "ns": float(ns), "n_frames": n_frames,
                "jsd": jsd_squared(gt_hist, h),
                "coverage": coverage(gt_hist, h),
                "entropy": entropy_norm(gt_hist, h),
                "f_helix": f_helix(nhb_all[:n_frames]),
                "f_helix_error": abs(f_helix(nhb_all[:n_frames]) - gt_f_helix),
                "transitions": count_transitions(nhb_all[:n_frames]),
                "trans_rate_per_ns": count_transitions(nhb_all[:n_frames]) / ns,
            })

        # Extended only
        nf_ext = min(n_frames, n_ext)
        if nf_ext > 0:
            ns_ext = nf_ext * FRAME_SPACING_NS
            h = hist2d(rmsd_ext[:nf_ext], nhb_ext[:nf_ext], edges_r, edges_nh)
            eq_curves["extended_only"].append({
                "ns": float(ns_ext), "n_frames": nf_ext,
                "jsd": jsd_squared(gt_hist, h),
                "coverage": coverage(gt_hist, h),
                "entropy": entropy_norm(gt_hist, h),
                "f_helix": f_helix(nhb_ext[:nf_ext]),
                "f_helix_error": abs(f_helix(nhb_ext[:nf_ext]) - gt_f_helix),
                "transitions": count_transitions(nhb_ext[:nf_ext]),
                "trans_rate_per_ns": count_transitions(nhb_ext[:nf_ext]) / ns_ext,
            })

        # Helix only
        nf_hel = min(n_frames, n_hel)
        if nf_hel > 0:
            ns_hel = nf_hel * FRAME_SPACING_NS
            h = hist2d(rmsd_hel[:nf_hel], nhb_hel[:nf_hel], edges_r, edges_nh)
            eq_curves["helix_only"].append({
                "ns": float(ns_hel), "n_frames": nf_hel,
                "jsd": jsd_squared(gt_hist, h),
                "coverage": coverage(gt_hist, h),
                "entropy": entropy_norm(gt_hist, h),
                "f_helix": f_helix(nhb_hel[:nf_hel]),
                "f_helix_error": abs(f_helix(nhb_hel[:nf_hel]) - gt_f_helix),
                "transitions": count_transitions(nhb_hel[:nf_hel]),
                "trans_rate_per_ns": count_transitions(nhb_hel[:nf_hel]) / ns_hel,
            })

        # Interleaved (take equal frames from each, alternating)
        nf_each = min(n_frames // 2, n_ext, n_hel)
        if nf_each > 0:
            ns_interleaved = 2 * nf_each * FRAME_SPACING_NS
            r_int = np.concatenate([rmsd_ext[:nf_each], rmsd_hel[:nf_each]])
            nh_int = np.concatenate([nhb_ext[:nf_each], nhb_hel[:nf_each]])
            h = hist2d(r_int, nh_int, edges_r, edges_nh)
            eq_curves["interleaved"].append({
                "ns": float(ns_interleaved), "n_frames": 2 * nf_each,
                "jsd": jsd_squared(gt_hist, h),
                "coverage": coverage(gt_hist, h),
                "entropy": entropy_norm(gt_hist, h),
                "f_helix": f_helix(nh_int),
                "f_helix_error": abs(f_helix(nh_int) - gt_f_helix),
                "transitions": count_transitions(nhb_ext[:nf_each]) + count_transitions(nhb_hel[:nf_each]),
                "trans_rate_per_ns": (count_transitions(nhb_ext[:nf_each]) + count_transitions(nhb_hel[:nf_each])) / ns_interleaved,
            })

    # ── Seeding data ──────────────────────────────────────────────────────────
    print("\nLoading seeding convergence data...")
    fm = list(csv.DictReader(
        (OUT / "full_md_analysis/full_metrics_long.csv").open("r", encoding="utf-8")
    ))

    seeding_curves = {}
    for strat in ["random", "native", "transferred"]:
        seeding_curves[strat] = []
        for r in fm:
            if r["strategy"] == strat:
                seeding_curves[strat].append({
                    "ns": float(r["cumulative_ns"]),
                    "n_frames": int(r["n_frames"]),
                    "jsd": float(r["jsd_vs_gt"]),
                    "coverage": float(r["coverage_vs_gt"]),
                    "entropy": float(r["entropy_vs_gt"]),
                    "f_helix": float(r["f_helix_estimate"]),
                    "f_helix_error": float(r["f_helix_error_vs_gt"]),
                    "transitions": None,
                    "trans_rate_per_ns": float(r["transitions_nhb_rate_per_ns"]),
                })

    # ── Effective sample sizes ────────────────────────────────────────────────
    print("\nComputing effective sample sizes...")
    ess_ext = effective_sample_size(nhb_ext)
    ess_hel = effective_sample_size(nhb_hel)
    ess_pooled = effective_sample_size(nhb_all)
    acf1_ext = lag1_autocorr(nhb_ext)
    acf1_hel = lag1_autocorr(nhb_hel)
    acf1_pooled = lag1_autocorr(nhb_all)

    ess_results = {
        "extended": {"n_frames": n_ext, "ns": n_ext * FRAME_SPACING_NS,
                     "ess": ess_ext, "ess_per_ns": ess_ext / (n_ext * FRAME_SPACING_NS),
                     "lag1_acf": acf1_ext},
        "helix": {"n_frames": n_hel, "ns": n_hel * FRAME_SPACING_NS,
                  "ess": ess_hel, "ess_per_ns": ess_hel / (n_hel * FRAME_SPACING_NS),
                  "lag1_acf": acf1_hel},
        "pooled": {"n_frames": n_total, "ns": total_ns,
                   "ess": ess_pooled, "ess_per_ns": ess_pooled / total_ns,
                   "lag1_acf": acf1_pooled},
    }

    # ── Find crossover times ─────────────────────────────────────────────────
    print("\nComputing acceleration factors...")
    noise_floor = float(
        json.loads((OUT / "full_md_analysis/analysis_metadata.json").read_text())["noise_floor_jsd2"]
    )

    thresholds_jsd = {
        "noise_floor": noise_floor,
        "2x_noise": 2 * noise_floor,
        "3x_noise": 3 * noise_floor,
        "jsd_0.02": 0.02,
        "jsd_0.01": 0.01,
    }

    thresholds_fhelix = {
        "err_0.01": 0.01,
        "err_0.005": 0.005,
        "err_0.002": 0.002,
    }

    thresholds_coverage = {
        "cov_0.90": 0.90,
        "cov_0.95": 0.95,
        "cov_1.00": 1.00,
    }

    def find_first_crossing(curve, metric, threshold, direction="below"):
        """Find first ns where metric crosses threshold."""
        for pt in curve:
            v = pt[metric]
            if v is None or np.isnan(v):
                continue
            if direction == "below" and v <= threshold:
                return pt["ns"]
            if direction == "above" and v >= threshold:
                return pt["ns"]
        return None

    acceleration_table = []
    
    for label, thresh_dict, metric, direction in [
        ("JSD", thresholds_jsd, "jsd", "below"),
        ("f_helix_error", thresholds_fhelix, "f_helix_error", "below"),
        ("coverage", thresholds_coverage, "coverage", "above"),
    ]:
        for tname, tval in thresh_dict.items():
            row = {"metric": label, "threshold": tname, "threshold_value": tval}

            for eq_label in ["sequential", "extended_only", "helix_only", "interleaved"]:
                ns = find_first_crossing(eq_curves[eq_label], metric, tval, direction)
                row[f"eq_{eq_label}_ns"] = ns

            for strat in ["random", "native", "transferred"]:
                ns = find_first_crossing(seeding_curves[strat], metric, tval, direction)
                row[f"seed_{strat}_ns"] = ns

            # Compute speedups: eq_interleaved / seed_X
            eq_ref = row.get("eq_interleaved_ns")
            for strat in ["random", "native", "transferred"]:
                seed_ns = row.get(f"seed_{strat}_ns")
                if eq_ref and seed_ns and seed_ns > 0:
                    row[f"speedup_{strat}_vs_interleaved"] = eq_ref / seed_ns
                else:
                    row[f"speedup_{strat}_vs_interleaved"] = None

            # Also vs single extended run
            eq_ext = row.get("eq_extended_only_ns")
            for strat in ["random", "native", "transferred"]:
                seed_ns = row.get(f"seed_{strat}_ns")
                if eq_ext and seed_ns and seed_ns > 0:
                    row[f"speedup_{strat}_vs_extended"] = eq_ext / seed_ns
                else:
                    row[f"speedup_{strat}_vs_extended"] = None

            acceleration_table.append(row)

    # ── AUC comparison ────────────────────────────────────────────────────────
    print("\nComputing AUC for JSD convergence...")
    
    def compute_auc(curve, metric="jsd", max_ns=600):
        pts = [(p["ns"], p[metric]) for p in curve if p[metric] is not None and not np.isnan(p[metric]) and p["ns"] <= max_ns]
        if len(pts) < 2:
            return None
        pts.sort()
        auc = 0.0
        for i in range(len(pts) - 1):
            dx = pts[i+1][0] - pts[i][0]
            avg_y = (pts[i][1] + pts[i+1][1]) / 2
            auc += dx * avg_y
        return auc

    auc_results = {}
    for label in ["sequential", "extended_only", "helix_only", "interleaved"]:
        auc_results[f"eq_{label}"] = compute_auc(eq_curves[label])
    for strat in ["random", "native", "transferred"]:
        auc_results[f"seed_{strat}"] = compute_auc(seeding_curves[strat])

    # ── Print everything ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n--- Effective Sample Sizes (NHB observable) ---")
    for k, v in ess_results.items():
        print(f"  {k}: n={v['n_frames']}, ns={v['ns']:.1f}, "
              f"ESS={v['ess']:.1f}, ESS/ns={v['ess_per_ns']:.2f}, "
              f"lag1_acf={v['lag1_acf']:.4f}")

    print(f"\n--- Noise floor JSD = {noise_floor:.6f} ---")
    print(f"--- Ground truth f_helix = {gt_f_helix:.6f} ---")

    print("\n--- Equilibrium convergence (interleaved, selected points) ---")
    for pt in eq_curves["interleaved"]:
        if pt["ns"] in [50, 100, 150, 200, 300, 450, 600, 750, 950]:
            print(f"  {pt['ns']:.0f} ns: JSD={pt['jsd']:.6f}, cov={pt['coverage']:.4f}, "
                  f"f_err={pt['f_helix_error']:.6f}, trans_rate={pt['trans_rate_per_ns']:.4f}")

    print("\n--- Equilibrium convergence (extended only, selected points) ---")
    for pt in eq_curves["extended_only"]:
        if pt["ns"] in [50, 100, 150, 200, 300, 450, 475]:
            print(f"  {pt['ns']:.0f} ns: JSD={pt['jsd']:.6f}, cov={pt['coverage']:.4f}, "
                  f"f_err={pt['f_helix_error']:.6f}, trans_rate={pt['trans_rate_per_ns']:.4f}")

    print("\n--- Seeding convergence ---")
    for strat in ["random", "native", "transferred"]:
        print(f"  Strategy: {strat}")
        for pt in seeding_curves[strat]:
            print(f"    {pt['ns']:.0f} ns: JSD={pt['jsd']:.6f}, cov={pt['coverage']:.4f}, "
                  f"f_err={pt['f_helix_error']:.6f}, trans_rate={pt['trans_rate_per_ns']:.4f}")

    print("\n--- AUC (JSD integrated over 0-600 ns, lower = faster convergence) ---")
    for k, v in sorted(auc_results.items()):
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    print("\n--- Time to reach thresholds (ns) and speedup ---")
    print(f"{'metric':<15} {'threshold':<15} {'eq_ext':<10} {'eq_interl':<10} "
          f"{'random':<10} {'native':<10} {'transf':<10} "
          f"{'spd_rand':<10} {'spd_nat':<10} {'spd_trans':<10}")
    for row in acceleration_table:
        def fmt(v):
            return f"{v:.0f}" if v is not None else "never"
        def fmts(v):
            return f"{v:.2f}x" if v is not None else "N/A"
        print(f"{row['metric']:<15} {row['threshold']:<15} "
              f"{fmt(row.get('eq_extended_only_ns')):<10} "
              f"{fmt(row.get('eq_interleaved_ns')):<10} "
              f"{fmt(row.get('seed_random_ns')):<10} "
              f"{fmt(row.get('seed_native_ns')):<10} "
              f"{fmt(row.get('seed_transferred_ns')):<10} "
              f"{fmts(row.get('speedup_random_vs_extended')):<10} "
              f"{fmts(row.get('speedup_native_vs_extended')):<10} "
              f"{fmts(row.get('speedup_transferred_vs_extended')):<10}")

    # ── Save results ──────────────────────────────────────────────────────────
    save = {
        "gt_f_helix": gt_f_helix,
        "noise_floor_jsd2": noise_floor,
        "n_frames_total": n_total,
        "total_ns": total_ns,
        "part_lengths": part_lengths,
        "ess": ess_results,
        "auc_jsd_0_600ns": auc_results,
        "eq_curves": eq_curves,
        "seeding_curves": seeding_curves,
        "acceleration_table": acceleration_table,
    }

    out_path = RESULTS_DIR / "acceleration_deep_results.json"
    with out_path.open("w") as f:
        json.dump(save, f, indent=2, default=lambda x: None if x is None else float(x))
    print(f"\nResults saved to {out_path}")

else:
    print("Cannot run full analysis without mdtraj. Install it and retry.")
