"""Calibrate M3 k-NN Euclidean distance tiers using forward-window composite JSD (oracle)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.compass.online_state import TrajectorySegments, parse_msm_segments_arg
from src.compass.v2.equivalence import EquivalenceMethod, apply_gates
from src.compass.v2.equivalence_index import build_and_save, load_index, load_macro_labels
from src.compass.v2.equivalence_index import _load_features
from src.compass.v2.forward_similarity import (
    forward_composite_scores,
    percentile_summary,
    youden_best_threshold,
)


def parse_neighbor_ranks(spec: str, ncols: int) -> list[int]:
    """1-based neighbor column in index (0 is self). Columns must be < ``ncols``."""

    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        out = list(range(int(a.strip()), int(b.strip()) + 1))
    else:
        out = [int(x.strip()) for x in spec.split(",") if x.strip()]
    k_max = ncols
    return [r for r in out if 0 < r < k_max]


def _decode_feature_note(eq: dict[str, Any]) -> str:
    fn = eq.get("feature_note")
    if isinstance(fn, np.ndarray) and fn.size:
        raw = fn.reshape(-1)[0]
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    return "a8q2_features_sin_cos_torsions"


def _collect_pairs(
    *,
    repo_root: Path,
    oracle_cache: Path,
    eq_k: int,
    chunk_frames: int,
    anchors: int,
    rng: np.random.Generator,
    segments_spec: str | None,
    same_segment_only_gate: bool,
    max_time_delta_frames: int,
    neighbor_ranks_spec: str,
    feat_weight: float,
    nhb_weight: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    repo_root = repo_root.resolve()
    feats = _load_features(repo_root)
    n_frames = feats.shape[0]
    segments = parse_msm_segments_arg(segments_spec, n_frames)
    seg = TrajectorySegments(segments)
    macro = load_macro_labels(repo_root, oracle_cache)
    nhb_path = repo_root / "outputs" / "artifacts" / "a8q2_nhb.npy"
    nhb = np.load(nhb_path) if nhb_path.is_file() else None

    path = oracle_cache / f"eq_{EquivalenceMethod.M3_TORSION.value}_k{eq_k}.npz"
    if not path.is_file():
        path = build_and_save(repo_root, oracle_cache, EquivalenceMethod.M3_TORSION, eq_k)
    eq = load_index(path)
    idx_tbl = np.asarray(eq["indices"], dtype=np.int64)
    dist_tbl = np.asarray(eq["distances"], dtype=np.float64)

    ncols = idx_tbl.shape[1]
    ranks = parse_neighbor_ranks(neighbor_ranks_spec, ncols)
    if not ranks:
        raise RuntimeError("No valid neighbor ranks after parsing spec (check --neighbor-ranks vs index width).")

    max_i = n_frames - chunk_frames
    if max_i <= 0:
        raise RuntimeError("Insufficient frames for chunk")
    picks = rng.choice(max_i + 1, size=min(int(anchors), max_i + 1), replace=False)

    ais: list[int] = []
    jis: list[int] = []
    dis: list[float] = []
    rk: list[int] = []

    max_td = None if same_segment_only_gate else int(max_time_delta_frames)

    for i in picks.tolist():
        i = int(i)
        for r in ranks:
            j = int(idx_tbl[i, r])
            dj = float(dist_tbl[i, r])
            if j < 0 or j >= n_frames:
                continue
            if j + chunk_frames > n_frames:
                continue
            if not apply_gates(
                i,
                j,
                seg,
                max_time_delta_frames=max_td,
                same_segment_only=same_segment_only_gate,
                macro_labels=macro,
                require_same_macro=False,
            ):
                continue
            ais.append(i)
            jis.append(j)
            dis.append(dj)
            rk.append(r)

    if not ais:
        raise RuntimeError("No valid (anchor, neighbor) pairs after gating.")

    anc = np.array(ais, dtype=np.int64)
    nei = np.array(jis, dtype=np.int64)
    d_arr = np.array(dis, dtype=np.float64)
    rank_arr = np.array(rk, dtype=np.int64)
    feat_js, nhb_js, comp = forward_composite_scores(
        feats,
        nhb,
        anc,
        nei,
        chunk_frames,
        feat_weight=feat_weight,
        nhb_weight=nhb_weight,
    )

    meta_extra = {
        "eq_method": EquivalenceMethod.M3_TORSION.value,
        "eq_k": eq_k,
        "chunk_frames": chunk_frames,
        "feat_weight": feat_weight,
        "nhb_weight": nhb_weight,
        "anchors_sampled_requested": anchors,
        "n_pairs": len(ais),
        "neighbor_ranks": ranks,
        "feature_note_from_index": _decode_feature_note(eq),
        "same_segment_only_gate": same_segment_only_gate,
        "max_time_delta_frames": None if same_segment_only_gate else max_time_delta_frames,
        "composite_includes_nhb": nhb is not None and getattr(nhb, "size", 0) == n_frames,
    }

    return meta_extra, d_arr, comp, feat_js, nhb_js, rank_arr, anc


def _tier_masks(comp: np.ndarray, q_close: float, q_far: float) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    tau_c = float(np.quantile(comp, q_close))
    tau_f = float(np.quantile(comp, q_far))
    close_m = comp <= tau_c
    far_m = comp >= tau_f
    amb_m = (~close_m) & (~far_m)
    return tau_c, tau_f, close_m, amb_m, far_m


def _distance_cutoffs_quantile(close_d: np.ndarray, far_d: np.ndarray, p_hi: float, p_lo: float) -> tuple[float, float]:
    """d_close_max ~ high percentile within close tier; d_not_similar_min ~ low percentile within far tier."""

    dc = float(np.percentile(close_d, p_hi)) if close_d.size else float("nan")
    df = float(np.percentile(far_d, p_lo)) if far_d.size else float("nan")
    return dc, df


def _bootstrap_cutoffs(
    anc: np.ndarray,
    *,
    sampler: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_boot: int,
    rng: np.random.Generator,
    q_close: float,
    q_far: float,
    p_hi: float,
    p_lo: float,
) -> dict[str, Any]:
    """Resample anchors (with replacement) and recompute JSD tau + distance summaries."""

    unique_anchors = np.unique(anc)
    if unique_anchors.size < 10 or n_boot <= 0:
        return {}

    tau_c_bs: list[float] = []
    tau_f_bs: list[float] = []
    dc_bs: list[float] = []
    df_bs: list[float] = []
    youden_bs: list[float] = []

    for _ in range(n_boot):
        pick = rng.choice(unique_anchors, size=unique_anchors.size, replace=True)
        sel = np.isin(anc, pick)
        if np.sum(sel) < 30:
            continue
        dd, cc, _ = sampler(sel)
        tau_c, tau_f, cl, _, fr = _tier_masks(cc, q_close, q_far)
        tau_c_bs.append(tau_c)
        tau_f_bs.append(tau_f)
        dcm, dfm = _distance_cutoffs_quantile(dd[cl], dd[fr], p_hi, p_lo)
        dc_bs.append(dcm)
        df_bs.append(dfm)
        y_mask = cc <= tau_c
        d_star, _ = youden_best_threshold(dd, positive_low_composite=y_mask)
        youden_bs.append(d_star)

    def band(xs: list[float]) -> dict[str, float]:
        arr = np.array([x for x in xs if np.isfinite(x)], dtype=np.float64)
        if arr.size == 0:
            return {"p025": float("nan"), "p975": float("nan")}
        return {"p025": float(np.percentile(arr, 2.5)), "p975": float(np.percentile(arr, 97.5))}

    return {
        "n_bootstrap": n_boot,
        "tau_close_Q": band(tau_c_bs),
        "tau_far_Q": band(tau_f_bs),
        "d_close_max_percentile_band": band(dc_bs),
        "d_not_similar_min_percentile_band": band(df_bs),
        "d_youden_close_band": band(youden_bs),
    }

def _distance_overlap_coefficient(close_d: np.ndarray, far_d: np.ndarray, n_grid: int = 512) -> float:
    """Rough overlap of KDE-smoothed distance distributions on a shared scaled axis."""

    if close_d.size < 5 or far_d.size < 5:
        return float("nan")
    lo = float(min(close_d.min(), far_d.min()))
    hi = float(max(close_d.max(), far_d.max()))
    span = hi - lo
    if span <= 0:
        return 1.0
    a = (close_d - lo) / span
    b = (far_d - lo) / span
    gx = np.linspace(0.0, 1.0, n_grid)
    sig = max(5.0 / n_grid, 0.03)

    def kde(xs: np.ndarray) -> np.ndarray:
        dens = np.zeros_like(gx, dtype=np.float64)
        for x in xs:
            dens += np.exp(-0.5 * ((gx - x) / sig) ** 2)
        denom = max(xs.size * sig * np.sqrt(2 * np.pi), 1e-12)
        dens /= denom
        s = dens.sum()
        return dens / max(s, 1e-15)

    pc, pf = kde(a), kde(b)
    overlap = np.minimum(pc, pf).sum()
    return float(overlap)


def _distance_separation_bundle(close_d: np.ndarray, far_d: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "mean_m3_distance_close": float(np.mean(close_d)) if close_d.size else float("nan"),
        "mean_m3_distance_not_similar": float(np.mean(far_d)) if far_d.size else float("nan"),
        "mean_gap_far_minus_close": float(np.mean(far_d) - np.mean(close_d))
        if close_d.size and far_d.size
        else float("nan"),
        "histogram_overlap_coeff_approx": float("nan"),
        "ks_statistic_distance_close_vs_far": float("nan"),
        "ks_pvalue_distance_close_vs_far": float("nan"),
    }
    try:
        from scipy.stats import ks_2samp

        if close_d.size and far_d.size:
            ks_stat, ks_p = ks_2samp(close_d, far_d)
            out["ks_statistic_distance_close_vs_far"] = float(ks_stat)
            out["ks_pvalue_distance_close_vs_far"] = float(ks_p)
    except Exception:
        pass
    out["histogram_overlap_coeff_approx"] = _distance_overlap_coefficient(close_d, far_d)
    ks = out["ks_statistic_distance_close_vs_far"]
    ol = out["histogram_overlap_coeff_approx"]
    gap = out["mean_gap_far_minus_close"]
    actionable = (
        np.isfinite(ks)
        and ks > 0.08
        and np.isfinite(ol)
        and ol < 0.92
        and np.isfinite(gap)
        and gap > 0.03
        and gap > 0.0
    )
    out["distance_separation_actionable_for_hard_cap"] = 1.0 if actionable else 0.0
    return out


def save_calibration_histograms(
    path_dir: Path,
    *,
    d_arr: np.ndarray,
    comp: np.ndarray,
    feat_js: np.ndarray,
    nhb_js: np.ndarray,
    rank_arr: np.ndarray,
    close_m: np.ndarray,
    amb_m: np.ndarray,
    far_m: np.ndarray,
    tau_c: float,
    tau_f: float,
    d_close_max: float,
    tag: str,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib required for --plots-dir") from exc

    path_dir.mkdir(parents=True, exist_ok=True)
    prefix = (tag.replace(" ", "_") + "_") if tag else ""
    outs: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(comp, bins=42, density=True, color="#4c72b0", alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.axvline(tau_c, color="#2ecc71", ls="--", lw=2, label=f"tau_close={tau_c:.4f}")
    ax.axvline(tau_f, color="#e74c3c", ls="--", lw=2, label=f"tau_far={tau_f:.4f}")
    ax.set_xlabel("forward composite JSD (lower = more similar dynamics)")
    ax.set_ylabel("density")
    ax.set_title("Pooled composite over sampled (anchor, neighbor_rank) pairs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p1 = path_dir / f"{prefix}hist_composite_jsd.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    outs.append(p1)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 40
    ax.hist(
        d_arr[close_m],
        bins=bins,
        density=True,
        alpha=0.55,
        color="#2ecc71",
        label="close (by composite)",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        d_arr[amb_m],
        bins=bins,
        density=True,
        alpha=0.45,
        color="#95a5a6",
        label="almost",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        d_arr[far_m],
        bins=bins,
        density=True,
        alpha=0.55,
        color="#e74c3c",
        label="not_similar",
        edgecolor="white",
        linewidth=0.4,
    )
    if np.isfinite(d_close_max):
        ax.axvline(d_close_max, color="#34495e", ls=":", lw=2, label=f"d_cap p90(close)={d_close_max:.3f}")
    ax.set_xlabel("M3 Euclidean distance (sin/cos torsion space)")
    ax.set_ylabel("density")
    ax.set_title("Distance by composite tier — overlap ⇒ distance cap weakly identified")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = path_dir / f"{prefix}hist_m3distance_by_composite_tier.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    outs.append(p2)

    n = len(comp)
    sub = slice(None)
    max_pts = 12_000
    if n > max_pts:
        rng = np.random.default_rng(0)
        sub = rng.choice(n, size=max_pts, replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(
        comp[sub],
        d_arr[sub],
        c=rank_arr[sub],
        cmap="viridis",
        s=10,
        alpha=0.45,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="neighbor rank column")
    ax.axvline(tau_c, color="#2ecc71", ls="--", alpha=0.8)
    ax.axvline(tau_f, color="#e74c3c", ls="--", alpha=0.8)
    ax.set_xlabel("composite JSD")
    ax.set_ylabel("M3 distance")
    ax.set_title("Geometry vs oracle score (rank colormap)")
    fig.tight_layout()
    p3 = path_dir / f"{prefix}scatter_composite_vs_distance.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    outs.append(p3)

    uniq_r = sorted(set(rank_arr.tolist()))
    positions = []
    data = []
    for r in uniq_r[:25]:
        m = rank_arr == r
        if np.sum(m) < 15:
            continue
        positions.append(r)
        data.append(comp[m])
    if data:
        fig, ax = plt.subplots(figsize=(8, 4))
        _ = ax.boxplot(data, positions=positions, widths=0.6, showfliers=False, medianprops=dict(color="orange"))
        ax.axhline(tau_c, color="#2ecc71", ls="--", alpha=0.7)
        ax.axhline(tau_f, color="#e74c3c", ls="--", alpha=0.7)
        ax.set_xlabel("k-NN column index j (distance sorted per anchor)")
        ax.set_ylabel("composite JSD")
        ax.set_title("Predictive disparity grows with geometry rank — expect higher ranks to score worse")
        fig.tight_layout()
        p4 = path_dir / f"{prefix}box_composite_vs_neighbor_rank.png"
        fig.savefig(p4, dpi=140)
        plt.close(fig)
        outs.append(p4)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    nhb_good = np.isfinite(nhb_js)
    if np.any(nhb_good):
        ax.hist(nhb_js[nhb_good], bins=32, density=True, color="#9b59b6", alpha=0.85)
        ax.set_xlabel("NHB channel JSD (when present)")
    else:
        ax.text(0.5, 0.5, "NHB not in composite / all NaN", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("NHB marginal (diagnostic degeneracy)")
    fig.tight_layout()
    p5 = path_dir / f"{prefix}hist_nhb_js.png"
    fig.savefig(p5, dpi=140)
    plt.close(fig)
    outs.append(p5)

    return outs


def calibrate(
    repo_root: Path,
    oracle_cache: Path,
    *,
    eq_k: int,
    chunk_frames: int,
    anchors: int,
    random_seed: int,
    segments_spec: str | None,
    same_segment_only_gate: bool,
    max_time_delta_frames: int,
    neighbor_ranks: str,
    feat_weight: float,
    nhb_weight: float,
    q_close: float,
    q_far: float,
    distance_cut_method: str,
    close_dist_percentile: float,
    far_dist_percentile: float,
    bootstrap: int,
    out_path: Path | None,
    plots_dir: Path | None = None,
    plots_tag: str = "",
) -> dict[str, Any]:
    if not (0 <= q_close < q_far <= 1):
        raise ValueError("Require 0 <= q_close < q_far <= 1")

    oracle_cache = oracle_cache.resolve()
    oracle_cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    meta, d_arr, comp, feat_js, nhb_js, rank_arr, anc = _collect_pairs(
        repo_root=repo_root,
        oracle_cache=oracle_cache,
        eq_k=eq_k,
        chunk_frames=chunk_frames,
        anchors=anchors,
        rng=rng,
        segments_spec=segments_spec,
        same_segment_only_gate=same_segment_only_gate,
        max_time_delta_frames=max_time_delta_frames,
        neighbor_ranks_spec=neighbor_ranks,
        feat_weight=feat_weight,
        nhb_weight=nhb_weight,
    )
    ranks_used: list[int] = list(meta["neighbor_ranks"])

    tau_c, tau_f, close_m, amb_m, far_m = _tier_masks(comp, q_close, q_far)
    tiers_def = (
        "close: composite <= Q_{q_close}(all pairs); "
        "almost_similar: tau_close < composite < tau_far; "
        "not_similar: composite >= Q_{q_far}(all pairs)"
    )

    tier_distances = {
        "close": percentile_summary(d_arr[close_m]),
        "almost_similar": percentile_summary(d_arr[amb_m]),
        "not_similar": percentile_summary(d_arr[far_m]),
    }

    d_close_max, d_not_similar_min = _distance_cutoffs_quantile(
        d_arr[close_m],
        d_arr[far_m],
        close_dist_percentile,
        far_dist_percentile,
    )

    youden_star, youden_j = youden_best_threshold(d_arr, positive_low_composite=close_m)

    rec: dict[str, Any] = {
        "d_close_cap_at_percentile": d_close_max,
        "d_not_similar_floor_at_percentile": d_not_similar_min,
        "primary_reporting_focus": distance_cut_method,
        "d_youden_close_star": youden_star,
        "d_youden_j": youden_j,
    }

    n_r1 = np.sum(rank_arr == 1) if ranks_used and min(ranks_used) <= 1 else 0
    sep = _distance_separation_bundle(d_arr[close_m], d_arr[far_m])
    try:
        from scipy.stats import spearmanr

        rho, rp = spearmanr(comp, d_arr)
        spearman_corr = {"rho_comp_vs_distance": float(rho), "p_value_two_sided": float(rp)}
    except Exception:
        spearman_corr = {"rho_comp_vs_distance": float("nan"), "p_value_two_sided": float("nan")}
    diagnostics = {
        "n_pairs": int(comp.size),
        "fraction_tier_close": float(np.mean(close_m)),
        "fraction_tier_almost": float(np.mean(amb_m)),
        "fraction_tier_not_similar": float(np.mean(far_m)),
        "rank1_pairs_total": int(n_r1),
        "fraction_rank1_given_close": float(np.mean(rank_arr[close_m] == 1)) if np.any(close_m) else float("nan"),
        "composite_feat_js_median_over_pairs": float(np.median(feat_js)),
        "composite_nhb_js_median_over_pairs": (
            float(np.nanmedian(nhb_js)) if np.any(np.isfinite(nhb_js)) else float("nan")
        ),
        "overlap_note_close_cap_vs_far_floor": (
            "ambiguous_overlap"
            if np.isfinite(d_close_max)
            and np.isfinite(d_not_similar_min)
            and d_close_max > d_not_similar_min
            else "ordered_or_degenerate"
        ),
        **sep,
        **spearman_corr,
    }

    actionable = diagnostics.get("distance_separation_actionable_for_hard_cap", 0.0) >= 0.99
    recommended_operational = {
        "authority": (
            "use_composite_tiers_as_primary_similarity_definition"
            if not actionable
            else "composite_and_geometry_align_use_both"
        ),
        "composite_similar_to_close_upper": tau_c,
        "composite_not_similar_lower": tau_f,
        "m3_distance_neighbor_mask_cap_when_actionable_only": (
            float(d_close_max)
            if actionable and distance_cut_method == "quantile" and np.isfinite(d_close_max)
            else None
        ),
        "note": (
            "If distance_separation_actionable_for_hard_cap is 0, prefer composite τ for labeling; "
            "do not trust a Euclidean mask without inspecting histograms."
        ),
    }

    plots_written: list[str] | None = None
    if plots_dir is not None:
        paths = save_calibration_histograms(
            Path(plots_dir),
            d_arr=d_arr,
            comp=comp,
            feat_js=feat_js,
            nhb_js=nhb_js,
            rank_arr=rank_arr,
            close_m=close_m,
            amb_m=amb_m,
            far_m=far_m,
            tau_c=tau_c,
            tau_f=tau_f,
            d_close_max=d_close_max,
            tag=plots_tag,
        )
        plots_written = [str(x) for x in paths]

    payload: dict[str, Any] = {
        "meta": {
            "repo_note": str(repo_root.resolve()),
            "oracle_cache": str(oracle_cache),
            **meta,
            "random_seed": random_seed,
            "tier_definition": tiers_def,
            "q_close": q_close,
            "q_far": q_far,
            "distance_cut_percentiles": {
                "close_tier_hi_percentile_for_d_cap": close_dist_percentile,
                "not_similar_tier_lo_percentile_for_d_floor": far_dist_percentile,
            },
        },
        "tiers": {
            "forward_composite_jsd": {
                "tau_close_inclusive_upper": tau_c,
                "tau_far_inclusive_lower": tau_f,
                "rules": tiers_def,
            },
            "m3_distance_euclidean": {
                "per_tier_percentiles": tier_distances,
                "recommended_hard_split_quantile_method": rec,
                "youden_close_vs_rest": {"d_star": youden_star, "j_max": youden_j},
            },
        },
        "recommended_operational": recommended_operational,
        "plots_written": plots_written if plots_written is not None else None,
        "diagnostics": diagnostics,
    }

    boot_payload: dict[str, Any] = {}
    if bootstrap > 0:

        def sampler(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return d_arr[mask], comp[mask], anc[mask]

        boot_payload = _bootstrap_cutoffs(
            anc,
            sampler=sampler,
            n_boot=bootstrap,
            rng=np.random.default_rng(random_seed + 1),
            q_close=q_close,
            q_far=q_far,
            p_hi=close_dist_percentile,
            p_lo=far_dist_percentile,
        )
        payload["bootstrap"] = boot_payload

    outp = out_path or (oracle_cache / "m3_similarity_thresholds.json")
    outp = Path(outp)
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_m3_distance_cap_from_json(calibration_json: Path) -> tuple[float | None, str]:
    """Read distance mask cap from calibration artifact.

    If diagnostics mark Euclidean separation actionable, prefers
    ``recommended_operational.m3_distance_neighbor_mask_cap_when_actionable_only``;
    otherwise returns ``None`` so bench does **not** over-trust geometry (legacy raw p90 cap).
    Falls back to the raw quantile ``d_close_cap_at_percentile`` only when actionable.
    """

    if not calibration_json.is_file():
        return None, "calibration_missing"
    data = json.loads(calibration_json.read_text(encoding="utf-8"))
    diag = data.get("diagnostics") or {}
    actionable = float(diag.get("distance_separation_actionable_for_hard_cap") or 0.0) >= 0.99

    rec = (
        data.get("tiers", {})
        .get("m3_distance_euclidean", {})
        .get("recommended_hard_split_quantile_method", {})
    )

    op = data.get("recommended_operational") or {}
    op_cap = op.get("m3_distance_neighbor_mask_cap_when_actionable_only")

    if actionable:
        if isinstance(op_cap, (int, float)) and np.isfinite(float(op_cap)):
            return float(op_cap), "recommended_operational_actionable_cap"
        raw = rec.get("d_close_cap_at_percentile")
        if raw is None:
            return None, "d_close_cap_missing"
        v = float(raw)
        if not np.isfinite(v):
            return None, "d_close_cap_nan"
        return v, "legacy_quantile_cap"

    return None, "geometry_weak_use_composite_skip_distance_mask"
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--oracle-cache", type=Path, default=None)
    p.add_argument("--eq-k", type=int, default=40)
    p.add_argument("--chunk-frames", type=int, default=20)
    p.add_argument("--anchors", type=int, default=400)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--msm-segments", type=str, default=None)
    p.add_argument("--same-segment-only-gate", action="store_true")
    p.add_argument("--max-time-delta-frames", type=int, default=800)
    p.add_argument("--neighbor-ranks", type=str, default="1-20", help='e.g. "1-20" or "1,2,5,10".')
    p.add_argument("--feat-weight", type=float, default=0.5)
    p.add_argument("--nhb-weight", type=float, default=0.5)
    p.add_argument("--q-close", type=float, default=0.25, help="Quantile defining *close* (composite upper bound).")
    p.add_argument("--q-far", type=float, default=0.75)
    p.add_argument(
        "--distance-cut-method",
        choices=("quantile", "youden"),
        default="quantile",
        help="Reporting focus; both summaries are emitted.",
    )
    p.add_argument(
        "--close-dist-percentile",
        type=float,
        default=90.0,
        help="Within *close* tier, cap distance at this percentile (d_close_max).",
    )
    p.add_argument(
        "--far-dist-percentile",
        type=float,
        default=10.0,
        help="Within *not_similar* tier, floor distance at this percentile (d_not_similar_min).",
    )
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Defaults to oracle-cache/m3_similarity_thresholds.json")
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG histogram diagnostics (matplotlib).",
    )
    p.add_argument("--plots-tag", type=str, default="", help="Filename prefix inside plots-dir.")
    args = p.parse_args(argv)
    repo = args.repo_root.resolve()
    cache = (args.oracle_cache or repo / "outputs" / "v2" / "oracle").resolve()

    payload = calibrate(
        repo,
        cache,
        eq_k=args.eq_k,
        chunk_frames=args.chunk_frames,
        anchors=args.anchors,
        random_seed=args.random_seed,
        segments_spec=args.msm_segments,
        same_segment_only_gate=args.same_segment_only_gate,
        max_time_delta_frames=args.max_time_delta_frames,
        neighbor_ranks=args.neighbor_ranks,
        feat_weight=args.feat_weight,
        nhb_weight=args.nhb_weight,
        q_close=args.q_close,
        q_far=args.q_far,
        distance_cut_method=args.distance_cut_method,
        close_dist_percentile=args.close_dist_percentile,
        far_dist_percentile=args.far_dist_percentile,
        bootstrap=args.bootstrap,
        out_path=args.out,
        plots_dir=args.plots_dir,
        plots_tag=args.plots_tag.strip(),
    )
    print(json.dumps(payload["meta"], indent=2))
    print(json.dumps(payload["tiers"], indent=2))
    print(json.dumps(payload["recommended_operational"], indent=2))
    print(json.dumps(payload["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
