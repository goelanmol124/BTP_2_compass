"""COMPASS v2 main bench: causal archive playback + equivalence-cloud candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.compass.online_state import ObservedTargetState, TrajectorySegments, parse_msm_segments_arg
from src.compass.replay_enhanced import ReplayConfig, _bin_indices, build_atlas, plot_curves
from src.compass.replay_enhanced import FRAME_SPACING_NS
from src.compass.replay_enhanced import _summarize_random as summarize_random_df
from src.compass.v2.equivalence import EquivalenceMethod, expand_candidates
from src.compass.v2.calibrate_m3_thresholds import load_m3_distance_cap_from_json
from src.compass.v2.equivalence_index import build_and_save, load_index, load_macro_labels
from src.compass.v2.evaluate import MilestoneTargets, cheating_audit_ok, effective_sample_size_timeseries, finalize_milestones
from src.compass.v2.evaluate import evaluate_state
from src.compass.v2.policy import (
    PolicyName,
    compass_pick,
    farthest_latent_pick,
    min_count_bin_pick,
    random_pick,
)


def _increment_bins_for_range(
    bin_attempts: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    start: int,
    stop: int,
) -> None:
    for fi in range(start, stop):
        bin_attempts[int(full_bx[fi]), int(full_by[fi])] += 1


def load_equivalence_choice_method(oracle_cache: Path, *, fallback: EquivalenceMethod) -> tuple[EquivalenceMethod, str]:
    path = oracle_cache / "equivalence_choice.json"
    if not path.is_file():
        return fallback, "no_equivalence_choice_json_using_fallback"
    data = json.loads(path.read_text(encoding="utf-8"))
    eco = data.get("equivalence_choice", data)
    win = eco.get("winner_method")
    if win is None or str(win).strip() == "":
        return fallback, "empty_winner_using_fallback"
    try:
        return EquivalenceMethod(str(win).strip()), f"equivalence_choice.json:{win}"
    except ValueError:
        return fallback, f"unknown_winner_{win}_fallback"


def _ensure_eq_index(repo_root: Path, cache_dir: Path, method: EquivalenceMethod, k: int) -> Path:
    path = cache_dir / f"eq_{method.value}_k{k}.npz"
    if not path.is_file():
        path = build_and_save(repo_root, cache_dir, method, k)
    return path


def build_candidate_pool(
    visited_idx: np.ndarray,
    eq_indices: np.ndarray,
    segments: TrajectorySegments,
    used_seeds: set[int],
    *,
    max_time_delta_frames: int | None,
    same_segment_only: bool,
    macro_labels: np.ndarray | None,
    require_same_macro: bool,
    max_neighbors_per_anchor: int,
    neighbor_mask: np.ndarray | None = None,
) -> np.ndarray:
    pool = expand_candidates(
        visited_idx,
        eq_indices,
        neighbor_mask,
        segments,
        max_time_delta_frames=max_time_delta_frames,
        same_segment_only=same_segment_only,
        macro_labels=macro_labels,
        require_same_macro=require_same_macro,
        max_neighbors_per_anchor=max_neighbors_per_anchor,
    )
    pool = np.array([i for i in pool.tolist() if i not in used_seeds], dtype=int)
    return pool


def run_v2_benchmark(
    cfg: ReplayConfig,
    *,
    policy: PolicyName,
    eq_method: EquivalenceMethod,
    eq_k: int,
    oracle_cache: Path,
    segments_spec: str | None,
    cold_start_frame: int | None,
    initial_seeds_extra: list[int],
    max_time_delta_frames: int | None,
    same_segment_only: bool,
    require_same_macro: bool,
    macro_n_clusters: int,
    max_neighbors_per_anchor: int,
    milestones: MilestoneTargets,
    allow_deterministic_reseed: bool,
    eq_mask_m3_by_calibration: bool = False,
    eq_m3_calibration_json: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    repo_root = cfg.repo_root
    atlas, arrays = build_atlas(cfg)
    nhb = arrays["a8q2_nhb"]
    n_frames = len(nhb)
    segments = parse_msm_segments_arg(segments_spec, n_frames)
    seg_helper = TrajectorySegments(segments)

    eq_path = _ensure_eq_index(repo_root, oracle_cache, eq_method, eq_k)
    eq_data = load_index(eq_path)
    neighbor_idx = np.asarray(eq_data["indices"], dtype=np.int64)
    macro_labels: np.ndarray | None = None
    if require_same_macro or eq_method == EquivalenceMethod.M7_MACRO_KMEANS:
        if "macro_labels" in eq_data:
            macro_labels = np.asarray(eq_data["macro_labels"], dtype=np.int64)
        else:
            macro_labels = load_macro_labels(repo_root, oracle_cache, n_clusters=macro_n_clusters)

    neighbor_mask: np.ndarray | None = None
    m3_cal_path: Path | None = None
    m3_cal_status = ""
    m3_d_cap: float | None = None
    if eq_mask_m3_by_calibration and eq_method == EquivalenceMethod.M3_TORSION:
        m3_cal_path = Path(eq_m3_calibration_json) if eq_m3_calibration_json is not None else oracle_cache / "m3_similarity_thresholds.json"
        d_cap, st = load_m3_distance_cap_from_json(m3_cal_path)
        m3_cal_status = st
        m3_d_cap = d_cap
        if d_cap is not None and "distances" in eq_data:
            dist = np.asarray(eq_data["distances"], dtype=np.float64)
            neighbor_mask = dist <= float(d_cap)
        else:
            neighbor_mask = None

    xy_policy = atlas.target_xy
    frontier_xy = atlas.frontier_xy if atlas.frontier_xy is not None else atlas.target_xy
    full_bx, full_by = _bin_indices(xy_policy, atlas.x_edges, atlas.y_edges)

    bin_attempts = np.zeros_like(atlas.source_hist, dtype=int)
    state = ObservedTargetState(
        n_frames=n_frames,
        segments=seg_helper,
        seed_reservoir_stride=1,
        seed_reservoir_max=None,
        allow_deterministic_reseed=allow_deterministic_reseed,
    )
    selected_seed_frames: list[int] = []
    used_seeds: set[int] = set()
    rng = np.random.default_rng(cfg.random_seed)

    rows: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    dec_rows: list[dict[str, Any]] = []

    def record_metrics(step: int) -> dict[str, float]:
        idx = np.flatnonzero(state.observed_mask)
        m = evaluate_state(atlas, nhb, idx, cfg.helix_nhb_threshold, cfg.open_nhb_threshold)
        rows.append(
            {
                "policy": policy.value,
                "eq_method": eq_method.value,
                "step": step,
                "selected_chunks": step,
                "sampled_frames": int(state.n_observed_frames),
                "sampled_ns": float(state.n_observed_frames) * FRAME_SPACING_NS,
                **m,
            }
        )
        return m

    # --- cold start + optional extra initial seeds (first step bundles all) ---
    if cold_start_frame is not None:
        init_frames = [int(cold_start_frame)]
    else:
        init_frames = [int(rng.integers(0, n_frames))]
    if policy == PolicyName.B0_CHRONOLOGICAL:
        initial_seeds_extra = []
    init_frames.extend(initial_seeds_extra)
    first_write = True
    for fi in init_frames:
        fi = int(fi)
        if fi in used_seeds:
            continue
        a, b = seg_helper.clip_continuation(fi, cfg.chunk_frames)
        state.mark_observed(a, b)
        _increment_bins_for_range(bin_attempts, full_bx, full_by, a, b)
        selected_seed_frames.append(fi)
        used_seeds.add(fi)
        state.register_seed(fi)
        if first_write:
            record_metrics(1)
            visited = np.flatnonzero(state.observed_mask)
            pool0 = build_candidate_pool(
                visited,
                neighbor_idx,
                seg_helper,
                used_seeds,
                max_time_delta_frames=max_time_delta_frames,
                same_segment_only=same_segment_only,
                macro_labels=macro_labels,
                require_same_macro=require_same_macro,
                max_neighbors_per_anchor=max_neighbors_per_anchor,
                neighbor_mask=neighbor_mask,
            )
            cand_rows.append(
                {
                    "step": 1,
                    "pool_size": int(pool0.size),
                    "n_visited_frames": int(visited.size),
                    "fraction_cloud": float(max(0, pool0.size - visited.size) / max(pool0.size, 1)),
                }
            )
            dec_rows.append(
                {
                    "step": 1,
                    "seed_frame": fi,
                    "policy": policy.value,
                    "reason": "cold_start_or_extra_init",
                    "candidate_pool_size": int(pool0.size),
                }
            )
            first_write = False

    # B0 chronological: single causal chain along saved frames (first seed only).
    anchor0 = selected_seed_frames[0]
    _a0, b0 = seg_helper.clip_continuation(anchor0, cfg.chunk_frames)
    chrono_cursor: int | None = int(b0) if policy == PolicyName.B0_CHRONOLOGICAL else None

    n_steps = cfg.n_steps
    last_pool = int(cand_rows[-1]["pool_size"]) if cand_rows else -1

    for step in range(2, n_steps + 1):
        visited = np.flatnonzero(state.observed_mask)
        if policy == PolicyName.B0_CHRONOLOGICAL:
            assert chrono_cursor is not None
            seed_f = int(chrono_cursor)
            seg = seg_helper.containing(seed_f)
            if seg is None:
                break
            s, e = seg
            if seed_f >= e:
                break
            _, b_end = seg_helper.clip_continuation(seed_f, cfg.chunk_frames)
            last_pool = 0
            chrono_cursor = int(b_end)
        else:
            pool = build_candidate_pool(
                visited,
                neighbor_idx,
                seg_helper,
                used_seeds,
                max_time_delta_frames=max_time_delta_frames,
                same_segment_only=same_segment_only,
                macro_labels=macro_labels,
                require_same_macro=require_same_macro,
                max_neighbors_per_anchor=max_neighbors_per_anchor,
                neighbor_mask=neighbor_mask,
            )
            if pool.size == 0:
                pool = visited.astype(int)
                pool = np.array([i for i in pool.tolist() if i not in used_seeds], dtype=int)
            if pool.size == 0:
                break
            fraction_done = (step - 1) / max(float(n_steps), 1.0)
            if policy == PolicyName.B5_COMPASS:
                _, seed_f = compass_pick(
                    atlas, cfg, pool, selected_seed_frames, bin_attempts, fraction_done, rng, uniform_prior=False
                )
            elif policy == PolicyName.B4_COMPASS_UNIFORM_PRIOR:
                _, seed_f = compass_pick(
                    atlas, cfg, pool, selected_seed_frames, bin_attempts, fraction_done, rng, uniform_prior=True
                )
            elif policy == PolicyName.B1_RANDOM:
                _, seed_f = random_pick(pool, rng)
            elif policy == PolicyName.B2_MIN_COUNT_BIN:
                _, seed_f = min_count_bin_pick(pool, bin_attempts, atlas, frontier_xy, rng)
            elif policy == PolicyName.B3_FARTHEST_LATENT:
                _, seed_f = farthest_latent_pick(pool, selected_seed_frames, frontier_xy, rng)
            else:
                raise ValueError(policy)

            cand_rows.append(
                {
                    "step": step,
                    "pool_size": int(pool.size),
                    "n_visited_frames": int(visited.size),
                    "fraction_cloud": float(max(0, pool.size - visited.size) / max(pool.size, 1)),
                }
            )
            last_pool = int(pool.size)



        selected_seed_frames.append(seed_f)
        used_seeds.add(seed_f)
        state.register_seed(seed_f)
        a, b = seg_helper.clip_continuation(seed_f, cfg.chunk_frames)
        state.mark_observed(a, b)
        _increment_bins_for_range(bin_attempts, full_bx, full_by, a, b)
        record_metrics(step)
        dec_rows.append(
            {
                "step": step,
                "seed_frame": int(seed_f),
                "policy": policy.value,
                "reason": policy.value,
                "candidate_pool_size": last_pool,
            }
        )

    metrics_df = pd.DataFrame(rows)
    cand_df = pd.DataFrame(cand_rows)
    dec_df = pd.DataFrame(dec_rows)

    idx_time = np.sort(np.flatnonzero(state.observed_mask))
    ess = effective_sample_size_timeseries(idx_time)
    summary_extra = finalize_milestones(metrics_df, milestones)
    summary_extra["effective_sample_size_proxy"] = ess
    summary_extra["cheating_audit"] = cheating_audit_ok(
        {"policy_features": ["atlas_xy", "bin_attempts"], "oracle_leak_check": False},
        {"rmsd_neighbors", "nhb"},
    )
    summary_extra["m3_distance_calibration"] = {
        "requested": eq_mask_m3_by_calibration,
        "calibration_path": str(m3_cal_path) if m3_cal_path else None,
        "status": m3_cal_status if eq_mask_m3_by_calibration else "",
        "d_close_cap_used": m3_d_cap if eq_mask_m3_by_calibration else None,
        "neighbor_mask_applied": bool(neighbor_mask is not None) if eq_mask_m3_by_calibration else False,
    }
    return metrics_df, cand_df, dec_df, summary_extra


def _random_baseline_matching(
    cfg: ReplayConfig,
    eq_method: EquivalenceMethod,
    eq_k: int,
    oracle_cache: Path,
    segments_spec: str | None,
    cold_start_frame: int | None,
    n_repeats: int,
    *,
    benchmark_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []
    for rep in range(n_repeats):
        sub = replace(cfg, random_seed=cfg.random_seed + 100 + rep)
        mdf, _, _, _ = run_v2_benchmark(
            sub,
            policy=PolicyName.B1_RANDOM,
            eq_method=eq_method,
            eq_k=eq_k,
            oracle_cache=oracle_cache,
            segments_spec=segments_spec,
            cold_start_frame=cold_start_frame,
            **benchmark_kwargs,
        )
        mdf = mdf.copy()
        mdf["repeat"] = rep
        all_rows.append(mdf)
    big = pd.concat(all_rows, ignore_index=True)
    summary = summarize_random_df(big)
    return big, summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--output-dir", type=Path, default=Path("outputs/v2/a8q2_default"))
    p.add_argument("--policy", type=str, default=PolicyName.B5_COMPASS.value, choices=[e.value for e in PolicyName])
    p.add_argument(
        "--eq-method",
        type=str,
        default="auto",
        help="'auto' reads outputs/v2/oracle/equivalence_choice.json winner; else M1|M3|M5|M7|M8.",
    )
    p.add_argument("--eq-k", type=int, default=40)
    p.add_argument("--oracle-cache", type=Path, default=None, help="Default: repo/outputs/v2/oracle")
    p.add_argument("--chunk-frames", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--map-kind", type=str, default="tica2d")
    p.add_argument("--schedule", type=str, default="two_phase")
    p.add_argument("--deficit-mode", type=str, default="information_gain")
    p.add_argument("--variant", type=str, default="balanced")
    p.add_argument("--n-bins-x", type=int, default=30)
    p.add_argument("--msm-segments", type=str, default=None)
    p.add_argument("--cold-start-frame", type=int, default=None)
    p.add_argument("--extra-init-frames", type=str, default="", help="Comma-separated extra initial seeds.")
    p.add_argument("--max-time-delta-frames", type=int, default=500)
    p.add_argument("--same-segment-only", action="store_true")
    p.add_argument("--require-same-macro", action="store_true")
    p.add_argument("--macro-n-clusters", type=int, default=16)
    p.add_argument("--max-neighbors-per-anchor", type=int, default=40)
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--run-random-summary", action="store_true")
    p.add_argument("--allow-deterministic-reseed", action="store_true")
    p.add_argument("--eq-mask-m3-by-calibration", action="store_true", help="M3 only: mask k-NN by d_close cap from calibration JSON.")
    p.add_argument(
        "--eq-m3-calibration-json",
        type=Path,
        default=None,
        help="Default: <oracle-cache>/m3_similarity_thresholds.json",
    )
    p.add_argument("--build-eq-only", action="store_true")

    args = p.parse_args(argv)
    repo = args.repo_root.resolve()
    out = (repo / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    oracle_cache = args.oracle_cache or (repo / "outputs" / "v2" / "oracle")
    oracle_cache.mkdir(parents=True, exist_ok=True)

    eq_raw = args.eq_method.strip()
    fallback = EquivalenceMethod.M3_TORSION
    eq_note = ""
    if eq_raw.lower() == "auto":
        eq_m, eq_note = load_equivalence_choice_method(oracle_cache, fallback=fallback)
    else:
        eq_m = EquivalenceMethod(eq_raw)
    if args.build_eq_only:
        path = build_and_save(repo, oracle_cache, eq_m, args.eq_k, n_clusters=args.macro_n_clusters)
        print(f"Built equivalence index: {path}")
        return

    extra = []
    if args.extra_init_frames.strip():
        extra = [int(x) for x in args.extra_init_frames.split(",") if x.strip()]
    kw = dict(
        eq_k=args.eq_k,
        oracle_cache=oracle_cache,
        segments_spec=args.msm_segments,
        cold_start_frame=args.cold_start_frame,
        initial_seeds_extra=extra,
        max_time_delta_frames=args.max_time_delta_frames,
        same_segment_only=args.same_segment_only,
        require_same_macro=args.require_same_macro,
        macro_n_clusters=args.macro_n_clusters,
        max_neighbors_per_anchor=args.max_neighbors_per_anchor,
        milestones=MilestoneTargets(),
        allow_deterministic_reseed=args.allow_deterministic_reseed,
        eq_mask_m3_by_calibration=args.eq_mask_m3_by_calibration,
        eq_m3_calibration_json=args.eq_m3_calibration_json,
    )

    cfg = ReplayConfig(
        repo_root=repo,
        output_dir=out,
        chunk_frames=args.chunk_frames,
        n_steps=args.n_steps,
        random_repeats=args.random_repeats,
        n_bins_x=args.n_bins_x,
        random_seed=args.random_seed,
        helix_nhb_threshold=3,
        open_nhb_threshold=1,
        variant=args.variant,
        schedule=args.schedule,
        deficit_mode=args.deficit_mode,
        prior_quality="none",
        prior_strength=0.5,
        map_kind=args.map_kind,
        n_bins_y=None,
        softmax_temperature=0.0,
    )

    policy = PolicyName(args.policy)
    metrics_df, cand_df, dec_df, summ = run_v2_benchmark(cfg, policy=policy, eq_method=eq_m, **kw)

    metrics_df.to_csv(out / "v2_metrics.csv", index=False)
    cand_df.to_csv(out / "candidate_pool_sizes.csv", index=False)
    dec_df.to_csv(out / "seed_decisions.csv", index=False)
    summary_path = out / "summary.json"
    merge = {
        "final_row": metrics_df.iloc[-1].to_dict(),
        **summ,
        "equivalence_method": eq_m.value,
        "equivalence_resolution": eq_note or "explicit_cli",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merge, f, indent=2)

    rand_df = None
    rand_sum = None
    if args.run_random_summary:
        bench_kw = {k: v for k, v in kw.items() if k not in {"cold_start_frame", "segments_spec", "oracle_cache", "eq_k"}}
        rand_df, rand_sum = _random_baseline_matching(
            cfg,
            eq_m,
            args.eq_k,
            oracle_cache,
            args.msm_segments,
            args.cold_start_frame,
            args.random_repeats,
            benchmark_kwargs=bench_kw,
        )
        rand_df.to_csv(out / "random_metrics.csv", index=False)
        rand_sum.to_csv(out / "random_summary.csv", index=False)

    compass_sum = summarize_random_df(metrics_df.assign(repeat=0)) if len(metrics_df) else None

    plot_curves(metrics_df.assign(repeat=0), rand_sum if rand_sum is not None else compass_sum, out)

    print(f"Wrote v2 benchmark to {out}")


if __name__ == "__main__":
    main()
