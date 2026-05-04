"""Online-faithful COMPASS replay: seed only from observed mutant frames.

.. deprecated:: COMPASS-v2

    The ``oracle`` environment (chunk library replay) is diagnostic only for v2 headlines.
    The ``msm`` environment is retained as **ablation-only** MSM emulation; headline seed
    benchmarks use :mod:`src.compass.v2` (see ``COMPASS_PLAN.md``).

Environments:
- oracle: legacy chunk replay (diagnostic; same as replay_enhanced).
- trajectory_replay: deterministic continuation from cached A8Q2 (diagnostic).
- msm: segment-aware Markov emulator (primary offline validation).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance

from src.compass.msm import (
    build_msm,
    microstate_observables,
    simulate_walk,
    state_flat,
    unflatten_state,
)
from src.compass.online_state import ObservedTargetState, TrajectorySegments, parse_msm_segments_arg
from src.compass.replay_enhanced import (
    DEFAULT_HELIX_NHB_THRESHOLD,
    DEFAULT_OPEN_NHB_THRESHOLD,
    FRAME_SPACING_NS,
    Atlas,
    ReplayConfig,
    _apply_trust_controller,
    _bin_indices,
    _chunk_slices,
    _coverage,
    _hist2d_xy,
    _important_source_mask,
    _jsd,
    _mmd_rbf,
    _prior_agreement,
    _softmax_choice,
    _source_deficit_grid,
    _summarize_compass,
    _summarize_random,
    _weights_for_step,
    _zscore,
    build_atlas,
    plot_component_decomposition,
    plot_curves,
    plot_reason_counts,
    run_compass_replay,
    run_random_replays,
    write_first_passage,
)


def _n_bins_xy(atlas: Atlas) -> tuple[int, int]:
    return int(len(atlas.x_edges) - 1), int(len(atlas.y_edges) - 1)


def _evaluate_observed_frames(
    idx: np.ndarray,
    xy: np.ndarray,
    ref_hist: np.ndarray,
    atlas: Atlas,
    nhb: np.ndarray,
    helix_th: int,
    open_th: int,
) -> dict[str, float]:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        empty = np.empty((0, xy.shape[1]), dtype=float)
        return {
            "jsd": 1.0,
            "latent_wasserstein": 1.0,
            "latent_mmd": 1.0,
            "coverage": 0.0,
            "important_coverage": 0.0,
            "source_mass_visited": 0.0,
            "f_helix": 0.0,
            "f_helix_error": abs(float(np.mean(nhb >= helix_th))),
            "f_open": 0.0,
            "f_open_error": abs(float(np.mean(nhb <= open_th))),
        }
    sample_xy = xy[idx]
    hist = _hist2d_xy(sample_xy, atlas.x_edges, atlas.y_edges)
    f_helix = float(np.mean(nhb[idx] >= helix_th))
    f_open = float(np.mean(nhb[idx] <= open_th))
    f_helix_ref = float(np.mean(nhb >= helix_th))
    f_open_ref = float(np.mean(nhb <= open_th))
    visited = hist > 0.0
    imp = _important_source_mask(atlas)

    def _lw(ref: np.ndarray, sample: np.ndarray) -> float:
        if sample.size == 0:
            return 1.0
        dims = min(ref.shape[1], sample.shape[1], 2)
        vals = [wasserstein_distance(ref[:, i], sample[:, i]) for i in range(dims)]
        return float(np.mean(vals))

    return {
        "jsd": _jsd(ref_hist, hist),
        "latent_wasserstein": _lw(xy, sample_xy),
        "latent_mmd": _mmd_rbf(xy, sample_xy),
        "coverage": _coverage(ref_hist, hist),
        "important_coverage": float(np.mean(visited[imp])) if np.any(imp) else 0.0,
        "source_mass_visited": float(atlas.source_prob[visited].sum()),
        "f_helix": f_helix,
        "f_helix_error": abs(f_helix - f_helix_ref),
        "f_open": f_open,
        "f_open_error": abs(f_open - f_open_ref),
    }


def _compass_scores_online(
    atlas: Atlas,
    candidate_frames: np.ndarray,
    selected_seed_frames: list[int],
    bin_attempts: np.ndarray,
    cfg: ReplayConfig,
    fraction_done: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    cf = np.asarray(candidate_frames, dtype=int)
    xy = atlas.target_xy[cf]
    bx, by = _bin_indices(xy, atlas.x_edges, atlas.y_edges)
    source_boundary = atlas.free_energy_score[bx, by]
    trust = atlas.source_support[bx, by].astype(float)
    unvisited = 1.0 / (1.0 + bin_attempts[bx, by].astype(float))
    failure_decay = unvisited.copy()
    source_representative = _zscore(atlas.source_prob[bx, by])
    if selected_seed_frames:
        deficit_grid = _source_deficit_grid(atlas, bin_attempts, cfg.deficit_mode)
        source_deficit = _zscore(deficit_grid[bx, by])
    else:
        source_deficit = source_representative.copy()

    frontier_xy = atlas.frontier_xy if atlas.frontier_xy is not None else atlas.target_xy
    acq_frontier = frontier_xy[cf, :2]
    if selected_seed_frames:
        sf = np.array(selected_seed_frames, dtype=int)
        sel_f = frontier_xy[sf, :2]
        d2 = np.sum((acq_frontier[:, None, :] - sel_f[None, :, :]) ** 2, axis=2)
        distance_novelty = _zscore(np.sqrt(np.min(d2, axis=1)))
        center = sel_f.mean(axis=0, keepdims=True)
        frontier = _zscore(np.sqrt(np.sum((acq_frontier - center) ** 2, axis=1)))
        redundancy = np.exp(-np.min(d2, axis=1) / 0.02)
        if cfg.novelty_mode == "distance":
            novelty = distance_novelty
        elif cfg.novelty_mode == "epistemic":
            uncertainty_grid = np.sqrt(atlas.source_prob * (1.0 - atlas.source_prob)) / np.sqrt(
                1.0 + bin_attempts.astype(float)
            )
            novelty = _zscore(uncertainty_grid[bx, by]) * unvisited
        else:
            raise ValueError(cfg.novelty_mode)
    else:
        novelty = np.ones(len(cf), dtype=float)
        frontier = np.ones(len(cf), dtype=float)
        redundancy = np.zeros(len(cf), dtype=float)

    frontier_disagreement = frontier * (1.0 - trust)
    components: dict[str, np.ndarray] = {
        "source_deficit": source_deficit,
        "source_representative": source_representative,
        "source_boundary_trust": source_boundary * trust,
        "frontier_disagreement": frontier_disagreement,
        "unvisited": unvisited,
        "novelty": novelty,
        "redundancy": redundancy,
        "failure_decay": failure_decay,
        "trust": trust,
        "prior_trust": np.full(len(cf), _prior_agreement(atlas, bin_attempts), dtype=float),
        "source_boundary": source_boundary,
    }
    weights = _weights_for_step(cfg.variant, cfg.schedule, fraction_done)
    weights, trust_factor = _apply_trust_controller(weights, cfg, atlas, bin_attempts)
    weighted = {k: weights[k] * components[k] for k in weights}
    raw_score = np.zeros(len(cf), dtype=float)
    for arr in weighted.values():
        raw_score += arr
    if cfg.restart_cost > 0.0 and selected_seed_frames:
        last = np.array(selected_seed_frames[-1], dtype=int)
        last_f = frontier_xy[last, :2]
        restart_distance = np.sqrt(np.sum((acq_frontier - last_f) ** 2, axis=1))
        raw_score -= cfg.restart_cost * _zscore(restart_distance)
    components["controller_trust_factor"] = np.full(len(cf), trust_factor, dtype=float)
    score = raw_score * failure_decay
    return score, components, weights, weighted


def _scores_for_msm(
    atlas: Atlas,
    cand_states: np.ndarray,
    selected_states: list[int],
    bin_attempts: np.ndarray,
    cfg: ReplayConfig,
    frac: float,
    n_by: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Score macrostates using bin (bx,by) of each state id."""

    nx, ny = _n_bins_xy(atlas)
    bx = np.array([unflatten_state(int(s), n_by)[0] for s in cand_states], dtype=int)
    by = np.array([unflatten_state(int(s), n_by)[1] for s in cand_states], dtype=int)
    source_boundary = atlas.free_energy_score[bx, by]
    trust = atlas.source_support[bx, by].astype(float)
    unvisited = 1.0 / (1.0 + bin_attempts[bx, by].astype(float))
    failure_decay = unvisited.copy()
    source_representative = _zscore(atlas.source_prob[bx, by])
    if selected_states:
        deficit_grid = _source_deficit_grid(atlas, bin_attempts, cfg.deficit_mode)
        source_deficit = _zscore(deficit_grid[bx, by])
    else:
        source_deficit = source_representative.copy()

    cx = (atlas.x_edges[bx] + atlas.x_edges[bx + 1]) / 2.0
    cy = (atlas.y_edges[by] + atlas.y_edges[by + 1]) / 2.0
    acq_frontier = np.column_stack([cx, cy])
    if selected_states:
        sb = np.array([unflatten_state(int(s), n_by)[0] for s in selected_states], dtype=int)
        sby = np.array([unflatten_state(int(s), n_by)[1] for s in selected_states], dtype=int)
        scx = (atlas.x_edges[sb] + atlas.x_edges[sb + 1]) / 2.0
        scy = (atlas.y_edges[sby] + atlas.y_edges[sby + 1]) / 2.0
        sel_f = np.column_stack([scx, scy])
        d2 = np.sum((acq_frontier[:, None, :] - sel_f[None, :, :]) ** 2, axis=2)
        distance_novelty = _zscore(np.sqrt(np.min(d2, axis=1)))
        center = sel_f.mean(axis=0, keepdims=True)
        frontier = _zscore(np.sqrt(np.sum((acq_frontier - center) ** 2, axis=1)))
        redundancy = np.exp(-np.min(d2, axis=1) / 0.02)
        if cfg.novelty_mode == "distance":
            novelty = distance_novelty
        elif cfg.novelty_mode == "epistemic":
            uncertainty_grid = np.sqrt(atlas.source_prob * (1.0 - atlas.source_prob)) / np.sqrt(
                1.0 + bin_attempts.astype(float)
            )
            novelty = _zscore(uncertainty_grid[bx, by]) * unvisited
        else:
            raise ValueError(cfg.novelty_mode)
    else:
        novelty = np.ones(len(cand_states), dtype=float)
        frontier = np.ones(len(cand_states), dtype=float)
        redundancy = np.zeros(len(cand_states), dtype=float)

    frontier_disagreement = frontier * (1.0 - trust)
    components = {
        "source_deficit": source_deficit,
        "source_representative": source_representative,
        "source_boundary_trust": source_boundary * trust,
        "frontier_disagreement": frontier_disagreement,
        "unvisited": unvisited,
        "novelty": novelty,
        "redundancy": redundancy,
        "failure_decay": failure_decay,
        "trust": trust,
        "prior_trust": np.full(len(cand_states), _prior_agreement(atlas, bin_attempts), dtype=float),
        "source_boundary": source_boundary,
    }
    weights = _weights_for_step(cfg.variant, cfg.schedule, frac)
    weights, trust_factor = _apply_trust_controller(weights, cfg, atlas, bin_attempts)
    weighted = {k: weights[k] * components[k] for k in weights}
    raw_score = np.zeros(len(cand_states), dtype=float)
    for arr in weighted.values():
        raw_score += arr
    components["controller_trust_factor"] = np.full(len(cand_states), trust_factor, dtype=float)
    score = raw_score * failure_decay
    return score, components, weights, weighted


def _parse_initial_seeds(spec: str, n_frames: int) -> list[int]:
    if spec.strip().lower() == "auto":
        mid = n_frames // 2
        return [0, mid]
    out = []
    for p in spec.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out or [0]


def run_oracle_branch(cfg: ReplayConfig, args: argparse.Namespace) -> None:
    atlas, arrays = build_atlas(cfg)
    target_nhb = arrays["a8q2_nhb"]
    if cfg.compass_repeats <= 1:
        compass_df, decisions = run_compass_replay(
            atlas, target_nhb, cfg, start_chunk=args.start_chunk, rng=np.random.default_rng(cfg.random_seed)
        )
    else:
        n_chunks = len(_chunk_slices(len(target_nhb), cfg.chunk_frames))
        compass_dfs = []
        decisions = None
        seed_rng = np.random.default_rng(cfg.random_seed)
        for rep in range(cfg.compass_repeats):
            sub = np.random.default_rng(cfg.random_seed + rep + 1)
            st = int(seed_rng.integers(0, n_chunks))
            df, dec = run_compass_replay(atlas, target_nhb, cfg, start_chunk=st, rng=sub)
            df = df.copy()
            df["repeat"] = rep
            compass_dfs.append(df)
            if rep == 0:
                decisions = dec
        compass_df = pd.concat(compass_dfs, ignore_index=True)
    random_df = run_random_replays(atlas, target_nhb, cfg)
    random_summary = _summarize_random(random_df)
    out = cfg.output_dir
    _write_outputs_common(cfg, atlas, compass_df, decisions, random_df, random_summary, out, environment="oracle")


def run_trajectory_branch(
    cfg: ReplayConfig,
    atlas: Atlas,
    target_nhb: np.ndarray,
    segments: list[tuple[int, int]],
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(cfg.random_seed)
    n_frames = len(target_nhb)
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    seg_helper = TrajectorySegments(segments)
    state = ObservedTargetState(
        n_frames=n_frames,
        segments=seg_helper,
        seed_reservoir_stride=args.seed_reservoir_stride,
        seed_reservoir_max=args.seed_reservoir_max,
        allow_deterministic_reseed=bool(args.allow_deterministic_reseed),
    )
    init_frames = _parse_initial_seeds(args.initial_seeds, n_frames)
    selected_seed_frames: list[int] = []
    bin_attempts = np.zeros_like(atlas.source_hist, dtype=int)
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    full_bx, full_by = _bin_indices(atlas.target_xy, atlas.x_edges, atlas.y_edges)

    def metrics_row(step: int, policy: str) -> dict[str, Any]:
        idx = np.flatnonzero(state.observed_mask)
        m = _evaluate_observed_frames(
            idx, atlas.target_xy, ref_hist, atlas, target_nhb, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
        )
        return {
            "policy": policy,
            "step": step,
            "selected_chunks": step,
            "sampled_frames": int(np.sum(state.observed_mask)),
            "sampled_ns": float(np.sum(state.observed_mask)) * FRAME_SPACING_NS,
            **m,
        }

    for fi in init_frames:
        a, b = seg_helper.clip_continuation(int(fi), cfg.chunk_frames)
        state.mark_observed(a, b)
        selected_seed_frames.append(int(fi))
        bx0, by0 = int(full_bx[fi]), int(full_by[fi])
        bin_attempts[bx0, by0] += 1

    step = 1
    metric_rows.append(metrics_row(step, "COMPASS-Online"))
    decision_rows.append(
        {
            "step": step,
            "seed_frame": selected_seed_frames[0],
            "bin_x": int(full_bx[selected_seed_frames[0]]),
            "bin_y": int(full_by[selected_seed_frames[0]]),
            "score": float("nan"),
            "reason": "init_seed",
            "selected_before_count": 0,
        }
    )
    pool = state.seed_pool()
    curve_rows.append(
        {
            "step": step,
            "sampled_ns": metric_rows[-1]["sampled_ns"],
            "seed_pool_size": int(pool.size),
            "observed_frame_count": state.n_observed_frames,
            "observed_bin_count": state.observed_bins_count(full_bx, full_by),
            "used_seed_count": len(state.used_seed_frames),
        }
    )

    n_steps = min(cfg.n_steps, n_frames // max(cfg.chunk_frames, 1))
    for step in range(2, n_steps + 1):
        pool = state.seed_pool()
        if pool.size == 0:
            pool = np.flatnonzero(state.observed_mask)
        fraction_done = (step - 1) / max(float(n_steps), 1.0)
        score, comp, weights, weighted = _compass_scores_online(
            atlas, pool, selected_seed_frames,
            bin_attempts,
            cfg,
            fraction_done,
        )
        mask = np.ones(len(pool), dtype=bool)
        pick = _softmax_choice(score, mask, rng, cfg.softmax_temperature)
        seed_f = int(pool[pick])
        sbc = int(sum(1 for s in selected_seed_frames if s == seed_f))
        selected_seed_frames.append(seed_f)
        state.register_seed(seed_f)
        a, b = seg_helper.clip_continuation(seed_f, cfg.chunk_frames)
        state.mark_observed(a, b)
        bx, by = int(full_bx[seed_f]), int(full_by[seed_f])
        bin_attempts[bx, by] += 1
        contrib = {k: float(weighted[k][pick]) for k in weighted}
        pos = {k: v for k, v in contrib.items() if v > 0}
        reason = max(pos.items(), key=lambda kv: kv[1])[0] if pos else "penalty_limited"
        metric_rows.append(metrics_row(step, "COMPASS-Online"))
        decision_rows.append(
            {
                "step": step,
                "seed_frame": seed_f,
                "bin_x": bx,
                "bin_y": by,
                "score": float(score[pick]),
                "reason": reason,
                "selected_before_count": sbc,
            }
        )
        pool2 = state.seed_pool()
        curve_rows.append(
            {
                "step": step,
                "sampled_ns": metric_rows[-1]["sampled_ns"],
                "seed_pool_size": int(pool2.size),
                "observed_frame_count": state.n_observed_frames,
                "observed_bin_count": state.observed_bins_count(full_bx, full_by),
                "used_seed_count": len(state.used_seed_frames),
            }
        )

    compass_df = pd.DataFrame(metric_rows)
    decisions = pd.DataFrame(decision_rows)
    random_df = _run_trajectory_random(cfg, atlas, target_nhb, segments, args)
    random_summary = _summarize_random(random_df)
    _write_outputs_common(cfg, atlas, compass_df, decisions, random_df, random_summary, cfg.output_dir, environment="trajectory_replay")
    pd.DataFrame(curve_rows).to_csv(cfg.output_dir / "online_state_curve.csv", index=False)


def _run_trajectory_random(
    cfg: ReplayConfig,
    atlas: Atlas,
    target_nhb: np.ndarray,
    segments: list[tuple[int, int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    n_frames = len(target_nhb)
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    seg_helper = TrajectorySegments(segments)
    rows: list[dict[str, Any]] = []
    full_bx, full_by = _bin_indices(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    n_steps = min(cfg.n_steps, n_frames // max(cfg.chunk_frames, 1))

    for rep in range(cfg.random_repeats):
        rng = np.random.default_rng(cfg.random_seed + 1000 + rep)
        state = ObservedTargetState(
            n_frames=n_frames,
            segments=seg_helper,
            seed_reservoir_stride=args.seed_reservoir_stride,
            seed_reservoir_max=args.seed_reservoir_max,
            allow_deterministic_reseed=bool(args.allow_deterministic_reseed),
        )
        init_frames = _parse_initial_seeds(args.initial_seeds, n_frames)
        selected: list[int] = []
        for fi in init_frames:
            a, b = seg_helper.clip_continuation(int(fi), cfg.chunk_frames)
            state.mark_observed(a, b)
            selected.append(int(fi))
        for step in range(1, n_steps + 1):
            idx = np.flatnonzero(state.observed_mask)
            m = _evaluate_observed_frames(
                idx, atlas.target_xy, ref_hist, atlas, target_nhb, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
            )
            rows.append(
                {
                    "policy": "Random-Online",
                    "repeat": rep,
                    "step": step,
                    "selected_chunks": step,
                    "sampled_frames": int(np.sum(state.observed_mask)),
                    "sampled_ns": float(np.sum(state.observed_mask)) * FRAME_SPACING_NS,
                    **m,
                }
            )
            if step >= n_steps:
                break
            pool = state.seed_pool()
            if pool.size == 0:
                pool = np.flatnonzero(state.observed_mask)
            seed_f = int(rng.choice(pool))
            state.register_seed(seed_f)
            a, b = seg_helper.clip_continuation(seed_f, cfg.chunk_frames)
            state.mark_observed(a, b)
    return pd.DataFrame(rows)


def run_msm_branch(
    cfg: ReplayConfig,
    atlas: Atlas,
    target_nhb: np.ndarray,
    segments: list[tuple[int, int]],
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(cfg.random_seed)
    n_frames = len(target_nhb)
    nx, ny = _n_bins_xy(atlas)
    n_by = ny
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    lag = int(args.msm_lag) if args.msm_lag else cfg.chunk_frames
    chunk_frames = cfg.chunk_frames
    n_msm_steps = max(1, chunk_frames // lag)
    if chunk_frames % lag != 0:
        sys.stderr.write(f"Warning: chunk_frames={chunk_frames} not divisible by msm_lag={lag}; using {n_msm_steps} MSM steps\n")

    full_bx, full_by = _bin_indices(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    t_mat, msm_meta = build_msm(full_bx, full_by, nx, ny, lag, segments)
    f_helix_s, f_open_s, occ_s = microstate_observables(
        target_nhb, full_bx, full_by, nx, ny, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
    )
    msm_meta["msm_lag_frames"] = lag
    msm_meta["chunk_frames"] = chunk_frames
    msm_meta["n_msm_steps_per_decision"] = n_msm_steps
    with open(cfg.output_dir / "msm_metadata.json", "w", encoding="utf-8") as f:
        json.dump(msm_meta, f, indent=2)

    observed_states: set[int] = set()
    bin_attempts = np.zeros_like(atlas.source_hist, dtype=int)
    cumulative_hist = np.zeros((nx, ny), dtype=float)
    lag_weight = float(lag)
    selected_states: list[int] = []

    def apply_path(path: np.ndarray) -> None:
        nonlocal cumulative_hist
        for _i, s in enumerate(path):
            bx, by = unflatten_state(int(s), n_by)
            cumulative_hist[bx, by] += lag_weight

    def msm_metrics(hist: np.ndarray) -> dict[str, float]:
        flat = hist.ravel()
        tot = flat.sum()
        if tot <= 0:
            jsd_v, cov_v = 1.0, 0.0
        else:
            jsd_v = _jsd(ref_hist, hist)
            cov_v = _coverage(ref_hist, hist)
        w = flat / max(tot, 1e-12)
        f_helix = float(np.sum(w * f_helix_s))
        f_open = float(np.sum(w * f_open_s))
        f_helix_ref = float(np.mean(target_nhb >= cfg.helix_nhb_threshold))
        f_open_ref = float(np.mean(target_nhb <= cfg.open_nhb_threshold))
        visited = hist > 0.0
        imp = _important_source_mask(atlas)
        return {
            "jsd": jsd_v,
            "latent_wasserstein": 0.0,
            "latent_mmd": 0.0,
            "coverage": cov_v,
            "important_coverage": float(np.mean(visited[imp])) if np.any(imp) else 0.0,
            "source_mass_visited": float(atlas.source_prob[visited].sum()),
            "f_helix": f_helix,
            "f_helix_error": abs(f_helix - f_helix_ref),
            "f_open": f_open,
            "f_open_error": abs(f_open - f_open_ref),
        }

    init_idx = _parse_initial_seeds(args.initial_seeds, n_frames)
    metric_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    first_s0: int | None = None
    for fi in init_idx:
        s0 = int(state_flat(full_bx[fi : fi + 1], full_by[fi : fi + 1], n_by)[0])
        if first_s0 is None:
            first_s0 = s0
        path = simulate_walk(t_mat, s0, n_msm_steps, rng)
        observed_states.update(int(x) for x in path.tolist())
        selected_states.append(s0)
        bx, by = unflatten_state(s0, n_by)
        bin_attempts[bx, by] += 1
        apply_path(path)

    n_steps = min(cfg.n_steps, 10**9)
    total_ns = float(np.sum(cumulative_hist)) * FRAME_SPACING_NS
    m = msm_metrics(cumulative_hist)
    bxf0, byf0 = unflatten_state(int(first_s0 or 0), n_by)
    metric_rows.append(
        {
            "policy": "COMPASS-Online-MSM",
            "step": 1,
            "selected_chunks": 1,
            "sampled_frames": int(np.sum(cumulative_hist) / max(lag_weight, 1.0)),
            "sampled_ns": total_ns,
            **m,
        }
    )
    decision_rows.append(
        {
            "step": 1,
            "seed_state": int(first_s0 or 0),
            "score": float("nan"),
            "reason": "init_seed",
            "bin_x": int(bxf0),
            "bin_y": int(byf0),
            "selected_before_count": 0,
        }
    )
    curve_rows.append(
        {
            "step": 1,
            "sampled_ns": total_ns,
            "seed_pool_size": len(observed_states),
            "observed_frame_count": int(np.sum(cumulative_hist) / max(lag_weight, 1.0)),
            "observed_bin_count": int(np.sum(cumulative_hist > 0)),
            "used_seed_count": len(selected_states),
        }
    )

    for step in range(2, n_steps + 1):
        cand = np.array(sorted(observed_states), dtype=int)
        if cand.size == 0:
            break
        fraction_done = (step - 1) / max(float(n_steps), 1.0)
        score, comp, weights, weighted = _scores_for_msm(
            atlas, cand, selected_states, bin_attempts, cfg, fraction_done, n_by
        )
        pick = _softmax_choice(score, np.ones(len(cand), dtype=bool), rng, cfg.softmax_temperature)
        s_pick = int(cand[pick])
        sbc = int(sum(1 for s in selected_states if s == s_pick))
        selected_states.append(s_pick)
        path = simulate_walk(t_mat, s_pick, n_msm_steps, rng)
        observed_states.update(int(x) for x in path.tolist())
        bx, by = unflatten_state(s_pick, n_by)
        bin_attempts[bx, by] += 1
        apply_path(path)
        contrib = {k: float(weighted[k][pick]) for k in weighted}
        pos = {k: v for k, v in contrib.items() if v > 0}
        reason = max(pos.items(), key=lambda kv: kv[1])[0] if pos else "penalty_limited"
        total_ns = float(np.sum(cumulative_hist)) * FRAME_SPACING_NS
        m = msm_metrics(cumulative_hist)
        metric_rows.append(
            {
                "policy": "COMPASS-Online-MSM",
                "step": step,
                "selected_chunks": step,
                "sampled_frames": int(np.sum(cumulative_hist) / max(lag_weight, 1.0)),
                "sampled_ns": total_ns,
                **m,
            }
        )
        decision_rows.append(
            {
                "step": step,
                "seed_state": s_pick,
                "score": float(score[pick]),
                "reason": reason,
                "bin_x": bx,
                "bin_y": by,
                "selected_before_count": sbc,
            }
        )
        curve_rows.append(
            {
                "step": step,
                "sampled_ns": total_ns,
                "seed_pool_size": len(observed_states),
                "observed_frame_count": int(np.sum(cumulative_hist) / max(lag_weight, 1.0)),
                "observed_bin_count": int(np.sum(cumulative_hist > 0)),
                "used_seed_count": len(selected_states),
            }
        )

    compass_df = pd.DataFrame(metric_rows)
    decisions = pd.DataFrame(decision_rows)
    random_df = _run_msm_random(cfg, atlas, target_nhb, t_mat, full_bx, full_by, nx, ny, lag, n_msm_steps, segments, args)
    random_summary = _summarize_random(random_df)
    _write_outputs_common(cfg, atlas, compass_df, decisions, random_df, random_summary, cfg.output_dir, environment="msm")
    pd.DataFrame(curve_rows).to_csv(cfg.output_dir / "online_state_curve.csv", index=False)

    s0_boot = int(state_flat(full_bx[init_idx[0] : init_idx[0] + 1], full_by[init_idx[0] : init_idx[0] + 1], n_by)[0])
    boot_rows = []
    for bi in range(min(50, max(1, cfg.random_repeats))):
        br = np.random.default_rng(cfg.random_seed + 5000 + bi)
        p = simulate_walk(t_mat, s0_boot, n_msm_steps, br)
        h = np.zeros((nx, ny), dtype=float)
        for s in p:
            bx, by = unflatten_state(int(s), n_by)
            h[bx, by] += lag_weight
        boot_rows.append({"repeat": bi, **msm_metrics(h)})
    pd.DataFrame(boot_rows).to_csv(cfg.output_dir / "msm_bootstrap_curves.csv", index=False)


def _run_msm_random(
    cfg: ReplayConfig,
    atlas: Atlas,
    target_nhb: np.ndarray,
    t_mat: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    nx: int,
    ny: int,
    lag: int,
    n_msm_steps: int,
    segments: list[tuple[int, int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    n_by = ny
    lag_weight = float(lag)
    rows: list[dict[str, Any]] = []

    def msm_metrics(hist: np.ndarray) -> dict[str, float]:
        ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
        f_helix_s, f_open_s, _ = microstate_observables(
            target_nhb, full_bx, full_by, nx, ny, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
        )
        flat = hist.ravel()
        tot = flat.sum()
        w = flat / max(tot, 1e-12) if tot > 0 else flat
        f_helix = float(np.sum(w * f_helix_s))
        f_open = float(np.sum(w * f_open_s))
        f_helix_ref = float(np.mean(target_nhb >= cfg.helix_nhb_threshold))
        f_open_ref = float(np.mean(target_nhb <= cfg.open_nhb_threshold))
        visited = hist > 0.0
        imp = _important_source_mask(atlas)
        jsd_v = _jsd(ref_hist, hist) if tot > 0 else 1.0
        cov_v = _coverage(ref_hist, hist) if tot > 0 else 0.0
        return {
            "jsd": jsd_v,
            "latent_wasserstein": 0.0,
            "latent_mmd": 0.0,
            "coverage": cov_v,
            "important_coverage": float(np.mean(visited[imp])) if np.any(imp) else 0.0,
            "source_mass_visited": float(atlas.source_prob[visited].sum()),
            "f_helix": f_helix,
            "f_helix_error": abs(f_helix - f_helix_ref),
            "f_open": f_open,
            "f_open_error": abs(f_open - f_open_ref),
        }

    n_steps = cfg.n_steps
    for rep in range(cfg.random_repeats):
        rng = np.random.default_rng(cfg.random_seed + 3000 + rep)
        observed: set[int] = set()
        cumulative_hist = np.zeros((nx, ny), dtype=float)
        init_idx = _parse_initial_seeds(args.initial_seeds, len(target_nhb))
        for fi in init_idx:
            s0 = int(state_flat(full_bx[fi : fi + 1], full_by[fi : fi + 1], n_by)[0])
            path = simulate_walk(t_mat, s0, n_msm_steps, rng)
            observed.update(int(x) for x in path.tolist())
            for s in path:
                bx, by = unflatten_state(int(s), n_by)
                cumulative_hist[bx, by] += lag_weight
        for step in range(1, n_steps + 1):
            m = msm_metrics(cumulative_hist)
            rows.append(
                {
                    "policy": "Random-Online-MSM",
                    "repeat": rep,
                    "step": step,
                    "selected_chunks": step,
                    "sampled_frames": int(np.sum(cumulative_hist) / max(lag_weight, 1.0)),
                    "sampled_ns": float(np.sum(cumulative_hist)) * FRAME_SPACING_NS,
                    **m,
                }
            )
            if step >= n_steps:
                break
            cand = np.array(sorted(observed), dtype=int)
            if cand.size == 0:
                continue
            s_pick = int(rng.choice(cand))
            path = simulate_walk(t_mat, s_pick, n_msm_steps, rng)
            observed.update(int(x) for x in path.tolist())
            for s in path:
                bx, by = unflatten_state(int(s), n_by)
                cumulative_hist[bx, by] += lag_weight
    return pd.DataFrame(rows)


def _action_set_for_environment(environment: str) -> str:
    if environment == "oracle":
        return "full_cached_target_chunk_starts_oracle"
    return "observed_target_frames_or_states_only"


def _write_outputs_common(
    cfg: ReplayConfig,
    atlas: Atlas,
    compass_df: pd.DataFrame,
    decisions: pd.DataFrame,
    random_df: pd.DataFrame,
    random_summary: pd.DataFrame,
    output_dir: Path,
    environment: str,
    msm_segments: list[list[int]] | None = None,
    action_set: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compass_df.to_csv(output_dir / "compass_metrics.csv", index=False)
    decisions.to_csv(output_dir / "seed_decisions.csv", index=False)
    random_df.to_csv(output_dir / "random_metrics.csv", index=False)
    random_summary.to_csv(output_dir / "random_summary.csv", index=False)
    plot_curves(compass_df, random_summary, output_dir)
    plot_reason_counts(decisions.rename(columns={"reason": "reason"}), output_dir)
    if any(c.startswith("weighted_") for c in decisions.columns):
        plot_component_decomposition(decisions, output_dir)
    summary_compass = _summarize_compass(compass_df)
    final_c = summary_compass.iloc[-1].to_dict()
    final_r = random_summary.iloc[-1].to_dict()
    summary: dict[str, Any] = {
        "method": "COMPASS online-faithful replay",
        "environment": environment,
        "action_set": action_set or _action_set_for_environment(environment),
        "map": atlas.map_label,
        "chunk_frames": cfg.chunk_frames,
        "variant": cfg.variant,
        "schedule": cfg.schedule,
        "deficit_mode": cfg.deficit_mode,
        "prior_quality": cfg.prior_quality,
        "frame_spacing_ns": FRAME_SPACING_NS,
        "n_steps": int(compass_df["step"].max()),
        "random_repeats": cfg.random_repeats,
        "bin_edges_mode": cfg.bin_edges_mode,
        "final_compass": final_c,
        "final_random_mean": final_r,
        "notes": [
            "Oracle mode is legacy chunk replay (diagnostic).",
            "trajectory_replay is deterministic cached continuation (diagnostic).",
            "msm uses segment-aware counts; policy does not read T during decisions.",
        ],
    }
    if msm_segments is not None:
        summary["msm_trajectory_segments"] = msm_segments
    with open(output_dir / "compass_replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if environment == "oracle":
        write_first_passage(atlas, decisions.rename(columns={"seed_frame": "start_frame"}), output_dir, cfg)


def main(argv: list[str] | None = None) -> None:
    warnings.warn(
        "src.compass.replay_online is deprecated for headline benchmarks in COMPASS v2. "
        "Use src.compass.v2.bench for primary seed-selection (oracle/msm remain ablations).",
        DeprecationWarning,
        stacklevel=1,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compass/replay_online"))
    parser.add_argument(
        "--environment",
        choices=["oracle", "trajectory_replay", "msm"],
        default="msm",
    )
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--n-bins-x", type=int, default=30)
    parser.add_argument("--n-bins-y", type=int, default=None)
    parser.add_argument("--map-kind", default="tica2d")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--helix-nhb-threshold", type=int, default=DEFAULT_HELIX_NHB_THRESHOLD)
    parser.add_argument("--open-nhb-threshold", type=int, default=DEFAULT_OPEN_NHB_THRESHOLD)
    parser.add_argument("--variant", default="balanced")
    parser.add_argument("--schedule", default="two_phase")
    parser.add_argument("--deficit-mode", default="information_gain")
    parser.add_argument("--prior-quality", default="none")
    parser.add_argument("--prior-strength", type=float, default=0.5)
    parser.add_argument("--softmax-temperature", type=float, default=0.0)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--compass-repeats", type=int, default=1)
    parser.add_argument("--trust-controller", default="none")
    parser.add_argument("--trust-threshold", type=float, default=0.65)
    parser.add_argument("--restart-cost", type=float, default=0.0)
    parser.add_argument("--novelty-mode", default="distance")
    parser.add_argument("--bin-edges", default="source_only", choices=["source_only", "pooled"])
    parser.add_argument("--initial-seeds", type=str, default="0", help="Comma frames or 'auto' for [0,N/2].")
    parser.add_argument("--msm-lag", type=int, default=0, help="MSM lag in frames; 0 means chunk_frames.")
    parser.add_argument("--msm-segments", type=str, default=None, help="'0:4750,4750:9500' or omit for half/half.")
    parser.add_argument("--seed-reservoir-stride", type=int, default=1)
    parser.add_argument("--seed-reservoir-max", type=int, default=None)
    parser.add_argument("--allow-deterministic-reseed", action="store_true")
    parser.add_argument("--no-bootstrap-random", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.environment == "msm" and args.msm_lag == 0:
        args.msm_lag = args.chunk_frames

    cfg = ReplayConfig(
        repo_root=repo_root,
        output_dir=output_dir,
        chunk_frames=args.chunk_frames,
        n_steps=args.n_steps,
        random_repeats=args.random_repeats,
        n_bins_x=args.n_bins_x,
        random_seed=args.random_seed,
        helix_nhb_threshold=args.helix_nhb_threshold,
        open_nhb_threshold=args.open_nhb_threshold,
        variant=args.variant,
        schedule=args.schedule,
        deficit_mode=args.deficit_mode,
        prior_quality=args.prior_quality,
        prior_strength=args.prior_strength,
        compass_repeats=args.compass_repeats,
        softmax_temperature=args.softmax_temperature,
        map_kind=args.map_kind,
        n_bins_y=args.n_bins_y,
        bootstrap_random=not args.no_bootstrap_random,
        trust_controller=args.trust_controller,
        trust_threshold=args.trust_threshold,
        restart_cost=args.restart_cost,
        novelty_mode=args.novelty_mode,
        bin_edges_mode=args.bin_edges,
    )

    if args.environment == "oracle":
        run_oracle_branch(cfg, args)
        print(f"Wrote oracle replay to {output_dir}")
        return

    atlas, arrays = build_atlas(cfg)
    target_nhb = arrays["a8q2_nhb"]
    segments = parse_msm_segments_arg(args.msm_segments, len(target_nhb))
    segs_json = [[int(a), int(b)] for a, b in segments]

    if args.environment == "trajectory_replay":
        run_trajectory_branch(cfg, atlas, target_nhb, segments, args)
        _write_first_passage_online(atlas, cfg, output_dir)
    else:
        run_msm_branch(cfg, atlas, target_nhb, segments, args)
        _write_first_passage_online(atlas, cfg, output_dir)

    summ_path = output_dir / "compass_replay_summary.json"
    if summ_path.is_file():
        with open(summ_path, encoding="utf-8") as f:
            summ = json.load(f)
        summ["msm_trajectory_segments"] = segs_json
        summ["initial_seed_frame_indices"] = _parse_initial_seeds(args.initial_seeds, len(target_nhb))
        if args.environment == "msm":
            summ["msm_lag"] = int(args.msm_lag)
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)

    print(f"Wrote {args.environment} replay to {output_dir}")


def _write_first_passage_online(atlas: Atlas, cfg: ReplayConfig, output_dir: Path) -> None:
    decisions = pd.read_csv(output_dir / "seed_decisions.csv")
    important = _important_source_mask(atlas)
    rows = []
    for bx, by in zip(*np.where(important)):
        if "bin_x" in decisions.columns:
            hits = decisions[(decisions["bin_x"] == bx) & (decisions["bin_y"] == by)]
        else:
            hits = pd.DataFrame()
        first_step = int(hits["step"].min()) if not hits.empty else -1
        rows.append(
            {
                "bin_x": int(bx),
                "bin_y": int(by),
                "source_prob": float(atlas.source_prob[bx, by]),
                "first_step": first_step,
                "first_ns": float(first_step * cfg.chunk_frames * FRAME_SPACING_NS) if first_step > 0 else np.nan,
                "observed_at_step": first_step,
            }
        )
    df = pd.DataFrame(rows).sort_values("source_prob", ascending=False)
    df.to_csv(output_dir / "compass_first_passage_bins.csv", index=False)


if __name__ == "__main__":
    main()
