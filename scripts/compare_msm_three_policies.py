"""MSM adaptive-sampling benchmark: B0–B6 vs COMPASS-full with matched compute.

Baselines share the same segment-aware MSM T, lag, atlas bins, and init seeds.
Outputs per-step metrics, first-passage tables, references (JSD noise floor), and
optional final histogram stacks for plotting.

Default output root: outputs/msm_benchmark/ (new runs). Completed benchmark tables used in this repo live under archive/outputs_legacy/msm_benchmark/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
from src.compass.online_state import parse_msm_segments_arg
from src.compass.replay_enhanced import (
    DEFAULT_HELIX_NHB_THRESHOLD,
    DEFAULT_OPEN_NHB_THRESHOLD,
    FRAME_SPACING_NS,
    ReplayConfig,
    _bin_indices,
    _coverage,
    _hist2d_xy,
    _important_source_mask,
    _jsd,
    build_atlas,
)
from src.compass.replay_online import (
    _n_bins_xy,
    _parse_initial_seeds,
    _scores_for_msm,
    _softmax_choice,
)

POLICY_IDS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")


def _mean_nhb_per_microstate(
    target_nhb: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    n_by = ny
    n_states = nx * ny
    sum_n = np.zeros(n_states, dtype=float)
    cnt = np.zeros(n_states, dtype=float)
    for i in range(len(target_nhb)):
        k = int(state_flat(full_bx[i : i + 1], full_by[i : i + 1], n_by)[0])
        if 0 <= k < n_states:
            sum_n[k] += float(target_nhb[i])
            cnt[k] += 1.0
    return np.divide(sum_n, cnt, out=np.zeros(n_states, dtype=float), where=cnt > 0)


def _nhb_wasserstein_weighted(
    hist: np.ndarray,
    mean_nhb_by_state: np.ndarray,
    target_nhb: np.ndarray,
) -> float:
    flat = hist.ravel().astype(float)
    tot = flat.sum()
    if tot <= 0:
        return 1.0
    idx = np.flatnonzero(flat > 1e-15)
    if idx.size == 0:
        return 1.0
    vals = mean_nhb_by_state[idx]
    w = flat[idx] / tot
    return float(
        wasserstein_distance(
            target_nhb.astype(float),
            vals.astype(float),
            v_weights=w,
        )
    )


def _row_metrics(
    hist: np.ndarray,
    ref_hist: np.ndarray,
    atlas,
    target_nhb: np.ndarray,
    cfg: ReplayConfig,
    f_helix_s: np.ndarray,
    f_open_s: np.ndarray,
    mean_nhb_by_state: np.ndarray,
) -> dict[str, float]:
    flat = hist.ravel()
    tot = flat.sum()
    if tot <= 0:
        jsd_v, cov_v = 1.0, 0.0
        f_helix = f_open = 0.0
    else:
        jsd_v = _jsd(ref_hist, hist)
        cov_v = _coverage(ref_hist, hist)
        w = flat / tot
        f_helix = float(np.sum(w * f_helix_s.ravel()))
        f_open = float(np.sum(w * f_open_s.ravel()))
    f_helix_ref = float(np.mean(target_nhb >= cfg.helix_nhb_threshold))
    f_open_ref = float(np.mean(target_nhb <= cfg.open_nhb_threshold))
    visited = hist > 0.0
    imp = _important_source_mask(atlas)
    nhb_w = _nhb_wasserstein_weighted(hist, mean_nhb_by_state.reshape(-1), target_nhb)
    denom_imp = float(atlas.source_prob[imp].sum()) if np.any(imp) else 0.0
    if denom_imp <= 0:
        src_mass_imp_visited = 0.0
    else:
        src_mass_imp_visited = float(
            np.sum(atlas.source_prob * visited.astype(float) * imp.astype(float))
        ) / denom_imp

    return {
        "jsd": float(jsd_v),
        "coverage": float(cov_v),
        "important_coverage": float(np.mean(visited[imp])) if np.any(imp) else 0.0,
        "source_mass_visited": float(atlas.source_prob[visited].sum()),
        "source_mass_important_visited_frac": float(src_mass_imp_visited),
        "f_helix_est": float(f_helix) if tot > 0 else 0.0,
        "f_open_est": float(f_open) if tot > 0 else 0.0,
        "f_helix_error": float(abs(f_helix - f_helix_ref)) if tot > 0 else abs(f_helix_ref),
        "f_open_error": float(abs(f_open - f_open_ref)) if tot > 0 else abs(f_open_ref),
        "f_helix_ref": float(f_helix_ref),
        "f_open_ref": float(f_open_ref),
        "nhb_wasserstein": float(nhb_w),
        "sampled_ns": float(np.sum(hist)) * FRAME_SPACING_NS,
        "hist_sum_lag_units": float(np.sum(hist)),
    }


def _update_first_passage(
    fp: dict[tuple[int, int], float],
    hist: np.ndarray,
    sampled_ns: float,
    important_bins: list[tuple[int, int]],
) -> None:
    for bx, by in important_bins:
        if hist[bx, by] > 0 and (bx, by) not in fp:
            fp[(bx, by)] = sampled_ns


def _bin_centers_2d(atlas) -> tuple[np.ndarray, np.ndarray]:
    nx = len(atlas.x_edges) - 1
    ny = len(atlas.y_edges) - 1
    cx = (atlas.x_edges[:-1] + atlas.x_edges[1:]) / 2.0
    cy = (atlas.y_edges[:-1] + atlas.y_edges[1:]) / 2.0
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    return gx, gy


def _state_center_xy(
    s: int, n_by: int, gx: np.ndarray, gy: np.ndarray
) -> tuple[float, float]:
    bx, by = unflatten_state(int(s), n_by)
    return float(gx[bx, by]), float(gy[bx, by])


def run_policy_adaptive(
    policy: str,
    cfg: ReplayConfig,
    atlas,
    target_nhb: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    t_mat: np.ndarray,
    nx: int,
    ny: int,
    lag: int,
    n_msm_steps: int,
    initial_seeds: str,
    rng: np.random.Generator,
    ref_hist: np.ndarray,
    f_helix_s: np.ndarray,
    f_open_s: np.ndarray,
    mean_nhb_by_state: np.ndarray,
    important_bins: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[tuple[int, int], float]]:
    """B2 uniform, B3 min-count, B4 farthest, B5 pure_frontier COMPASS, B6 full COMPASS."""
    n_by = ny
    n_frames = len(target_nhb)
    lag_weight = float(lag)
    gx, gy = _bin_centers_2d(atlas)

    run_cfg = replace(cfg, variant="pure_frontier", schedule="static") if policy == "B5" else cfg

    observed_states: set[int] = set()
    cumulative_hist = np.zeros((nx, ny), dtype=float)
    bin_attempts = np.zeros_like(atlas.source_hist, dtype=int)
    selected_states: list[int] = []
    selected_restarts_b4: list[int] = []
    rows: list[dict[str, Any]] = []
    first_pass: dict[tuple[int, int], float] = {}

    def apply_path(path: np.ndarray) -> None:
        for s in path:
            bx, by = unflatten_state(int(s), n_by)
            cumulative_hist[bx, by] += lag_weight

    init_idx = _parse_initial_seeds(initial_seeds, n_frames)
    n_steps = min(cfg.n_steps, 10**9)

    def record_row(step: int) -> None:
        m = _row_metrics(
            cumulative_hist,
            ref_hist,
            atlas,
            target_nhb,
            cfg,
            f_helix_s,
            f_open_s,
            mean_nhb_by_state,
        )
        _update_first_passage(first_pass, cumulative_hist, m["sampled_ns"], important_bins)
        rows.append({"step": step, **m})

    step_idx = 0
    for fi in init_idx:
        step_idx += 1
        s0 = int(state_flat(full_bx[fi : fi + 1], full_by[fi : fi + 1], n_by)[0])
        path = simulate_walk(t_mat, s0, n_msm_steps, rng)
        observed_states.update(int(x) for x in path.tolist())
        if policy in ("B6", "B5"):
            selected_states.append(s0)
            bx, by = unflatten_state(s0, n_by)
            bin_attempts[bx, by] += 1
        if policy == "B4":
            selected_restarts_b4.append(s0)
        apply_path(path)
        record_row(step_idx)

    for _ in range(n_steps - 1):
        cand = np.array(sorted(observed_states), dtype=int)
        if cand.size == 0:
            break

        fraction_done = step_idx / max(float(n_steps), 1.0)

        if policy == "B2":
            pick = int(rng.choice(cand))
        elif policy == "B3":
            masses = np.array([cumulative_hist[unflatten_state(int(s), n_by)] for s in cand])
            min_m = float(np.min(masses))
            ties = np.flatnonzero(masses <= min_m + 1e-15)
            pick = int(cand[int(rng.choice(ties))])
        elif policy == "B4":
            restart_xy = np.array(
                [_state_center_xy(s, n_by, gx, gy) for s in cand],
                dtype=float,
            )
            if len(selected_restarts_b4) == 0:
                pick = int(rng.choice(cand))
            else:
                pts = np.array(
                    [_state_center_xy(s, n_by, gx, gy) for s in selected_restarts_b4],
                    dtype=float,
                )
                sel_xy = np.mean(pts, axis=0, keepdims=True)
                d2 = np.sum((restart_xy - sel_xy) ** 2, axis=1)
                pick = int(cand[int(np.argmax(d2))])
            selected_restarts_b4.append(pick)
        elif policy in ("B5", "B6"):
            score, _, _, _ = _scores_for_msm(
                atlas, cand, selected_states, bin_attempts, run_cfg, fraction_done, n_by
            )
            pick_ix = _softmax_choice(
                score, np.ones(len(cand), dtype=bool), rng, cfg.softmax_temperature
            )
            pick = int(cand[pick_ix])
            selected_states.append(pick)
            bx, by = unflatten_state(pick, n_by)
            bin_attempts[bx, by] += 1
        else:
            raise ValueError(policy)

        path = simulate_walk(t_mat, pick, n_msm_steps, rng)
        observed_states.update(int(x) for x in path.tolist())
        apply_path(path)
        step_idx += 1
        record_row(step_idx)

    return rows, cumulative_hist, first_pass


def run_policy_b1_replicas(
    cfg: ReplayConfig,
    atlas,
    target_nhb: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    t_mat: np.ndarray,
    nx: int,
    ny: int,
    lag: int,
    n_msm_steps: int,
    initial_seeds: str,
    rng: np.random.Generator,
    ref_hist: np.ndarray,
    f_helix_s: np.ndarray,
    f_open_s: np.ndarray,
    mean_nhb_by_state: np.ndarray,
    important_bins: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[tuple[int, int], float]]:
    n_by = ny
    n_frames = len(target_nhb)
    lag_weight = float(lag)
    init_idx = _parse_initial_seeds(initial_seeds, n_frames)
    n_steps = min(cfg.n_steps, 10**9)
    n_walks = len(init_idx) + n_steps - 1

    cumulative_hist = np.zeros((nx, ny), dtype=float)
    rows: list[dict[str, Any]] = []
    first_pass: dict[tuple[int, int], float] = {}

    def apply_path(path: np.ndarray) -> None:
        for s in path:
            bx, by = unflatten_state(int(s), n_by)
            cumulative_hist[bx, by] += lag_weight

    for w in range(n_walks):
        fi = init_idx[w % len(init_idx)]
        s0 = int(state_flat(full_bx[fi : fi + 1], full_by[fi : fi + 1], n_by)[0])
        path = simulate_walk(t_mat, s0, n_msm_steps, rng)
        apply_path(path)
        m = _row_metrics(
            cumulative_hist,
            ref_hist,
            atlas,
            target_nhb,
            cfg,
            f_helix_s,
            f_open_s,
            mean_nhb_by_state,
        )
        _update_first_passage(first_pass, cumulative_hist, m["sampled_ns"], important_bins)
        rows.append({"step": w + 1, **m})

    return rows, cumulative_hist, first_pass


def run_policy_b0_long_sliced(
    cfg: ReplayConfig,
    atlas,
    target_nhb: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    t_mat: np.ndarray,
    nx: int,
    ny: int,
    lag: int,
    n_msm_steps: int,
    initial_seeds: str,
    rng: np.random.Generator,
    ref_hist: np.ndarray,
    f_helix_s: np.ndarray,
    f_open_s: np.ndarray,
    mean_nhb_by_state: np.ndarray,
    important_bins: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[tuple[int, int], float]]:
    n_by = ny
    n_frames = len(target_nhb)
    lag_weight = float(lag)
    init_idx = _parse_initial_seeds(initial_seeds, n_frames)
    fi0 = init_idx[0]
    first_s0 = int(state_flat(full_bx[fi0 : fi0 + 1], full_by[fi0 : fi0 + 1], n_by)[0])
    n_steps = min(cfg.n_steps, 10**9)
    n_walks = len(init_idx) + n_steps - 1
    total_transitions = int(n_walks * n_msm_steps)

    path = simulate_walk(t_mat, first_s0, total_transitions, rng)
    cumulative_hist = np.zeros((nx, ny), dtype=float)
    rows: list[dict[str, Any]] = []
    first_pass: dict[tuple[int, int], float] = {}

    for w in range(n_walks):
        lo = w * n_msm_steps
        hi = (w + 1) * n_msm_steps + 1
        segment = path[lo:hi]
        for s in segment:
            bx, by = unflatten_state(int(s), n_by)
            cumulative_hist[bx, by] += lag_weight
        m = _row_metrics(
            cumulative_hist,
            ref_hist,
            atlas,
            target_nhb,
            cfg,
            f_helix_s,
            f_open_s,
            mean_nhb_by_state,
        )
        _update_first_passage(first_pass, cumulative_hist, m["sampled_ns"], important_bins)
        rows.append({"step": w + 1, **m})

    return rows, cumulative_hist, first_pass


def _bootstrap_jsd_noise_floor(
    target_xy: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    ref_full = _hist2d_xy(target_xy, x_edges, y_edges)
    n = len(target_xy)
    mid = n // 2
    h_a = _hist2d_xy(target_xy[:mid], x_edges, y_edges)
    h_b = _hist2d_xy(target_xy[mid:], x_edges, y_edges)
    split_jsd = float(_jsd(ref_full, 0.5 * (h_a + h_b)))

    jsds: list[float] = []
    for _ in range(n_bootstrap):
        perm = rng.permutation(n)
        half = n // 2
        ha = _hist2d_xy(target_xy[perm[:half]], x_edges, y_edges)
        hb = _hist2d_xy(target_xy[perm[half:]], x_edges, y_edges)
        jsds.append(_jsd(ha, hb))
    arr = np.array(jsds, dtype=float)
    return {
        "jsd_split_halves_vs_full_mix": split_jsd,
        "jsd_bootstrap_mean": float(np.mean(arr)),
        "jsd_bootstrap_p95": float(np.percentile(arr, 95)),
        "jsd_bootstrap_p50": float(np.percentile(arr, 50)),
        "n_bootstrap": int(n_bootstrap),
    }


def build_references(
    atlas,
    target_nhb: np.ndarray,
    target_xy: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    nx: int,
    ny: int,
    cfg: ReplayConfig,
    rng: np.random.Generator,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    ref_hist = _hist2d_xy(target_xy, atlas.x_edges, atlas.y_edges)
    f_helix_ref = float(np.mean(target_nhb >= cfg.helix_nhb_threshold))
    f_open_ref = float(np.mean(target_nhb <= cfg.open_nhb_threshold))
    imp = _important_source_mask(atlas)
    important_bins = [(int(bx), int(by)) for bx, by in zip(*np.where(imp))]
    noise = _bootstrap_jsd_noise_floor(target_xy, atlas.x_edges, atlas.y_edges, n_bootstrap, rng)
    f_helix_s, f_open_s, _ = microstate_observables(
        target_nhb, full_bx, full_by, nx, ny, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
    )
    return {
        "f_helix_GT": f_helix_ref,
        "f_open_GT": f_open_ref,
        "ref_hist_shape": list(ref_hist.shape),
        "source_prob_sum_important": float(atlas.source_prob[imp].sum()) if np.any(imp) else 0.0,
        "n_important_bins": int(np.sum(imp)),
        "important_bins": important_bins,
        "noise_floor": noise,
        "f_helix_s_shape": list(f_helix_s.shape),
        "f_open_s_shape": list(f_open_s.shape),
    }


@dataclass(frozen=True)
class Preset:
    name: str
    map_kind: str
    chunk_frames: int
    n_steps: int
    variant: str
    schedule: str
    initial_seeds: str


PRESETS: list[Preset] = [
    Preset("01_tica2d_baseline_m01", "tica2d", 10, 80, "balanced", "two_phase", "0"),
    Preset("02_tica2d_chunk20_m12", "tica2d", 20, 40, "balanced", "two_phase", "0"),
    Preset("03_spib1d_m20", "spib1d", 10, 80, "balanced", "two_phase", "0"),
    Preset("04_tica2d_long200_m64", "tica2d", 10, 200, "balanced", "two_phase", "0"),
    Preset("05_tica2d_two_seed_m02", "tica2d", 10, 80, "balanced", "two_phase", "auto"),
]


def _parse_policy_list(s: str, with_b5: bool) -> list[str]:
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    valid: list[str] = []
    for p in parts:
        if p not in POLICY_IDS:
            raise ValueError(f"Unknown policy {p}, expected one of {POLICY_IDS}")
        valid.append(p)
    if with_b5 and "B5" not in valid:
        valid.append("B5")
    order = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
    return sorted(set(valid), key=lambda x: order.index(x))


POLICY_INDEX = {p: i for i, p in enumerate(["B0", "B1", "B2", "B3", "B4", "B5", "B6"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/msm_benchmark"))
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--n-bins-x", type=int, default=30)
    parser.add_argument("--n-bins-y", type=int, default=None)
    parser.add_argument("--msm-segments", type=str, default=None)
    parser.add_argument("--deficit-mode", default="information_gain")
    parser.add_argument("--prior-quality", default="none")
    parser.add_argument("--softmax-temperature", type=float, default=0.0)
    parser.add_argument(
        "--policies",
        type=str,
        default="B0,B1,B2,B3,B4,B6",
        help="Comma-separated policy ids",
    )
    parser.add_argument(
        "--with-b5",
        action="store_true",
        help="Include B5 pure_frontier COMPASS in addition to --policies",
    )
    parser.add_argument("--save-hists", action="store_true", help="Write final_hists.npz per policy dir")
    parser.add_argument("--noise-bootstrap", type=int, default=200)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_root = (repo_root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    active_policies = _parse_policy_list(args.policies, args.with_b5)

    all_long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metric_keys_final = [
        "jsd",
        "coverage",
        "important_coverage",
        "source_mass_visited",
        "f_helix_error",
        "f_open_error",
        "nhb_wasserstein",
        "sampled_ns",
    ]

    ref_rng = np.random.default_rng(args.random_seed)

    for preset in PRESETS:
        chunk_frames = preset.chunk_frames
        lag = chunk_frames
        n_msm_steps = max(1, chunk_frames // lag)

        cfg = ReplayConfig(
            repo_root=repo_root,
            output_dir=out_root / preset.name,
            chunk_frames=chunk_frames,
            n_steps=preset.n_steps,
            random_repeats=args.random_repeats,
            n_bins_x=args.n_bins_x,
            random_seed=args.random_seed,
            helix_nhb_threshold=DEFAULT_HELIX_NHB_THRESHOLD,
            open_nhb_threshold=DEFAULT_OPEN_NHB_THRESHOLD,
            variant=preset.variant,
            schedule=preset.schedule,
            deficit_mode=args.deficit_mode,
            prior_quality=args.prior_quality,
            prior_strength=0.5,
            compass_repeats=1,
            softmax_temperature=args.softmax_temperature,
            map_kind=preset.map_kind,
            n_bins_y=args.n_bins_y,
            bootstrap_random=True,
            trust_controller="none",
            trust_threshold=0.65,
            restart_cost=0.0,
            novelty_mode="distance",
            bin_edges_mode="source_only",
        )

        atlas, arrays = build_atlas(cfg)
        target_nhb = arrays["a8q2_nhb"]
        target_xy = atlas.target_xy
        segments = parse_msm_segments_arg(args.msm_segments, len(target_nhb))
        nx, ny = _n_bins_xy(atlas)
        full_bx, full_by = _bin_indices(target_xy, atlas.x_edges, atlas.y_edges)
        t_mat, msm_meta = build_msm(full_bx, full_by, nx, ny, lag, segments)
        ref_hist = _hist2d_xy(target_xy, atlas.x_edges, atlas.y_edges)
        f_helix_s, f_open_s, _ = microstate_observables(
            target_nhb, full_bx, full_by, nx, ny, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
        )
        mean_nhb_by_state = _mean_nhb_per_microstate(target_nhb, full_bx, full_by, nx, ny)

        ref_data = build_references(
            atlas, target_nhb, target_xy, full_bx, full_by, nx, ny, cfg, ref_rng, args.noise_bootstrap
        )
        important_bins = [(int(a), int(b)) for a, b in ref_data["important_bins"]]

        ref_dir = out_root / "references" / preset.name
        ref_dir.mkdir(parents=True, exist_ok=True)
        with open(ref_dir / "references.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in ref_data.items() if k != "important_bins"}, f, indent=2)
        np.savez_compressed(
            ref_dir / "references.npz",
            ref_hist=ref_hist,
            source_prob=np.asarray(atlas.source_prob, dtype=float),
            important_mask=np.asarray(_important_source_mask(atlas), dtype=bool),
            f_helix_s=f_helix_s,
            f_open_s=f_open_s,
            mean_nhb_by_state=mean_nhb_by_state.reshape(nx, ny),
        )

        policy_label = {
            "B0": "B0_long_msm",
            "B1": "B1_replicas_from_s0",
            "B2": "B2_uniform_adaptive",
            "B3": "B3_min_count_adaptive",
            "B4": "B4_farthest_point_adaptive",
            "B5": "B5_compass_pure_frontier",
            "B6": "B6_compass_full",
        }

        row_base = {
            "preset": preset.name,
            "map_kind": preset.map_kind,
            "chunk_frames": chunk_frames,
            "n_steps": preset.n_steps,
            "initial_seeds": preset.initial_seeds,
            "msm_lag": lag,
            "n_msm_steps": n_msm_steps,
        }

        preset_dir = out_root / preset.name
        preset_dir.mkdir(parents=True, exist_ok=True)
        with open(preset_dir / "msm_build_meta.json", "w", encoding="utf-8") as f:
            json.dump(msm_meta, f, indent=2)

        for pol in active_policies:
            pol_dir = preset_dir / pol
            pol_dir.mkdir(parents=True, exist_ok=True)
            fp_rows_all: list[dict[str, Any]] = []
            hists: list[np.ndarray] = []

            for rep in range(args.random_repeats):
                rng = np.random.default_rng(args.random_seed + 50_000 + rep * 20 + POLICY_INDEX[pol])

                if pol == "B0":
                    rows, hist, fp = run_policy_b0_long_sliced(
                        cfg,
                        atlas,
                        target_nhb,
                        full_bx,
                        full_by,
                        t_mat,
                        nx,
                        ny,
                        lag,
                        n_msm_steps,
                        preset.initial_seeds,
                        rng,
                        ref_hist,
                        f_helix_s,
                        f_open_s,
                        mean_nhb_by_state,
                        important_bins,
                    )
                elif pol == "B1":
                    rows, hist, fp = run_policy_b1_replicas(
                        cfg,
                        atlas,
                        target_nhb,
                        full_bx,
                        full_by,
                        t_mat,
                        nx,
                        ny,
                        lag,
                        n_msm_steps,
                        preset.initial_seeds,
                        rng,
                        ref_hist,
                        f_helix_s,
                        f_open_s,
                        mean_nhb_by_state,
                        important_bins,
                    )
                else:
                    rows, hist, fp = run_policy_adaptive(
                        pol,
                        cfg,
                        atlas,
                        target_nhb,
                        full_bx,
                        full_by,
                        t_mat,
                        nx,
                        ny,
                        lag,
                        n_msm_steps,
                        preset.initial_seeds,
                        rng,
                        ref_hist,
                        f_helix_s,
                        f_open_s,
                        mean_nhb_by_state,
                        important_bins,
                    )

                for r in rows:
                    all_long_rows.append(
                        {
                            **row_base,
                            "policy": pol,
                            "policy_label": policy_label[pol],
                            "repeat": rep,
                            **r,
                        }
                    )

                for (bx, by), ns in fp.items():
                    fp_rows_all.append(
                        {
                            "repeat": rep,
                            "bin_x": bx,
                            "bin_y": by,
                            "first_sampled_ns": ns,
                        }
                    )

                if args.save_hists:
                    hists.append(hist.copy())

            df_step = pd.DataFrame(
                [x for x in all_long_rows if x["preset"] == preset.name and x["policy"] == pol]
            )
            df_step.to_csv(pol_dir / "per_step_metrics.csv", index=False)
            pd.DataFrame(fp_rows_all).to_csv(pol_dir / "first_passage.csv", index=False)

            if args.save_hists and hists:
                np.savez_compressed(pol_dir / "final_hists.npz", hists=np.stack(hists, axis=0))

            last_per_rep = (
                df_step.sort_values("step").groupby("repeat", as_index=False).tail(1)
            )
            for _, lr in last_per_rep.iterrows():
                summary_rows.append(
                    {
                        **row_base,
                        "policy": pol,
                        "policy_label": policy_label[pol],
                        **{k: lr[k] for k in metric_keys_final if k in lr},
                    }
                )

    df_all = pd.DataFrame(all_long_rows)
    try:
        df_all.to_parquet(out_root / "per_step_all.parquet")
    except Exception:
        df_all.to_csv(out_root / "per_step_all.csv", index=False)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_root / "benchmark_final_step_summary.csv", index=False)

    if len(df_sum) > 0:
        means = df_sum.groupby(["preset", "policy"])[metric_keys_final].mean().reset_index()
        stds = df_sum.groupby(["preset", "policy"])[metric_keys_final].std().reset_index()
        stds = stds.rename(columns={c: f"{c}_std" for c in metric_keys_final})
        summary_wide = means.merge(stds, on=["preset", "policy"])
        summary_wide.to_csv(out_root / "three_policy_summary.csv", index=False)

    with open(out_root / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "B0: single long MSM walk, sliced into n_walks segments for per-step curves; "
            "transitions = n_walks * n_msm_steps.\n"
            "B1: n_walks independent walks, start round-robins initial_seeds.\n"
            "B2: uniform adaptive over observed states.\n"
            "B3: min cumulative_hist mass adaptive.\n"
            "B4: farthest-point adaptive from restart centroid.\n"
            "B5: COMPASS pure_frontier.\n"
            "B6: COMPASS full (balanced two_phase).\n"
            "sampled_ns = sum(cumulative_hist) * FRAME_SPACING_NS.\n"
        )

    print(f"Wrote benchmark under {out_root}")


if __name__ == "__main__":
    main()
