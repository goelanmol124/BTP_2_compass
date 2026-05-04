"""Animated side-by-side COMPASS vs random MSM replay on the source-derived FES.

Layout:
  - Two panels show the latent bin grid ( transferred TICA or SPIB from A10→A8Q2 )
    with the source atlas free-energy score as background,
    mean backbone H-bonds per bin as contours (interpretable chemistry),
    and the running MSM-emulated visitation histogram from offline replay.
  - Strategy sidebar summarizes policy hyperparameters and the latest scoring reason.
  - Bottom strip plots JSD and coverage vs step for both policies.

This mirrors ``src.compass.replay_online.run_msm_branch`` and ``_run_msm_random`` logic
so the animation matches the numerical experiments.

Examples:
  python scripts/compass_msm_sampling_animation.py --n-steps 40 --output outputs/animations/msm_compass.gif
  python scripts/compass_msm_sampling_animation.py --format mp4 --fps 2 --n-steps 60
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from src.compass.msm import build_msm, simulate_walk, state_flat, unflatten_state
from src.compass.online_state import parse_msm_segments_arg
from src.compass.replay_enhanced import (
    FRAME_SPACING_NS,
    ReplayConfig,
    _hist2d_xy,
    _jsd,
    _coverage,
    _softmax_choice,
    build_atlas,
)
from src.compass.replay_online import _scores_for_msm, _parse_initial_seeds


@dataclass
class MSMStepSnap:
    step: int
    cumulative_hist_c: np.ndarray
    cumulative_hist_r: np.ndarray
    last_path_c: np.ndarray
    last_path_r: np.ndarray
    compass_reason: str
    compass_bin: tuple[int, int]
    compass_score: float
    weights_c: dict[str, float]


def _n_bins_xy(atlas) -> tuple[int, int]:
    return int(len(atlas.x_edges) - 1), int(len(atlas.y_edges) - 1)


def _bin_indices_compat(atlas) -> tuple[np.ndarray, np.ndarray]:
    from src.compass.replay_enhanced import _bin_indices

    xy = atlas.target_xy
    return _bin_indices(xy, atlas.x_edges, atlas.y_edges)


def _msm_metrics_hist(
    hist: np.ndarray,
    ref_hist: np.ndarray,
    atlas,
    target_nhb: np.ndarray,
    full_bx: np.ndarray,
    full_by: np.ndarray,
    nx: int,
    ny: int,
    helix_th: int,
    open_th: int,
) -> dict[str, float]:
    from src.compass.msm import microstate_observables

    f_helix_s, f_open_s, _ = microstate_observables(target_nhb, full_bx, full_by, nx, ny, helix_th, open_th)
    flat = hist.ravel()
    tot = flat.sum()
    if tot <= 0:
        jsd_v, cov_v = 1.0, 0.0
        w = flat
    else:
        jsd_v = _jsd(ref_hist, hist)
        cov_v = _coverage(ref_hist, hist)
        w = flat / tot
    f_helix = float(np.sum(w * f_helix_s)) if tot > 0 else 0.0
    f_open = float(np.sum(w * f_open_s)) if tot > 0 else 0.0
    f_helix_ref = float(np.mean(target_nhb >= helix_th))
    f_open_ref = float(np.mean(target_nhb <= open_th))
    visited = hist > 0.0
    from src.compass.replay_enhanced import _important_source_mask

    imp = _important_source_mask(atlas)
    return {
        "jsd": jsd_v,
        "coverage": cov_v,
        "important_coverage": float(np.mean(visited[imp])) if np.any(imp) else 0.0,
        "f_helix_error": abs(f_helix - f_helix_ref) if tot > 0 else abs(f_helix_ref),
        "f_open_error": abs(f_open - f_open_ref) if tot > 0 else abs(f_open_ref),
    }


def _mean_nhb_grid(target_nhb: np.ndarray, bx: np.ndarray, by: np.ndarray, nx: int, ny: int) -> np.ndarray:
    s = np.zeros((nx, ny), dtype=float)
    c = np.zeros((nx, ny), dtype=float)
    for i in range(len(target_nhb)):
        s[int(bx[i]), int(by[i])] += float(target_nhb[i])
        c[int(bx[i]), int(by[i])] += 1.0
    return np.divide(s, np.maximum(c, 1.0))


def run_paired_msm_snapshots(
    cfg: ReplayConfig,
    atlas,
    target_nhb: np.ndarray,
    segments: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[list[MSMStepSnap], dict[str, np.ndarray]]:
    rng_c = np.random.default_rng(cfg.random_seed)
    rng_r = np.random.default_rng(cfg.random_seed + 3000)

    n_frames = len(target_nhb)
    nx, ny = _n_bins_xy(atlas)
    n_by = ny
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    lag = int(args.msm_lag) if args.msm_lag else cfg.chunk_frames
    chunk_frames = cfg.chunk_frames
    n_msm_steps = max(1, chunk_frames // lag)

    full_bx, full_by = _bin_indices_compat(atlas)
    t_mat, msm_meta = build_msm(full_bx, full_by, nx, ny, lag, segments)

    observ_c: set[int] = set()
    observ_r: set[int] = set()
    attempts_c = np.zeros_like(atlas.source_hist, dtype=int)
    hist_c = np.zeros((nx, ny), dtype=float)
    hist_r = np.zeros((nx, ny), dtype=float)
    lag_w = float(lag)
    selected_c: list[int] = []

    snaps: list[MSMStepSnap] = []
    jsd_c: list[float] = []
    jsd_r: list[float] = []
    cov_c: list[float] = []
    cov_r: list[float] = []

    def apply_path(hist: np.ndarray, path: np.ndarray) -> None:
        for s in path:
            bx, by = unflatten_state(int(s), n_by)
            hist[bx, by] += lag_w

    last_path_c = np.array([0], dtype=int)
    last_path_r = np.array([0], dtype=int)
    reason = "init_seed"
    score_v = float("nan")
    weights_out: dict[str, float] = {}
    bin_pick = (0, 0)

    init_idx = _parse_initial_seeds(args.initial_seeds, n_frames)
    first_s0: int | None = None

    # --- Compass init (possibly multiple seeds) ---
    for fi in init_idx:
        s0 = int(state_flat(full_bx[fi : fi + 1], full_by[fi : fi + 1], n_by)[0])
        if first_s0 is None:
            first_s0 = s0
        path = simulate_walk(t_mat, s0, n_msm_steps, rng_c)
        observ_c.update(int(x) for x in path.tolist())
        selected_c.append(s0)
        bx, by = unflatten_state(s0, n_by)
        attempts_c[bx, by] += 1
        apply_path(hist_c, path)
        last_path_c = path.copy()

    # --- Random init ---
    for fi in init_idx:
        s0 = int(state_flat(full_bx[fi : fi + 1], full_by[fi : fi + 1], n_by)[0])
        path = simulate_walk(t_mat, s0, n_msm_steps, rng_r)
        observ_r.update(int(x) for x in path.tolist())
        apply_path(hist_r, path)
        last_path_r = path.copy()

    n_steps = cfg.n_steps

    def push_snap(step: int) -> None:
        mc = _msm_metrics_hist(
            hist_c, ref_hist, atlas, target_nhb, full_bx, full_by, nx, ny, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
        )
        mr = _msm_metrics_hist(
            hist_r, ref_hist, atlas, target_nhb, full_bx, full_by, nx, ny, cfg.helix_nhb_threshold, cfg.open_nhb_threshold
        )
        jsd_c.append(mc["jsd"])
        jsd_r.append(mr["jsd"])
        cov_c.append(mc["coverage"])
        cov_r.append(mr["coverage"])
        snaps.append(
            MSMStepSnap(
                step=step,
                cumulative_hist_c=hist_c.copy(),
                cumulative_hist_r=hist_r.copy(),
                last_path_c=last_path_c.copy(),
                last_path_r=last_path_r.copy(),
                compass_reason=reason,
                compass_bin=bin_pick,
                compass_score=score_v,
                weights_c=dict(weights_out),
            )
        )

    bxf0, byf0 = unflatten_state(int(first_s0 or 0), n_by)
    bin_pick = (int(bxf0), int(byf0))
    push_snap(1)

    for step in range(2, n_steps + 1):
        cand = np.array(sorted(observ_c), dtype=int)
        if cand.size == 0:
            break
        fraction_done = (step - 1) / max(float(n_steps), 1.0)
        score, _comp, weights, weighted = _scores_for_msm(
            atlas, cand, selected_c, attempts_c, cfg, fraction_done, n_by
        )
        pick = _softmax_choice(score, np.ones(len(cand), dtype=bool), rng_c, cfg.softmax_temperature)
        s_pick = int(cand[pick])
        selected_c.append(s_pick)
        path = simulate_walk(t_mat, s_pick, n_msm_steps, rng_c)
        observ_c.update(int(x) for x in path.tolist())
        bx, by = unflatten_state(s_pick, n_by)
        attempts_c[bx, by] += 1
        apply_path(hist_c, path)
        last_path_c = path.copy()
        contrib = {k: float(weighted[k][pick]) for k in weighted}
        pos = {k: v for k, v in contrib.items() if v > 0}
        reason = max(pos.items(), key=lambda kv: kv[1])[0] if pos else "penalty_limited"
        score_v = float(score[pick])
        weights_out = {k: float(weights[k]) for k in weights}
        bin_pick = (int(bx), int(by))

        # Random step
        cand_r = np.array(sorted(observ_r), dtype=int)
        if cand_r.size > 0:
            s_r = int(rng_r.choice(cand_r))
            path_r = simulate_walk(t_mat, s_r, n_msm_steps, rng_r)
            observ_r.update(int(x) for x in path_r.tolist())
            apply_path(hist_r, path_r)
            last_path_r = path_r.copy()

        push_snap(step)

    meta = {
        "msm": msm_meta,
        "ref_hist": ref_hist,
        "mean_nhb": _mean_nhb_grid(target_nhb, full_bx, full_by, nx, ny),
        "jsd_compass": np.array(jsd_c),
        "jsd_random": np.array(jsd_r),
        "cov_compass": np.array(cov_c),
        "cov_random": np.array(cov_r),
    }
    return snaps, meta


def _draw_heatmap_panel(
    ax,
    atlas,
    fe_bg: np.ndarray,
    mean_nhb: np.ndarray,
    visit: np.ndarray,
    path: np.ndarray,
    n_by: int,
    title: str,
    vmax_visit: float,
):
    xe, ye = atlas.x_edges, atlas.y_edges
    # Background FES
    ax.pcolormesh(xe, ye, fe_bg.T, shading="auto", cmap="bone_r", rasterized=True)
    ax.set_title(title, fontsize=11)
    # Visitation overlay
    v = np.log1p(visit.T)
    ax.pcolormesh(xe, ye, v, shading="auto", cmap="YlOrRd", alpha=0.45, vmin=0.0, vmax=np.log1p(vmax_visit), rasterized=True)

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xg, yg = np.meshgrid(xc, yc, indexing="ij")
    try:
        cs = ax.contour(
            xg,
            yg,
            mean_nhb,
            levels=6,
            colors="cyan",
            linewidths=0.8,
            alpha=0.85,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    except ValueError:
        pass

    if path is not None and len(path) > 0:
        px = [0.5 * (xe[unflatten_state(int(s), n_by)[0]] + xe[unflatten_state(int(s), n_by)[0] + 1]) for s in path]
        py = [0.5 * (ye[unflatten_state(int(s), n_by)[1]] + ye[unflatten_state(int(s), n_by)[1] + 1]) for s in path]
        ax.plot(px, py, "w-", linewidth=2.0, alpha=0.95)
        ax.scatter(px, py, c="yellow", s=28, zorder=5, edgecolors="k", linewidths=0.4)
        ax.scatter([px[0]], [py[0]], c="lime", s=55, zorder=6, edgecolors="k", linewidths=0.5, marker="*")

    ax.set_xlabel("Latent dim 1 (transferred)", fontsize=9)
    ax.set_ylabel("Latent dim 2 (transferred)", fontsize=9)
    ax.set_aspect("auto")


def render_animation(
    cfg: ReplayConfig,
    atlas,
    snaps: list[MSMStepSnap],
    meta: dict[str, np.ndarray],
    out: Path,
    fps: float,
    fmt: str,
) -> None:
    if not snaps:
        raise ValueError("No snapshots to animate")
    fe = np.asarray(atlas.free_energy_score, dtype=float)
    mean_nhb = meta["mean_nhb"]
    n_by = fe.shape[1]
    vmax_visit = max(float(np.max(s.cumulative_hist_c)) for s in snaps)
    vmax_visit = max(vmax_visit, max(float(np.max(s.cumulative_hist_r)) for s in snaps), 1.0)

    fig = plt.figure(figsize=(14, 9), dpi=120)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3.2, 1.0], width_ratios=[1.15, 1.15, 0.85], hspace=0.35, wspace=0.32)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, :2])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")

    _draw_heatmap_panel(ax_c, atlas, fe, mean_nhb, snaps[0].cumulative_hist_c, snaps[0].last_path_c, n_by, "COMPASS (MSM)", vmax_visit)
    _draw_heatmap_panel(ax_r, atlas, fe, mean_nhb, snaps[0].cumulative_hist_r, snaps[0].last_path_r, n_by, "Random (same initial seeds)", vmax_visit)

    line_c_j, = ax_curve.plot([], [], "b-", label="COMPASS JSD", linewidth=1.8)
    line_r_j, = ax_curve.plot([], [], "r--", label="Random JSD", linewidth=1.8)
    ax_cov = ax_curve.twinx()
    line_c_c, = ax_cov.plot([], [], color="steelblue", linestyle=":", label="COMPASS coverage", linewidth=1.5, alpha=0.85)
    line_r_c, = ax_cov.plot([], [], color="salmon", linestyle=":", label="Random coverage", linewidth=1.5, alpha=0.85)
    ax_curve.set_xlabel("Decision step")
    ax_curve.set_ylabel("Jensen–Shannon vs full target")
    ax_cov.set_ylabel("Histogram coverage")
    ax_curve.set_ylim(0.0, 1.0)
    ax_cov.set_ylim(0.0, 1.0)
    ax_curve.grid(True, alpha=0.3)
    h1, l1 = ax_curve.get_legend_handles_labels()
    h2, l2 = ax_cov.get_legend_handles_labels()
    ax_curve.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    side_text = ax_side.text(
        0.02,
        0.98,
        "",
        transform=ax_side.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        wrap=True,
    )
    ax_side.set_axis_off()

    leg = (
        "Cyan contours: mean heavy-atom H-bonds / frame in each bin (A8Q2).\n"
        "Background: normalized source free-energy prior on the same latent grid.\n"
        "Yellow path: last MSM walk from the chosen macrostate; green star: walk start.\n"
        "Orange heat: log-scaled emulated sampling mass (lag-weighted bin visits).\n"
        "Random matches initial seeds and MSM lag/steps, but picks seeds uniformly from the observed pool."
    )
    ax_leg.text(0.0, 1.0, leg, va="top", ha="left", fontsize=8, linespacing=1.35)

    n_frames_anim = len(snaps)

    def init():
        return (line_c_j, line_r_j, line_c_c, line_r_c)

    def update(i: int):
        snap = snaps[i]
        ax_c.clear()
        ax_r.clear()
        _draw_heatmap_panel(ax_c, atlas, fe, mean_nhb, snap.cumulative_hist_c, snap.last_path_c, n_by, "COMPASS (MSM)", vmax_visit)
        _draw_heatmap_panel(ax_r, atlas, fe, mean_nhb, snap.cumulative_hist_r, snap.last_path_r, n_by, "Random (same initial seeds)", vmax_visit)
        steps_axis = np.arange(1, len(meta["jsd_compass"]) + 1)
        line_c_j.set_data(steps_axis[: i + 1], meta["jsd_compass"][: i + 1])
        line_r_j.set_data(steps_axis[: i + 1], meta["jsd_random"][: i + 1])
        line_c_c.set_data(steps_axis[: i + 1], meta["cov_compass"][: i + 1])
        line_r_c.set_data(steps_axis[: i + 1], meta["cov_random"][: i + 1])
        ax_curve.set_xlim(1, max(snaps[-1].step, 2))
        total_ns = float(np.sum(snap.cumulative_hist_c)) * FRAME_SPACING_NS
        wtxt = snap.weights_c
        wlines = "\n".join(f"  {k}: {v:.3f}" for k, v in sorted(wtxt.items(), key=lambda kv: -abs(kv[1]))[:8])
        side_text.set_text(
            f"Step {snap.step} / {snaps[-1].step}\n"
            f"Emulated budget ≈ {total_ns:.1f} ns\n"
            f"Map: {atlas.map_label}\n"
            f"Variant: {cfg.variant}  |  schedule: {cfg.schedule}\n"
            f"Deficit: {cfg.deficit_mode}\n"
            f"Softmax T: {cfg.softmax_temperature}\n"
            f"Restart cost: {cfg.restart_cost}\n"
            f"Novelty: {cfg.novelty_mode}\n"
            f"Trust ctrl: {cfg.trust_controller}\n"
            f"--- last COMPASS pick ---\n"
            f"Bin (x,y): {snap.compass_bin}\n"
            f"Top term: {snap.compass_reason}\n"
            f"Score: {snap.compass_score:.4f}\n"
            f"Score weights:\n{wlines if wlines else '  (init)'}\n"
        )
        fig.suptitle(
            f"Offline MSM replay — {atlas.map_label}  |  chunk={cfg.chunk_frames} frames, lag={int(meta['msm']['lag'])}",
            fontsize=12,
            y=0.98,
        )
        return (line_c_j, line_r_j, line_c_c, line_r_c)

    anim = FuncAnimation(fig, update, frames=n_frames_anim, init_func=init, blit=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "gif":
        anim.save(out, writer=PillowWriter(fps=fps))
    elif fmt == "mp4":
        anim.save(out, writer="ffmpeg", fps=fps)
    else:
        raise ValueError(fmt)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=_REPO)
    p.add_argument("--output", type=Path, default=_REPO / "outputs/animations/msm_compass_vs_random.gif")
    p.add_argument("--format", choices=["gif", "mp4"], default="gif")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--chunk-frames", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--n-bins-x", type=int, default=30)
    p.add_argument("--n-bins-y", type=int, default=None)
    p.add_argument("--map-kind", default="tica2d")
    p.add_argument("--variant", default="balanced")
    p.add_argument("--schedule", default="two_phase")
    p.add_argument("--deficit-mode", default="information_gain")
    p.add_argument("--prior-quality", default="none")
    p.add_argument("--prior-strength", type=float, default=0.5)
    p.add_argument("--softmax-temperature", type=float, default=0.0)
    p.add_argument("--trust-controller", default="none")
    p.add_argument("--trust-threshold", type=float, default=0.65)
    p.add_argument("--restart-cost", type=float, default=0.0)
    p.add_argument("--novelty-mode", default="distance")
    p.add_argument("--bin-edges", default="source_only", choices=["source_only", "pooled"])
    p.add_argument("--initial-seeds", type=str, default="0")
    p.add_argument("--msm-lag", type=int, default=0)
    p.add_argument("--msm-segments", type=str, default=None)
    p.add_argument("--helix-nhb-threshold", type=int, default=3)
    p.add_argument("--open-nhb-threshold", type=int, default=1)
    p.add_argument("--write-meta", type=Path, default=None, help="Optional JSON path for run metadata")

    args = p.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.msm_lag == 0:
        args.msm_lag = args.chunk_frames

    out_path = args.output
    if out_path.suffix.lower() not in {".gif", ".mp4"}:
        out_path = out_path.with_suffix(".gif" if args.format == "gif" else ".mp4")

    cfg = ReplayConfig(
        repo_root=repo_root,
        output_dir=out_path.parent,
        chunk_frames=args.chunk_frames,
        n_steps=args.n_steps,
        random_repeats=20,
        n_bins_x=args.n_bins_x,
        random_seed=args.random_seed,
        helix_nhb_threshold=args.helix_nhb_threshold,
        open_nhb_threshold=args.open_nhb_threshold,
        variant=args.variant,
        schedule=args.schedule,
        deficit_mode=args.deficit_mode,
        prior_quality=args.prior_quality,
        prior_strength=args.prior_strength,
        softmax_temperature=args.softmax_temperature,
        map_kind=args.map_kind,
        n_bins_y=args.n_bins_y,
        trust_controller=args.trust_controller,
        trust_threshold=args.trust_threshold,
        restart_cost=args.restart_cost,
        novelty_mode=args.novelty_mode,
        bin_edges_mode=args.bin_edges,
    )

    atlas, arrays = build_atlas(cfg)
    target_nhb = arrays["a8q2_nhb"]
    segments = parse_msm_segments_arg(args.msm_segments, len(target_nhb))

    snaps, meta = run_paired_msm_snapshots(cfg, atlas, target_nhb, segments, args)
    render_animation(cfg, atlas, snaps, meta, out_path, fps=args.fps, fmt=args.format)

    if args.write_meta:
        dump = {
            "output": str(out_path),
            "n_snapshots": len(snaps),
            "msm_lag": int(meta["msm"]["lag"]),
            "map": atlas.map_label,
            "variant": cfg.variant,
        }
        args.write_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_meta, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
