"""Retrospective COMPASS-Lite replay on cached A10/A8Q2 feature arrays.

This is the first-layer BTP test: no DCD reading and no new MD.  It treats the
existing A8Q2 production frames as a library of short chunks, then asks whether a
source-prior policy selects chunks that reconstruct the full A8Q2 distribution
faster than random chunk acquisition.

Selection rule
--------------
The policy must not use RMSD, NHB, or helix labels for guidance.  Those are
reserved for evaluation only.  COMPASS-Lite selects chunks using only transferred
source-map coordinates derived from generic torsion features (A10 TICA here).

Important limitation
--------------------
This is an active-subset replay, not a faithful online adaptive-MD emulator: the
candidate chunks are already available on disk.  The purpose is to validate the
scoring logic and generate presentation-ready diagnostics before spending GPU
time on new target MD.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


FRAME_SPACING_NS = 0.1
DEFAULT_HELIX_NHB_THRESHOLD = 3


@dataclass(frozen=True)
class Atlas:
    """Source map used by COMPASS-Lite."""

    source_xy: np.ndarray
    target_xy: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    source_hist: np.ndarray
    source_prob: np.ndarray
    source_support: np.ndarray
    free_energy_score: np.ndarray
    support_threshold: float


@dataclass(frozen=True)
class ReplayConfig:
    repo_root: Path
    output_dir: Path
    chunk_frames: int
    n_steps: int
    random_repeats: int
    n_bins_x: int
    random_seed: int
    helix_nhb_threshold: int
    variant: str
    compass_repeats: int = 1
    softmax_temperature: float = 0.0
    map_kind: str = "tica2d"
    n_bins_y: int | None = None


def _load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def project_tica(features: np.ndarray, mean: np.ndarray, eigenvectors: np.ndarray, dim: int = 2) -> np.ndarray:
    """Project raw torsion features into the donor A10 TICA space."""

    features = np.asarray(features, dtype=float)
    mean = np.asarray(mean, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    dim = min(dim, eigenvectors.shape[1])
    return (features - mean) @ eigenvectors[:, :dim]


def _hist2d_xy(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """2D histogram in transferred latent space."""

    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return hist.astype(float)


def _jsd(hist_ref: np.ndarray, hist_sample: np.ndarray, min_count: int = 3) -> float:
    p = hist_ref.ravel().astype(float)
    q = hist_sample.ravel().astype(float)
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


def _coverage(hist_ref: np.ndarray, hist_sample: np.ndarray, min_count: int = 3) -> float:
    mask = hist_ref >= min_count
    if not np.any(mask):
        return 0.0
    return float(np.mean(hist_sample[mask] > 0.0))


def _build_edges(source_x: np.ndarray, target_x: np.ndarray, n_bins_x: int) -> np.ndarray:
    """Robust shared histogram edges for source and target projections."""

    both = np.concatenate([source_x, target_x])
    lo, hi = np.percentile(both, [0.5, 99.5])
    pad = 0.08 * max(float(hi - lo), 1e-6)
    x_edges = np.linspace(float(lo - pad), float(hi + pad), n_bins_x + 1)
    return x_edges


def _build_xy_edges(source_xy: np.ndarray, target_xy: np.ndarray, n_bins_x: int) -> tuple[np.ndarray, np.ndarray]:
    x_edges = _build_edges(source_xy[:, 0], target_xy[:, 0], n_bins_x)
    y_edges = _build_edges(source_xy[:, 1], target_xy[:, 1], n_bins_x)
    return x_edges, y_edges


def build_tica_atlas(
    repo_root: Path,
    n_bins_x: int = 30,
    n_bins_y: int | None = None,
    map_kind: str = "tica2d",
) -> tuple[Atlas, dict[str, np.ndarray]]:
    """Load cached arrays and build the A10 source atlas in transferred TICA space."""

    artifacts = repo_root / "outputs" / "artifacts"
    tica_dir = repo_root / "outputs" / "tica_artifacts_a10"

    a10_features = _load_npy(artifacts / "a10_features.npy")
    a8_features = _load_npy(artifacts / "a8q2_features.npy")
    a8_nhb = _load_npy(artifacts / "a8q2_nhb.npy")
    mean = _load_npy(tica_dir / "a10_tica_mean0.npy")
    ev = _load_npy(tica_dir / "a10_tica_singular_vectors_left.npy")

    if map_kind == "tica2d":
        source_xy = project_tica(a10_features, mean, ev, dim=2)
        target_xy = project_tica(a8_features, mean, ev, dim=2)
    elif map_kind == "tica1d":
        src1 = project_tica(a10_features, mean, ev, dim=1)
        tgt1 = project_tica(a8_features, mean, ev, dim=1)
        source_xy = np.column_stack([src1[:, 0], np.zeros(len(src1))])
        target_xy = np.column_stack([tgt1[:, 0], np.zeros(len(tgt1))])
    else:
        raise ValueError(f"Unsupported map_kind: {map_kind}")

    if n_bins_y is None:
        n_bins_y = n_bins_x
    x_edges = _build_edges(source_xy[:, 0], target_xy[:, 0], n_bins_x)
    if map_kind == "tica1d":
        y_edges = np.array([-1.0, 1.0])
    else:
        y_edges = _build_edges(source_xy[:, 1], target_xy[:, 1], n_bins_y)
    source_hist = _hist2d_xy(source_xy, x_edges, y_edges)

    smooth = source_hist + 1.0
    source_prob = smooth / smooth.sum()
    support_threshold = float(np.quantile(source_prob[source_prob > 0.0], 0.20))
    source_support = source_prob >= support_threshold

    free_energy = -np.log(source_prob)
    valid = source_support
    fe_score = np.zeros_like(source_prob, dtype=float)
    if np.any(valid):
        fe_valid = free_energy[valid]
        denom = max(float(fe_valid.max() - fe_valid.min()), 1e-12)
        fe_score[valid] = (fe_valid - fe_valid.min()) / denom

    atlas = Atlas(
        source_xy=source_xy,
        target_xy=target_xy,
        x_edges=x_edges,
        y_edges=y_edges,
        source_hist=source_hist,
        source_prob=source_prob,
        source_support=source_support,
        free_energy_score=fe_score,
        support_threshold=support_threshold,
    )
    arrays = {
        "a8q2_nhb": a8_nhb,
        "a10_features": a10_features,
        "a8q2_features": a8_features,
    }
    return atlas, arrays


def _chunk_slices(n_frames: int, chunk_frames: int) -> list[slice]:
    n_chunks = n_frames // chunk_frames
    return [slice(i * chunk_frames, (i + 1) * chunk_frames) for i in range(n_chunks)]


def _bin_indices(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bx = np.searchsorted(x_edges, xy[:, 0], side="right") - 1
    by = np.searchsorted(y_edges, xy[:, 1], side="right") - 1
    bx = np.clip(bx, 0, len(x_edges) - 2)
    by = np.clip(by, 0, len(y_edges) - 2)
    return bx.astype(int), by.astype(int)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _selected_hist(chunks: list[slice], selected: Iterable[int], xy: np.ndarray, atlas: Atlas) -> np.ndarray:
    chosen = list(selected)
    if not chosen:
        return np.zeros_like(atlas.source_hist, dtype=float)
    idx = np.concatenate([np.arange(chunks[i].start, chunks[i].stop) for i in chosen])
    return _hist2d_xy(xy[idx], atlas.x_edges, atlas.y_edges)


def _evaluate(
    chunks: list[slice],
    selected: list[int],
    xy: np.ndarray,
    ref_hist: np.ndarray,
    atlas: Atlas,
    nhb: np.ndarray,
    helix_threshold: int,
) -> dict[str, float]:
    hist = _selected_hist(chunks, selected, xy, atlas)
    if selected:
        idx = np.concatenate([np.arange(chunks[i].start, chunks[i].stop) for i in selected])
        f_helix = float(np.mean(nhb[idx] >= helix_threshold))
    else:
        f_helix = 0.0
    f_ref = float(np.mean(nhb >= helix_threshold))
    return {
        "jsd": _jsd(ref_hist, hist),
        "coverage": _coverage(ref_hist, hist),
        "f_helix": f_helix,
        "f_helix_error": abs(f_helix - f_ref),
    }


def _compass_scores(
    atlas: Atlas,
    chunk_starts: np.ndarray,
    selected: list[int],
    chunk_bins: tuple[np.ndarray, np.ndarray],
    bin_attempts: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute COMPASS-Lite score for each chunk start."""

    bx, by = chunk_bins
    source_boundary = atlas.free_energy_score[bx, by]
    trust = atlas.source_support[bx, by].astype(float)

    unvisited = (bin_attempts[bx, by] == 0).astype(float)
    overuse = bin_attempts[bx, by].astype(float)
    failure_decay = 1.0 / (1.0 + overuse)

    source_representative = _zscore(atlas.source_prob[bx, by])
    if selected:
        observed = bin_attempts.astype(float)
        observed = observed / max(float(observed.sum()), 1.0)
        desired = atlas.source_prob / atlas.source_prob.sum()
        deficit_grid = np.maximum(desired - observed, 0.0)
        source_deficit = _zscore(deficit_grid[bx, by])

    else:
        source_deficit = source_representative.copy()

    # Frontier/novelty is measured only in transferred latent space.
    acq = atlas.target_xy[chunk_starts, :2]
    if selected:
        selected_acq = acq[selected]
        d2 = np.sum((acq[:, None, :] - selected_acq[None, :, :]) ** 2, axis=2)
        novelty = _zscore(np.sqrt(np.min(d2, axis=1)))
        center = selected_acq.mean(axis=0, keepdims=True)
        frontier = _zscore(np.sqrt(np.sum((acq - center) ** 2, axis=1)))
        redundancy = np.exp(-np.min(d2, axis=1) / 0.02)
    else:
        novelty = np.ones(len(chunk_starts), dtype=float)
        frontier = np.ones(len(chunk_starts), dtype=float)
        redundancy = np.zeros(len(chunk_starts), dtype=float)

    frontier_disagreement = frontier * (1.0 - trust)

    components = {
        "source_boundary": source_boundary,
        "trust": trust,
        "frontier_disagreement": frontier_disagreement,
        "unvisited": unvisited,
        "novelty": novelty,
        "source_representative": source_representative,
        "source_deficit": source_deficit,
        "failure_decay": failure_decay,
        "redundancy": redundancy,
    }
    if variant == "exploratory":
        score = (
            1.20 * source_boundary * trust
            + 0.90 * frontier_disagreement
            + 0.75 * unvisited
            + 0.65 * novelty
            - 0.90 * redundancy
        ) * failure_decay
    elif variant == "balanced":
        score = (
            1.25 * source_deficit
            + 0.65 * source_representative
            + 0.25 * source_boundary * trust
            + 0.35 * frontier_disagreement
            + 0.25 * unvisited
            + 0.25 * novelty
            - 0.80 * redundancy
        ) * failure_decay
    elif variant == "source_only":
        score = (
            1.25 * source_deficit
            + 0.65 * source_representative
            + 0.25 * source_boundary * trust
            - 0.40 * redundancy
        ) * failure_decay
    elif variant == "pure_frontier":
        score = (
            0.80 * frontier_disagreement
            + 0.80 * unvisited
            + 0.80 * novelty
            - 0.80 * redundancy
        ) * failure_decay
    else:
        raise ValueError(f"Unknown COMPASS variant: {variant}")
    return score, components


def _softmax_choice(score: np.ndarray, remaining: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int = 5) -> int:
    masked = np.where(remaining, score, -np.inf)
    if temperature <= 0.0:
        return int(np.argmax(masked))
    valid_idx = np.flatnonzero(remaining)
    if valid_idx.size == 0:
        return int(np.argmax(masked))
    top = valid_idx[np.argsort(masked[valid_idx])[-top_k:]]
    sub = masked[top]
    sub = sub - sub.max()
    p = np.exp(sub / max(temperature, 1e-6))
    p = p / p.sum()
    return int(rng.choice(top, p=p))


def run_compass_replay(
    atlas: Atlas,
    target_nhb: np.ndarray,
    cfg: ReplayConfig,
    start_chunk: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one COMPASS-Lite retrospective replay (deterministic if temperature == 0)."""

    chunks = _chunk_slices(len(target_nhb), cfg.chunk_frames)
    n_chunks = len(chunks)
    chunk_starts = np.array([s.start for s in chunks], dtype=int)
    chunk_xy = atlas.target_xy[chunk_starts]
    chunk_bins = _bin_indices(chunk_xy, atlas.x_edges, atlas.y_edges)
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    rng = rng if rng is not None else np.random.default_rng(cfg.random_seed)

    selected: list[int] = []
    remaining = np.ones(n_chunks, dtype=bool)
    bin_attempts = np.zeros_like(atlas.source_hist, dtype=int)
    metric_rows: list[dict[str, float | int | str]] = []
    decision_rows: list[dict[str, float | int | str]] = []

    if 0 <= start_chunk < n_chunks:
        first = int(start_chunk)
        selected.append(first)
        remaining[first] = False
        bx0, by0 = int(chunk_bins[0][first]), int(chunk_bins[1][first])
        bin_attempts[bx0, by0] += 1
        metrics = _evaluate(chunks, selected, atlas.target_xy, ref_hist, atlas, target_nhb, cfg.helix_nhb_threshold)
        metric_rows.append({
            "policy": "COMPASS-Lite",
            "variant": cfg.variant,
            "step": 1,
            "selected_chunks": 1,
            "sampled_frames": cfg.chunk_frames,
            "sampled_ns": cfg.chunk_frames * FRAME_SPACING_NS,
            **metrics,
        })
        decision_rows.append({
            "step": 1,
            "chunk": first,
            "start_frame": int(chunk_starts[first]),
            "start_tica1": float(chunk_xy[first, 0]),
            "start_tica2": float(chunk_xy[first, 1]) if chunk_xy.shape[1] > 1 else 0.0,
            "eval_start_nhb": float(target_nhb[chunk_starts[first]]),
            "bin_x": bx0,
            "bin_y": by0,
            "score": float("nan"),
            "reason": "init_seed",
        })
        start_step = 2
    else:
        start_step = 1

    for step in range(start_step, min(cfg.n_steps, n_chunks) + 1):
        score, comp = _compass_scores(
            atlas,
            chunk_starts,
            selected,
            chunk_bins,
            bin_attempts,
            cfg.variant,
        )
        chosen = _softmax_choice(score, remaining, rng, cfg.softmax_temperature)
        selected.append(chosen)
        remaining[chosen] = False

        bx, by = int(chunk_bins[0][chosen]), int(chunk_bins[1][chosen])
        bin_attempts[bx, by] += 1

        reason = max(
            {
                "source_boundary": float(comp["source_boundary"][chosen] * comp["trust"][chosen]),
                "frontier_disagreement": float(comp["frontier_disagreement"][chosen]),
                "unvisited_region": float(comp["unvisited"][chosen]),
                "novelty": float(comp["novelty"][chosen]),
            }.items(),
            key=lambda kv: kv[1],
        )[0]

        metrics = _evaluate(chunks, selected, atlas.target_xy, ref_hist, atlas, target_nhb, cfg.helix_nhb_threshold)
        metric_rows.append(
            {
                "policy": "COMPASS-Lite",
                "variant": cfg.variant,
                "step": step,
                "selected_chunks": len(selected),
                "sampled_frames": len(selected) * cfg.chunk_frames,
                "sampled_ns": len(selected) * cfg.chunk_frames * FRAME_SPACING_NS,
                **metrics,
            }
        )
        decision_rows.append(
            {
                "step": step,
                "chunk": chosen,
                "start_frame": int(chunk_starts[chosen]),
                "start_tica1": float(chunk_xy[chosen, 0]),
                "start_tica2": float(chunk_xy[chosen, 1]) if chunk_xy.shape[1] > 1 else 0.0,
                "eval_start_nhb": float(target_nhb[chunk_starts[chosen]]),
                "bin_x": bx,
                "bin_y": by,
                "score": float(score[chosen]),
                "reason": reason,
                **{k: float(v[chosen]) for k, v in comp.items()},
            }
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(decision_rows)


def run_random_replays(atlas: Atlas, target_nhb: np.ndarray, cfg: ReplayConfig) -> pd.DataFrame:
    chunks = _chunk_slices(len(target_nhb), cfg.chunk_frames)
    n_chunks = len(chunks)
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    rng = np.random.default_rng(cfg.random_seed)
    rows: list[dict[str, float | int | str]] = []
    n_steps = min(cfg.n_steps, n_chunks)
    for rep in range(cfg.random_repeats):
        order = rng.permutation(n_chunks)[:n_steps]
        selected: list[int] = []
        for step, chunk in enumerate(order, start=1):
            selected.append(int(chunk))
            metrics = _evaluate(chunks, selected, atlas.target_xy, ref_hist, atlas, target_nhb, cfg.helix_nhb_threshold)
            rows.append(
                {
                    "policy": "Random",
                    "repeat": rep,
                    "step": step,
                    "selected_chunks": len(selected),
                    "sampled_frames": len(selected) * cfg.chunk_frames,
                    "sampled_ns": len(selected) * cfg.chunk_frames * FRAME_SPACING_NS,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _summarize_random(random_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["jsd", "coverage", "f_helix", "f_helix_error"]
    out = random_df.groupby(["policy", "step", "sampled_ns"], as_index=False)[metric_cols].agg(["mean", "std"])
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    return out.rename(columns={"policy_": "policy", "step_": "step", "sampled_ns_": "sampled_ns"})


def _summarize_compass(compass_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["jsd", "coverage", "f_helix", "f_helix_error"]
    out = compass_df.groupby(["step", "sampled_ns"], as_index=False)[metric_cols].agg(["mean", "std"])
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    return out.rename(columns={"step_": "step", "sampled_ns_": "sampled_ns"})


def plot_curves(compass_df: pd.DataFrame, random_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    specs = [
        ("jsd", "JSD vs A8Q2 reference", "lower is better"),
        ("coverage", "Coverage vs A8Q2 reference", "higher is better"),
        ("f_helix_error", "Helix-fraction error", "lower is better"),
    ]
    compass_summary = _summarize_compass(compass_df)
    for ax, (metric, title, ylabel) in zip(axes, specs):
        cm = compass_summary[f"{metric}_mean"]
        cs = compass_summary[f"{metric}_std"].fillna(0.0)
        cx = compass_summary["sampled_ns"]
        ax.plot(cx, cm, label="COMPASS-Lite", lw=2.4, color="#D1495B")
        if (cs > 0).any():
            ax.fill_between(cx, cm - cs, cm + cs, color="#D1495B", alpha=0.18)
        mean = random_summary[f"{metric}_mean"]
        std = random_summary[f"{metric}_std"].fillna(0.0)
        x = random_summary["sampled_ns"]
        ax.plot(x, mean, label="Random mean", color="0.25", lw=1.8)
        ax.fill_between(x, mean - std, mean + std, color="0.7", alpha=0.35, label="Random +/- 1 sd")
        ax.set_title(title)
        ax.set_xlabel("sampled ns")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output_dir / "compass_replay_curves.png", dpi=250)
    plt.close(fig)


def plot_map(atlas: Atlas, target_nhb: np.ndarray, decisions: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sc = ax.scatter(
        atlas.target_xy[:, 0],
        atlas.target_xy[:, 1],
        c=target_nhb,
        s=6,
        alpha=0.22,
        cmap="viridis",
        linewidths=0,
    )
    first = decisions.head(30)
    ax.scatter(
        first["start_tica1"],
        first["start_tica2"],
        c=first["step"],
        cmap="plasma",
        s=42,
        edgecolor="black",
        linewidth=0.4,
        label="COMPASS first 30 choices",
    )
    for _, row in first.head(12).iterrows():
        ax.text(row["start_tica1"], row["start_tica2"], str(int(row["step"])), fontsize=7)
    ax.set_xlabel("A10 transferred TICA-1")
    ax.set_ylabel("A10 transferred TICA-2")
    ax.set_title("COMPASS-Lite choices in transferred TICA space")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="NHB")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_dir / "compass_selected_probe_map.png", dpi=250)
    plt.close(fig)


def plot_reason_counts(decisions: pd.DataFrame, output_dir: Path) -> None:
    counts = decisions["reason"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_ylabel("selected chunks")
    ax.set_title("Why COMPASS-Lite selected chunks")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "compass_reason_counts.png", dpi=250)
    plt.close(fig)


def write_summary(
    cfg: ReplayConfig,
    atlas: Atlas,
    compass_df: pd.DataFrame,
    random_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary_compass = _summarize_compass(compass_df)
    final_compass = summary_compass.iloc[-1].to_dict()
    final_random = random_summary.iloc[-1].to_dict()
    summary = {
        "method": "COMPASS-Lite retrospective chunk acquisition",
        "map": "A10 transferred TICA-1 x TICA-2",
        "chunk_frames": cfg.chunk_frames,
        "variant": cfg.variant,
        "frame_spacing_ns": FRAME_SPACING_NS,
        "n_steps": int(compass_df["step"].max()),
        "random_repeats": cfg.random_repeats,
        "support_threshold": atlas.support_threshold,
        "final_compass": final_compass,
        "final_random_mean": final_random,
        "notes": [
            "No DCD reading or new MD; this is a first-layer replay over cached arrays.",
            "Candidate chunks are known retrospectively, so results validate scoring logic rather than online feasibility.",
            "RMSD/NHB/helix labels are not used for selection; NHB is used only for post-hoc evaluation.",
        ],
    }
    with open(output_dir / "compass_replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compass/replay_tica"))
    parser.add_argument("--chunk-frames", type=int, default=10, help="Frames revealed per selected chunk (10 frames = 1 ns).")
    parser.add_argument("--n-steps", type=int, default=80, help="Number of chunks to acquire.")
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--n-bins-x", type=int, default=30)
    parser.add_argument("--n-bins-y", type=int, default=None)
    parser.add_argument("--map-kind", choices=["tica2d", "tica1d"], default="tica2d")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--helix-nhb-threshold", type=int, default=DEFAULT_HELIX_NHB_THRESHOLD)
    parser.add_argument(
        "--variant",
        choices=["balanced", "exploratory", "source_only", "pure_frontier"],
        default="balanced",
        help="COMPASS-Lite scoring variant.",
    )
    parser.add_argument("--compass-repeats", type=int, default=1)
    parser.add_argument("--softmax-temperature", type=float, default=0.0)
    parser.add_argument("--start-chunk", type=int, default=0)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReplayConfig(
        repo_root=repo_root,
        output_dir=output_dir,
        chunk_frames=args.chunk_frames,
        n_steps=args.n_steps,
        random_repeats=args.random_repeats,
        n_bins_x=args.n_bins_x,
        random_seed=args.random_seed,
        helix_nhb_threshold=args.helix_nhb_threshold,
        variant=args.variant,
        compass_repeats=args.compass_repeats,
        softmax_temperature=args.softmax_temperature,
        map_kind=args.map_kind,
        n_bins_y=args.n_bins_y,
    )

    atlas, arrays = build_tica_atlas(
        repo_root,
        n_bins_x=cfg.n_bins_x,
        n_bins_y=cfg.n_bins_y,
        map_kind=cfg.map_kind,
    )
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
            sub_rng = np.random.default_rng(cfg.random_seed + rep + 1)
            start = int(seed_rng.integers(0, n_chunks))
            df, dec = run_compass_replay(atlas, target_nhb, cfg, start_chunk=start, rng=sub_rng)
            df = df.copy()
            df["repeat"] = rep
            compass_dfs.append(df)
            if rep == 0:
                decisions = dec
        compass_df = pd.concat(compass_dfs, ignore_index=True)
    random_df = run_random_replays(atlas, target_nhb, cfg)
    random_summary = _summarize_random(random_df)

    compass_df.to_csv(output_dir / "compass_metrics.csv", index=False)
    decisions.to_csv(output_dir / "compass_decisions.csv", index=False)
    random_df.to_csv(output_dir / "random_metrics.csv", index=False)
    random_summary.to_csv(output_dir / "random_summary.csv", index=False)

    plot_curves(compass_df, random_summary, output_dir)
    plot_map(atlas, target_nhb, decisions, output_dir)
    plot_reason_counts(decisions, output_dir)
    write_summary(cfg, atlas, compass_df, random_summary, output_dir)

    summary_compass = _summarize_compass(compass_df)
    final_c = summary_compass.iloc[-1]
    final_r = random_summary.iloc[-1]
    print(f"Wrote COMPASS replay outputs to {output_dir}")
    print(
        "Final @ %.1f ns: COMPASS JSD=%.4f coverage=%.3f f_err=%.4f | "
        "Random mean JSD=%.4f coverage=%.3f f_err=%.4f"
        % (
            final_c["sampled_ns"],
            final_c["jsd_mean"],
            final_c["coverage_mean"],
            final_c["f_helix_error_mean"],
            final_r["jsd_mean"],
            final_r["coverage_mean"],
            final_r["f_helix_error_mean"],
        )
    )


if __name__ == "__main__":
    main()

