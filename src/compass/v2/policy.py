"""Policy wrappers: COMPASS (full prior vs uniform prior) and baseline pickers."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import numpy as np

from src.compass.replay_enhanced import Atlas, ReplayConfig, _softmax_choice
from src.compass.replay_online import _compass_scores_online


class PolicyName(str, Enum):
    B5_COMPASS = "compass"
    B4_COMPASS_UNIFORM_PRIOR = "uniform_prior"
    B1_RANDOM = "random"
    B2_MIN_COUNT_BIN = "min_count"
    B3_FARTHEST_LATENT = "farthest"
    B0_CHRONOLOGICAL = "chrono"
    # P2: seeds chosen once at start from initial candidate expansion; never updated
    P2_INDEPENDENT_REPLICAS = "independent_replicas"


def uniform_prior_atlas(atlas: Atlas) -> Atlas:
    flat = atlas.source_hist.size
    u = np.full_like(atlas.source_hist, 1.0 / flat, dtype=np.float64)
    support = np.ones_like(atlas.source_hist, dtype=bool)
    logp = -np.log(np.maximum(u, 1e-300))
    fe = np.zeros_like(u)
    fv = logp[support]
    denom = max(float(fv.max() - fv.min()), 1e-12)
    fe[support] = (fv - fv.min()) / denom
    return replace(atlas, source_prob=u, source_support=support, free_energy_score=fe)


def compass_pick(
    atlas: Atlas,
    cfg: ReplayConfig,
    candidates: np.ndarray,
    selected_seed_frames: list[int],
    bin_attempts: np.ndarray,
    fraction_done: float,
    rng: np.random.Generator,
    *,
    uniform_prior: bool,
) -> tuple[int, int]:
    """Return (pool_index, global_seed_frame)."""

    work_atlas = uniform_prior_atlas(atlas) if uniform_prior else atlas
    scores, *_rest = _compass_scores_online(
        work_atlas, candidates, selected_seed_frames, bin_attempts, cfg, fraction_done
    )
    idx = _softmax_choice(scores, np.ones(len(candidates), dtype=bool), rng, cfg.softmax_temperature)
    return int(idx), int(candidates[idx])


def random_pick(candidates: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    if len(candidates) == 0:
        raise ValueError("random_pick: empty candidate set")
    ii = int(rng.integers(0, len(candidates)))
    return ii, int(candidates[ii])


def min_count_bin_pick(
    candidates: np.ndarray,
    bin_attempts: np.ndarray,
    atlas: Atlas,
    frontier_xy: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    from src.compass.replay_enhanced import _bin_indices

    xy = frontier_xy[np.asarray(candidates, dtype=int)]
    bx, by = _bin_indices(xy, atlas.x_edges, atlas.y_edges)
    counts = bin_attempts[bx, by].astype(np.float64)
    min_c = counts.min()
    contenders = np.flatnonzero(counts <= min_c + 1e-9)
    ii = int(rng.choice(contenders))
    return ii, int(candidates[ii])


def independent_replicas_pick(
    candidates: np.ndarray,
    preselected_queue: list[int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Pop next seed from a pre-committed queue; falls back to random if exhausted."""
    if preselected_queue:
        seed_f = preselected_queue.pop(0)
        where = np.flatnonzero(candidates == seed_f)
        if where.size > 0:
            return int(where[0]), seed_f
    ii = int(rng.integers(0, len(candidates)))
    return ii, int(candidates[ii])


def farthest_latent_pick(
    candidates: np.ndarray,
    selected_seed_frames: list[int],
    frontier_xy: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if not selected_seed_frames:
        ii = int(rng.integers(0, len(candidates)))
        return ii, int(candidates[ii])
    cand_xy = frontier_xy[np.asarray(candidates, dtype=int)]
    sel = np.array(selected_seed_frames, dtype=int)
    center = frontier_xy[sel].mean(axis=0, keepdims=True)
    d = np.sqrt(np.sum((cand_xy - center) ** 2, axis=1))
    ii = int(np.argmax(d))
    return ii, int(candidates[ii])
