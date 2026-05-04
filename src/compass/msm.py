"""Segment-aware MSM utilities for offline stochastic replay.

.. deprecated:: COMPASS-v2

    MSM emulation is **ablation-only** for v2 headlines. Primary benchmarks use
    archived-trajectory continuations in :mod:`src.compass.v2` (see ``COMPASS_PLAN.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def state_flat(bx: np.ndarray, by: np.ndarray, n_by: int) -> np.ndarray:
    return (bx * n_by + by).astype(np.int64)


def unflatten_state(k: int, n_by: int) -> tuple[int, int]:
    return int(k // n_by), int(k % n_by)


def build_msm(
    bx: np.ndarray,
    by: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    lag: int,
    segments: list[tuple[int, int]],
    laplace: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Row-stochastic T with Laplace smoothing; counts only intra-segment (i, i+lag) pairs."""

    n_by = n_bins_y
    n_sx = n_bins_x
    n_states = n_sx * n_by
    c = np.zeros((n_states, n_states), dtype=float)
    per_segment: list[dict[str, int | tuple[int, int]]] = []

    for start, end in segments:
        start = max(0, int(start))
        end = min(len(bx), int(end))
        seg_trans = 0
        if end - start > lag:
            for i in range(start, end - lag):
                a = int(state_flat(bx[i : i + 1], by[i : i + 1], n_by)[0])
                b = int(state_flat(bx[i + lag : i + lag + 1], by[i + lag : i + lag + 1], n_by)[0])
                if 0 <= a < n_states and 0 <= b < n_states:
                    c[a, b] += 1.0
                    seg_trans += 1
        per_segment.append(
            {
                "segment": (start, end),
                "n_lagged_transitions": seg_trans,
            }
        )

    c_smooth = c + laplace / max(n_states, 1)
    row = c_smooth.sum(axis=1, keepdims=True)
    row = np.maximum(row, 1e-12)
    t = c_smooth / row

    nonzero = int(np.count_nonzero(c))
    row_entropy = -np.sum(t * np.log(np.maximum(t, 1e-300)), axis=1)
    sinks = int(np.sum(row_entropy < 0.01))

    meta = {
        "n_states": n_states,
        "n_bins_x": n_bins_x,
        "n_bins_y": n_bins_y,
        "lag": lag,
        "segments": [[int(a), int(b)] for a, b in segments],
        "per_segment_transition_counts": per_segment,
        "count_matrix_nonzeros": nonzero,
        "laplace": laplace,
        "sink_like_rows_entropy_lt_0p01": sinks,
    }
    return t, meta


def simulate_walk(transition: np.ndarray, s0: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Markov chain; n_steps transitions -> path length n_steps+1."""

    s0 = int(s0)
    path = [s0]
    s = s0
    n_states = transition.shape[0]
    for _ in range(int(n_steps)):
        probs = transition[s]
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            s = int(rng.integers(0, n_states))
        else:
            s = int(rng.choice(n_states, p=probs))
        path.append(s)
    return np.array(path, dtype=int)


def microstate_observables(
    nhb: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    helix_threshold: int,
    open_threshold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per microstate mean helix indicator, open indicator, and occupancy counts."""

    n_by = n_bins_y
    n_states = n_bins_x * n_by
    counts = np.zeros(n_states, dtype=float)
    sum_h = np.zeros(n_states, dtype=float)
    sum_o = np.zeros(n_states, dtype=float)
    for i in range(len(nhb)):
        k = int(state_flat(bx[i : i + 1], by[i : i + 1], n_by)[0])
        counts[k] += 1.0
        sum_h[k] += 1.0 if nhb[i] >= helix_threshold else 0.0
        sum_o[k] += 1.0 if nhb[i] <= open_threshold else 0.0
    f_h = np.divide(sum_h, counts, out=np.zeros_like(sum_h), where=counts > 0)
    f_o = np.divide(sum_o, counts, out=np.zeros_like(sum_o), where=counts > 0)
    return f_h, f_o, counts


def path_to_hist2d(path: np.ndarray, n_bins_x: int, n_bins_y: int) -> np.ndarray:
    hist = np.zeros((n_bins_x, n_bins_y), dtype=float)
    n_by = n_bins_y
    for s in path:
        bx, by = unflatten_state(int(s), n_by)
        if 0 <= bx < n_bins_x and 0 <= by < n_bins_y:
            hist[bx, by] += 1.0
    return hist


def bootstrap_walks(
    transition: np.ndarray,
    s0: int,
    n_steps: int,
    rng: np.random.Generator,
    n_boot: int,
) -> list[np.ndarray]:
    return [simulate_walk(transition, s0, n_steps, np.random.default_rng(int(rng.integers(0, 2**31)))) for _ in range(n_boot)]
