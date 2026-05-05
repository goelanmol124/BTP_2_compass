"""Metrics, milestones, ESS, and cheating-audit helpers for COMPASS v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.compass.replay_enhanced import Atlas, DEFAULT_HELIX_NHB_THRESHOLD, DEFAULT_OPEN_NHB_THRESHOLD
from src.compass.replay_online import _evaluate_observed_frames, _hist2d_xy


@dataclass
class MilestoneTargets:
    coverage: float | None = None
    jsd: float | None = None
    f_helix_err: float | None = None


def effective_sample_size_timeseries(idx_sorted_by_time: np.ndarray, max_lag: int = 50) -> float:
    """Rough ESS from autocorr of a binary mask along time order (diagnostic only)."""

    idx_sorted_by_time = np.asarray(idx_sorted_by_time, dtype=int)
    if idx_sorted_by_time.size < 3:
        return float(idx_sorted_by_time.size)
    tags = np.zeros(idx_sorted_by_time.max() + 1, dtype=np.int8)
    tags[idx_sorted_by_time] = 1
    x = tags[idx_sorted_by_time.min() : idx_sorted_by_time.max() + 1].astype(np.float64)
    if x.var() < 1e-18:
        return float(x.sum())
    acf = np.ones(max_lag + 1)
    mu = x.mean()
    for lag in range(1, max_lag + 1):
        acf[lag] = np.mean((x[:-lag] - mu) * (x[lag:] - mu)) / (x.var() + 1e-18)
    tau = 1.0 + 2 * max(0.0, float(np.sum(np.clip(acf[1:], 0, None))))
    return float(x.sum() / max(tau, 1.0))


def first_milestone_step(df: pd.DataFrame, col: str, op: str, thr: float) -> int | None:
    if thr is None or col not in df.columns:
        return None
    vals = df[col].to_numpy(dtype=float)
    if op == "<=":
        hit = np.flatnonzero(vals <= thr)
    elif op == ">=":
        hit = np.flatnonzero(vals >= thr)
    else:
        raise ValueError(op)
    if hit.size == 0:
        return None
    steps = df["step"].to_numpy()
    return int(steps[hit[0]])


def finalize_milestones(metrics_df: pd.DataFrame, milestones: MilestoneTargets) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if milestones.coverage is not None:
        out["milestone_step_coverage"] = first_milestone_step(
            metrics_df, "coverage", ">=", float(milestones.coverage)
        )
    else:
        out["milestone_step_coverage"] = None
    if milestones.jsd is not None:
        out["milestone_step_jsd"] = first_milestone_step(metrics_df, "jsd", "<=", milestones.jsd)
    if milestones.f_helix_err is not None:
        out["milestone_step_f_helix_err"] = first_milestone_step(
            metrics_df, "f_helix_error", "<=", milestones.f_helix_err
        )
    return out


def cheating_audit_ok(policy_inputs: dict[str, Any], forbidden_keys: set[str]) -> bool:
    keys = set(policy_inputs.keys())
    return len(keys & forbidden_keys) == 0


def reweighted_observable(
    observed_frames: np.ndarray,
    bin_assignments: np.ndarray,
    pi_ref_bins: np.ndarray,
    f_obs_per_bin: np.ndarray,
    n_total_bins: int | None = None,
) -> float:
    """Reference-reweighted importance sampling estimate of a per-frame observable.

    Corrects for adaptive sampling restart bias. Equivalent to WHAM/MBAR with zero
    bias potential: each observed chunk is unbiased MD, but restart over-represents
    some regions. We reweight using the known reference equilibrium distribution.

    Args:
        observed_frames: indices of observed frames (subset of archive).
        bin_assignments: integer bin index for every frame in the archive (shape [n_archive]).
        pi_ref_bins: equilibrium bin weights from reference long MD (shape [n_bins]).
        f_obs_per_bin: observable value per bin from reference (shape [n_bins]).
        n_total_bins: if given, used as denominator floor.

    Returns:
        Reweighted scalar estimate of the observable.
    """
    observed_frames = np.asarray(observed_frames, dtype=int)
    if observed_frames.size == 0:
        return float(np.sum(pi_ref_bins * f_obs_per_bin))

    obs_bins = bin_assignments[observed_frames]
    n_bins = len(pi_ref_bins)
    counts = np.bincount(obs_bins, minlength=n_bins).astype(float)
    n_obs = float(observed_frames.size)
    pi_sample = counts / max(n_obs, 1.0)

    # Importance weight: target / proposal; floor prevents zero-division
    floor = 1.0 / max(n_obs, float(n_total_bins or n_bins))
    w = pi_ref_bins / np.maximum(pi_sample, floor)
    # Only weight bins that are actually visited; zero out unvisited bins
    w = np.where(counts > 0, w, 0.0)
    w_sum = w.sum()
    if w_sum <= 0.0:
        return float(np.sum(pi_ref_bins * f_obs_per_bin))
    w_hat = w / w_sum
    return float(np.sum(w_hat * f_obs_per_bin))


def count_transitions(nhb: np.ndarray, threshold: int = 3) -> int:
    """Count helix <-> coil transitions in an NHB time-series.

    A transition is counted each time the signal crosses the threshold boundary
    (from helix to coil or vice-versa), ignoring consecutive identical states.
    """
    nhb = np.asarray(nhb, dtype=int)
    if nhb.size < 2:
        return 0
    states = (nhb >= threshold).astype(np.int8)
    changes = np.diff(states)
    return int(np.count_nonzero(changes))


def evaluate_state(
    atlas: Atlas,
    nhb: np.ndarray,
    observed_idx: np.ndarray,
    helix_th: int = DEFAULT_HELIX_NHB_THRESHOLD,
    open_th: int = DEFAULT_OPEN_NHB_THRESHOLD,
) -> dict[str, float]:
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    return _evaluate_observed_frames(
        observed_idx, atlas.target_xy, ref_hist, atlas, nhb, helix_th, open_th
    )
