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
