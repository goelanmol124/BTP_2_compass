"""Shared sweep aggregation: wide metrics from compass_replay_summary.json, plots, long timeseries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_CORE = [
    "jsd",
    "latent_wasserstein",
    "latent_mmd",
    "coverage",
    "important_coverage",
    "source_mass_visited",
    "f_helix_error",
    "f_open_error",
]


def _json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_sanitize(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
    return x


def load_summary(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_summary(summary: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """One flat row: config + all final_compass + all final_random_mean + deltas."""

    out: dict[str, Any] = dict(base)
    notes = summary.get("notes")
    if isinstance(notes, list):
        out["summary_notes_joined"] = " | ".join(str(x) for x in notes)

    skip_root = {"final_compass", "final_random_mean", "notes"}
    for k, v in summary.items():
        if k in skip_root:
            continue
        key = f"cfg_{k}"
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        elif isinstance(v, list):
            out[key] = json.dumps(v)

    fc = summary.get("final_compass") or {}
    for k, v in fc.items():
        out[f"compass_{k}"] = v

    fr = summary.get("final_random_mean") or {}
    for k, v in fr.items():
        out[f"random_{k}"] = v

    for metric in METRIC_CORE:
        c = out.get(f"compass_{metric}_mean")
        if c is None:
            c = out.get(f"compass_{metric}")
        r = out.get(f"random_{metric}_mean")
        if r is None:
            r = out.get(f"random_{metric}")
        try:
            c = float(c) if c is not None else float("nan")
            r = float(r) if r is not None else float("nan")
        except (TypeError, ValueError):
            c, r = float("nan"), float("nan")
        if metric in {"coverage", "important_coverage", "source_mass_visited"}:
            out[f"delta_{metric}"] = c - r
        else:
            out[f"delta_{metric}"] = r - c

    for short in (
        "map",
        "variant",
        "schedule",
        "deficit_mode",
        "prior_quality",
        "chunk_frames",
        "n_steps",
        "environment",
        "action_set",
        "msm_lag",
    ):
        ck = f"cfg_{short}"
        if ck in out:
            out[short] = out[ck]
    if "compass_sampled_ns" in out:
        out["sampled_ns"] = out["compass_sampled_ns"]

    return out


def write_summaries_jsonl(
    output_root: Path,
    experiment_names: Iterable[str],
    jsonl_path: Path,
) -> int:
    n = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for name in experiment_names:
            p = output_root / name / "compass_replay_summary.json"
            if not p.exists():
                continue
            rec = {"experiment": name, "summary": _json_sanitize(load_summary(p))}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def slim_columns_wide_df(df: pd.DataFrame) -> pd.DataFrame:
    """Readable subset for sweep2_results.csv / quick tables."""

    want = [
        "experiment",
        "description",
        "family",
        "environment",
        "sweep_driver",
        "map",
        "variant",
        "schedule",
        "deficit_mode",
        "prior_quality",
        "chunk_frames",
        "n_steps",
        "sampled_ns",
        "msm_lag",
    ]
    want.extend([f"delta_{m}" for m in METRIC_CORE])
    want.extend([f"compass_{m}_mean" for m in METRIC_CORE])
    want.extend([f"random_{m}_mean" for m in METRIC_CORE])
    cols = [c for c in want if c in df.columns]
    for extra in (
        "cfg_trust_controller",
        "cfg_restart_cost",
        "cfg_novelty_mode",
        "cfg_msm_lag",
        "cfg_prior_strength",
        "cfg_action_set",
        "cfg_environment",
        "cfg_method",
    ):
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    return df[cols].copy() if cols else df.copy()


def export_timeseries_long(
    output_root: Path,
    experiment_names: Iterable[str],
    out_csv: Path,
) -> int:
    """Stack compass_metrics and random_metrics per experiment (long format)."""

    parts: list[pd.DataFrame] = []
    for name in experiment_names:
        for policy, fname in (
            ("COMPASS", "compass_metrics.csv"),
            ("Random", "random_metrics.csv"),
        ):
            p = output_root / name / fname
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df.insert(0, "experiment", name)
            df.insert(1, "curve_policy", policy)
            parts.append(df)
    if not parts:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_csv, index=False)
        return 0
    long_df = pd.concat(parts, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_csv, index=False)
    return len(long_df)


def plot_delta_correlation(df: pd.DataFrame, out_path: Path, title: str) -> None:
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    if len(delta_cols) < 2:
        return
    sub = df[delta_cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(how="all")
    if len(sub) < 3:
        return
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_compass_vs_random_grid(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, m in zip(axes, METRIC_CORE):
        cx = f"compass_{m}_mean"
        rx = f"random_{m}_mean"
        if cx not in df.columns or rx not in df.columns:
            ax.set_visible(False)
            continue
        d = df[["experiment", cx, rx]].apply(pd.to_numeric, errors="coerce").dropna()
        if d.empty:
            ax.set_visible(False)
            continue
        ax.scatter(d[rx], d[cx], alpha=0.7, s=28)
        lo = float(min(d[cx].min(), d[rx].min()))
        hi = float(max(d[cx].max(), d[rx].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"Random {m}")
        ax.set_ylabel(f"COMPASS {m}")
        ax.set_title(m)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_delta_ranked(df: pd.DataFrame, out_path: Path, title: str, top_n: int = 25) -> None:
    if "delta_f_helix_error" not in df.columns or "experiment" not in df.columns:
        return
    sub = df[["experiment", "delta_f_helix_error"]].copy()
    sub["delta_f_helix_error"] = pd.to_numeric(sub["delta_f_helix_error"], errors="coerce")
    sub = sub.dropna().sort_values("delta_f_helix_error", ascending=False).head(top_n)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(sub))))
    colors = np.where(sub["delta_f_helix_error"].values >= 0, "#4C78A8", "#E45756")
    ax.barh(sub["experiment"], sub["delta_f_helix_error"], color=colors)
    ax.axvline(0, color="0.35", lw=0.9)
    ax.set_xlabel("Δ helix error (random − COMPASS); positive ⇒ COMPASS better")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
