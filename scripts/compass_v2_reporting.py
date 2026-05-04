#!/usr/bin/env python3
"""Roll up COMPASS v2 planned runs: final budgets, TTT / speedup, Tier-1 figures.

Reads ``outputs/v2/plan/manifest.json`` (from ``compass_v2_experiment_suite``) or
explicit directories containing ``v2_metrics.csv`` and ``random_metrics.csv``.
Writes tables and figures under ``outputs/v2/plan/report/`` by default.

Examples::

    python scripts/compass_v2_reporting.py
    python scripts/compass_v2_reporting.py --manifest outputs/v2/plan/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Default threshold grids (plan § stratified acceleration)
COVERAGE_TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
IMPORTANT_COV_TAUS = [0.25, 0.5, 0.75]
JSD_TAUS = [0.20, 0.15, 0.10, 0.07, 0.05]
FERR_TAUS = [0.05, 0.02, 0.01]

MAP_METRICS = ["coverage", "important_coverage", "jsd", "source_mass_visited"]
OBS_METRICS = ["f_helix_error", "f_open_error"]


def _out_relpath(argv: list[str]) -> Path | None:
    for i, a in enumerate(argv):
        if a == "--output-dir" and i + 1 < len(argv):
            return Path(argv[i + 1])
    return None


def run_dirs_from_manifest(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for r in data.get("runs", []):
        name = r["name"]
        od = _out_relpath(r.get("argv", []))
        if od is None:
            continue
        out[name] = (REPO_ROOT / od).resolve()
    return out


def first_crossing(
    sub: pd.DataFrame,
    metric: str,
    threshold: float,
    *,
    sense: str,
) -> float | None:
    """First ``sampled_ns`` where condition holds; None if never."""
    if metric not in sub.columns or "sampled_ns" not in sub.columns:
        return None
    s = sub.sort_values("step")
    v = s[metric].to_numpy(dtype=float)
    t = s["sampled_ns"].to_numpy(dtype=float)
    if sense == "ge":
        hit = np.where(v >= threshold)[0]
    elif sense == "le":
        hit = np.where(v <= threshold)[0]
    else:
        raise ValueError(sense)
    if hit.size == 0:
        return None
    return float(t[hit[0]])


def ttt_compass(v2_metrics: pd.DataFrame, metric: str, threshold: float, sense: str) -> float | None:
    return first_crossing(v2_metrics, metric, threshold, sense=sense)


def ttt_random(random_metrics: pd.DataFrame, metric: str, threshold: float, sense: str) -> dict[str, Any]:
    reps = sorted(int(x) for x in random_metrics["repeat"].unique())
    times: list[float] = []
    for rep in reps:
        t = first_crossing(random_metrics[random_metrics["repeat"] == rep], metric, threshold, sense=sense)
        if t is not None:
            times.append(t)
    n = len(reps)
    nr = n - len(times)
    return {
        "n_repeats": n,
        "n_not_reached": nr,
        "frac_nr": float(nr / max(n, 1)),
        "mean_ns": float(np.mean(times)) if times else None,
        "median_ns": float(np.median(times)) if times else None,
    }


def speedup_ratio(t_rand: float | None, t_comp: float | None) -> float | None:
    if t_rand is None or t_comp is None or t_comp <= 0 or t_rand <= 0:
        return None
    return float(t_rand / t_comp)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    vm = run_dir / "v2_metrics.csv"
    rm = run_dir / "random_metrics.csv"
    if not vm.is_file():
        raise FileNotFoundError(vm)
    v2 = pd.read_csv(vm)
    rnd = pd.read_csv(rm) if rm.is_file() else None
    return v2, rnd


def final_row_table(run_map: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, d in sorted(run_map.items()):
        v2, rnd = load_run(d)
        last = v2.iloc[-1].to_dict()
        row = {
            "run": name,
            "dir": str(d),
            **{k: last.get(k) for k in MAP_METRICS + OBS_METRICS + ["sampled_ns", "policy", "eq_method"]},
        }
        if rnd is not None and len(rnd) and "repeat" in rnd.columns:
            last_per = rnd.sort_values("step").groupby("repeat", as_index=False).last()
            for col in MAP_METRICS + OBS_METRICS:
                if col in last_per.columns:
                    row[f"random_mean_final_{col}"] = float(last_per[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_ttt_table(run_map: dict[str, Path]) -> pd.DataFrame:
    spec: list[tuple[str, float, str]] = []
    for tau in COVERAGE_TAUS:
        spec.append(("coverage", tau, "ge"))
    for tau in IMPORTANT_COV_TAUS:
        spec.append(("important_coverage", tau, "ge"))
    for tau in JSD_TAUS:
        spec.append(("jsd", tau, "le"))
    for tau in FERR_TAUS:
        spec.append(("f_helix_error", tau, "le"))
        spec.append(("f_open_error", tau, "le"))

    out_rows: list[dict[str, Any]] = []
    for name, d in sorted(run_map.items()):
        v2, rnd = load_run(d)
        for metric, tau, sense in spec:
            tc = ttt_compass(v2, metric, tau, sense)
            row: dict[str, Any] = {
                "run": name,
                "metric": metric,
                "threshold": tau,
                "sense": sense,
                "compass_t_ns": tc,
            }
            if rnd is not None:
                tr = ttt_random(rnd, metric, tau, sense)
                row.update(
                    {
                        "random_mean_t_ns": tr["mean_ns"],
                        "random_median_t_ns": tr["median_ns"],
                        "random_frac_nr": tr["frac_nr"],
                    }
                )
                row["speedup_vs_random_mean"] = speedup_ratio(tr["mean_ns"], tc)
                row["speedup_vs_random_median"] = speedup_ratio(tr["median_ns"], tc)
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def plot_speedup_bars(
    ttt: pd.DataFrame,
    runs_filter: list[str],
    metrics_subset: list[str],
    taus_pick: dict[str, float],
    out_path: Path,
) -> None:
    sub = ttt[ttt["run"].isin(runs_filter)].copy()
    xs: list[str] = []
    vals: list[float] = []
    for m in metrics_subset:
        tau = taus_pick.get(m, COVERAGE_TAUS[4] if m == "coverage" else JSD_TAUS[2])
        sense = "ge" if m in {"coverage", "important_coverage", "source_mass_visited"} else "le"
        for run in runs_filter:
            row = sub[(sub["run"] == run) & (sub["metric"] == m) & (sub["threshold"] == tau) & (sub["sense"] == sense)]
            if row.empty:
                continue
            su = row.iloc[0].get("speedup_vs_random_mean")
            if su is not None and np.isfinite(su):
                xs.append(f"{run}\n{m}@{tau}")
                vals.append(float(su))
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(vals)), vals, color="#4C78A8")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=25, ha="right", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Speedup vs random mean T (TTT)")
    ax.set_title("Stratified acceleration (first-crossing time)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tica_vs_spib_effects(
    summary_final: pd.DataFrame,
    out_path: Path,
) -> None:
    """Bar chart of (compass − random) style deltas using final-row map metrics where available."""
    tica = summary_final[summary_final["run"] == "a_compass"]
    spib = summary_final[summary_final["run"] == "b_spib_compass"]
    if tica.empty or spib.empty:
        return
    tr = tica.iloc[0]
    sr = spib.iloc[0]
    metrics = ["coverage", "jsd", "important_coverage", "f_helix_error"]
    deltas_a: list[float] = []
    deltas_b: list[float] = []
    ok_labels: list[str] = []
    for m in metrics:
        if m not in tr or m not in sr:
            continue
        rv_a = tr.get(f"random_mean_final_{m}") if f"random_mean_final_{m}" in tr else None
        rv_b = sr.get(f"random_mean_final_{m}") if f"random_mean_final_{m}" in tr else None
        if m == "jsd":
            # lower is better: random - policy
            da = float(rv_a) - float(tr[m]) if rv_a is not None else np.nan
            db = float(rv_b) - float(sr[m]) if rv_b is not None else np.nan
        else:
            da = float(tr[m]) - float(rv_a) if rv_a is not None else np.nan
            db = float(sr[m]) - float(rv_b) if rv_b is not None else np.nan
        if np.isfinite(da) and np.isfinite(db):
            deltas_a.append(da)
            deltas_b.append(db)
            ok_labels.append(m)
    if not ok_labels:
        return
    x = np.arange(len(ok_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, deltas_a, width=w, label="TICA2D (block A compass)", color="#F58518")
    ax.bar(x + w / 2, deltas_b, width=w, label="SPIB1D (block B)", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(ok_labels)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("Δ vs random mean (final budget)")
    ax.legend()
    ax.set_title("Per-map effect sizes (do not rank absolute levels across maps)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=REPO_ROOT / "outputs" / "v2" / "plan" / "manifest.json")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "v2" / "plan" / "report")
    p.add_argument("--runs", type=str, default="", help="Comma-separated run names to include (default: all in manifest)")
    args = p.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"Missing manifest: {args.manifest}")

    run_map = run_dirs_from_manifest(args.manifest)
    if args.runs.strip():
        keep = {x.strip() for x in args.runs.split(",") if x.strip()}
        run_map = {k: v for k, v in run_map.items() if k in keep}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fin = final_row_table(run_map)
    fin.to_csv(args.out_dir / "final_budget_by_run.csv", index=False)

    ttt = build_ttt_table(run_map)
    ttt.to_csv(args.out_dir / "ttt_lattice.csv", index=False)

    # Fig 2: speedups for block A compass-focused
    blk_a = [k for k in run_map if k.startswith("a_") and "compass" in k and "cs4500" not in k]
    if blk_a:
        plot_speedup_bars(
            ttt,
            blk_a[:1],
            ["coverage", "jsd"],
            {"coverage": 0.5, "jsd": JSD_TAUS[2]},
            args.out_dir / "fig2_speedup_block_a_compass.png",
        )

    plot_tica_vs_spib_effects(fin, args.out_dir / "fig3_tica_vs_spib_effect_bars.png")

    meta = {
        "n_runs": len(run_map),
        "manifest": str(args.manifest),
        "outputs": [
            "final_budget_by_run.csv",
            "ttt_lattice.csv",
            "fig2_speedup_block_a_compass.png",
            "fig3_tica_vs_spib_effect_bars.png",
        ],
        "note": "Per-run learning curves remain under each bench output dir (compass_replay_curves.png).",
    }
    (args.out_dir / "report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote report under {args.out_dir}")


if __name__ == "__main__":
    main()
