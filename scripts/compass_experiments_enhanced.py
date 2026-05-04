"""Run the second COMPASS sweep with scheduler, SPIB and robustness tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compass_sweep_collect import (
    METRIC_CORE,
    export_timeseries_long,
    flatten_summary,
    load_summary,
    plot_compass_vs_random_grid,
    plot_delta_correlation,
    plot_delta_ranked,
    slim_columns_wide_df,
    write_summaries_jsonl,
)


def _budget_replace(argv: list[str], flag: str, value: str) -> None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            argv[i + 1] = value
            return


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    args: list[str] = field(default_factory=list)


BASE = [
    "--variant",
    "balanced",
    "--chunk-frames",
    "10",
    "--n-steps",
    "80",
    "--random-repeats",
    "100",
    "--deficit-mode",
    "information_gain",
]


EXPERIMENTS: list[Experiment] = [
    Experiment(
        "e01_tica2d_static_raw_80ns",
        "Original-ish balanced TICA2D baseline with raw deficit.",
        ["--map-kind", "tica2d", "--variant", "balanced", "--schedule", "static", "--deficit-mode", "raw", "--chunk-frames", "10", "--n-steps", "80", "--random-repeats", "100"],
    ),
    Experiment(
        "e02_tica2d_static_ig_80ns",
        "Balanced TICA2D with information-gain deficit only.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "static"],
    ),
    Experiment(
        "e03_tica2d_two_phase_ig_80ns",
        "Balanced TICA2D with two-phase scheduler and information-gain deficit.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase"],
    ),
    Experiment(
        "e04_tica2d_anneal_ig_80ns",
        "Balanced TICA2D with smooth annealed scheduler.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "anneal"],
    ),
    Experiment(
        "e05_tica1d_two_phase_ig_80ns",
        "TICA1D source map with two-phase scheduler.",
        [*BASE, "--map-kind", "tica1d", "--schedule", "two_phase"],
    ),
    Experiment(
        "e06_spib1d_two_phase_ig_80ns",
        "SPIB z=1 source map with two-phase scheduler.",
        [*BASE, "--map-kind", "spib1d", "--schedule", "two_phase"],
    ),
    Experiment(
        "e07_spib2d_two_phase_ig_80ns",
        "SPIB z=2 source map with two-phase scheduler.",
        [*BASE, "--map-kind", "spib2d", "--schedule", "two_phase"],
    ),
    Experiment(
        "e08_tica2d_two_phase_ig_200ns",
        "Longer 200 ns TICA2D two-phase test.",
        ["--variant", "balanced", "--map-kind", "tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "200", "--random-repeats", "100"],
    ),
    Experiment(
        "e09_spib1d_two_phase_ig_200ns",
        "Longer 200 ns SPIB z=1 two-phase test.",
        ["--variant", "balanced", "--map-kind", "spib1d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "200", "--random-repeats", "100"],
    ),
    Experiment(
        "e10_spib2d_two_phase_ig_200ns",
        "Longer 200 ns SPIB z=2 two-phase test.",
        ["--variant", "balanced", "--map-kind", "spib2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "200", "--random-repeats", "100"],
    ),
    Experiment(
        "e11_tica2d_two_phase_chunk5_80ns",
        "TICA2D two-phase with 0.5 ns chunks.",
        ["--variant", "balanced", "--map-kind", "tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "5", "--n-steps", "160", "--random-repeats", "100"],
    ),
    Experiment(
        "e12_tica2d_two_phase_chunk20_80ns",
        "TICA2D two-phase with 2 ns chunks.",
        ["--variant", "balanced", "--map-kind", "tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "20", "--n-steps", "40", "--random-repeats", "100"],
    ),
    Experiment(
        "e13_tica2d_uniform_prior_80ns",
        "Robustness: no useful source density prior (uniform prior).",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--prior-quality", "uniform"],
    ),
    Experiment(
        "e14_tica2d_shuffled_prior_80ns",
        "Robustness: source density randomly shuffled over bins.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--prior-quality", "shuffled"],
    ),
    Experiment(
        "e15_tica2d_inverted_prior_80ns",
        "Robustness: deliberately bad inverted prior.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--prior-quality", "inverted"],
    ),
    Experiment(
        "e16_tica2d_noisy_prior_80ns",
        "Robustness: 50 percent noisy density prior.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--prior-quality", "noisy", "--prior-strength", "0.5"],
    ),
    Experiment(
        "e17_tica2d_pure_frontier_schedule_80ns",
        "Ablation: pure frontier with two-phase flag (stays frontier by variant).",
        ["--variant", "pure_frontier", "--map-kind", "tica2d", "--schedule", "static", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "80", "--random-repeats", "100"],
    ),
    Experiment(
        "e18_tica2d_exploratory_static_80ns",
        "Ablation: exploratory static policy under enhanced metrics.",
        ["--variant", "exploratory", "--map-kind", "tica2d", "--schedule", "static", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "80", "--random-repeats", "100"],
    ),
    Experiment(
        "e19_tica2d_two_phase_softmax_repeats_80ns",
        "Robustness to initial seed/stochastic top-k selection.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--compass-repeats", "12", "--softmax-temperature", "0.4"],
    ),
    Experiment(
        "e20_spib1d_two_phase_softmax_repeats_80ns",
        "SPIB z=1 with repeated stochastic starts.",
        [*BASE, "--map-kind", "spib1d", "--schedule", "two_phase", "--compass-repeats", "12", "--softmax-temperature", "0.4"],
    ),
    Experiment(
        "e21_tica2d_chunk20_softmax_confirm_80ns",
        "Confirmation of the e12 headline run with stochastic starts.",
        ["--variant", "balanced", "--map-kind", "tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "20", "--n-steps", "40", "--random-repeats", "200", "--compass-repeats", "24", "--softmax-temperature", "0.4"],
    ),
    Experiment(
        "e22_hybrid_spib1d_tica2d_80ns",
        "Hybrid: SPIB z=1 frontier discovery plus TICA2D density correction.",
        ["--variant", "hybrid", "--map-kind", "hybrid_spib1d_tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "80", "--random-repeats", "100"],
    ),
    Experiment(
        "e23_hybrid_spib2d_tica2d_80ns",
        "Hybrid: SPIB z=2 frontier discovery plus TICA2D density correction.",
        ["--variant", "hybrid", "--map-kind", "hybrid_spib2d_tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "80", "--random-repeats", "100"],
    ),
    Experiment(
        "e24_hybrid_spib1d_tica2d_200ns",
        "Long-budget hybrid with SPIB z=1 frontier and TICA2D density.",
        ["--variant", "hybrid", "--map-kind", "hybrid_spib1d_tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "200", "--random-repeats", "100"],
    ),
    Experiment(
        "e25_hybrid_spib2d_tica2d_200ns",
        "Long-budget hybrid with SPIB z=2 frontier and TICA2D density.",
        ["--variant", "hybrid", "--map-kind", "hybrid_spib2d_tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "200", "--random-repeats", "100"],
    ),
    Experiment(
        "e26_hybrid_spib1d_tica2d_noisy_prior_80ns",
        "Hybrid robustness: SPIB z=1 frontier plus noisy TICA2D density prior.",
        ["--variant", "hybrid", "--map-kind", "hybrid_spib1d_tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "80", "--random-repeats", "100", "--prior-quality", "noisy", "--prior-strength", "0.5"],
    ),
    Experiment(
        "e27_tica2d_chunk20_prior_trust_80ns",
        "Trust-controller version of the e12 observable run.",
        ["--variant", "balanced", "--map-kind", "tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "20", "--n-steps", "40", "--random-repeats", "100", "--trust-controller", "prior_agreement"],
    ),
    Experiment(
        "e28_tica2d_noisy_prior_trust_80ns",
        "Trust controller under 50 percent noisy TICA2D prior.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--prior-quality", "noisy", "--prior-strength", "0.5", "--trust-controller", "prior_agreement"],
    ),
    Experiment(
        "e29_tica2d_inverted_prior_trust_80ns",
        "Trust controller under deliberately inverted TICA2D prior.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--prior-quality", "inverted", "--trust-controller", "prior_agreement"],
    ),
    Experiment(
        "e30_tica2d_chunk20_restart_cost_80ns",
        "Cost-aware restart penalty on the e12 observable run.",
        ["--variant", "balanced", "--map-kind", "tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "20", "--n-steps", "40", "--random-repeats", "100", "--restart-cost", "0.35"],
    ),
    Experiment(
        "e31_tica2d_epistemic_novelty_80ns",
        "Epistemic novelty instead of pure distance novelty.",
        [*BASE, "--map-kind", "tica2d", "--schedule", "two_phase", "--novelty-mode", "epistemic"],
    ),
    Experiment(
        "e32_hybrid_spib1d_tica2d_prior_trust_80ns",
        "SPIB frontier plus TICA density with prior-agreement controller.",
        ["--variant", "hybrid", "--map-kind", "hybrid_spib1d_tica2d", "--schedule", "two_phase", "--deficit-mode", "information_gain", "--chunk-frames", "10", "--n-steps", "80", "--random-repeats", "100", "--trust-controller", "prior_agreement"],
    ),
]


METRICS = list(METRIC_CORE)


def run_experiment(
    exp: Experiment,
    output_root: Path,
    n_steps: int | None = None,
    random_repeats: int | None = None,
) -> Path:
    out_dir = output_root / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = list(exp.args)
    if n_steps is not None:
        _budget_replace(argv, "--n-steps", str(n_steps))
    if random_repeats is not None:
        _budget_replace(argv, "--random-repeats", str(random_repeats))
    cmd = [
        sys.executable,
        "-m",
        "src.compass.replay_enhanced",
        "--output-dir",
        str(out_dir.relative_to(REPO_ROOT)),
        *argv,
    ]
    print(f"\n=== {exp.name} ===\n{exp.description}\n  cmd: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    with open(out_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump({"name": exp.name, "description": exp.description, "args": exp.args}, f, indent=2)
    return out_dir


def collect_results(output_root: Path, experiments_list: list[Experiment] | None = None) -> pd.DataFrame:
    explist = experiments_list if experiments_list is not None else EXPERIMENTS
    rows: list[dict[str, object]] = []
    for exp in explist:
        summary_path = output_root / exp.name / "compass_replay_summary.json"
        if not summary_path.exists():
            continue
        summary = load_summary(summary_path)
        base = {
            "experiment": exp.name,
            "description": exp.description,
            "sweep_driver": "oracle_enhanced",
            "family": "oracle_enhanced",
        }
        rows.append(flatten_summary(summary, base))
    return pd.DataFrame(rows)


def _read_compass_mean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    available = [m for m in METRICS if m in df.columns]
    return df.groupby(["step", "sampled_ns"], as_index=False)[available].mean()


def plot_overlay(output_root: Path, sweep_dir: Path, experiments_list: list[Experiment]) -> None:
    specs = [
        ("f_helix_error", "Helix-fraction error"),
        ("coverage", "Coverage"),
        ("important_coverage", "Important-bin discovery"),
        ("jsd", "JSD sanity check"),
        ("source_mass_visited", "Source-prior mass visited"),
        ("f_open_error", "Open-fraction error"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    cmap = plt.cm.tab20
    random_summary = None
    for i, exp in enumerate(experiments_list):
        compass_csv = output_root / exp.name / "compass_metrics.csv"
        random_csv = output_root / exp.name / "random_summary.csv"
        if not compass_csv.exists():
            continue
        df = _read_compass_mean(compass_csv)
        label = exp.name.replace("_", " ")
        for ax, (metric, _title) in zip(axes.ravel(), specs):
            ax.plot(df["sampled_ns"], df[metric], label=label, lw=1.25, color=cmap(i % 20), alpha=0.9)
        if random_summary is None and random_csv.exists():
            random_summary = pd.read_csv(random_csv)
    for ax, (metric, title) in zip(axes.ravel(), specs):
        if random_summary is not None:
            x = random_summary["sampled_ns"]
            mean = random_summary[f"{metric}_mean"]
            lo = random_summary.get(f"{metric}_lo", mean)
            hi = random_summary.get(f"{metric}_hi", mean)
            ax.plot(x, mean, label="Random bootstrap mean", color="black", lw=2.1, linestyle="--")
            ax.fill_between(x, lo, hi, color="0.6", alpha=0.20)
        ax.set_title(title)
        ax.set_xlabel("sampled ns")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle("Enhanced COMPASS sweep: scheduler, SPIB, robustness")
    fig.savefig(sweep_dir / "sweep2_overlay_curves.png", dpi=220)
    plt.close(fig)


def plot_summary_bars(table: pd.DataFrame, sweep_dir: Path) -> None:
    if table.empty:
        return
    specs = [
        ("delta_f_helix_error", "Helix-error improvement"),
        ("delta_coverage", "Coverage improvement"),
        ("delta_important_coverage", "Important-bin discovery improvement"),
        ("delta_jsd", "JSD improvement"),
        ("delta_source_mass_visited", "Source-prior mass improvement"),
        ("delta_f_open_error", "Open-error improvement"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    order = table.sort_values("delta_f_helix_error", ascending=True)["experiment"]
    for ax, (col, title) in zip(axes.ravel(), specs):
        sub = table.set_index("experiment").loc[order].reset_index()
        colors = np.where(sub[col].fillna(0) >= 0, "#4C78A8", "#E45756")
        ax.barh(sub["experiment"], sub[col], color=colors)
        ax.axvline(0.0, color="0.35", lw=0.8)
        ax.set_title(f"{title}\npositive = COMPASS better")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(sweep_dir / "sweep2_summary_bars.png", dpi=220)
    plt.close(fig)


def write_markdown(table: pd.DataFrame, sweep_dir: Path) -> None:
    with open(sweep_dir / "sweep2_results.md", "w", encoding="utf-8") as f:
        f.write("# Enhanced COMPASS sweep results\n\n")
        if table.empty:
            f.write("No completed experiment summaries found.\n")
            return
        cols = [
            "experiment",
            "map",
            "schedule",
            "prior_quality",
            "sampled_ns",
            "delta_f_helix_error",
            "delta_coverage",
            "delta_important_coverage",
            "delta_jsd",
            "delta_source_mass_visited",
            "delta_f_open_error",
        ]
        cols = [c for c in cols if c in table.columns]
        sub = table[cols].copy()
        for c in sub.columns:
            if sub[c].dtype == float:
                sub[c] = sub[c].round(4)
        f.write("| " + " | ".join(sub.columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in sub.columns) + " |\n")
        for _, row in sub.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in sub.columns) + " |\n")
        f.write("\n## Descriptions\n\n")
        if "description" in table.columns:
            for _, row in table.iterrows():
                f.write(f"- **{row['experiment']}**: {row['description']}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/compass_sweep2"))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override all --n-steps values in experiment arg lists.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=None,
        help="Override all --random-repeats values in experiment arg lists.",
    )
    args = parser.parse_args(argv)
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [e for e in EXPERIMENTS if e.name in wanted]
        if not selected:
            raise SystemExit(f"No experiments matched {sorted(wanted)}")
    if not args.aggregate_only:
        for exp in selected:
            run_experiment(exp, output_root, args.n_steps, args.random_repeats)
    sweep_dir = output_root / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    names = [e.name for e in selected]
    table_full = collect_results(output_root, selected)
    table_full.to_csv(sweep_dir / "sweep2_results_full.csv", index=False)
    slim = slim_columns_wide_df(table_full)
    slim.to_csv(sweep_dir / "sweep2_results.csv", index=False)
    write_summaries_jsonl(output_root, names, sweep_dir / "sweep2_summaries.jsonl")
    export_timeseries_long(output_root, names, sweep_dir / "sweep2_timeseries_long.csv")
    write_markdown(slim, sweep_dir)
    plot_overlay(output_root, sweep_dir, selected)
    plot_summary_bars(slim, sweep_dir)
    plot_delta_correlation(
        table_full,
        sweep_dir / "sweep2_delta_correlation.png",
        "Enhanced sweep: correlation of Δ metrics across experiments",
    )
    plot_compass_vs_random_grid(
        table_full,
        sweep_dir / "sweep2_compass_vs_random_scatter.png",
        "Enhanced: COMPASS vs random (final means)",
    )
    plot_delta_ranked(
        table_full,
        sweep_dir / "sweep2_top_helix_delta.png",
        "Top experiments by helix-error advantage",
    )
    print(f"\nEnhanced sweep summary written to {sweep_dir}")


if __name__ == "__main__":
    main()
