"""Run a sweep of COMPASS-Lite retrospective replay experiments.

Each experiment is a small variation: variant, map kind, chunk size, total
budget, or stochasticity.  All outputs are stored under
``outputs/compass/<exp_name>/`` and a roll-up CSV plus a comparison plot are
written to ``outputs/compass/sweep/``.

Run::

    python scripts/compass_experiments.py
    python scripts/compass_experiments.py --only exp_balanced_short exp_explor_long
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    args: list[str] = field(default_factory=list)


EXPERIMENTS: list[Experiment] = [
    Experiment(
        "exp01_balanced_tica2d_chunk10_80ns",
        "Balanced policy on TICA 2D map, 1 ns chunks, 80 ns budget (baseline).",
        ["--variant", "balanced", "--map-kind", "tica2d", "--chunk-frames", "10", "--n-steps", "80"],
    ),
    Experiment(
        "exp02_exploratory_tica2d_chunk10_80ns",
        "Exploratory policy on TICA 2D map, 1 ns chunks, 80 ns budget.",
        ["--variant", "exploratory", "--map-kind", "tica2d", "--chunk-frames", "10", "--n-steps", "80"],
    ),
    Experiment(
        "exp03_source_only_tica2d_chunk10_80ns",
        "Source-only ablation: source map drives all guidance, no novelty/frontier.",
        ["--variant", "source_only", "--map-kind", "tica2d", "--chunk-frames", "10", "--n-steps", "80"],
    ),
    Experiment(
        "exp04_pure_frontier_tica2d_chunk10_80ns",
        "Pure-frontier ablation: only novelty/frontier, source map ignored.",
        ["--variant", "pure_frontier", "--map-kind", "tica2d", "--chunk-frames", "10", "--n-steps", "80"],
    ),
    Experiment(
        "exp05_balanced_tica2d_chunk5_80ns",
        "Balanced policy with smaller 0.5 ns chunks (more decisions per ns).",
        ["--variant", "balanced", "--map-kind", "tica2d", "--chunk-frames", "5", "--n-steps", "160"],
    ),
    Experiment(
        "exp06_balanced_tica2d_chunk20_80ns",
        "Balanced policy with larger 2 ns chunks (fewer, longer decisions).",
        ["--variant", "balanced", "--map-kind", "tica2d", "--chunk-frames", "20", "--n-steps", "40"],
    ),
    Experiment(
        "exp07_balanced_tica2d_chunk10_200ns",
        "Balanced policy extended to a 200 ns budget (do COMPASS curves cross random?).",
        ["--variant", "balanced", "--map-kind", "tica2d", "--chunk-frames", "10", "--n-steps", "200"],
    ),
    Experiment(
        "exp08_exploratory_tica2d_chunk10_200ns",
        "Exploratory policy extended to a 200 ns budget for full coverage curves.",
        ["--variant", "exploratory", "--map-kind", "tica2d", "--chunk-frames", "10", "--n-steps", "200"],
    ),
    Experiment(
        "exp09_balanced_tica2d_chunk10_80ns_repeats",
        "Balanced policy with 8 random restart seeds + softmax exploration (fair vs random).",
        [
            "--variant",
            "balanced",
            "--map-kind",
            "tica2d",
            "--chunk-frames",
            "10",
            "--n-steps",
            "80",
            "--compass-repeats",
            "8",
            "--softmax-temperature",
            "0.5",
        ],
    ),
    Experiment(
        "exp10_balanced_tica1d_chunk10_80ns",
        "Balanced policy on a 1D TICA-1 source map (worst-case dimensionality).",
        ["--variant", "balanced", "--map-kind", "tica1d", "--chunk-frames", "10", "--n-steps", "80"],
    ),
    Experiment(
        "exp11_balanced_tica2d_chunk10_80ns_finegrid",
        "Balanced policy with a finer 50x50 source grid (resolution sensitivity).",
        [
            "--variant",
            "balanced",
            "--map-kind",
            "tica2d",
            "--chunk-frames",
            "10",
            "--n-steps",
            "80",
            "--n-bins-x",
            "50",
            "--n-bins-y",
            "50",
        ],
    ),
]


def run_experiment(exp: Experiment, output_root: Path) -> Path:
    out_dir = output_root / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.compass.replay", "--output-dir", str(out_dir.relative_to(REPO_ROOT)), *exp.args]
    print(f"\n=== {exp.name} ===\n{exp.description}\n  cmd: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    desc = {
        "name": exp.name,
        "description": exp.description,
        "args": exp.args,
    }
    with open(out_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=2)
    return out_dir


def collect_results(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp in EXPERIMENTS:
        summary_path = output_root / exp.name / "compass_replay_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        final_c = summary["final_compass"]
        final_r = summary["final_random_mean"]
        rows.append(
            {
                "experiment": exp.name,
                "description": exp.description,
                "variant": summary["variant"],
                "map": summary["map"],
                "chunk_frames": summary["chunk_frames"],
                "n_steps": summary["n_steps"],
                "sampled_ns": final_c.get("sampled_ns"),
                "compass_jsd": final_c.get("jsd_mean", final_c.get("jsd")),
                "compass_coverage": final_c.get("coverage_mean", final_c.get("coverage")),
                "compass_f_helix_error": final_c.get("f_helix_error_mean", final_c.get("f_helix_error")),
                "random_jsd": final_r.get("jsd_mean"),
                "random_coverage": final_r.get("coverage_mean"),
                "random_f_helix_error": final_r.get("f_helix_error_mean"),
                "delta_jsd": (final_r.get("jsd_mean") or 0.0)
                - (final_c.get("jsd_mean", final_c.get("jsd")) or 0.0),
                "delta_coverage": (final_c.get("coverage_mean", final_c.get("coverage")) or 0.0)
                - (final_r.get("coverage_mean") or 0.0),
                "delta_f_helix_error": (final_r.get("f_helix_error_mean") or 0.0)
                - (final_c.get("f_helix_error_mean", final_c.get("f_helix_error")) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def plot_overlay(output_root: Path, sweep_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    specs = [
        ("jsd", "JSD vs A8Q2 reference (lower is better)"),
        ("coverage", "Coverage of A8Q2 reference (higher is better)"),
        ("f_helix_error", "Helix-fraction error (lower is better)"),
    ]

    cmap = plt.cm.tab10

    compass_summaries: dict[str, pd.DataFrame] = {}
    random_summary = None
    for exp in EXPERIMENTS:
        compass_csv = output_root / exp.name / "compass_metrics.csv"
        random_csv = output_root / exp.name / "random_summary.csv"
        if not compass_csv.exists():
            continue
        compass_df = pd.read_csv(compass_csv)
        agg = compass_df.groupby(["step", "sampled_ns"], as_index=False)[["jsd", "coverage", "f_helix_error"]].mean()
        compass_summaries[exp.name] = agg
        if random_summary is None and random_csv.exists():
            random_summary = pd.read_csv(random_csv)

    for ax, (metric, title) in zip(axes, specs):
        for i, (name, df) in enumerate(compass_summaries.items()):
            ax.plot(df["sampled_ns"], df[metric], label=name, lw=1.6, color=cmap(i % 10))
        if random_summary is not None:
            mean = random_summary[f"{metric}_mean"]
            std = random_summary[f"{metric}_std"].fillna(0.0)
            x = random_summary["sampled_ns"]
            ax.plot(x, mean, label="Random mean", color="black", lw=2.0, linestyle="--")
            ax.fill_between(x, mean - std, mean + std, color="0.65", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("sampled ns")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("COMPASS-Lite sweep: all experiments vs random baseline")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(sweep_dir / "sweep_overlay_curves.png", dpi=220)
    plt.close(fig)


def plot_summary_bars(table: pd.DataFrame, sweep_dir: Path) -> None:
    if table.empty:
        return
    table = table.sort_values("delta_jsd", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    specs = [
        ("delta_jsd", "JSD improvement vs random\n(positive = better)"),
        ("delta_coverage", "Coverage improvement vs random\n(positive = better)"),
        ("delta_f_helix_error", "Helix-error improvement vs random\n(positive = better)"),
    ]
    for ax, (col, title) in zip(axes, specs):
        ax.barh(table["experiment"], table[col], color="#4C78A8")
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(sweep_dir / "sweep_summary_bars.png", dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="If given, only run experiments whose name matches one of these.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/compass"),
        help="Root directory under which experiment outputs are stored.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip running experiments and only re-aggregate existing outputs.",
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
            run_experiment(exp, output_root)

    sweep_dir = output_root / "sweep"
    table = collect_results(output_root)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(sweep_dir / "sweep_results.csv", index=False)
    with open(sweep_dir / "sweep_results.md", "w", encoding="utf-8") as f:
        f.write("# COMPASS-Lite sweep results\n\n")
        if not table.empty:
            cols = [
                "experiment",
                "variant",
                "chunk_frames",
                "sampled_ns",
                "compass_jsd",
                "random_jsd",
                "delta_jsd",
                "compass_coverage",
                "random_coverage",
                "delta_coverage",
                "compass_f_helix_error",
                "random_f_helix_error",
                "delta_f_helix_error",
            ]
            sub = table[cols].round(4)
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join("---" for _ in cols) + " |\n")
            for _, row in sub.iterrows():
                f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
            f.write("\n## Descriptions\n\n")
            for _, row in table.iterrows():
                f.write(f"- **{row['experiment']}**: {row['description']}\n")
    plot_overlay(output_root, sweep_dir)
    plot_summary_bars(table, sweep_dir)
    print(f"\nSweep summary written to {sweep_dir}")


if __name__ == "__main__":
    main()
