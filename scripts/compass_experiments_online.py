"""Online-faithful COMPASS replay sweep (observed-seed action set only).

Supports a **core** matrix (plan §9) and an **extended** matrix mirroring the old
enhanced sweep axes: schedules, chunk sizes, map kinds, priors, trust controller,
MSM lag, long budget, and trajectory/MSM/oracle families.
"""

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


def apply_budget_args(args: list[str], n_steps: int | None, random_repeats: int | None) -> list[str]:
    out = list(args)
    if n_steps is not None:
        _replace_after_flag(out, "--n-steps", str(n_steps))
    if random_repeats is not None:
        _replace_after_flag(out, "--random-repeats", str(random_repeats))
    return out


def _replace_after_flag(argv: list[str], flag: str, value: str) -> None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            argv[i + 1] = value
            return


# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    args: list[str] = field(default_factory=list)
    family: str = "online"  # oracle | msm | trajectory


def _base(minimal: list[str]) -> list[str]:
    """Start from balanced two-phase TICA2D 80 ns @ 1 ns chunks, 100 repeats."""
    return [
        *minimal,
        "--map-kind",
        "tica2d",
        "--chunk-frames",
        "10",
        "--n-steps",
        "80",
        "--random-repeats",
        "100",
        "--variant",
        "balanced",
        "--schedule",
        "two_phase",
        "--deficit-mode",
        "information_gain",
    ]


def build_core_experiments() -> list[Experiment]:
    b = lambda **_: _base([])
    return [
        Experiment(
            "o00_oracle_baseline_80ns",
            "Legacy oracle chunk replay (full cached chunk library); diagnostic only.",
            [*b(), "--environment", "oracle"],
            "oracle",
        ),
        Experiment(
            "t01_trajectory_single_seed_80ns",
            "Trajectory continuation; single seed frame 0.",
            [*b(), "--environment", "trajectory_replay", "--initial-seeds", "0"],
            "trajectory",
        ),
        Experiment(
            "t02_trajectory_two_initial_seeds_80ns",
            "Trajectory continuation; extended + helix starts (auto).",
            [*b(), "--environment", "trajectory_replay", "--initial-seeds", "auto"],
            "trajectory",
        ),
        Experiment(
            "t03_trajectory_allow_reseed_cost_80ns",
            "Trajectory: allow deterministic re-seed + restart cost.",
            [
                *b(),
                "--environment",
                "trajectory_replay",
                "--initial-seeds",
                "0",
                "--allow-deterministic-reseed",
                "--restart-cost",
                "0.35",
            ],
            "trajectory",
        ),
        Experiment(
            "m01_msm_single_seed_80ns",
            "MSM emulator; single initial seed; lag defaults to chunk_frames.",
            [*b(), "--environment", "msm", "--initial-seeds", "0"],
            "msm",
        ),
        Experiment(
            "m02_msm_two_initial_seeds_80ns",
            "MSM; two initial seeds (auto).",
            [*b(), "--environment", "msm", "--initial-seeds", "auto"],
            "msm",
        ),
        Experiment(
            "m03_msm_lag_sensitivity_80ns",
            "MSM; msm-lag 5 frames.",
            [*b(), "--environment", "msm", "--initial-seeds", "0", "--msm-lag", "5"],
            "msm",
        ),
        Experiment(
            "t04_trajectory_pure_frontier_80ns",
            "Trajectory; pure_frontier variant.",
            [
                "--map-kind",
                "tica2d",
                "--chunk-frames",
                "10",
                "--n-steps",
                "80",
                "--random-repeats",
                "100",
                "--variant",
                "pure_frontier",
                "--schedule",
                "static",
                "--deficit-mode",
                "information_gain",
                "--environment",
                "trajectory_replay",
                "--initial-seeds",
                "0",
            ],
            "trajectory",
        ),
        Experiment(
            "m04_msm_pure_frontier_80ns",
            "MSM; pure_frontier variant.",
            [
                "--map-kind",
                "tica2d",
                "--chunk-frames",
                "10",
                "--n-steps",
                "80",
                "--random-repeats",
                "100",
                "--variant",
                "pure_frontier",
                "--schedule",
                "static",
                "--deficit-mode",
                "information_gain",
                "--environment",
                "msm",
                "--initial-seeds",
                "0",
            ],
            "msm",
        ),
    ]


def build_extended_experiments() -> list[Experiment]:
    """Additional experiments (beyond core) for an extensive suite."""
    xs: list[Experiment] = []

    def add(name: str, desc: str, args: list[str], fam: str) -> None:
        xs.append(Experiment(name, desc, args, fam))

    # --- Oracle mirrors of headline axes (diagnostic reference) ---
    add(
        "o10_oracle_chunk20_80ns",
        "Oracle; 2 ns chunks, 40 steps (80 ns).",
        [
            "--environment",
            "oracle",
            "--map-kind",
            "tica2d",
            "--chunk-frames",
            "20",
            "--n-steps",
            "40",
            "--random-repeats",
            "100",
            "--variant",
            "balanced",
            "--schedule",
            "two_phase",
            "--deficit-mode",
            "information_gain",
        ],
        "oracle",
    )
    add(
        "o11_oracle_chunk5_80ns",
        "Oracle; 0.5 ns chunks, 160 steps.",
        [
            "--environment",
            "oracle",
            "--map-kind",
            "tica2d",
            "--chunk-frames",
            "5",
            "--n-steps",
            "160",
            "--random-repeats",
            "100",
            "--variant",
            "balanced",
            "--schedule",
            "two_phase",
            "--deficit-mode",
            "information_gain",
        ],
        "oracle",
    )
    for mk, tag in [
        ("tica1d", "tica1d"),
        ("spib1d", "spib1d"),
        ("spib2d", "spib2d"),
    ]:
        add(
            f"o20_oracle_{tag}_80ns",
            f"Oracle; {mk} map.",
            [
                "--environment",
                "oracle",
                "--map-kind",
                mk,
                "--chunk-frames",
                "10",
                "--n-steps",
                "80",
                "--random-repeats",
                "100",
                "--variant",
                "balanced",
                "--schedule",
                "two_phase",
                "--deficit-mode",
                "information_gain",
            ],
            "oracle",
        )
    for hybrid_map, hid in [
        ("hybrid_spib1d_tica2d", "hybrid_spib1d_tica2d"),
        ("hybrid_spib2d_tica2d", "hybrid_spib2d_tica2d"),
    ]:
        add(
            f"o30_oracle_{hid}_80ns",
            f"Oracle; {hybrid_map}.",
            [
                "--environment",
                "oracle",
                "--map-kind",
                hybrid_map,
                "--chunk-frames",
                "10",
                "--n-steps",
                "80",
                "--random-repeats",
                "100",
                "--variant",
                "hybrid",
                "--schedule",
                "two_phase",
                "--deficit-mode",
                "information_gain",
            ],
            "oracle",
        )

    # --- MSM: schedules & deficit ---
    for sched in ["static", "anneal"]:
        add(
            f"m10_msm_sched_{sched}_80ns",
            f"MSM; schedule={sched}.",
            [*_base([]), "--environment", "msm", "--initial-seeds", "0", "--schedule", sched],
            "msm",
        )
    add(
        "m11_msm_deficit_raw_80ns",
        "MSM; raw deficit mode.",
        [
            *_base([]),
            "--environment",
            "msm",
            "--initial-seeds",
            "0",
            "--deficit-mode",
            "raw",
            "--schedule",
            "static",
        ],
        "msm",
    )

    # --- MSM: chunk sizes ---
    add(
        "m12_msm_chunk20_80ns",
        "MSM; 2 ns chunks.",
        [
            *_base([]),
            "--environment",
            "msm",
            "--initial-seeds",
            "0",
            "--chunk-frames",
            "20",
            "--n-steps",
            "40",
        ],
        "msm",
    )
    add(
        "m13_msm_chunk5_80ns",
        "MSM; 0.5 ns chunks.",
        [
            *_base([]),
            "--environment",
            "msm",
            "--initial-seeds",
            "0",
            "--chunk-frames",
            "5",
            "--n-steps",
            "160",
        ],
        "msm",
    )

    # --- MSM: map kinds ---
    for mk, tag in [
        ("tica1d", "tica1d"),
        ("spib1d", "spib1d"),
        ("spib2d", "spib2d"),
    ]:
        add(
            f"m20_msm_{tag}_80ns",
            f"MSM; {mk}.",
            [*_base([]), "--environment", "msm", "--initial-seeds", "0", "--map-kind", mk],
            "msm",
        )
    for hybrid_map, hid in [
        ("hybrid_spib1d_tica2d", "hybrid_spib1d_tica2d"),
        ("hybrid_spib2d_tica2d", "hybrid_spib2d_tica2d"),
    ]:
        add(
            f"m30_msm_{hid}_80ns",
            f"MSM; {hybrid_map}.",
            [
                "--environment",
                "msm",
                "--initial-seeds",
                "0",
                "--map-kind",
                hybrid_map,
                "--chunk-frames",
                "10",
                "--n-steps",
                "80",
                "--random-repeats",
                "100",
                "--variant",
                "hybrid",
                "--schedule",
                "two_phase",
                "--deficit-mode",
                "information_gain",
            ],
            "msm",
        )

    # --- MSM: lag grid ---
    for lag, lid in [(20, "lag20"), (10, "lag10")]:
        add(
            f"m40_msm_{lid}_80ns",
            f"MSM; msm-lag={lag}.",
            [*_base([]), "--environment", "msm", "--initial-seeds", "0", "--msm-lag", str(lag)],
            "msm",
        )

    # --- MSM: priors & robustness ---
    for pq, label in [
        ("uniform", "prior_uniform"),
        ("noisy", "prior_noisy"),
        ("inverted", "prior_inverted"),
        ("shuffled", "prior_shuffled"),
    ]:
        extra = []
        if pq == "noisy":
            extra = ["--prior-quality", pq, "--prior-strength", "0.5"]
        else:
            extra = ["--prior-quality", pq]
        add(
            f"m50_msm_{label}_80ns",
            f"MSM; prior_quality={pq}.",
            [*_base([]), "--environment", "msm", "--initial-seeds", "0", *extra],
            "msm",
        )

    # --- MSM: trust, restart, novelty, softmax, long budget, reservoir ---
    add(
        "m60_msm_trust_prior_80ns",
        "MSM; prior_agreement trust controller.",
        [
            *_base([]),
            "--environment",
            "msm",
            "--initial-seeds",
            "0",
            "--trust-controller",
            "prior_agreement",
        ],
        "msm",
    )
    add(
        "m61_msm_restart_cost_80ns",
        "MSM; restart cost 0.35.",
        [*_base([]), "--environment", "msm", "--initial-seeds", "0", "--restart-cost", "0.35"],
        "msm",
    )
    add(
        "m62_msm_epistemic_novelty_80ns",
        "MSM; epistemic novelty.",
        [*_base([]), "--environment", "msm", "--initial-seeds", "0", "--novelty-mode", "epistemic"],
        "msm",
    )
    add(
        "m63_msm_softmax_stoch_80ns",
        "MSM; stochastic softmax + multi compass repeats.",
        [
            *_base([]),
            "--environment",
            "msm",
            "--initial-seeds",
            "0",
            "--softmax-temperature",
            "0.4",
            "--compass-repeats",
            "8",
        ],
        "msm",
    )
    add(
        "m64_msm_long200ns",
        "MSM; 200 ns budget (chunk 10, 200 steps).",
        [
            *_base([]),
            "--environment",
            "msm",
            "--initial-seeds",
            "0",
            "--n-steps",
            "200",
        ],
        "msm",
    )
    add(
        "m65_msm_seed_stride2_80ns",
        "MSM; seed reservoir stride 2.",
        [*_base([]), "--environment", "msm", "--initial-seeds", "0", "--seed-reservoir-stride", "2"],
        "msm",
    )
    add(
        "m66_msm_exploratory_80ns",
        "MSM; exploratory variant.",
        [
            "--map-kind",
            "tica2d",
            "--chunk-frames",
            "10",
            "--n-steps",
            "80",
            "--random-repeats",
            "100",
            "--variant",
            "exploratory",
            "--schedule",
            "static",
            "--deficit-mode",
            "information_gain",
            "--environment",
            "msm",
            "--initial-seeds",
            "0",
        ],
        "msm",
    )

    # --- Trajectory mirrors (subset) ---
    add(
        "t10_traj_sched_static_80ns",
        "Trajectory; static schedule.",
        [*_base([]), "--environment", "trajectory_replay", "--initial-seeds", "0", "--schedule", "static"],
        "trajectory",
    )
    add(
        "t11_traj_chunk20_80ns",
        "Trajectory; chunk 20, 40 steps.",
        [
            *_base([]),
            "--environment",
            "trajectory_replay",
            "--initial-seeds",
            "0",
            "--chunk-frames",
            "20",
            "--n-steps",
            "40",
        ],
        "trajectory",
    )
    add(
        "t12_traj_spib1d_80ns",
        "Trajectory; SPIB 1D map.",
        [
            *_base([]),
            "--environment",
            "trajectory_replay",
            "--initial-seeds",
            "0",
            "--map-kind",
            "spib1d",
        ],
        "trajectory",
    )
    add(
        "t13_traj_prior_noisy_80ns",
        "Trajectory; noisy prior.",
        [
            *_base([]),
            "--environment",
            "trajectory_replay",
            "--initial-seeds",
            "0",
            "--prior-quality",
            "noisy",
            "--prior-strength",
            "0.5",
        ],
        "trajectory",
    )

    return xs


def build_experiments(suite: str) -> list[Experiment]:
    core = build_core_experiments()
    if suite == "core":
        return core
    ext = build_extended_experiments()
    if suite == "extended":
        return ext
    if suite == "all":
        # De-duplicate by name (core first)
        seen: set[str] = set()
        out: list[Experiment] = []
        for e in core + ext:
            if e.name not in seen:
                seen.add(e.name)
                out.append(e)
        return out
    raise ValueError(suite)


METRICS = list(METRIC_CORE)


def run_experiment(
    exp: Experiment,
    output_root: Path,
    n_steps: int | None,
    random_repeats: int | None,
) -> Path:
    out_dir = output_root / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = apply_budget_args(exp.args, n_steps, random_repeats)
    cmd = [
        sys.executable,
        "-m",
        "src.compass.replay_online",
        "--output-dir",
        str(out_dir.relative_to(REPO_ROOT)),
        *argv,
    ]
    print(f"\n=== {exp.name} ({exp.family}) ===\n{exp.description}\n  cmd: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    with open(out_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"name": exp.name, "description": exp.description, "args": argv, "family": exp.family},
            f,
            indent=2,
        )
    return out_dir


def collect_results(output_root: Path, experiments: list[Experiment]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp in experiments:
        summary_path = output_root / exp.name / "compass_replay_summary.json"
        if not summary_path.exists():
            continue
        summary = load_summary(summary_path)
        base = {
            "experiment": exp.name,
            "description": exp.description,
            "family": exp.family,
            "sweep_driver": "online_faithful",
        }
        rows.append(flatten_summary(summary, base))
    return pd.DataFrame(rows)


def _read_compass_mean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    available = [m for m in METRICS if m in df.columns]
    group_cols = [c for c in ("step", "sampled_ns") if c in df.columns]
    if not group_cols:
        return df
    return df.groupby(group_cols, as_index=False)[available].mean()


def plot_overlay(output_root: Path, sweep_dir: Path, experiments: list[Experiment]) -> None:
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
    for i, exp in enumerate(experiments):
        compass_csv = output_root / exp.name / "compass_metrics.csv"
        random_csv = output_root / exp.name / "random_summary.csv"
        if not compass_csv.exists():
            continue
        df = _read_compass_mean(compass_csv)
        if df.empty:
            continue
        label = f"{exp.family[:1]}:{exp.name.replace('_', ' ')[:40]}"
        for ax, (metric, _title) in zip(axes.ravel(), specs):
            if metric in df.columns and "sampled_ns" in df.columns:
                ax.plot(df["sampled_ns"], df[metric], label=label, lw=1.0, color=cmap(i % 20), alpha=0.85)
        if random_summary is None and random_csv.exists():
            random_summary = pd.read_csv(random_csv)
    for ax, (metric, title) in zip(axes.ravel(), specs):
        if random_summary is not None and metric in random_summary.columns:
            x = random_summary["sampled_ns"]
            mean = random_summary[f"{metric}_mean"]
            lo = random_summary.get(f"{metric}_lo", mean)
            hi = random_summary.get(f"{metric}_hi", mean)
            ax.plot(x, mean, label="Random ref (last exp)", color="black", lw=2.0, linestyle="--")
            ax.fill_between(x, lo, hi, color="0.6", alpha=0.20)
        ax.set_title(title)
        ax.set_xlabel("sampled ns")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(frameon=False, fontsize=5, loc="upper right")
    fig.suptitle("Online-faithful COMPASS sweep — all completed runs")
    fig.savefig(sweep_dir / "online_sweep_overlay_curves.png", dpi=220)
    plt.close(fig)


def plot_overlays_by_family(
    output_root: Path, sweep_dir: Path, experiments: list[Experiment]
) -> None:
    specs = [
        ("f_helix_error", "Helix error"),
        ("coverage", "Coverage"),
        ("important_coverage", "Important coverage"),
    ]
    for family, title_sfx in [("msm", "MSM"), ("trajectory", "Trajectory"), ("oracle", "Oracle")]:
        sub = [e for e in experiments if e.family == family]
        if len(sub) < 2:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        cmap = plt.cm.tab20
        random_summary = None
        for i, exp in enumerate(sub):
            compass_csv = output_root / exp.name / "compass_metrics.csv"
            random_csv = output_root / exp.name / "random_summary.csv"
            if not compass_csv.exists():
                continue
            df = _read_compass_mean(compass_csv)
            if df.empty:
                continue
            label = exp.name.replace("_", " ")[:45]
            for ax, (metric, _) in zip(axes, specs):
                if metric in df.columns and "sampled_ns" in df.columns:
                    ax.plot(
                        df["sampled_ns"],
                        df[metric],
                        label=label,
                        lw=1.1,
                        color=cmap(i % 20),
                        alpha=0.9,
                    )
            if random_summary is None and random_csv.exists():
                random_summary = pd.read_csv(random_csv)
        for ax, (metric, tl) in zip(axes, specs):
            if random_summary is not None and metric in random_summary.columns:
                x = random_summary["sampled_ns"]
                mean = random_summary[f"{metric}_mean"]
                lo = random_summary.get(f"{metric}_lo", mean)
                hi = random_summary.get(f"{metric}_hi", mean)
                ax.plot(x, mean, color="black", lw=1.8, linestyle="--", label="Random (ref)")
                ax.fill_between(x, lo, hi, color="0.75", alpha=0.2)
            ax.set_title(tl)
            ax.set_xlabel("sampled ns")
            ax.grid(alpha=0.25)
        axes[0].legend(frameon=False, fontsize=5, loc="best")
        fig.suptitle(f"COMPASS online sweep — {title_sfx} experiments")
        fig.savefig(sweep_dir / f"online_sweep_curves_{family}.png", dpi=200)
        plt.close(fig)


def plot_delta_heatmap(table: pd.DataFrame, sweep_dir: Path) -> None:
    if table.empty or "family" not in table.columns:
        return
    col = "delta_f_helix_error"
    if col not in table.columns:
        return
    sub = table.dropna(subset=[col]).copy()
    if sub.empty:
        return
    # Pivot: family x map (coarse view)
    try:
        pivot = sub.pivot_table(index="family", columns="map", values=col, aggfunc="mean")
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Mean Δ helix error (random − COMPASS)\nblue = COMPASS better")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(sweep_dir / "online_sweep_delta_heatmap.png", dpi=180)
    plt.close(fig)


def plot_summary_bars(table: pd.DataFrame, sweep_dir: Path) -> None:
    if table.empty:
        return
    specs = [
        ("delta_f_helix_error", "Helix-error Δ (R−C)"),
        ("delta_coverage", "Coverage Δ"),
        ("delta_important_coverage", "Important coverage Δ"),
        ("delta_jsd", "JSD Δ"),
        ("delta_source_mass_visited", "Source mass Δ"),
        ("delta_f_open_error", "Open-error Δ"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    metric = "delta_f_helix_error"
    order = table.sort_values(metric, ascending=True)["experiment"]
    for ax, (col, title) in zip(axes.ravel(), specs):
        if col not in table.columns:
            continue
        sub = table.set_index("experiment").reindex(order).reset_index()
        colors = np.where(sub[col].fillna(0) >= 0, "#4C78A8", "#E45756")
        ax.barh(sub["experiment"], sub[col], color=colors)
        ax.axvline(0.0, color="0.35", lw=0.8)
        ax.set_title(f"{title}\n(positive ⟹ COMPASS better for error metrics)")
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(sweep_dir / "online_sweep_summary_bars.png", dpi=220)
    plt.close(fig)


def write_markdown(table: pd.DataFrame, sweep_dir: Path) -> None:
    with open(sweep_dir / "online_sweep_results.md", "w", encoding="utf-8") as f:
        f.write("# Online-faithful COMPASS sweep results\n\n")
        if table.empty:
            f.write("No completed experiment summaries found.\n")
            return
        cols = [
            "experiment",
            "family",
            "environment",
            "map",
            "variant",
            "sampled_ns",
            "delta_f_helix_error",
            "delta_coverage",
            "delta_important_coverage",
            "delta_jsd",
            "delta_source_mass_visited",
            "delta_f_open_error",
        ]
        use = [c for c in cols if c in table.columns]
        sub = table[use].copy()
        for c in sub.columns:
            if sub[c].dtype == float:
                sub[c] = sub[c].round(4)
        f.write("| " + " | ".join(sub.columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in sub.columns) + " |\n")
        for _, row in sub.iterrows():
            f.write("| " + " | ".join(str(row[c]) for c in sub.columns) + " |\n")
        f.write("\n## Descriptions\n\n")
        for _, row in table.iterrows():
            dcol = "description" if "description" in table.columns else None
            desc = row[dcol] if dcol else ""
            f.write(f"- **{row['experiment']}**: {desc}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="Filter by experiment name.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/compass_sweep_online"))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--suite",
        choices=["core", "extended", "all"],
        default="all",
        help="core = plan §9 only; extended = extra matrix only; all = union (recommended).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override every --n-steps in experiment args (quick sweeps).",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=None,
        help="Override every --random-repeats in experiment args.",
    )
    args = parser.parse_args(argv)
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args.suite)
    if args.only:
        wanted = set(args.only)
        experiments = [e for e in experiments if e.name in wanted]
        if not experiments:
            raise SystemExit(f"No experiments matched {sorted(wanted)}")

    if not args.aggregate_only:
        for exp in experiments:
            run_experiment(exp, output_root, args.n_steps, args.random_repeats)

    sweep_dir = output_root / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    names = [e.name for e in experiments]
    table_full = collect_results(output_root, experiments)
    table_full.to_csv(sweep_dir / "online_sweep_results_full.csv", index=False)
    slim = slim_columns_wide_df(table_full)
    slim.to_csv(sweep_dir / "online_sweep_results.csv", index=False)
    write_summaries_jsonl(output_root, names, sweep_dir / "online_sweep_summaries.jsonl")
    export_timeseries_long(output_root, names, sweep_dir / "online_sweep_timeseries_long.csv")
    write_markdown(slim, sweep_dir)
    plot_overlay(output_root, sweep_dir, experiments)
    plot_overlays_by_family(output_root, sweep_dir, experiments)
    plot_delta_heatmap(slim, sweep_dir)
    plot_summary_bars(slim, sweep_dir)
    plot_delta_correlation(
        table_full,
        sweep_dir / "online_sweep_delta_correlation.png",
        "Online-faithful sweep: Δ metric correlation across experiments",
    )
    plot_compass_vs_random_grid(
        table_full,
        sweep_dir / "online_sweep_compass_vs_random_scatter.png",
        "Online-faithful: COMPASS vs random (final means)",
    )
    plot_delta_ranked(
        table_full,
        sweep_dir / "online_sweep_top_helix_delta.png",
        "Top online-faithful experiments by helix-error advantage",
    )
    print(f"\nOnline sweep summary written to {sweep_dir}")


if __name__ == "__main__":
    main()
