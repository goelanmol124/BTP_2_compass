"""Create slide-ready COMPASS figures and a concise BTP story note."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP2 = REPO_ROOT / "archive" / "outputs_legacy" / "compass_sweep2"
OUT = SWEEP2 / "slides"


def _compass_mean(exp_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(exp_dir / "compass_metrics.csv")
    return df.groupby(["step", "sampled_ns"], as_index=False).mean(numeric_only=True)


def _random_summary(exp_dir: Path) -> pd.DataFrame:
    return pd.read_csv(exp_dir / "random_summary.csv")


def _style(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("sampled ns")


def _plot_metric(
    ax: plt.Axes,
    compass: pd.DataFrame,
    random_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    lower_is_better: bool,
    label: str = "COMPASS",
) -> None:
    x = compass["sampled_ns"]
    ax.plot(x, compass[metric], color="#D1495B", lw=2.8, label=label)
    std_col = f"{metric}_std"
    if std_col in compass and compass[std_col].fillna(0).max() > 0:
        ax.fill_between(
            x,
            compass[metric] - compass[std_col].fillna(0),
            compass[metric] + compass[std_col].fillna(0),
            color="#D1495B",
            alpha=0.16,
            linewidth=0,
        )
    r_x = random_df["sampled_ns"]
    r_mean = random_df[f"{metric}_mean"]
    lo = random_df.get(f"{metric}_lo", r_mean)
    hi = random_df.get(f"{metric}_hi", r_mean)
    ax.plot(r_x, r_mean, color="0.20", lw=2.0, label="random mean")
    ax.fill_between(r_x, lo, hi, color="0.70", alpha=0.35, label="random 95%")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    _style(ax)
    final_c = compass[metric].iloc[-1]
    final_r = r_mean.iloc[-1]
    arrow = "lower" if lower_is_better else "higher"
    ax.text(
        0.98,
        0.92,
        f"final: {final_c:.4f} vs {final_r:.4f}\n{arrow} is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.82"},
    )


def plot_observable_slide() -> None:
    det_dir = SWEEP2 / "e12_tica2d_two_phase_chunk20_80ns"
    confirm_dir = SWEEP2 / "e21_tica2d_chunk20_softmax_confirm_80ns"
    det = _compass_mean(det_dir)
    confirm = _compass_mean(confirm_dir)
    random_df = _random_summary(confirm_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    _plot_metric(axes[0], confirm, random_df, "f_helix_error", "Helix-fraction error", "absolute error", True, "COMPASS stochastic")
    axes[0].plot(det["sampled_ns"], det["f_helix_error"], color="#8B0000", lw=2.2, ls="--", label="best deterministic")
    _plot_metric(axes[1], confirm, random_df, "f_open_error", "Open-fraction error", "absolute error", True, "COMPASS stochastic")
    axes[1].plot(det["sampled_ns"], det["f_open_error"], color="#8B0000", lw=2.2, ls="--", label="best deterministic")
    fig.suptitle("COMPASS observable convergence: TICA2D two-phase, 2 ns chunks", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(OUT / "btp_observable_e12_confirm.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_discovery_slide() -> None:
    exp_dir = SWEEP2 / "e06_spib1d_two_phase_ig_80ns"
    compass = _compass_mean(exp_dir)
    random_df = _random_summary(exp_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    _plot_metric(axes[0], compass, random_df, "coverage", "Latent coverage", "fraction of reference bins", False)
    _plot_metric(axes[1], compass, random_df, "important_coverage", "Important-bin discovery", "fraction discovered", False)
    fig.suptitle("COMPASS discovery mode: SPIB1D finds latent regions early", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(OUT / "btp_discovery_spib1d.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_long_budget_slide() -> None:
    exp_dir = SWEEP2 / "e08_tica2d_two_phase_ig_200ns"
    compass = _compass_mean(exp_dir)
    random_df = _random_summary(exp_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    _plot_metric(axes[0], compass, random_df, "jsd", "JSD vs A8Q2 reference", "JSD", True)
    _plot_metric(axes[1], compass, random_df, "coverage", "Coverage recovery", "fraction of reference bins", False)
    fig.suptitle("Longer budget: TICA2D COMPASS catches and overtakes random", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(OUT / "btp_long_budget_e08.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_score_slide() -> None:
    decisions = pd.read_csv(SWEEP2 / "e12_tica2d_two_phase_chunk20_80ns" / "compass_decisions.csv")
    weighted_cols = [c for c in decisions.columns if c.startswith("weighted_")]
    sub = decisions[decisions["step"] > 1].head(40).copy()
    x = sub["step"]
    fig, ax = plt.subplots(figsize=(12, 4.6), constrained_layout=True)
    colors = {
        "weighted_source_boundary_trust": "#59A14F",
        "weighted_unvisited": "#E15759",
        "weighted_novelty": "#B07AA1",
        "weighted_source_deficit": "#4E79A7",
        "weighted_source_representative": "#F28E2B",
    }
    bottom = pd.Series(0.0, index=sub.index)
    for col in [c for c in colors if c in weighted_cols]:
        vals = sub[col].clip(lower=0)
        ax.bar(x, vals, bottom=bottom, width=0.82, label=col.replace("weighted_", ""), color=colors[col])
        bottom = bottom + vals
    if "weighted_redundancy" in weighted_cols:
        ax.bar(x, sub["weighted_redundancy"].clip(upper=0), width=0.82, label="redundancy penalty", color="#8C564B")
    ax.axvspan(1, 12.5, color="#59A14F", alpha=0.08, label="early discovery phase")
    ax.axvspan(12.5, 40.5, color="#4E79A7", alpha=0.06, label="density cleanup phase")
    ax.set_xlabel("decision step")
    ax.set_ylabel("weighted score contribution")
    ax.set_title("COMPASS decisions are interpretable: frontier first, density later", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.savefig(OUT / "btp_score_decomposition_e12.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_story() -> Path:
    story = """# COMPASS BTP Story

## One-Slide Thesis

We already have a long A10 trajectory. COMPASS asks whether that old WT map can guide a new A8Q2 mutant simulation so the next short probe is chosen for a reason, not randomly.

## Method in One Paragraph

Project both A10 and A8Q2 frames into a transferred latent map. At every step, only already visited A8Q2 chunks are eligible. COMPASS scores candidate restart chunks using source-prior density, source-boundary trust, target novelty, unvisited-bin pressure, and redundancy penalty. NHB/helix labels are never used for guidance; they are only evaluation observables.

## Result 1: Observable Mode

TICA2D two-phase COMPASS with 2 ns chunks gives the cleanest short-budget observable result. The deterministic headline run reaches helix-fraction error 0.00075 versus random 0.02080 at 80 ns. The stochastic confirmation is more conservative but still improves mean helix/open error versus random.

Use plot: `archive/outputs_legacy/compass_sweep2/slides/btp_observable_e12_confirm.png`.

## Result 2: Discovery Mode

SPIB1D is not a good equilibrium prior, but it is a strong discovery coordinate. It reaches near-saturated latent coverage and important-bin discovery around 12 ns, while random takes roughly 70 ns.

Use plot: `archive/outputs_legacy/compass_sweep2/slides/btp_discovery_spib1d.png`.

## Result 3: Long-Budget Recovery

At 200 ns, TICA2D two-phase COMPASS also overtakes random on global JSD and coverage. This means the short-budget JSD loss is transient, not a fundamental failure.

Use plot: `archive/outputs_legacy/compass_sweep2/slides/btp_long_budget_e08.png`.

## Mechanistic Support

The score-decomposition plot shows the intended behavior: early decisions are dominated by source-boundary/unvisited/novelty terms, and later decisions shift toward source-deficit/source-representative density cleanup. This makes the method explainable.

Use plot: `archive/outputs_legacy/compass_sweep2/slides/btp_score_decomposition_e12.png`.

## Honest Caveats

Replay-random is a very strong baseline because it samples from the completed A8Q2 chunk pool. These results prove the policy logic and provide a conservative retrospective benchmark; they do not yet prove wall-clock online MD speedup. SPIB should be described as discovery mode, not thermodynamic mode.

## Next Scientific Step

Hybrid scoring was tested as a first pass after the main sweep: SPIB frontier discovery plus TICA density correction. The simple hybrid is not the new headline. At 80 ns it hurts helix/open observables relative to random; at 200 ns it improves JSD but only modestly. The lesson is that hybridization needs a trust/controller layer, not just addition of SPIB frontier terms.

Next version: use SPIB only when the objective is discovery or when TICA density has stalled; otherwise keep TICA2D density correction in control.
"""
    story_out = REPO_ROOT / "archive" / "docs_deprecated" / "GENERATED_COMPASS_BTP_STORY.md"
    story_out.parent.mkdir(parents=True, exist_ok=True)
    story_out.write_text(story, encoding="utf-8")
    return story_out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_observable_slide()
    plot_discovery_slide()
    plot_long_budget_slide()
    plot_score_slide()
    story_path = write_story()
    print(f"Wrote slide assets to {OUT}")
    print(f"Wrote {story_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
