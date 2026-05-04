"""Generate interpretable figures from MSM benchmark outputs (compare_msm_three_policies.py).

Reads ``archive/outputs_legacy/msm_benchmark/per_step_all.parquet`` or ``per_step_all.csv`` plus
references/<preset>/references.json and references.npz.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _apply_plot_style(*, compact: bool) -> None:
    """Publication-friendly defaults; ``compact`` drops footer essays and bumps PNG DPI."""
    base = {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "font.size": 9 if compact else 9,
        "axes.titlesize": 9 if compact else 9,
        "axes.labelsize": 8.5 if compact else 8.5,
        "legend.fontsize": 7 if compact else 6.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    plt.rcParams.update(base)


def _png_save_kw(*, compact: bool) -> dict:
    return {
        "dpi": 240 if compact else 200,
        "bbox_inches": "tight",
        "facecolor": "white",
        "pad_inches": 0.05 if compact else 0.08,
    }


# ---------------------------------------------------------------------------
# Display constants — written for readers who do not have the paper/repo open
# ---------------------------------------------------------------------------

POLICY_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
# Short labels for tight spaces (bar charts, violins)
POLICY_LABELS_SHORT = {
    "B0": "B0 · long MSM",
    "B1": "B1 · replicas",
    "B2": "B2 · uniform",
    "B3": "B3 · min-count",
    "B4": "B4 · farthest",
    "B5": "B5 · COMPASS frontier",
    "B6": "B6 · COMPASS full",
}
# Long labels for legends
POLICY_LABELS = {
    "B0": "B0 · one continuous MSM walk (same # of transitions as others)",
    "B1": "B1 · many independent walks; start state cycles fixed seeds",
    "B2": "B2 · each step: uniform over states seen so far",
    "B3": "B3 · each step: restart from bin with least accumulated MSM mass",
    "B4": "B4 · each step: restart from bin farthest from prior restarts (2D map)",
    "B5": "B5 · COMPASS with frontier-only scoring (no source-density weight)",
    "B6": "B6 · COMPASS: prior × MSM information score (full objective)",
}
POLICY_COLORS = {
    "B0": "#7f7f7f",
    "B1": "#bcbd22",
    "B2": "#ff7f0e",
    "B3": "#9467bd",
    "B4": "#8c564b",
    "B5": "#e377c2",
    "B6": "#1f77b4",
}

PRESET_ORDER = [
    "01_tica2d_baseline_m01",
    "02_tica2d_chunk20_m12",
    "03_spib1d_m20",
    "04_tica2d_long200_m64",
    "05_tica2d_two_seed_m02",
]
# One-line scenario description for column titles
PRESET_SCENARIO = {
    "01_tica2d_baseline_m01": "Latent: 2D tICA · 80 MSM rounds · 10 frames / segment · 1 seed",
    "02_tica2d_chunk20_m12": "Latent: 2D tICA · 40 MSM rounds · 20 frames / segment · 1 seed",
    "03_spib1d_m20": "Latent: 1D SPIB · 80 MSM rounds · 10 frames / segment · 1 seed",
    "04_tica2d_long200_m64": "Latent: 2D tICA · 200 MSM rounds · 10 frames / segment · 1 seed",
    "05_tica2d_two_seed_m02": "Latent: 2D tICA · 80 MSM rounds · 10 frames / segment · 2 seeds",
}

XAXIS_NS = (
    "Cumulative sampled time (ns)\n"
    "(sum of MSM histogram counts × 0.1 ns/frame; emulator budget, not wall-clock MD)"
)

GLOSSARY_FIG1 = textwrap.fill(
    "How to read rows: "
    "JSD measures how different the current 2D bin histogram is from the reference target histogram (lower is better). "
    "Coverage is the fraction of reference-supported bins—those with reference count ≥ 3—that have received any sampled mass "
    "(higher is better). "
    "|Δf_helix| and |Δf_open| are absolute errors in helix and open-state population fractions vs the segment-weighted target "
    "(lower is better). "
    "Shaded bands: 25th–75th percentile across independent repeats; lines: mean. "
    "Horizontal dashed line on JSD row: 2× the median bootstrap JSD split noise floor for that geometry (sanity reference, not a tuned target).",
    width=118,
)

GLOSSARY_FIG2 = textwrap.fill(
    "Vertical axis: cumulative prior probability mass on bins flagged as “important” that have been visited at least once "
    "(0 = none, 1 = all important prior mass touched). "
    "Shaded band = inter-quartile range across repeats; line = mean. "
    "This answers: “How much of the informative region has the policy actually seen?”",
    width=118,
)

GLOSSARY_FIG3 = textwrap.fill(
    "Each bar: median time (ns) until the inequality in the row label is first satisfied along the run; "
    "shorter is better. "
    "Red dashed vertical line: maximum budget for that preset (runs that never hit the target are drawn to this line). "
    "JSD target uses twice the median bootstrap split noise for that map. "
    "If many bars sit on the red line, the threshold is too strict for the fixed transition budget.",
    width=118,
)

GLOSSARY_FIG5 = textwrap.fill(
    "Distributions of metric values at the final MSM step (final budget), over all repeats. "
    "Violin width ∝ frequency. "
    "JSD & errors: lower median is better. Coverage: higher median is better.",
    width=118,
)

GLOSSARY_FIG6 = textwrap.fill(
    "Pairwise policy differences along the same random repeats. "
    "For JSD, |Δf_helix|, |Δf_open|: a negative B6−B3 means B6 is closer to truth than B3 (better). "
    "For coverage & important coverage: a positive B6−B3 means B6 exceeds B3 (better). "
    "Bands: ±1 standard error of the mean across repeats. "
    "B3−B1 isolates adaptive restarts vs independent replicas; B6−B3 isolates the COMPASS objective vs pure min-count.",
    width=118,
)

PAIR_LINE_COLORS = {"B6−B3": "#1f77b4", "B3−B1": "#ff7f0e", "B6−B1": "#2ca02c"}


def _panel_title(preset: str) -> str:
    name = preset.replace("_", " ")
    scen = PRESET_SCENARIO.get(preset, "")
    return f"{name}\n{scen}" if scen else name


def _load_long(repo_root: Path, benchmark_dir: Path) -> pd.DataFrame:
    p = benchmark_dir / "per_step_all.parquet"
    if p.is_file():
        return pd.read_parquet(p)
    return pd.read_csv(benchmark_dir / "per_step_all.csv")


def _load_ref_json(repo_root: Path, benchmark_dir: Path, preset: str) -> dict:
    with open(benchmark_dir / "references" / preset / "references.json", encoding="utf-8") as f:
        return json.load(f)


def _load_ref_npz(benchmark_dir: Path, preset: str) -> dict:
    p = benchmark_dir / "references" / preset / "references.npz"
    return dict(np.load(p))


def fig1_convergence_grid(df: pd.DataFrame, benchmark_dir: Path, fig_dir: Path, *, compact: bool) -> None:
    metrics = [
        ("jsd", "Jensen–Shannon divergence\n(2D bin histogram vs reference target)"),
        ("coverage", "Coverage of ref.-supported bins\n(fraction with ref. count ≥3 that have sample >0)"),
        ("f_helix_error", "|Δf_helix|\n(abs. error vs target helix fraction)"),
        ("f_open_error", "|Δf_open|\n(abs. error vs target open fraction)"),
    ]
    presets = [p for p in PRESET_ORDER if p in df["preset"].unique()]
    policies_in = [p for p in POLICY_ORDER if p in df["policy"].unique()]
    fig, axes = plt.subplots(
        len(metrics),
        len(presets),
        figsize=(4.1 * len(presets), 3.5 * len(metrics)),
        squeeze=False,
    )
    fig.suptitle(
        (
            "MSM benchmark: convergence vs cumulative sampled time\n"
            "(identical MSM lag and bin grid per column)"
            if compact
            else (
                "Fig. 1 · MSM sampling policies — convergence vs cumulative emulator time\n"
                "(Markov state model replay benchmark; identical MSM lag and bin grid per column)"
            )
        ),
        fontsize=11 if compact else 11,
        fontweight="bold",
        y=0.995,
    )
    for j, preset in enumerate(presets):
        noise = _load_ref_json(Path(), benchmark_dir, preset)["noise_floor"]["jsd_bootstrap_p50"]
        sub = df[df["preset"] == preset]
        for i, (col, ytitle) in enumerate(metrics):
            ax = axes[i][j]
            for pol in policies_in:
                subp = sub[sub["policy"] == pol]
                g = subp.groupby("sampled_ns")[col].agg(
                    ["mean", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
                )
                g.columns = ["mean", "q25", "q75"]
                x = g.index.values.astype(float)
                ax.fill_between(x, g["q25"], g["q75"], color=POLICY_COLORS[pol], alpha=0.2)
                ax.plot(
                    x,
                    g["mean"],
                    label=POLICY_LABELS[pol],
                    color=POLICY_COLORS[pol],
                    linewidth=1.8,
                )
            if col == "jsd":
                ax.axhline(
                    noise * 2,
                    color="0.35",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=0,
                )
            else:
                ax.axhline(0.0, color="0.45", linestyle=":", linewidth=1, alpha=0.7, zorder=0)
            if i == 0:
                ax.set_title(_panel_title(preset), fontsize=8.5)
            if j == 0:
                ax.set_ylabel(ytitle, fontsize=8.5)
            if i == len(metrics) - 1:
                ax.set_xlabel(XAXIS_NS, fontsize=7.5)

    policy_handles = [
        Line2D([0], [0], color=POLICY_COLORS[p], lw=2.4, label=POLICY_LABELS[p]) for p in policies_in
    ]
    ref_handle = Line2D(
        [0],
        [0],
        color="0.35",
        ls="--",
        lw=1.5,
        alpha=0.85,
        label="Reference only (JSD row): 2 × median bootstrap split-noise estimate",
    )
    ncol_leg = 3 if compact else 1
    leg_y = -0.02 if compact else 0.002
    leg = fig.legend(
        handles=policy_handles + [ref_handle],
        loc="lower center",
        ncol=ncol_leg,
        fontsize=plt.rcParams["legend.fontsize"],
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        title="Legend",
        title_fontsize=7,
        bbox_to_anchor=(0.5, leg_y),
    )
    leg.get_title().set_fontweight("bold")

    if not compact:
        fig.text(0.5, -0.16, GLOSSARY_FIG1, ha="center", va="top", fontsize=6.5, linespacing=1.25)
    bottom_m = 0.18 if compact else 0.31
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=bottom_m, hspace=0.45, wspace=0.28)
    fig.savefig(fig_dir / "fig1_convergence_grid.png", **_png_save_kw(compact=compact))
    fig.savefig(fig_dir / "fig1_convergence_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_source_mass_discovery(df: pd.DataFrame, benchmark_dir: Path, fig_dir: Path, *, compact: bool) -> None:
    presets = [p for p in PRESET_ORDER if p in df["preset"].unique()]
    policies_in = [p for p in POLICY_ORDER if p in df["policy"].unique()]
    fig, axes = plt.subplots(1, len(presets), figsize=(4.2 * len(presets), 4.8), squeeze=False)
    fig.suptitle(
        (
            "Important-bin discovery under the source prior"
            if compact
            else (
                "Fig. 2 · Discovery of important bins under the source prior\n"
                "(how much prior mass on analyst-important regions has been visited?)"
            )
        ),
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    for j, preset in enumerate(presets):
        ax = axes[0][j]
        sub = df[df["preset"] == preset]
        col = "source_mass_important_visited_frac"
        for pol in policies_in:
            subp = sub[sub["policy"] == pol]
            g = subp.groupby("sampled_ns")[col].agg(
                ["mean", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            )
            g.columns = ["mean", "q25", "q75"]
            x = g.index.values.astype(float)
            ax.fill_between(x, g["q25"], g["q75"], color=POLICY_COLORS[pol], alpha=0.2)
            ax.plot(x, g["mean"], label=POLICY_LABELS[pol], color=POLICY_COLORS[pol], linewidth=1.8)
        ax.set_title(_panel_title(preset), fontsize=8.5)
        ax.set_xlabel(XAXIS_NS, fontsize=7.5)
        ax.set_ylabel(
            "Important-region prior mass visited (0–1)\n"
            "Σ prior prob. on important bins with hist. mass > 0",
            fontsize=8,
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    policy_handles = [
        Line2D([0], [0], color=POLICY_COLORS[p], lw=2.4, label=POLICY_LABELS[p]) for p in policies_in
    ]
    fig.legend(
        handles=policy_handles,
        loc="lower center",
        ncol=3 if compact else 1,
        fontsize=plt.rcParams["legend.fontsize"],
        title="Policies",
        title_fontsize=7,
        bbox_to_anchor=(0.5, -0.06 if compact else -0.02),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )
    if not compact:
        fig.text(0.5, -0.26, GLOSSARY_FIG2, ha="center", va="top", fontsize=6.5, linespacing=1.25)
    bot = 0.22 if compact else 0.38
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=bot, wspace=0.28)
    fig.savefig(fig_dir / "fig2_source_mass_discovery.png", **_png_save_kw(compact=compact))
    fig.savefig(fig_dir / "fig2_source_mass_discovery.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_compute_equivalent(df: pd.DataFrame, benchmark_dir: Path, fig_dir: Path, *, compact: bool) -> None:
    presets = [p for p in PRESET_ORDER if p in df["preset"].unique()]
    policies_in = [p for p in POLICY_ORDER if p in df["policy"].unique()]
    max_ns = df.groupby("preset")["sampled_ns"].max()
    target_specs: list[tuple[str, str, object, str]] = [
        ("jsd", "le", "noise2", "First time JSD ≤ 2× split-noise median"),
        ("f_helix_error", "le", 0.005, "First time |Δf_helix| ≤ 0.005"),
        ("f_open_error", "le", 0.01, "First time |Δf_open| ≤ 0.01"),
        ("important_coverage", "ge", 0.95, "First time important-bin coverage ≥ 0.95"),
    ]
    rows = []
    for preset in presets:
        noise = float(_load_ref_json(Path(), benchmark_dir, preset)["noise_floor"]["jsd_bootstrap_p50"])
        sub = df[df["preset"] == preset]
        budget = float(max_ns.loc[preset])
        for col, op, thr, _ in target_specs:
            thr_v = noise * 2 if thr == "noise2" else float(thr)
            for pol in policies_in:
                sp = sub[sub["policy"] == pol]
                for rep, grp in sp.groupby("repeat"):
                    grp = grp.sort_values("step")
                    s = grp["sampled_ns"].values
                    v = grp[col].values
                    hit = np.nan
                    if op == "le":
                        m = np.where(v <= thr_v)[0]
                    else:
                        m = np.where(v >= thr_v)[0]
                    if len(m):
                        hit = float(s[m[0]])
                    rows.append(
                        {
                            "preset": preset,
                            "policy": pol,
                            "repeat": int(rep),
                            "target_key": f"{col}_{thr}",
                            "first_hit_ns": hit,
                            "budget": budget,
                        }
                    )
    rdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(
        len(target_specs),
        len(presets),
        figsize=(4.1 * len(presets), 3.0 * len(target_specs)),
        squeeze=False,
    )
    fig.suptitle(
        (
            "Time-to-target (median across repeats; bar at budget ⇒ target not met)"
            if compact
            else (
                "Fig. 3 · Time-to-target (median across repeats)\n"
                "BAR reaches red line ⇒ target never met within preset budget"
            )
        ),
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    for i, (col, op, thr, lab) in enumerate(target_specs):
        key = f"{col}_{thr}"
        for j, preset in enumerate(presets):
            ax = axes[i][j]
            budget = float(max_ns.loc[preset])
            slice_t = rdf[(rdf["preset"] == preset) & (rdf["target_key"] == key)]
            meds = slice_t.groupby("policy")["first_hit_ns"].median()
            y = np.arange(len(policies_in))
            xvals = [
                float(meds[pol]) if pol in meds.index and np.isfinite(meds[pol]) else budget * 1.05
                for pol in policies_in
            ]
            ax.barh(y, xvals, color=[POLICY_COLORS[p] for p in policies_in], height=0.72)
            ax.set_yticks(y)
            ax.set_yticklabels([POLICY_LABELS_SHORT[p] for p in policies_in], fontsize=7.5)
            ax.axvline(
                budget,
                color="firebrick",
                linestyle="--",
                lw=1.5,
                alpha=0.85,
                label="Max budget",
            )
            ax.set_xlim(0, budget * 1.15)
            if i == 0:
                ax.set_title(_panel_title(preset), fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            ax.set_xlabel("Median time when target is first hit (ns)", fontsize=7.5)

    if not compact:
        fig.text(0.5, 0.015, GLOSSARY_FIG3, ha="center", va="bottom", fontsize=6.5, linespacing=1.2)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.12 if compact else 0.14, hspace=0.52, wspace=0.32)
    fig.savefig(fig_dir / "fig3_compute_equivalent_bars.png", **_png_save_kw(compact=compact))
    fig.savefig(fig_dir / "fig3_compute_equivalent_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def fig4_map_overlay(benchmark_dir: Path, fig_dir: Path, preset: str, *, compact: bool) -> None:
    npz = _load_ref_npz(benchmark_dir, preset)
    ref_hist = npz["ref_hist"]
    src = npz["source_prob"]
    nx, ny = ref_hist.shape
    pol_dirs = []
    for name in POLICY_ORDER:
        d = benchmark_dir / preset / name
        if (d / "final_hists.npz").is_file():
            pol_dirs.append(d)
    if not pol_dirs:
        return
    n_pol = min(6, len(pol_dirs))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=False)
    axes = axes.ravel()
    eps = 1e-12
    im = None
    n_rep = 0
    for ax, pol_dir in zip(axes, pol_dirs[:6]):
        pol = pol_dir.name
        z = np.load(pol_dir / "final_hists.npz")["hists"]
        n_rep = int(z.shape[0])
        mean_hist = np.mean(z, axis=0)
        ratio = np.log((mean_hist + eps) / (ref_hist + eps))
        im = ax.imshow(
            ratio.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
        )
        ax.contour(
            src.T,
            levels=5,
            colors="k",
            linewidths=0.45,
            alpha=0.55,
        )
        ax.set_title(POLICY_LABELS_SHORT.get(pol, pol), fontsize=9, fontweight="bold")
        ax.set_xlabel("Bin index (dim. 1)", fontsize=8)
        ax.set_ylabel("Bin index (dim. 2)", fontsize=8)

    for k in range(n_pol, len(axes)):
        axes[k].axis("off")

    cbar = fig.colorbar(
        im,
        ax=list(axes[:n_pol]),
        shrink=0.82,
        pad=0.02,
        location="right",
    )
    cbar.set_label(
        "ln ratio: ln((mean sampled histogram + ε) / (reference histogram + ε))\n"
        "(natural log; blue = under-sampled vs reference, red = over-sampled)",
        fontsize=8,
    )

    fig.suptitle(
        (
            f"Sampling bias vs reference (mean over {n_rep} repeats)\n{_panel_title(preset)}\n"
            "Contours: source prior (5 levels); same grid as reference."
            if compact
            else (
                f"Fig. 4 · Spatial sampling bias vs reference (mean over {n_rep} repeats)\n"
                f"{_panel_title(preset)}\n"
                "Black contours: source prior probability on bins (5 levels). "
                "Compared panels use the same bin grid and reference."
            )
        ),
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.88, top=0.82, bottom=0.08, wspace=0.35, hspace=0.4)
    fig.savefig(fig_dir / f"fig4_map_overlay_{preset}.png", **_png_save_kw(compact=compact))
    fig.savefig(fig_dir / f"fig4_map_overlay_{preset}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig5_violins(df: pd.DataFrame, fig_dir: Path, *, compact: bool) -> None:
    metric_meta = [
        ("jsd", "JSD vs reference\n(lower median is better)"),
        ("coverage", "Coverage (ref.-supported bins)\n(higher median is better)"),
        ("f_helix_error", "|Δf_helix|\n(lower median is better)"),
        ("f_open_error", "|Δf_open|\n(lower median is better)"),
    ]
    presets = [p for p in PRESET_ORDER if p in df["preset"].unique()]
    policies_in = [p for p in POLICY_ORDER if p in df["policy"].unique()]
    last = df.sort_values("step").groupby(["preset", "policy", "repeat"], as_index=False).tail(1)
    fig, axes = plt.subplots(
        len(metric_meta), len(presets), figsize=(4.1 * len(presets), 3.3 * len(metric_meta)), squeeze=False
    )
    fig.suptitle(
        (
            "Final-budget variability across repeats"
            if compact
            else ("Fig. 5 · Final-budget variability across repeats\n(violin = spread at last MSM step)")
        ),
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    for j, preset in enumerate(presets):
        for i, (m, ytitle) in enumerate(metric_meta):
            ax = axes[i][j]
            sub = last[last["preset"] == preset]
            data = [sub[sub["policy"] == pol][m].values for pol in policies_in]
            parts = ax.violinplot(
                data,
                positions=range(len(policies_in)),
                showmeans=True,
                showmedians=True,
            )
            for b, pol in zip(parts["bodies"], policies_in):
                b.set_facecolor(POLICY_COLORS[pol])
                b.set_alpha(0.7)
            ax.set_xticks(range(len(policies_in)))
            ax.set_xticklabels(
                [POLICY_LABELS_SHORT[p] for p in policies_in],
                rotation=40,
                ha="right",
                fontsize=7.5,
            )
            if i == 0:
                ax.set_title(_panel_title(preset), fontsize=8.5)
            if j == 0:
                ax.set_ylabel(ytitle, fontsize=8.5)

    if not compact:
        fig.text(0.5, 0.01, GLOSSARY_FIG5, ha="center", fontsize=6.5, linespacing=1.2)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.12 if compact else 0.14, hspace=0.55, wspace=0.3)
    fig.savefig(fig_dir / "fig5_final_budget_violins.png", **_png_save_kw(compact=compact))
    fig.savefig(fig_dir / "fig5_final_budget_violins.pdf", bbox_inches="tight")
    plt.close(fig)


def fig6_pair_delta(df: pd.DataFrame, fig_dir: Path, *, compact: bool) -> None:
    metrics = ["jsd", "coverage", "f_helix_error", "important_coverage"]
    metric_titles = {
        "jsd": "Pairwise ΔJSD (see legend)\n(negative ⇒ left-hand policy has lower JSD)",
        "coverage": "Pairwise Δ coverage (see legend)\n(positive ⇒ left-hand policy covers more)",
        "f_helix_error": "Pairwise Δ|Δf_helix| (see legend)\n(negative ⇒ left-hand policy more accurate)",
        "important_coverage": "Pairwise Δ important coverage (see legend)\n(positive ⇒ left-hand policy higher)",
    }
    need = {"B6", "B3", "B1"}
    presets = [p for p in PRESET_ORDER if p in df["preset"].unique()]
    if not need.issubset(set(df["policy"].unique())):
        return
    fig, axes = plt.subplots(len(metrics), len(presets), figsize=(4.1 * len(presets), 3.2 * len(metrics)), squeeze=False)
    n_rep = int(df["repeat"].nunique())
    for j, preset in enumerate(presets):
        sub = df[df["preset"] == preset]
        piv: dict[str, pd.DataFrame] = {}
        for pol in ["B6", "B3", "B1"]:
            psub = sub[sub["policy"] == pol]
            piv[pol] = psub.pivot_table(index=["repeat", "step"], values=metrics, aggfunc="first")
        common = piv["B6"].index.intersection(piv["B3"].index).intersection(piv["B1"].index)
        step_to_ns = (
            sub[sub["policy"] == "B6"].groupby("step")["sampled_ns"].median()
            if "B6" in sub["policy"].values
            else sub.groupby("step")["sampled_ns"].median()
        )
        for i, m in enumerate(metrics):
            ax = axes[i][j]
            for tag, a, b in [("B6−B3", "B6", "B3"), ("B3−B1", "B3", "B1"), ("B6−B1", "B6", "B1")]:
                d = piv[a].loc[common][m] - piv[b].loc[common][m]
                tmp = d.reset_index()
                val_col = [c for c in tmp.columns if c not in ("repeat", "step")][0]
                g = tmp.groupby("step")[val_col].agg(["mean", "std"])
                sem = g["std"] / np.sqrt(max(n_rep, 1))
                x_ns = step_to_ns.reindex(g.index).ffill().values.astype(float)
                ax.plot(
                    x_ns,
                    g["mean"],
                    label=tag,
                    color=PAIR_LINE_COLORS[tag],
                    lw=1.8,
                )
                ax.fill_between(
                    x_ns,
                    g["mean"] - sem,
                    g["mean"] + sem,
                    color=PAIR_LINE_COLORS[tag],
                    alpha=0.18,
                )
            ax.axhline(0, color="0.35", linestyle="-", linewidth=0.9, alpha=0.8)
            if i == 0:
                ax.set_title(_panel_title(preset), fontsize=8.5)
            if j == 0:
                ax.set_ylabel(metric_titles.get(m, m), fontsize=8)
            ax.set_xlabel(XAXIS_NS, fontsize=7)

    h = [
        Line2D([0], [0], color=PAIR_LINE_COLORS["B6−B3"], lw=2, label="B6 − B3  (COMPASS vs min-count)"),
        Line2D([0], [0], color=PAIR_LINE_COLORS["B3−B1"], lw=2, label="B3 − B1  (min-count vs replicas)"),
        Line2D([0], [0], color=PAIR_LINE_COLORS["B6−B1"], lw=2, label="B6 − B1  (COMPASS vs replicas)"),
    ]
    fig.legend(
        handles=h,
        loc="lower center",
        ncol=3 if compact else 1,
        fontsize=7,
        title="Difference curves (same repeats)",
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, -0.08 if compact else -0.06),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )
    fig.suptitle(
        (
            "Head-to-head policy differences (zero = tie)"
            if compact
            else (
                "Fig. 6 · Head-to-head policy differences along sampling\n"
                "Zero line = identical; sign interpretation varies by row (see y-labels)."
            )
        ),
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    if not compact:
        fig.text(0.5, -0.22, GLOSSARY_FIG6, ha="center", va="top", fontsize=6.5, linespacing=1.2)
    bot = 0.14 if compact else 0.22
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=bot, hspace=0.55, wspace=0.32)
    fig.savefig(fig_dir / "fig6_pair_delta_curves.png", **_png_save_kw(compact=compact))
    fig.savefig(fig_dir / "fig6_pair_delta_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--benchmark-dir", type=Path, default=Path("archive/outputs_legacy/msm_benchmark"))
    ap.add_argument(
        "--compact",
        action="store_true",
        help="Publication/BTP mode: higher PNG resolution, no footer text blocks, shorter titles.",
    )
    ap.add_argument(
        "--btp-figures-dir",
        type=Path,
        default=None,
        help="After plotting, copy key PNGs here with names used by COMPASS_BTP_REPORT.tex (figures/).",
    )
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    bench = (repo / args.benchmark_dir).resolve()
    fig_dir = bench / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _apply_plot_style(compact=args.compact)

    df = _load_long(repo, bench)
    compact = args.compact
    fig1_convergence_grid(df, bench, fig_dir, compact=compact)
    fig2_source_mass_discovery(df, bench, fig_dir, compact=compact)
    fig3_compute_equivalent(df, bench, fig_dir, compact=compact)
    fig5_violins(df, fig_dir, compact=compact)
    fig6_pair_delta(df, fig_dir, compact=compact)

    for preset in ["03_spib1d_m20", "04_tica2d_long200_m64"]:
        if (bench / preset).is_dir():
            try:
                fig4_map_overlay(bench, fig_dir, preset, compact=compact)
            except Exception:
                pass

    print(f"Wrote figures under {fig_dir}")

    if args.btp_figures_dir is not None:
        import shutil

        btp = args.btp_figures_dir
        dest = btp.resolve() if btp.is_absolute() else (repo / btp).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        mapping = [
            ("fig1_convergence_grid.png", "msm_convergence_grid.png"),
            ("fig6_pair_delta_curves.png", "msm_pair_delta_curves.png"),
            ("fig5_final_budget_violins.png", "msm_final_budget_violins.png"),
        ]
        preset_long = "04_tica2d_long200_m64"
        for src_name, dst_name in mapping:
            src = fig_dir / src_name
            if src.is_file():
                shutil.copy2(src, dest / dst_name)
                print(f"  copied -> {dest / dst_name}")
        mp_src = fig_dir / f"fig4_map_overlay_{preset_long}.png"
        if mp_src.is_file():
            shutil.copy2(mp_src, dest / "msm_map_overlay_tica2d_long400ns.png")
            print(f"  copied -> {dest / 'msm_map_overlay_tica2d_long400ns.png'}")


if __name__ == "__main__":
    main()
