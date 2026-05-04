"""Generate additional manuscript figures from existing output data."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

warnings.filterwarnings("ignore")

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
LEGACY_SWP2 = Path("archive/outputs_legacy/compass_sweep2")


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# FIGURE 1: TICA Transfer Diagnostics composite
# ─────────────────────────────────────────────────────────────────────
def make_transfer_diagnostics():
    _apply_style()
    fig = plt.figure(figsize=(7.2, 6.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.34, left=0.10, right=0.98, top=0.94, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    metrics = ["$D_0$", "$D_{\\tau}$", "$D_{KM}$", "$|r|$"]
    values = [0.984, 0.648, 14.04, 0.786]
    colors = ["#CC3311", "#EE7733", "#CC3311", "#009E73"]
    bars = ax_a.bar(metrics, values, color=colors, edgecolor="#222222", linewidth=0.6, width=0.58, alpha=0.92)
    for bar, v in zip(bars, values):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{v:.3f}" if v < 1 else f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_a.axhline(0.3, color="#666666", ls="--", lw=1.0, alpha=0.75)
    ax_a.text(3.55, 0.52, "good transfer\nthreshold", fontsize=8, color="#555555", ha="right", va="bottom")
    ax_a.set_ylabel("Metric value")
    ax_a.set_title("(a) TICA transfer scores (A10 → A8Q2)", fontweight="600")
    ax_a.set_ylim(0, 16)

    scatter_path = Path("outputs/transfer_a8q2/transfer_pearson_scatter.png")
    if scatter_path.exists():
        ax_b = fig.add_subplot(gs[0, 1])
        img = imread(str(scatter_path))
        h, w = img.shape[:2]
        crop = img[int(h * 0.08) : int(h * 0.95), int(w * 0.06) : int(w * 0.92)]
        ax_b.imshow(crop, aspect="auto")
        ax_b.set_axis_off()
        ax_b.set_title("(b) Transferred TIC-1 vs A8Q2 helix H-bonds", fontweight="600")

    fes_path = Path("outputs/transfer_a8q2/transfer_fes_comparison.png")
    if fes_path.exists():
        ax_c = fig.add_subplot(gs[1, :])
        img = imread(str(fes_path))
        h, w = img.shape[:2]
        crop = img[int(h * 0.05) : int(h * 0.95), int(w * 0.02) : int(w * 0.98)]
        ax_c.imshow(crop, aspect="auto")
        ax_c.set_axis_off()
        ax_c.set_title(
            "(c) 1D free-energy profiles: transferred A10 TIC-1 vs native A8Q2 TIC-1",
            fontweight="600",
        )

    _save(fig, "transfer_diagnostics.png")
    print("  OK: transfer_diagnostics.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURE 2: Score decomposition (embed sweep asset at print resolution)
# ─────────────────────────────────────────────────────────────────────
def make_score_decomposition():
    _apply_style()
    src = LEGACY_SWP2 / "e12_tica2d_two_phase_chunk20_80ns/compass_score_decomposition.png"
    if not src.exists():
        print("  FAIL: score decomposition source not found")
        return

    img = imread(str(src))
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.imshow(img, aspect="auto")
    ax.set_axis_off()
    _save(fig, "score_decomposition_e12.png")
    print("  OK: score_decomposition_e12.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURE 3: Prior corruption probe maps
# ─────────────────────────────────────────────────────────────────────
def make_prior_corruption_maps():
    _apply_style()
    correct_path = LEGACY_SWP2 / "e12_tica2d_two_phase_chunk20_80ns/compass_selected_probe_map.png"
    inverted_path = LEGACY_SWP2 / "e15_tica2d_inverted_prior_80ns/compass_selected_probe_map.png"

    if not correct_path.exists() or not inverted_path.exists():
        print("  FAIL: probe map sources not found")
        return

    img_correct = imread(str(correct_path))
    img_inverted = imread(str(inverted_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.45))

    for ax, img, title in [
        (ax1, img_correct, "(a) Correct A10 TICA prior"),
        (ax2, img_inverted, "(b) Inverted TICA prior"),
    ]:
        h, w = img.shape[:2]
        crop = img[int(h * 0.04) : int(h * 0.96), int(w * 0.02) : int(w * 0.96)]
        ax.imshow(crop, aspect="auto")
        ax.set_axis_off()
        ax.set_title(title, fontweight="600")

    fig.suptitle(
        "COMPASS selections in TICA space (correct vs inverted prior)",
        fontsize=11,
        fontweight="700",
        y=1.03,
    )
    fig.subplots_adjust(top=0.88, wspace=0.06)
    _save(fig, "prior_corruption_probe_maps.png")
    print("  OK: prior_corruption_probe_maps.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURE 4: Prior corruption bar chart
# ─────────────────────────────────────────────────────────────────────
def make_prior_corruption_bars():
    _apply_style()
    priors = ["None\n(correct)", "Uniform", "Shuffled", "Inverted", "Noisy"]
    jsd = [0.147, 0.200, 0.151, 0.273, 0.140]
    helix_err = [0.0045, 0.0695, 0.0208, 0.0645, 0.0245]
    coverage = [0.528, 0.765, 0.725, 0.725, 0.563]
    random_jsd = 0.132
    random_helix = 0.0208
    random_cov = 0.583

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))

    x = np.arange(len(priors))
    w = 0.55
    colors = ["#009E73", "#0072B2", "#E69F00", "#CC3311", "#AA4499"]

    axes[0].bar(x, jsd, w, color=colors, edgecolor="#222222", linewidth=0.45)
    axes[0].axhline(random_jsd, color="#444444", ls="--", lw=1.1, label=f"Random ({random_jsd})")
    axes[0].set_ylabel("JSD")
    axes[0].set_title("(a) JSD", fontweight="600")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.95)

    axes[1].bar(x, helix_err, w, color=colors, edgecolor="#222222", linewidth=0.45)
    axes[1].axhline(random_helix, color="#444444", ls="--", lw=1.1, label=f"Random (~{random_helix})")
    axes[1].set_ylabel(r"$\epsilon_{\mathrm{helix}}$")
    axes[1].set_title("(b) Helix error", fontweight="600")
    axes[1].legend(fontsize=8, loc="upper left", framealpha=0.95)

    axes[2].bar(x, coverage, w, color=colors, edgecolor="#222222", linewidth=0.45)
    axes[2].axhline(random_cov, color="#444444", ls="--", lw=1.1, label=f"Random ({random_cov})")
    axes[2].set_ylabel("Coverage")
    axes[2].set_title("(c) Coverage", fontweight="600")
    axes[2].legend(fontsize=8, loc="upper left", framealpha=0.95)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(priors, fontsize=8)

    fig.suptitle("Prior-quality controls (80 ns, TICA2D)", fontsize=11, fontweight="700", y=1.05)
    fig.subplots_adjust(top=0.82)
    _save(fig, "prior_corruption_bars.png")
    print("  OK: prior_corruption_bars.png")


if __name__ == "__main__":
    print("Generating manuscript figures...")
    make_transfer_diagnostics()
    make_score_decomposition()
    make_prior_corruption_maps()
    make_prior_corruption_bars()
    print("Done.")
