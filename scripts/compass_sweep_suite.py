"""Run COMPASS sweep suites together: oracle (enhanced replay) + online-faithful replay.

Example:
  python scripts/compass_sweep_suite.py --preset medium
  python scripts/compass_sweep_suite.py --enhanced-only --preset full
  python scripts/compass_sweep_suite.py --online-only --suite core --preset quick

Presets adjust --n-steps and --random-repeats for both drivers (uniform override).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compass_sweep_collect import plot_delta_correlation

PRESETS = {
    "quick": (12, 12),
    "medium": (40, 40),
    "full": (None, None),
}


def _run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=list(PRESETS), default="medium")
    p.add_argument("--enhanced-root", type=Path, default=Path("archive/outputs_legacy/compass_sweep2_suite"))
    p.add_argument("--online-root", type=Path, default=Path("outputs/compass_sweep_online_suite"))
    p.add_argument("--combined-root", type=Path, default=Path("outputs/compass_sweep_combined"))
    p.add_argument("--enhanced-only", action="store_true")
    p.add_argument("--online-only", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--suite", choices=["core", "extended", "all"], default="all")
    p.add_argument("--enhanced-only-names", nargs="*", default=None, help="--only for enhanced driver.")
    p.add_argument("--online-only-names", nargs="*", default=None, help="--only for online driver.")
    p.add_argument(
        "--final",
        action="store_true",
        help="Paper-style run: outputs/compass_sweep{2,online}_final, combined_final, preset=full (no step/repeat shrink).",
    )
    args = p.parse_args(argv)

    if args.final:
        args.enhanced_root = Path("archive/outputs_legacy/compass_sweep2_final")
        args.online_root = Path("outputs/compass_sweep_online_final")
        args.combined_root = Path("outputs/compass_sweep_combined_final")
        args.preset = "full"

    ns, nr = PRESETS[args.preset]
    py = sys.executable

    if not args.online_only:
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "compass_experiments_enhanced.py"),
            "--output-root",
            str(args.enhanced_root),
        ]
        if args.aggregate_only:
            cmd.append("--aggregate-only")
        else:
            if ns is not None:
                cmd.extend(["--n-steps", str(ns)])
            if nr is not None:
                cmd.extend(["--random-repeats", str(nr)])
        if args.enhanced_only_names:
            cmd.append("--only")
            cmd.extend(args.enhanced_only_names)
        _run(cmd)

    if not args.enhanced_only:
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "compass_experiments_online.py"),
            "--output-root",
            str(args.online_root),
            "--suite",
            args.suite,
        ]
        if args.aggregate_only:
            cmd.append("--aggregate-only")
        else:
            if ns is not None:
                cmd.extend(["--n-steps", str(ns)])
            if nr is not None:
                cmd.extend(["--random-repeats", str(nr)])
        if args.online_only_names:
            cmd.append("--only")
            cmd.extend(args.online_only_names)
        _run(cmd)

    comb = (REPO_ROOT / args.combined_root).resolve() / "sweep"
    comb.mkdir(parents=True, exist_ok=True)
    enhanced_slim = REPO_ROOT / args.enhanced_root / "sweep" / "sweep2_results.csv"
    enhanced_full = REPO_ROOT / args.enhanced_root / "sweep" / "sweep2_results_full.csv"
    online_slim = REPO_ROOT / args.online_root / "sweep" / "online_sweep_results.csv"
    online_full = REPO_ROOT / args.online_root / "sweep" / "online_sweep_results_full.csv"
    frames: list[pd.DataFrame] = []
    frames_full: list[pd.DataFrame] = []
    if enhanced_slim.is_file():
        he = pd.read_csv(enhanced_slim)
        if "sweep_driver" not in he.columns:
            he.insert(0, "sweep_driver", "oracle_enhanced")
        frames.append(he)
    if enhanced_full.is_file():
        hef = pd.read_csv(enhanced_full)
        if "sweep_driver" not in hef.columns:
            hef.insert(0, "sweep_driver", "oracle_enhanced")
        frames_full.append(hef)
    if online_slim.is_file():
        ho = pd.read_csv(online_slim)
        if "sweep_driver" not in ho.columns:
            ho.insert(0, "sweep_driver", "online_faithful")
        frames.append(ho)
    if online_full.is_file():
        hof = pd.read_csv(online_full)
        if "sweep_driver" not in hof.columns:
            hof.insert(0, "sweep_driver", "online_faithful")
        frames_full.append(hof)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(comb / "combined_sweep_results.csv", index=False)

        if "delta_f_helix_error" in all_df.columns:
            sub = all_df.dropna(subset=["delta_f_helix_error"])
            if not sub.empty and "sweep_driver" in sub.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                drivers = sub["sweep_driver"].unique()
                x = np.arange(len(drivers))
                means = [float(sub.loc[sub["sweep_driver"] == d, "delta_f_helix_error"].mean()) for d in drivers]
                ax.bar(x, means, color=["#72B7B2", "#F58518"])
                ax.set_xticks(x)
                ax.set_xticklabels(drivers, rotation=15, ha="right")
                ax.axhline(0, color="0.4", lw=0.9)
                ax.set_ylabel("Mean Δ f_helix error (random − COMPASS)")
                ax.set_title("Suite comparison: mean helix-error advantage (roll-up)")
                fig.tight_layout()
                fig.savefig(comb / "combined_mean_delta_helix_by_driver.png", dpi=180)
                plt.close(fig)
    if frames_full:
        all_full = pd.concat(frames_full, ignore_index=True)
        all_full.to_csv(comb / "combined_sweep_results_full.csv", index=False)
        plot_delta_correlation(
            all_full,
            comb / "combined_delta_correlation.png",
            "Combined drivers: correlation of Δ metrics (all experiments)",
        )

    print(f"\nCombined summary (if any) -> {comb}")


if __name__ == "__main__":
    main()
