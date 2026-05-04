#!/usr/bin/env python3
"""Regenerate polished PNGs under ``figures/`` for COMPASS_BTP_REPORT.tex.

1. MSM benchmark panels (needs ``archive/outputs_legacy/msm_benchmark/per_step_all.{csv,parquet}``):
     python scripts/plot_msm_benchmark.py --compact --btp-figures-dir figures

2. Composite figures from sweep outputs:
     python scripts/make_manuscript_figures.py

3. Replay curves (copied when sweep outputs exist):
     e12 -> replay_tica2d_chunk20_80ns.png
     e06 -> replay_spib1d_80ns.png
     e08 -> replay_tica2d_200ns.png

Run from repository root.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY = ROOT / "archive" / "outputs_legacy"

REPLAY_SOURCES = [
    (
        LEGACY / "compass_sweep2/e12_tica2d_two_phase_chunk20_80ns/compass_replay_curves.png",
        ROOT / "figures/replay_tica2d_chunk20_80ns.png",
    ),
    (
        LEGACY / "compass_sweep2/e06_spib1d_two_phase_ig_80ns/compass_replay_curves.png",
        ROOT / "figures/replay_spib1d_80ns.png",
    ),
    (
        LEGACY / "compass_sweep2/e08_tica2d_two_phase_ig_200ns/compass_replay_curves.png",
        ROOT / "figures/replay_tica2d_200ns.png",
    ),
]


def main() -> int:
    root = ROOT
    py = sys.executable

    msm_script = root / "scripts/plot_msm_benchmark.py"
    bench = root / "archive" / "outputs_legacy" / "msm_benchmark"
    parq = bench / "per_step_all.parquet"
    csv = bench / "per_step_all.csv"
    if parq.is_file() or csv.is_file():
        print("[export_btp_figures] Running MSM benchmark plots (compact -> figures/)...", flush=True)
        r = subprocess.run(
            [
                py,
                str(msm_script),
                "--repo-root",
                str(root),
                "--benchmark-dir",
                str(bench.relative_to(root)),
                "--compact",
                "--btp-figures-dir",
                "figures",
            ],
            cwd=str(root),
        )
        if r.returncode != 0:
            print("  WARNING: plot_msm_benchmark exited non-zero", file=sys.stderr)
    else:
        print("[export_btp_figures] Skip MSM plots (no per_step_all under archive/outputs_legacy/msm_benchmark/).")

    print("[export_btp_figures] Running make_manuscript_figures.py...", flush=True)
    r = subprocess.run([py, str(root / "scripts/make_manuscript_figures.py")], cwd=str(root))
    if r.returncode != 0:
        print("  WARNING: make_manuscript_figures exited non-zero", file=sys.stderr)

    print("[export_btp_figures] Copying replay curves when sweep outputs exist...")
    for src, dst in REPLAY_SOURCES:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied {src.name} -> {dst.relative_to(root)}")
        else:
            print(f"  skip (missing): {src.relative_to(root)}")

    print("[export_btp_figures] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
