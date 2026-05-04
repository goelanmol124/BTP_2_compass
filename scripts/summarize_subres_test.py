#!/usr/bin/env python3
"""Aggregate metrics across COMPASS subspace-residual ablation runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "archive/outputs_legacy/compass/subres_test"
RUNS = {
    "e12_baseline_det": f"{PREFIX}/e12_baseline",
    "e12_subres_w0p25_det": f"{PREFIX}/e12_subres_w0p25",
    "e12_subres_w0p5_det": f"{PREFIX}/e12_subres_w0p5",
    "e12_subres_w1p0_det": f"{PREFIX}/e12_subres_w1p0",
    "e12_baseline_smx12": f"{PREFIX}/e12_baseline_softmax",
    "e12_subres_w0p25_smx12": f"{PREFIX}/e12_subres_w0p25_softmax",
    "e16_noisy_baseline_det": f"{PREFIX}/e16_noisy_baseline",
    "e16_noisy_subres_w0p5_det": f"{PREFIX}/e16_noisy_subres_w0p5",
    "e16_noisy_subres_w1p0_det": f"{PREFIX}/e16_noisy_subres_w1p0",
    "e16_noisy_baseline_smx8": f"{PREFIX}/e16_noisy_baseline_softmax",
    "e16_noisy_subres_w1p0_smx8": f"{PREFIX}/e16_noisy_subres_w1p0_softmax",
}

cols = ("f_helix_error", "f_open_error", "jsd", "coverage", "important_coverage", "source_mass_visited")

print(f"{'run':<26} " + " ".join(f"{c:>13}" for c in cols))
for name, rel in RUNS.items():
    csv = ROOT / rel / "compass_metrics.csv"
    df = pd.read_csv(csv)
    last = df[df["step"] == df["step"].max()]
    means = [last[c].mean() for c in cols]
    sds = [last[c].std() for c in cols]
    n = len(last)
    cells = []
    for m, s in zip(means, sds):
        if n > 1:
            cells.append(f"{m:.4f}±{s:.4f}")
        else:
            cells.append(f"{m:.4f}")
    print(f"{name:<26} " + " ".join(f"{c:>13}" for c in cells) + f"  (n={n})")

random_csv = ROOT / next(iter(RUNS.values())) / "random_metrics.csv"
rdf = pd.read_csv(random_csv)
last_r = rdf[rdf["step"] == rdf["step"].max()]
means = [last_r[c].mean() for c in cols]
sds = [last_r[c].std() for c in cols]
print(f"{'random_bootstrap':<26} " + " ".join(f"{m:.4f}±{s:.4f}" for m, s in zip(means, sds)) + f"  (n={len(last_r)})")
