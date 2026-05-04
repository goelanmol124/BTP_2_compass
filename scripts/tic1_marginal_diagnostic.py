"""One-off: TIC1 marginals for A10 vs A8Q2 with independent vs pooled bin edges."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
N_BINS = 30


def main() -> None:
    art = REPO / "outputs" / "artifacts"
    tica = REPO / "outputs" / "tica_artifacts_a10"
    a10 = np.load(art / "a10_features.npy")
    a8 = np.load(art / "a8q2_features.npy")
    mu = np.load(tica / "a10_tica_mean0.npy")
    ev = np.load(tica / "a10_tica_singular_vectors_left.npy")

    def tic1(x: np.ndarray) -> np.ndarray:
        return ((x - mu) @ ev[:, :1]).ravel()

    t_a10 = tic1(a10)
    t_a8 = tic1(a8)

    def edges_from(z: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(z, [0.5, 99.5])
        pad = 0.08 * max(float(hi - lo), 1e-6)
        return np.linspace(float(lo - pad), float(hi + pad), N_BINS + 1)

    def hist1d(z: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, _ = np.histogram(z, bins=edges)
        p = (h.astype(float) + 1.0)
        p = p / p.sum()
        return h.astype(float), p

    e_a10 = edges_from(t_a10)
    e_a8 = edges_from(t_a8)
    h10_own, p10_own = hist1d(t_a10, e_a10)
    h8_own, p8_own = hist1d(t_a8, e_a8)

    both = np.concatenate([t_a10, t_a8])
    lo, hi = np.percentile(both, [0.5, 99.5])
    pad = 0.08 * max(float(hi - lo), 1e-6)
    e_pool = np.linspace(float(lo - pad), float(hi + pad), N_BINS + 1)
    h10_pool, p10_pool = hist1d(t_a10, e_pool)
    h8_pool, p8_pool = hist1d(t_a8, e_pool)

    def summarize(z: np.ndarray) -> dict:
        return {
            "n": len(z),
            "mean": float(np.mean(z)),
            "std": float(np.std(z)),
            "p01": float(np.percentile(z, 1)),
            "p50": float(np.percentile(z, 50)),
            "p99": float(np.percentile(z, 99)),
        }

    print("=== TIC1 (A10-trained TICA, first component) per peptide ===")
    print("A10:", json.dumps(summarize(t_a10), indent=2))
    print("A8Q2:", json.dumps(summarize(t_a8), indent=2))
    print()
    print("=== Bin edge range (30 bins), COMPASS-style percentiles ===")
    print(f"A10-only edges:    [{e_a10[0]:.6g}, {e_a10[-1]:.6g}]")
    print(f"A8Q2-only edges:   [{e_a8[0]:.6g}, {e_a8[-1]:.6g}]")
    print(f"Pooled A10+A8Q2:   [{e_pool[0]:.6g}, {e_pool[-1]:.6g}]")
    print()
    print("Argmax bin (raw counts, own edges): A10", int(np.argmax(h10_own)), " A8Q2", int(np.argmax(h8_own)))
    print("Argmax bin (pooled edges):          A10", int(np.argmax(h10_pool)), " A8Q2", int(np.argmax(h8_pool)))
    h8_on_a10, _ = hist1d(t_a8, e_a10)
    print()
    peak_a10 = int(np.argmax(h10_own))
    print(f"Fraction of frames in A10 peak bin ({peak_a10}) [A10 grid]: A10={h10_own[peak_a10]/len(t_a10):.4f}  A8Q2={h8_on_a10[peak_a10]/len(t_a8):.4f}")
    corr = float(np.corrcoef(p10_pool, p8_pool)[0, 1])
    print(f"Pearson r (pooled-bin smoothed probs A10 vs A8Q2): {corr:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
