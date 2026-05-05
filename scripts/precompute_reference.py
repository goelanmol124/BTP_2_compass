"""Phase 0 pre-computation: reference quantities shared across all experiments.

Computes and caches to outputs/v2/reference/:
  - pi_ref_bins.npy         equilibrium bin weights (target TICA-2D 20x20 grid)
  - f_helix_per_bin.npy     per-bin helix fraction from reference
  - f_open_per_bin.npy      per-bin open fraction from reference
  - ref_fes_2d.npy          2D PMF in kcal/mol
  - f_helix_ref_scalar.npy  global reference helix fraction
  - f_open_ref_scalar.npy   global reference open fraction
  - bin_assignments.npy     integer bin index for every archive frame (TICA-2D bins)
  - helix_coil_boundary_tic1.npy  TIC-1 threshold separating helix and coil basins
  - a8q2_tica2d.npy         TICA-2D projection of all A8Q2 frames (transferred A10 TICA)
  - a8q2_tica1d.npy         TICA-1D (first component only)
  - source_bin_assignments_tica2d.npy  A10 source frame bin assignments in TICA-2D
  - n_source_bins_tica2d.npy          number of populated A10 source bins

Usage:
    python scripts/precompute_reference.py [--repo-root .] [--n-bins 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.compass.replay_enhanced import project_tica, _hist2d_xy, _build_edges_source_only


KT_KCAL = 0.656  # kT in kcal/mol at 330 K (R * 330 K)


def _make_2d_edges(xy: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    def _edges1d(z: np.ndarray) -> np.ndarray:
        z = z[np.isfinite(z)]
        lo, hi = np.percentile(z, [0.5, 99.5])
        pad = 0.08 * max(float(hi - lo), 1e-6)
        return np.linspace(float(lo - pad), float(hi + pad), n_bins + 1)

    return _edges1d(xy[:, 0]), _edges1d(xy[:, 1])


def _bin_assign_2d(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """Assign each frame to a flat bin index."""
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    xi = np.clip(np.digitize(xy[:, 0], x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(xy[:, 1], y_edges) - 1, 0, ny - 1)
    return xi * ny + yi


def run(repo_root: Path, n_bins: int = 20) -> None:
    out_dir = repo_root / "outputs" / "v2" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = repo_root / "outputs" / "artifacts"
    tica_dir = repo_root / "outputs" / "tica_artifacts_a10"

    # --- Load A8Q2 features and NHB ---
    feats_path = artifacts / "a8q2_features.npy"
    nhb_path = artifacts / "a8q2_nhb.npy"
    if not feats_path.is_file():
        raise FileNotFoundError(feats_path)
    if not nhb_path.is_file():
        raise FileNotFoundError(nhb_path)

    feats = np.load(feats_path).astype(np.float64)
    nhb = np.load(nhb_path).astype(np.float64)
    n_frames = len(nhb)
    print(f"Loaded A8Q2: {n_frames} frames, {feats.shape[1]} features")

    # --- Project into TICA-2D (transferred A10 TICA) ---
    mean_path = tica_dir / "a10_tica_mean0.npy"
    ev_path = tica_dir / "a10_tica_singular_vectors_left.npy"
    if not mean_path.is_file() or not ev_path.is_file():
        raise FileNotFoundError(f"TICA artifacts not found in {tica_dir}. Run tica_train first.")

    mean = np.load(mean_path)
    ev = np.load(ev_path)
    tica2d = project_tica(feats, mean, ev, dim=2)
    tica1d = tica2d[:, :1]

    np.save(out_dir / "a8q2_tica2d.npy", tica2d)
    np.save(out_dir / "a8q2_tica1d.npy", tica1d)
    print(f"  TICA-2D shape: {tica2d.shape}")

    # --- Build 2D grid edges (based on target distribution) ---
    x_edges, y_edges = _make_2d_edges(tica2d, n_bins)
    np.save(out_dir / "bin_x_edges.npy", x_edges)
    np.save(out_dir / "bin_y_edges.npy", y_edges)

    # --- Bin assignments for all archive frames ---
    bin_assign = _bin_assign_2d(tica2d, x_edges, y_edges)
    np.save(out_dir / "bin_assignments.npy", bin_assign)

    # --- Reference equilibrium distribution ---
    n_total_bins = n_bins * n_bins
    counts = np.bincount(bin_assign, minlength=n_total_bins).astype(float)
    pi_ref = counts / counts.sum()
    np.save(out_dir / "pi_ref_bins.npy", pi_ref)

    # --- Per-bin helix and open fractions ---
    helix_th = 3
    open_th = 1
    is_helix = (nhb >= helix_th).astype(float)
    is_open = (nhb <= open_th).astype(float)

    f_helix_per_bin = np.zeros(n_total_bins)
    f_open_per_bin = np.zeros(n_total_bins)
    for b in range(n_total_bins):
        mask = bin_assign == b
        if mask.any():
            f_helix_per_bin[b] = float(is_helix[mask].mean())
            f_open_per_bin[b] = float(is_open[mask].mean())

    np.save(out_dir / "f_helix_per_bin.npy", f_helix_per_bin)
    np.save(out_dir / "f_open_per_bin.npy", f_open_per_bin)

    f_helix_ref = float(is_helix.mean())
    f_open_ref = float(is_open.mean())
    np.save(out_dir / "f_helix_ref_scalar.npy", np.array(f_helix_ref))
    np.save(out_dir / "f_open_ref_scalar.npy", np.array(f_open_ref))
    print(f"  f_helix_ref={f_helix_ref:.4f}, f_open_ref={f_open_ref:.4f}")

    # --- 2D PMF (FES in kcal/mol) ---
    hist2d, _, _ = np.histogram2d(tica2d[:, 0], tica2d[:, 1], bins=[x_edges, y_edges])
    hist2d = hist2d.astype(float)
    hist2d /= hist2d.sum()
    with np.errstate(divide="ignore"):
        fes = -KT_KCAL * np.log(np.where(hist2d > 0, hist2d, np.nan))
    fes_min = np.nanmin(fes)
    fes -= fes_min
    np.save(out_dir / "ref_fes_2d.npy", fes)
    np.save(out_dir / "bin_x_edges.npy", x_edges)
    np.save(out_dir / "bin_y_edges.npy", y_edges)

    # --- Helix/coil boundary in TIC-1 ---
    # Find TIC-1 value that best separates helix (NHB>=3) from coil (NHB<3) by midpoint
    tic1 = tica2d[:, 0]
    helix_mask = nhb >= helix_th
    coil_mask = ~helix_mask
    tic1_helix_mean = float(tic1[helix_mask].mean()) if helix_mask.any() else 0.0
    tic1_coil_mean = float(tic1[coil_mask].mean()) if coil_mask.any() else 0.0
    boundary = 0.5 * (tic1_helix_mean + tic1_coil_mean)
    np.save(out_dir / "helix_coil_boundary_tic1.npy", np.array(boundary))
    print(f"  TIC-1 helix mean={tic1_helix_mean:.3f}, coil mean={tic1_coil_mean:.3f}, boundary={boundary:.3f}")

    # --- Source (A10) coverage pre-computation ---
    # Project A10 features into same TICA space; get bin assignments
    a10_feats_path = artifacts / "a10_features.npy"
    if a10_feats_path.is_file():
        a10_feats = np.load(a10_feats_path).astype(np.float64)
        a10_tica2d = project_tica(a10_feats, mean, ev, dim=2)
        a10_bins = _bin_assign_2d(a10_tica2d, x_edges, y_edges)
        np.save(out_dir / "source_bin_assignments_tica2d.npy", a10_bins)
        n_src_bins = int(np.unique(a10_bins).size)
        np.save(out_dir / "n_source_bins_tica2d.npy", np.array(n_src_bins))
        print(f"  A10 source: {len(a10_feats)} frames, {n_src_bins} unique bins")
    else:
        print(f"  Warning: A10 features not found at {a10_feats_path}; source coverage unavailable")
        # Use A8Q2 as source proxy so experiments still run
        np.save(out_dir / "source_bin_assignments_tica2d.npy", bin_assign)
        np.save(out_dir / "n_source_bins_tica2d.npy", np.array(int(np.unique(bin_assign).size)))

    print(f"\nReference data written to: {out_dir}")
    print("Files:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} {size_kb:8.1f} KB")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--n-bins", type=int, default=20, help="Grid bins per axis (default: 20)")
    args = p.parse_args(argv)
    run(args.repo_root.resolve(), n_bins=args.n_bins)


if __name__ == "__main__":
    main()
