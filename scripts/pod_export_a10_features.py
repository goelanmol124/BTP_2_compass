#!/usr/bin/env python3
"""Export a10_features.npy + a10_nhb.npy from A10 production DCDs (Phase 6 inputs).

Uses the same burn-in and featurization as pod2_orchestrate for A8Q2.
Run from repo root on the pod::

    .venv-openmm/bin/python scripts/pod_export_a10_features.py \\
        --sim-dir outputs/btp_equil/A10 --prefix A10 \\
        --out-dir outputs/artifacts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        required=True,
        help="Directory with {prefix}_extended_prod.dcd and helix replica.",
    )
    parser.add_argument("--prefix", default="A10")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/artifacts"),
    )
    parser.add_argument(
        "--burnin-frames",
        type=int,
        default=-1,
        help="Override burn-in; default: analysis.BURNIN_FRAMES.",
    )
    args = parser.parse_args()

    import os
    import sys

    root = args.repo_root.resolve()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.analysis import BURNIN_FRAMES, compute_featurization, compute_nhb, load_trajectory

    bi = args.burnin_frames if args.burnin_frames >= 0 else BURNIN_FRAMES
    sim_dir = args.sim_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = args.prefix

    chunks_f: list[np.ndarray] = []
    chunks_n: list[np.ndarray] = []
    for lab in ("extended", "helix"):
        stem = f"{pfx}_{lab}"
        dcd = sim_dir / f"{stem}_prod.dcd"
        top = sim_dir / f"{stem}_prod_top.pdb"
        if not dcd.is_file() or not top.is_file():
            raise FileNotFoundError(f"Missing {dcd} or {top}")
        print(f"[export] {stem}: load + strip {bi} frames …", flush=True)
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        if traj.n_frames <= bi:
            raise RuntimeError(f"{dcd}: need > {bi} frames, got {traj.n_frames}")
        traj = traj[bi:]
        chunks_f.append(compute_featurization(traj))
        chunks_n.append(compute_nhb(traj).astype(np.float64))

    feat_path = out_dir / "a10_features.npy"
    nhb_path = out_dir / "a10_nhb.npy"
    np.save(feat_path, np.concatenate(chunks_f, axis=0))
    np.save(nhb_path, np.concatenate(chunks_n, axis=0))
    print(f"[export] Saved {feat_path}  shape={np.load(feat_path).shape}", flush=True)
    print(f"[export] Saved {nhb_path}  shape={np.load(nhb_path).shape}", flush=True)


if __name__ == "__main__":
    main()
