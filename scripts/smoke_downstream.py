#!/usr/bin/env python3
"""
Smoke-test downstream pipelines without a full production DCD.

Creates a short synthetic DCD from ``A10_extended_prod_top.pdb`` (jittered
replicas of frame 0), then runs:

  - ``analyze_system`` (Phase 5)
  - ``python -m src.tica_train`` (Phase 6)
  - ``python -m src.tica_transfer`` (Phase 7; A8Q2 arrays = copy of A10 for wiring only)
  - ``metrics.compute_jsd`` on the analysis histogram
  - ``adaptive_seeding._build_ground_truth_histogram`` (no OpenMM)

Full ``adaptive_seeding`` MD is **not** run (needs A8Q2 Phase 3 inputs + long GPU).

Usage (from repo root)::

    python scripts/smoke_downstream.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = REPO_ROOT / "outputs" / "smoke_pipeline"
N_FRAMES = 600  # > burn-in (250) + TICA bootstrap blocks


def _ensure_repo_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _subprocess_env() -> dict[str, str]:
    """Windows-friendly UTF-8 for child CLIs that log Greek/subscripts."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _reconfigure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows (analysis prints Greek letters)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass


def build_synthetic_dcd(prod_top_pdb: Path, dcd_out: Path) -> None:
    import mdtraj as md

    ref = md.load_pdb(str(prod_top_pdb))
    rng = np.random.default_rng(42)
    xyz = np.repeat(ref.xyz, N_FRAMES, axis=0).astype(np.float32)
    # Small Cartesian jitter so RMSD / N_HB vary frame-to-frame
    xyz += rng.normal(0.0, 0.002, xyz.shape).astype(np.float32)
    traj = md.Trajectory(xyz, ref.topology)
    dcd_out.parent.mkdir(parents=True, exist_ok=True)
    traj.save_dcd(str(dcd_out))
    print(f"[smoke] Wrote {dcd_out}  ({N_FRAMES} frames, {traj.n_atoms} atoms)")


def main() -> int:
    _reconfigure_stdio_utf8()
    _ensure_repo_path()

    prod_top = REPO_ROOT / "outputs" / "btp_equil" / "A10" / "A10_extended_prod_top.pdb"
    helix_ref = REPO_ROOT / "outputs" / "btp_equil" / "A10" / "A10_helix.pdb"
    if not prod_top.is_file():
        print(f"ERROR: missing {prod_top}")
        return 1
    if not helix_ref.is_file():
        print(f"ERROR: missing {helix_ref}")
        return 1

    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    dcd_path = SMOKE_ROOT / "A10_extended_smoke.dcd"
    build_synthetic_dcd(prod_top, dcd_path)

    from src.analysis import analyze_system
    from src.metrics import compute_jsd

    analysis_dir = SMOKE_ROOT / "analysis"
    print("[smoke] Running analyze_system …")
    summary = analyze_system(
        prefix="A10_extended",
        label="A10 smoke trajectory",
        prod_dcd=str(dcd_path),
        prod_top_pdb=str(prod_top),
        helix_ref_pdb=str(helix_ref),
        output_dir=str(analysis_dir),
    )

    feat_path = SMOKE_ROOT / "a10_features_smoke.npy"
    nhb_path = SMOKE_ROOT / "a10_nhb_smoke.npy"
    np.save(feat_path, summary["features"])
    np.save(nhb_path, summary["nhb"])
    print(f"[smoke] Saved {feat_path}  shape {summary['features'].shape}")
    print(f"[smoke] Saved {nhb_path}  shape {summary['nhb'].shape}")

    hist = summary["fes2d"]["hist"]
    jsd_self = compute_jsd(hist, hist)
    print(f"[smoke] metrics.compute_jsd(hist, hist) = {jsd_self:.6f} (expect ~0)")

    from src.adaptive_seeding import _build_ground_truth_histogram

    h_gt = _build_ground_truth_histogram(
        str(dcd_path), str(dcd_path), str(prod_top)
    )
    print(f"[smoke] _build_ground_truth_histogram shape {h_gt.shape}  sum {h_gt.sum():.0f}")

    tica_dir = SMOKE_ROOT / "tica"
    if tica_dir.exists():
        shutil.rmtree(tica_dir)
    tica_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    print("[smoke] Running: python -m src.tica_train …")
    r = subprocess.run(
        [
            py,
            "-m",
            "src.tica_train",
            "--features_npy",
            str(feat_path),
            "--nhb_npy",
            str(nhb_path),
            "--output_dir",
            str(tica_dir),
            # Use default lag scan from config (must extend to >= TICA_MIN_LAG_FRAMES).
        ],
        cwd=str(REPO_ROOT),
        env=_subprocess_env(),
        check=False,
    )
    if r.returncode != 0:
        print("ERROR: tica_train failed")
        return r.returncode

    lag_file = tica_dir / "a10_tica_chosen_lagtime.txt"
    if not lag_file.is_file():
        print(f"ERROR: missing {lag_file}")
        return 1
    lagtime = int(lag_file.read_text().strip())
    print(f"[smoke] TICA chosen lag (frames): {lagtime}")

    # Transfer assessment: use duplicate A10 features as stand-in A8Q2 (same 40-D backbone layout).
    a8_feat = SMOKE_ROOT / "a8q2_features_smoke.npy"
    a8_nhb = SMOKE_ROOT / "a8q2_nhb_smoke.npy"
    shutil.copy(feat_path, a8_feat)
    shutil.copy(nhb_path, a8_nhb)

    transfer_dir = SMOKE_ROOT / "transfer"
    if transfer_dir.exists():
        shutil.rmtree(transfer_dir)
    transfer_dir.mkdir(parents=True, exist_ok=True)

    print("[smoke] Running: python -m src.tica_transfer …")
    r2 = subprocess.run(
        [
            py,
            "-m",
            "src.tica_transfer",
            "--features_a10",
            str(feat_path),
            "--features_a8q2",
            str(a8_feat),
            "--nhb_a8q2",
            str(a8_nhb),
            "--tica_artifacts_dir",
            str(tica_dir),
            "--lagtime",
            str(lagtime),
            "--output_dir",
            str(transfer_dir),
        ],
        cwd=str(REPO_ROOT),
        env=_subprocess_env(),
        check=False,
    )
    if r2.returncode != 0:
        print("ERROR: tica_transfer failed")
        return r2.returncode

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("  Analysis outputs:", analysis_dir)
    print("  TICA outputs:    ", tica_dir)
    print("  Transfer outputs:", transfer_dir)
    print()
    print("Not run here: full adaptive_seeding (needs A8Q2 *_npt_final + GPU).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
