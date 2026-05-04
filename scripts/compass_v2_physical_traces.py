#!/usr/bin/env python3
"""Offline mean native hydrogen-bond count vs sampled budget for a v2 bench run.

Uses ``seed_decisions.csv`` to reconstruct which target frames were observed (same
segment/chunk rules as ``compass_v2_bench``) and plots mean ``a8q2_nhb`` over the
union of observed frames. **Post hoc** — not seen by policy.

Example::

    python scripts/compass_v2_physical_traces.py --run-dir outputs/v2/plan/block_a/compass_tica2d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.compass.online_state import TrajectorySegments, parse_msm_segments_arg
from src.compass.replay_enhanced import FRAME_SPACING_NS


def infer_chunk_frames(v2_metrics: pd.DataFrame) -> int:
    if len(v2_metrics) < 2:
        return 10
    sf = v2_metrics["sampled_frames"].astype(int).diff().dropna()
    if sf.empty:
        return 10
    c = int(round(sf.median()))
    return max(1, c)


def mean_nhb_trace(
    repo_root: Path,
    run_dir: Path,
    chunk_frames: int | None,
    msm_segments: str | None,
) -> pd.DataFrame:
    nhb_path = repo_root / "outputs" / "artifacts" / "a8q2_nhb.npy"
    if not nhb_path.is_file():
        raise FileNotFoundError(nhb_path)
    nhb = np.load(nhb_path)
    n_frames = len(nhb)
    segments = parse_msm_segments_arg(msm_segments, n_frames)
    seg = TrajectorySegments(segments)

    dec = pd.read_csv(run_dir / "seed_decisions.csv")
    v2 = pd.read_csv(run_dir / "v2_metrics.csv")
    if chunk_frames is None:
        chunk_frames = infer_chunk_frames(v2)

    obs = np.zeros(n_frames, dtype=bool)
    rows: list[dict[str, float]] = []
    for _, r in dec.sort_values("step").iterrows():
        sf = int(r["seed_frame"])
        a, b = seg.clip_continuation(sf, chunk_frames)
        obs[a:b] = True
        idx = np.flatnonzero(obs)
        rows.append(
            {
                "step": float(r["step"]),
                "mean_nhb": float(np.mean(nhb[idx])) if idx.size else float("nan"),
                "sampled_frames": float(idx.size),
                "sampled_ns": float(idx.size) * FRAME_SPACING_NS,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--chunk-frames", type=int, default=None)
    p.add_argument("--msm-segments", type=str, default=None)
    args = p.parse_args()

    df = mean_nhb_trace(args.repo_root, args.run_dir, args.chunk_frames, args.msm_segments)
    out_csv = args.run_dir / "physical_mean_nhb_trace.csv"
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["sampled_ns"], df["mean_nhb"], lw=2.0, color="#B279A2")
    ax.set_xlabel("Sampled time (ns)")
    ax.set_ylabel("Mean NHB (observed frames)")
    ax.set_title("Post hoc physical trace (evaluation only)")
    fig.tight_layout()
    fig.savefig(args.run_dir / "physical_mean_nhb_vs_ns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_csv} and physical_mean_nhb_vs_ns.png")


if __name__ == "__main__":
    main()
