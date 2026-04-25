"""
Pre-train SPIB on pooled A10 production trajectories (helix + extended).

Features: 85-D (45 Calpha 1/r pairs + 40 sin/cos phi/psi), z-scored on the
training pool. K-means (k=4) on standardized features provides initial labels.

Writes to ``--output-dir`` (default ``outputs/spib_a10_model``):
  - spib_feature_mean.npy, spib_feature_std.npy
  - spib_metadata.json
  - spib_train_final_cpt.pt  (state_dict for frozen transfer)
  - spib_train_train.log      (from spib_lib.train)

Run from repo root:
  set PYTHONPATH=.
  python scripts/train_spib_a10.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis import BURNIN_FRAMES, FRAME_SPACING_PS, load_trajectory  # noqa: E402
from src.spib_features import SPIB_FEATURE_DIM, compute_spib_features  # noqa: E402
from src.spib_lib import SPIB, data_init, train  # noqa: E402
from src.utils import remove_burn_in  # noqa: E402

# --- Defaults (match ratchet / TICA timescales) ---
A10_DIR = _ROOT / "outputs" / "btp_equil" / "A10"
HELIX_DCD = A10_DIR / "A10_helix_prod.dcd"
EXTENDED_DCD = A10_DIR / "A10_extended_prod.dcd"
HELIX_TOP = A10_DIR / "A10_helix_prod_top.pdb"
EXTENDED_TOP = A10_DIR / "A10_extended_prod_top.pdb"

DEFAULT_OUT = _ROOT / "outputs" / "spib_a10_model"

# Lag: 10 frames @ 100 ps = 1 ns (same order as TICA lag scan convention)
LAG_FRAMES = 10
BURN_IN_NS = BURNIN_FRAMES * FRAME_SPACING_PS / 1000.0

# SPIB hyperparameters (plan)
DEFAULT_Z_DIM = 1
OUTPUT_DIM = 4
BETA = 0.01
THRESHOLD = 0.01
PATIENCE = 5
REFINEMENTS = 3
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
LR_STEP = 50
LR_GAMMA = 0.5
# Avoid writing a checkpoint every N steps (very large = only final save from train())
LOG_INTERVAL = 10**9


def _pooled_a10_features() -> np.ndarray:
    blocks: list[np.ndarray] = []
    for dcd, top, label in [(HELIX_DCD, HELIX_TOP, "helix"), (EXTENDED_DCD, EXTENDED_TOP, "extended")]:
        if not dcd.is_file():
            raise FileNotFoundError(f"Missing A10 trajectory: {dcd}")
        if not top.is_file():
            raise FileNotFoundError(f"Missing A10 topology: {top}")
        print(f"Loading {label}: {dcd}")
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        feats = compute_spib_features(traj)
        feats = remove_burn_in(feats, FRAME_SPACING_PS, BURN_IN_NS)
        print(f"  frames after burn-in ({BURN_IN_NS:g} ns): {len(feats)}")
        blocks.append(feats)
    return np.vstack(blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SPIB on A10 for frozen transfer to A8Q2 ratchet.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Directory for checkpoints and scaler.")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for K-means and torch.")
    ap.add_argument("--z-dim", type=int, default=DEFAULT_Z_DIM, help="SPIB latent dimensionality.")
    args = ap.parse_args()

    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Computing pooled A10 features …")
    raw = _pooled_a10_features()
    n, dim = raw.shape
    if dim != SPIB_FEATURE_DIM:
        raise RuntimeError(f"Expected {SPIB_FEATURE_DIM} features, got {dim}")

    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    np.save(out_dir / "spib_feature_mean.npy", mean.astype(np.float64))
    np.save(out_dir / "spib_feature_std.npy", std.astype(np.float64))

    std_safe = np.where(std < 1e-12, 1.0, std)
    zdata = ((raw - mean) / std_safe).astype(np.float32)

    print("K-means (k=4) for initial labels …")
    km = KMeans(n_clusters=OUTPUT_DIM, random_state=args.seed, n_init=10)
    cluster_ids = km.fit_predict(zdata)
    traj_label_oh = np.eye(OUTPUT_DIM, dtype=np.float32)[cluster_ids]

    traj_data_t = torch.from_numpy(zdata)
    traj_label_t = torch.from_numpy(traj_label_oh)

    (
        data_shape,
        train_past,
        train_future,
        label_train,
        _wtr,
        test_past,
        test_future,
        label_test,
        _wte,
    ) = data_init(0, LAG_FRAMES, traj_data_t, traj_label_t, None)

    print(
        f"Time-lagged samples: train={len(train_past)} test={len(test_past)} "
        f"data_shape={data_shape} lag_frames={LAG_FRAMES}"
    )

    ib = SPIB(
        "Nonlinear",
        args.z_dim,
        OUTPUT_DIM,
        data_shape,
        device,
        UpdateLabel=True,
        neuron_num1=128,
        neuron_num2=128,
    )
    ib.to(device)
    ib.init_representative_inputs(train_past, label_train)

    # spib_lib.train concatenates output_path + "cpt" + index + "/IB"; it must end with
    # a path separator so checkpoints land in ``out_dir/cpt0/IB_*.pt``.
    try:
        output_path = os.path.relpath(out_dir.resolve(), _ROOT)
    except ValueError:
        output_path = str(out_dir.resolve())
    if not output_path.endswith(os.sep):
        output_path += os.sep

    failed = train(
        ib,
        BETA,
        train_past,
        train_future,
        label_train,
        None,
        test_past,
        test_future,
        label_test,
        None,
        LEARNING_RATE,
        LR_STEP,
        LR_GAMMA,
        BATCH_SIZE,
        THRESHOLD,
        PATIENCE,
        REFINEMENTS,
        output_path,
        LOG_INTERVAL,
        device,
        0,
    )
    if failed:
        raise RuntimeError("SPIB training aborted (NaN loss).")

    final_src = out_dir / "cpt0" / "IB_final_cpt.pt"
    if not final_src.is_file():
        alt = list(out_dir.glob("**/IB_final_cpt.pt"))
        if alt:
            final_src = alt[0]
        else:
            raise FileNotFoundError("Could not find IB_final_cpt.pt after training.")

    import shutil

    dst = out_dir / "spib_train_final_cpt.pt"
    shutil.copy2(final_src, dst)
    print(f"Copied final checkpoint to {dst}")

    meta = {
        "z_dim": args.z_dim,
        "output_dim": OUTPUT_DIM,
        "encoder_type": "Nonlinear",
        "data_shape": list(data_shape),
        "feature_dim": SPIB_FEATURE_DIM,
        "lag_frames": LAG_FRAMES,
        "frame_spacing_ps": FRAME_SPACING_PS,
        "burn_in_ns": BURN_IN_NS,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "kmeans_clusters": OUTPUT_DIM,
        "train_trajectories": [str(HELIX_DCD), str(EXTENDED_DCD)],
        "topology_pdb": {"helix": str(HELIX_TOP), "extended": str(EXTENDED_TOP)},
        "checkpoint_file": str(dst.name),
        "mean_file": "spib_feature_mean.npy",
        "std_file": "spib_feature_std.npy",
        "device_trained": str(device),
    }
    with open(out_dir / "spib_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_dir / 'spib_metadata.json'}")


if __name__ == "__main__":
    main()
