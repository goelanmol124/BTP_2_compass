"""
Analyze trained A10 SPIB states against helix/open conformational metrics.

Outputs, by default, to ``outputs/spib_a10_analysis``:
  - spib_a10_latent_conformation.png
  - spib_a10_state_summary.csv
  - spib_a10_summary.json
  - spib_a10_frame_assignments.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis import (  # noqa: E402
    BURNIN_FRAMES,
    FRAME_SPACING_NS,
    RMSD_COIL_NM,
    RMSD_HELIX_NM,
    compute_backbone_rmsd_to_ideal_helix,
    compute_nhb,
    load_trajectory,
)
from src.spib_features import apply_spib_standardization, compute_spib_features  # noqa: E402
from src.spib_lib import SPIB  # noqa: E402
from src.utils import remove_burn_in  # noqa: E402

A10_DIR = _ROOT / "outputs" / "btp_equil" / "A10"
A10_TRAJS = (
    ("helix-start", A10_DIR / "A10_helix_prod.dcd", A10_DIR / "A10_helix_prod_top.pdb"),
    ("open-start", A10_DIR / "A10_extended_prod.dcd", A10_DIR / "A10_extended_prod_top.pdb"),
)
HELIX_REF = A10_DIR / "A10_helix.pdb"

DEFAULT_MODEL_DIR = _ROOT / "outputs" / "spib_a10_model"
DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "spib_a10_analysis"

DEFAULT_SPIB_Z_DIM = 1
DEFAULT_SPIB_OUTPUT_DIM = 4
SPIB_FEATURE_DIM = 85


def _load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "spib_metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_model(model_dir: Path, device: torch.device) -> tuple[SPIB, np.ndarray, np.ndarray, int, int]:
    metadata = _load_metadata(model_dir)
    z_dim = int(metadata.get("z_dim", DEFAULT_SPIB_Z_DIM))
    output_dim = int(metadata.get("output_dim", DEFAULT_SPIB_OUTPUT_DIM))
    mean = np.load(model_dir / "spib_feature_mean.npy")
    std = np.load(model_dir / "spib_feature_std.npy")
    ckpt_path = model_dir / "spib_train_final_cpt.pt"
    if not ckpt_path.is_file():
        ckpt_path = model_dir / "cpt0" / "IB_final_cpt.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No SPIB checkpoint found under {model_dir}")

    model = SPIB(
        "Nonlinear",
        z_dim,
        output_dim,
        (SPIB_FEATURE_DIM,),
        device,
        UpdateLabel=True,
        neuron_num1=128,
        neuron_num2=128,
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, mean, std, z_dim, output_dim


def _ca_end_to_end_nm(traj: md.Trajectory) -> np.ndarray:
    ca = traj.topology.select("name CA")
    if len(ca) != 10:
        raise ValueError(f"Expected 10 CA atoms, found {len(ca)}")
    pair = np.asarray([[ca[0], ca[-1]]], dtype=np.int32)
    return md.compute_distances(traj, pair, periodic=False)[:, 0]


def _classify_conformation(rmsd_nm: np.ndarray, nhb: np.ndarray) -> np.ndarray:
    labels = np.full(len(rmsd_nm), "transition", dtype=object)
    labels[(rmsd_nm <= RMSD_HELIX_NM) | (nhb >= 3)] = "helix-like"
    labels[(rmsd_nm >= RMSD_COIL_NM) & (nhb < 3)] = "open-like"
    return labels


def _encode(
    model: SPIB,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    z_blocks: list[np.ndarray] = []
    p_blocks: list[np.ndarray] = []
    x = apply_spib_standardization(features, mean, std)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            z_mean, _z_logvar = model.encode(batch)
            log_p = model.decoder(z_mean)
            z_blocks.append(z_mean.cpu().numpy())
            p_blocks.append(log_p.exp().cpu().numpy())
    return np.vstack(z_blocks), np.vstack(p_blocks)


def _state_rows(
    state_id: np.ndarray,
    conformation: np.ndarray,
    z: np.ndarray,
    rmsd: np.ndarray,
    nhb: np.ndarray,
    e2e: np.ndarray,
    output_dim: int,
):
    rows = []
    for sid in range(output_dim):
        mask = state_id == sid
        n = int(mask.sum())
        if n == 0:
            row = {
                "spib_state": sid,
                "n_frames": 0,
                "fraction": 0.0,
                "dominant_conformation": "absent",
                "helix_like_fraction": 0.0,
                "transition_fraction": 0.0,
                "open_like_fraction": 0.0,
                "mean_rmsd_nm": float("nan"),
                "mean_nhb": float("nan"),
                "mean_ca_end_to_end_nm": float("nan"),
            }
            for dim in range(z.shape[1]):
                row[f"mean_z{dim + 1}"] = float("nan")
            rows.append(row)
            continue

        conf_counts = {name: int(np.sum(conformation[mask] == name)) for name in ("helix-like", "transition", "open-like")}
        dominant = max(conf_counts, key=conf_counts.get)
        row = {
            "spib_state": sid,
            "n_frames": n,
            "fraction": float(n / len(state_id)),
            "dominant_conformation": dominant,
            "helix_like_fraction": float(conf_counts["helix-like"] / n),
            "transition_fraction": float(conf_counts["transition"] / n),
            "open_like_fraction": float(conf_counts["open-like"] / n),
            "mean_rmsd_nm": float(np.mean(rmsd[mask])),
            "mean_nhb": float(np.mean(nhb[mask])),
            "mean_ca_end_to_end_nm": float(np.mean(e2e[mask])),
        }
        for dim in range(z.shape[1]):
            row[f"mean_z{dim + 1}"] = float(np.mean(z[mask, dim]))
        rows.append(row)
    return rows


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    preferred = [
        "spib_state",
        "n_frames",
        "fraction",
        "dominant_conformation",
        "helix_like_fraction",
        "transition_fraction",
        "open_like_fraction",
    ]
    z_fields = sorted([key for key in rows[0] if key.startswith("mean_z")])
    remaining = [key for key in rows[0] if key not in preferred and key not in z_fields]
    fieldnames = preferred + z_fields + remaining
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    out_png: Path,
    traj_name: np.ndarray,
    time_ns: np.ndarray,
    z: np.ndarray,
    state_id: np.ndarray,
    conformation: np.ndarray,
    rmsd: np.ndarray,
    nhb: np.ndarray,
    output_dim: int,
) -> None:
    colors = {"helix-like": "tab:green", "transition": "tab:gray", "open-like": "tab:orange"}
    state_markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for name in ("helix-start", "open-start"):
        mask_name = traj_name == name
        for conf, color in colors.items():
            mask = mask_name & (conformation == conf)
            ax.scatter(time_ns[mask], z[mask, 0], s=8, alpha=0.55, c=color, label=f"{name}, {conf}")
    ax.set_xlabel("Time after 25 ns burn-in (ns)")
    ax.set_ylabel("SPIB latent coordinate z1")
    ax.set_title("A10 SPIB latent coordinate colored by conformation")
    ax.legend(fontsize=7, ncols=2)

    ax = axes[0, 1]
    if z.shape[1] >= 2:
        for conf, color in colors.items():
            mask = conformation == conf
            ax.scatter(z[mask, 0], z[mask, 1], s=8, alpha=0.45, c=color, label=conf)
        ax.set_xlabel("SPIB latent coordinate z1")
        ax.set_ylabel("SPIB latent coordinate z2")
        ax.set_title("2D latent space colored by conformation")
        ax.legend(fontsize=8)
    else:
        for conf, color in colors.items():
            mask = conformation == conf
            ax.scatter(z[mask, 0], rmsd[mask], s=8, alpha=0.45, c=color, label=conf)
        ax.axhline(RMSD_HELIX_NM, color="tab:green", ls="--", lw=1, alpha=0.7)
        ax.axhline(RMSD_COIL_NM, color="tab:orange", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel("SPIB latent coordinate z1")
        ax.set_ylabel("Backbone RMSD to ideal helix (nm)")
        ax.set_title("Latent coordinate vs helix RMSD")
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    for sid in range(output_dim):
        marker = state_markers[sid % len(state_markers)]
        mask = state_id == sid
        ax.scatter(rmsd[mask], nhb[mask], s=10, alpha=0.45, marker=marker, label=f"SPIB state {sid}")
    ax.axvline(RMSD_HELIX_NM, color="tab:green", ls="--", lw=1, alpha=0.7)
    ax.axvline(RMSD_COIL_NM, color="tab:orange", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Backbone RMSD to ideal helix (nm)")
    ax.set_ylabel("N_HB")
    ax.set_title("SPIB state assignments on standard helix/open metrics")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    bins = np.linspace(float(np.min(z[:, 0])), float(np.max(z[:, 0])), 45)
    for conf, color in colors.items():
        mask = conformation == conf
        ax.hist(z[mask, 0], bins=bins, density=True, alpha=0.45, color=color, label=conf)
    ax.set_xlabel("SPIB latent coordinate z1")
    ax.set_ylabel("Density")
    ax.set_title("Conformational separation along z1")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_latent_space(out_png: Path, z: np.ndarray, conformation: np.ndarray, state_id: np.ndarray, output_dim: int) -> None:
    """Plot the learned SPIB latent space colored by helix/open assignment."""
    colors = {"helix-like": "tab:green", "transition": "tab:gray", "open-like": "tab:orange"}
    if z.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        for conf in ("open-like", "transition", "helix-like"):
            mask = conformation == conf
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=11,
                alpha=0.5,
                c=colors[conf],
                edgecolors="none",
                label=f"{conf} (n={int(mask.sum())})",
            )

        for sid in range(output_dim):
            mask = state_id == sid
            if not np.any(mask):
                continue
            center = np.mean(z[mask, :2], axis=0)
            ax.scatter(center[0], center[1], s=90, c="black", marker="x", linewidths=1.6)
            ax.text(center[0], center[1], f" S{sid}", ha="left", va="center", fontsize=9)

        ax.set_xlabel("SPIB latent coordinate z1")
        ax.set_ylabel("SPIB latent coordinate z2")
        ax.set_title("A10 two-dimensional SPIB latent space colored by helix/open conformation")
        ax.legend(loc="best", fontsize=9, frameon=True)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return

    y_centers = {"helix-like": 1.0, "transition": 0.0, "open-like": -1.0}
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for conf in ("open-like", "transition", "helix-like"):
        mask = conformation == conf
        jitter = rng.normal(0.0, 0.055, size=int(mask.sum()))
        y = y_centers[conf] + jitter
        ax.scatter(
            z[mask, 0],
            y,
            s=10,
            alpha=0.5,
            c=colors[conf],
            edgecolors="none",
            label=f"{conf} (n={int(mask.sum())})",
        )

    state_order = np.argsort([np.mean(z[state_id == sid, 0]) if np.any(state_id == sid) else np.inf for sid in range(output_dim)])
    for sid in state_order:
        mask = state_id == sid
        if not np.any(mask):
            continue
        center = float(np.mean(z[mask, 0]))
        ax.axvline(center, color="black", lw=0.8, alpha=0.25)
        ax.text(center, 1.28, f"S{sid}", ha="center", va="bottom", fontsize=8)

    ax.set_yticks([y_centers["open-like"], y_centers["transition"], y_centers["helix-like"]])
    ax.set_yticklabels(["open-like", "transition", "helix-like"])
    ax.set_xlabel("SPIB latent coordinate z1")
    ax.set_ylabel("Conformation class")
    ax.set_title("A10 one-dimensional SPIB latent space colored by helix/open conformation")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze A10 SPIB states against helix/open metrics.")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, mean, std, z_dim, output_dim = _load_model(args.model_dir.resolve(), device)

    feature_blocks = []
    traj_names = []
    times = []
    rmsd_blocks = []
    nhb_blocks = []
    e2e_blocks = []

    for name, dcd, top in A10_TRAJS:
        if not dcd.is_file() or not top.is_file():
            raise FileNotFoundError(f"Missing trajectory/topology for {name}: {dcd}, {top}")
        print(f"Loading {name}: {dcd}")
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        traj = traj[BURNIN_FRAMES:]
        feature_blocks.append(compute_spib_features(traj))
        traj_names.append(np.full(traj.n_frames, name, dtype=object))
        times.append(np.arange(traj.n_frames, dtype=float) * FRAME_SPACING_NS)
        rmsd_blocks.append(compute_backbone_rmsd_to_ideal_helix(traj, str(HELIX_REF)))
        nhb_blocks.append(compute_nhb(traj))
        e2e_blocks.append(_ca_end_to_end_nm(traj))
        print(f"  analyzed {traj.n_frames} frames")

    features = np.vstack(feature_blocks)
    traj_name = np.concatenate(traj_names)
    time_ns = np.concatenate(times)
    rmsd = np.concatenate(rmsd_blocks)
    nhb = np.concatenate(nhb_blocks)
    e2e = np.concatenate(e2e_blocks)

    z, probs = _encode(model, features, mean, std, device, args.batch_size)
    state_id = probs.argmax(axis=1)
    conformation = _classify_conformation(rmsd, nhb)
    rows = _state_rows(state_id, conformation, z, rmsd, nhb, e2e, output_dim)

    _write_summary_csv(out_dir / "spib_a10_state_summary.csv", rows)
    _plot(out_dir / "spib_a10_latent_conformation.png", traj_name, time_ns, z, state_id, conformation, rmsd, nhb, output_dim)
    latent_space_plot = out_dir / f"spib_a10_latent_space_{z_dim}d_conformations.png"
    _plot_latent_space(latent_space_plot, z, conformation, state_id, output_dim)

    summary = {
        "n_frames": int(len(state_id)),
        "z_dim": z_dim,
        "output_dim": output_dim,
        "trajectories": [name for name, _dcd, _top in A10_TRAJS],
        "classification_rule": {
            "helix_like": f"RMSD <= {RMSD_HELIX_NM} nm OR N_HB >= 3",
            "open_like": f"RMSD >= {RMSD_COIL_NM} nm AND N_HB < 3",
            "transition": "all remaining frames",
        },
        "state_summary": rows,
        "plot": str(out_dir / "spib_a10_latent_conformation.png"),
        "latent_space_plot": str(latent_space_plot),
    }
    with (out_dir / "spib_a10_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_dir / "spib_a10_frame_assignments.npz",
        traj_name=traj_name,
        time_ns=time_ns,
        z=z,
        probabilities=probs,
        state_id=state_id,
        conformation=conformation,
        rmsd_nm=rmsd,
        nhb=nhb,
        ca_end_to_end_nm=e2e,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
