"""
Kinetic/predictive diagnostics for A10 SPIB state assignments.

This checks whether SPIB states that look similar by RMSD/NHB still have
different time-lagged behavior. If they do, SPIB is learning kinetic substates;
if not, the extra states are likely label noise or an initialization artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ASSIGNMENTS = Path("outputs/spib_a10_analysis_z2/spib_a10_frame_assignments.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/spib_a10_analysis_z2")
FRAME_SPACING_NS = 0.1
CONFORMATIONS = ("helix-like", "transition", "open-like")


def _contiguous_trajectory_slices(traj_name: np.ndarray) -> list[tuple[str, np.ndarray]]:
    names: list[tuple[str, np.ndarray]] = []
    for name in dict.fromkeys(traj_name.tolist()):
        names.append((str(name), np.where(traj_name == name)[0]))
    return names


def _transition_counts_by_state(
    state_id: np.ndarray,
    traj_slices: list[tuple[str, np.ndarray]],
    n_states: int,
    lag_frames: int,
) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    for _name, idx in traj_slices:
        if len(idx) <= lag_frames:
            continue
        past = state_id[idx[:-lag_frames]]
        future = state_id[idx[lag_frames:]]
        for i, j in zip(past, future):
            counts[int(i), int(j)] += 1
    return counts


def _future_conformation_counts(
    state_id: np.ndarray,
    conformation: np.ndarray,
    traj_slices: list[tuple[str, np.ndarray]],
    n_states: int,
    lag_frames: int,
) -> np.ndarray:
    counts = np.zeros((n_states, len(CONFORMATIONS)), dtype=np.int64)
    conf_to_idx = {name: i for i, name in enumerate(CONFORMATIONS)}
    for _name, idx in traj_slices:
        if len(idx) <= lag_frames:
            continue
        past = state_id[idx[:-lag_frames]]
        future_conf = conformation[idx[lag_frames:]]
        for state, conf in zip(past, future_conf):
            counts[int(state), conf_to_idx[str(conf)]] += 1
    return counts


def _row_normalize(counts: np.ndarray) -> np.ndarray:
    denom = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, denom, out=np.zeros_like(counts, dtype=float), where=denom > 0)


def _dwell_summary(
    state_id: np.ndarray,
    traj_slices: list[tuple[str, np.ndarray]],
    n_states: int,
) -> list[dict]:
    dwell_by_state: list[list[float]] = [[] for _ in range(n_states)]
    for _name, idx in traj_slices:
        states = state_id[idx]
        if len(states) == 0:
            continue
        start = 0
        for pos in range(1, len(states) + 1):
            if pos == len(states) or states[pos] != states[start]:
                dwell_by_state[int(states[start])].append((pos - start) * FRAME_SPACING_NS)
                start = pos

    rows = []
    for state, values in enumerate(dwell_by_state):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            rows.append(
                {
                    "spib_state": state,
                    "n_runs": 0,
                    "mean_dwell_ns": float("nan"),
                    "median_dwell_ns": float("nan"),
                    "p90_dwell_ns": float("nan"),
                    "max_dwell_ns": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "spib_state": state,
                "n_runs": int(arr.size),
                "mean_dwell_ns": float(np.mean(arr)),
                "median_dwell_ns": float(np.median(arr)),
                "p90_dwell_ns": float(np.percentile(arr, 90)),
                "max_dwell_ns": float(np.max(arr)),
            }
        )
    return rows


def _state_geometry_rows(
    state_id: np.ndarray,
    conformation: np.ndarray,
    rmsd: np.ndarray,
    nhb: np.ndarray,
    e2e: np.ndarray,
    z: np.ndarray,
    n_states: int,
) -> list[dict]:
    rows = []
    for state in range(n_states):
        mask = state_id == state
        n = int(mask.sum())
        if n == 0:
            continue
        row = {
            "spib_state": state,
            "n_frames": n,
            "helix_like_fraction": float(np.mean(conformation[mask] == "helix-like")),
            "transition_fraction": float(np.mean(conformation[mask] == "transition")),
            "open_like_fraction": float(np.mean(conformation[mask] == "open-like")),
            "mean_rmsd_nm": float(np.mean(rmsd[mask])),
            "mean_nhb": float(np.mean(nhb[mask])),
            "mean_ca_end_to_end_nm": float(np.mean(e2e[mask])),
        }
        for dim in range(z.shape[1]):
            row[f"mean_z{dim + 1}"] = float(np.mean(z[mask, dim]))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_diagnostics(
    out_png: Path,
    transition_probs_1ns: np.ndarray,
    future_conf_probs_1ns: np.ndarray,
    future_conf_probs_5ns: np.ndarray,
    dwell_rows: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    im = ax.imshow(transition_probs_1ns, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_title("SPIB state transition probability at 1 ns")
    ax.set_xlabel("Future SPIB state")
    ax.set_ylabel("Current SPIB state")
    ax.set_xticks(range(transition_probs_1ns.shape[1]))
    ax.set_yticks(range(transition_probs_1ns.shape[0]))
    for i in range(transition_probs_1ns.shape[0]):
        for j in range(transition_probs_1ns.shape[1]):
            ax.text(j, i, f"{transition_probs_1ns[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    x = np.arange(future_conf_probs_1ns.shape[0])
    bottom = np.zeros_like(x, dtype=float)
    colors = {"helix-like": "tab:green", "transition": "tab:gray", "open-like": "tab:orange"}
    for k, conf in enumerate(CONFORMATIONS):
        ax.bar(x, future_conf_probs_1ns[:, k], bottom=bottom, label=conf, color=colors[conf], alpha=0.8)
        bottom += future_conf_probs_1ns[:, k]
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xlabel("Current SPIB state")
    ax.set_ylabel("Future conformation probability")
    ax.set_title("Conformation after 1 ns")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bottom = np.zeros_like(x, dtype=float)
    for k, conf in enumerate(CONFORMATIONS):
        ax.bar(x, future_conf_probs_5ns[:, k], bottom=bottom, label=conf, color=colors[conf], alpha=0.8)
        bottom += future_conf_probs_5ns[:, k]
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xlabel("Current SPIB state")
    ax.set_ylabel("Future conformation probability")
    ax.set_title("Conformation after 5 ns")

    ax = axes[1, 1]
    states = [row["spib_state"] for row in dwell_rows]
    means = [row["mean_dwell_ns"] for row in dwell_rows]
    p90 = [row["p90_dwell_ns"] for row in dwell_rows]
    ax.bar(np.asarray(states) - 0.18, means, width=0.36, label="mean", color="tab:blue", alpha=0.8)
    ax.bar(np.asarray(states) + 0.18, p90, width=0.36, label="p90", color="tab:purple", alpha=0.8)
    ax.set_xlabel("SPIB state")
    ax.set_ylabel("Dwell time (ns)")
    ax.set_title("Contiguous state dwell times")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SPIB state kinetics and future conformation prediction.")
    ap.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.assignments, allow_pickle=True)
    traj_name = data["traj_name"]
    state_id = data["state_id"].astype(int)
    conformation = data["conformation"]
    rmsd = data["rmsd_nm"]
    nhb = data["nhb"]
    e2e = data["ca_end_to_end_nm"]
    z = data["z"]

    n_states = int(state_id.max()) + 1
    traj_slices = _contiguous_trajectory_slices(traj_name)
    lag_specs = {"0.1ns": 1, "1ns": 10, "5ns": 50}

    transitions = {}
    future_conformations = {}
    future_rows = []
    for label, lag_frames in lag_specs.items():
        state_counts = _transition_counts_by_state(state_id, traj_slices, n_states, lag_frames)
        state_probs = _row_normalize(state_counts)
        conf_counts = _future_conformation_counts(state_id, conformation, traj_slices, n_states, lag_frames)
        conf_probs = _row_normalize(conf_counts)
        transitions[label] = {"counts": state_counts.tolist(), "probabilities": state_probs.tolist()}
        future_conformations[label] = {"counts": conf_counts.tolist(), "probabilities": conf_probs.tolist()}
        for state in range(n_states):
            future_rows.append(
                {
                    "lag": label,
                    "spib_state": state,
                    "p_future_helix_like": float(conf_probs[state, 0]),
                    "p_future_transition": float(conf_probs[state, 1]),
                    "p_future_open_like": float(conf_probs[state, 2]),
                    "p_same_spib_state": float(state_probs[state, state]),
                }
            )

    dwell_rows = _dwell_summary(state_id, traj_slices, n_states)
    geometry_rows = _state_geometry_rows(state_id, conformation, rmsd, nhb, e2e, z, n_states)

    _write_csv(out_dir / "spib_a10_state_future_conformation.csv", future_rows)
    _write_csv(out_dir / "spib_a10_state_dwell_times.csv", dwell_rows)
    _write_csv(out_dir / "spib_a10_state_geometry_kinetic_context.csv", geometry_rows)
    _plot_diagnostics(
        out_dir / "spib_a10_state_kinetics.png",
        np.asarray(transitions["1ns"]["probabilities"]),
        np.asarray(future_conformations["1ns"]["probabilities"]),
        np.asarray(future_conformations["5ns"]["probabilities"]),
        dwell_rows,
    )

    summary = {
        "assignments": str(args.assignments),
        "n_frames": int(len(state_id)),
        "n_states": n_states,
        "frame_spacing_ns": FRAME_SPACING_NS,
        "trajectory_blocks": {name: int(len(idx)) for name, idx in traj_slices},
        "geometry_by_state": geometry_rows,
        "dwell_by_state": dwell_rows,
        "future_conformation_by_state": future_conformations,
        "state_transition_by_lag": transitions,
        "plots": {"kinetics": str(out_dir / "spib_a10_state_kinetics.png")},
    }
    with (out_dir / "spib_a10_state_kinetics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
