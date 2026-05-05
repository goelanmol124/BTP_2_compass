"""Enhanced COMPASS retrospective replay experiments.

.. deprecated:: COMPASS-v2

    Full-library oracle chunk replay remains **diagnostic only**. Prefer
    :mod:`src.compass.v2` for the primary seed-selection benchmark (see ``COMPASS_PLAN.md``).

This runner extends ``src.compass.replay`` with:

- two-phase / annealed scheduler,
- information-gain source deficit,
- continuous unvisited term,
- bootstrap-with-replacement random baseline,
- first-passage / important-bin coverage curves,
- prior-quality corruption experiments,
- TICA-1D/TICA-2D and SPIB-z1/SPIB-z2 source maps,
- weighted reason classifier and component-decomposition plots.

Guidance still uses only generic feature-derived latent maps. NHB-derived labels
are used for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication-oriented defaults for replay/SI figures (override mpl defaults).
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.65,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    }
)

# Okabe–Ito–compatible distinct colors for COMPASS vs random.
_COLOR_COMPASS = "#0072B2"
_COLOR_RANDOM_MEAN = "#000000"
_COLOR_RANDOM_BAND = "#999999"
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


FRAME_SPACING_NS = 0.1
DEFAULT_HELIX_NHB_THRESHOLD = 3
DEFAULT_OPEN_NHB_THRESHOLD = 1


@dataclass(frozen=True)
class Atlas:
    """Source map used by COMPASS."""

    source_xy: np.ndarray
    target_xy: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    source_hist: np.ndarray
    source_prob: np.ndarray
    source_support: np.ndarray
    free_energy_score: np.ndarray
    support_threshold: float
    map_label: str
    prior_quality: str
    frontier_xy: np.ndarray | None = None


@dataclass(frozen=True)
class ReplayConfig:
    repo_root: Path
    output_dir: Path
    chunk_frames: int
    n_steps: int
    random_repeats: int
    n_bins_x: int
    random_seed: int
    helix_nhb_threshold: int
    open_nhb_threshold: int
    variant: str
    schedule: str
    deficit_mode: str
    prior_quality: str
    prior_strength: float
    compass_repeats: int = 1
    softmax_temperature: float = 0.0
    map_kind: str = "tica2d"
    n_bins_y: int | None = None
    bootstrap_random: bool = True
    trust_controller: str = "none"
    trust_threshold: float = 0.65
    restart_cost: float = 0.0
    novelty_mode: str = "distance"
    bin_edges_mode: str = "source_only"


def _load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def project_tica(features: np.ndarray, mean: np.ndarray, eigenvectors: np.ndarray, dim: int = 2) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    mean = np.asarray(mean, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    dim = min(dim, eigenvectors.shape[1])
    return (features - mean) @ eigenvectors[:, :dim]


def _hist2d_xy(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    rx = max(float(x_edges[-1] - x_edges[0]), 1e-12)
    ry = max(float(y_edges[-1] - y_edges[0]), 1e-12)
    eps_x = max(1e-12, 1e-6 * rx)
    eps_y = max(1e-12, 1e-6 * ry)
    x = np.clip(x, x_edges[0] + eps_x, x_edges[-1] - eps_x)
    y = np.clip(y, y_edges[0] + eps_y, y_edges[-1] - eps_y)
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return hist.astype(float)


def _jsd(hist_ref: np.ndarray, hist_sample: np.ndarray, min_count: int = 3) -> float:
    p = hist_ref.ravel().astype(float)
    q = hist_sample.ravel().astype(float)
    mask = p >= min_count
    if not np.any(mask):
        return 1.0
    p = p[mask]
    q = q[mask]
    p = p / p.sum()
    q_sum = q.sum()
    if q_sum <= 0.0:
        return 1.0
    q = q / q_sum
    return float(jensenshannon(p, q, base=2) ** 2)


def _coverage(hist_ref: np.ndarray, hist_sample: np.ndarray, min_count: int = 3) -> float:
    mask = hist_ref >= min_count
    if not np.any(mask):
        return 0.0
    return float(np.mean(hist_sample[mask] > 0.0))


def _latent_wasserstein(ref_xy: np.ndarray, sample_xy: np.ndarray) -> float:
    """Marginal Wasserstein distance in latent space; lower means closer geometry."""

    if sample_xy.size == 0:
        return 1.0
    dims = min(ref_xy.shape[1], sample_xy.shape[1], 2)
    vals = [wasserstein_distance(ref_xy[:, i], sample_xy[:, i]) for i in range(dims)]
    return float(np.mean(vals))


def _subsample_rows(xy: np.ndarray, max_rows: int = 96) -> np.ndarray:
    if len(xy) <= max_rows:
        return xy
    idx = np.linspace(0, len(xy) - 1, max_rows).astype(int)
    return xy[idx]


def _mmd_rbf(ref_xy: np.ndarray, sample_xy: np.ndarray) -> float:
    """Biased RBF MMD for latent distributions; deterministic subsampling keeps it cheap."""

    if sample_xy.size == 0:
        return 1.0
    x = _subsample_rows(np.asarray(ref_xy[:, :2], dtype=float))
    y = _subsample_rows(np.asarray(sample_xy[:, :2], dtype=float))
    both = np.vstack([x, y])
    scale = float(np.median(np.sum((both - np.median(both, axis=0)) ** 2, axis=1)))
    gamma = 1.0 / max(2.0 * scale, 1e-6)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * d2)

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def _build_edges_source_only(source_x: np.ndarray, n_bins: int) -> np.ndarray:
    z = np.asarray(source_x, dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.linspace(-1.0, 1.0, n_bins + 1)
    lo, hi = np.percentile(z, [0.5, 99.5])
    pad = 0.08 * max(float(hi - lo), 1e-6)
    return np.linspace(float(lo - pad), float(hi + pad), n_bins + 1)


def _build_edges_pooled(source_x: np.ndarray, target_x: np.ndarray, n_bins: int) -> np.ndarray:
    both = np.concatenate(
        [np.asarray(source_x, dtype=float).ravel(), np.asarray(target_x, dtype=float).ravel()]
    )
    both = both[np.isfinite(both)]
    if both.size == 0:
        return np.linspace(-1.0, 1.0, n_bins + 1)
    lo, hi = np.percentile(both, [0.5, 99.5])
    pad = 0.08 * max(float(hi - lo), 1e-6)
    return np.linspace(float(lo - pad), float(hi + pad), n_bins + 1)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _corrupt_prior(prob: np.ndarray, prior_quality: str, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Return a deliberately corrupted source probability grid for robustness tests."""

    p = np.asarray(prob, dtype=float)
    p = p / p.sum()
    flat = p.ravel().copy()
    if prior_quality == "none":
        out = flat
    elif prior_quality == "uniform":
        out = np.ones_like(flat) / flat.size
    elif prior_quality == "shuffled":
        out = rng.permutation(flat)
    elif prior_quality == "inverted":
        ranks = np.argsort(np.argsort(flat))
        inv = ranks.max() - ranks + 1
        out = inv.astype(float) / inv.sum()
    elif prior_quality == "noisy":
        noise = rng.dirichlet(np.ones_like(flat))
        out = (1.0 - strength) * flat + strength * noise
    else:
        raise ValueError(f"Unsupported prior_quality: {prior_quality}")
    out = np.maximum(out, 0.0)
    out = out / out.sum()
    return out.reshape(p.shape)


def _make_atlas(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    n_bins_x: int,
    n_bins_y: int | None,
    map_label: str,
    prior_quality: str,
    prior_strength: float,
    random_seed: int,
    bin_edges_mode: str = "source_only",
) -> Atlas:
    if n_bins_y is None:
        n_bins_y = n_bins_x
    flat_y = np.allclose(source_xy[:, 1], source_xy[0, 1])
    if bin_edges_mode == "source_only":
        x_edges = _build_edges_source_only(source_xy[:, 0], n_bins_x)
        y_edges = (
            np.array([-1.0, 1.0])
            if flat_y
            else _build_edges_source_only(source_xy[:, 1], n_bins_y)
        )
    elif bin_edges_mode == "pooled":
        x_edges = _build_edges_pooled(source_xy[:, 0], target_xy[:, 0], n_bins_x)
        y_edges = (
            np.array([-1.0, 1.0])
            if flat_y
            else _build_edges_pooled(source_xy[:, 1], target_xy[:, 1], n_bins_y)
        )
    else:
        raise ValueError(f"Unknown bin_edges_mode: {bin_edges_mode}")
    source_hist = _hist2d_xy(source_xy, x_edges, y_edges)
    smooth = source_hist + 1.0
    source_prob = smooth / smooth.sum()
    source_prob = _corrupt_prior(
        source_prob,
        prior_quality=prior_quality,
        strength=prior_strength,
        rng=np.random.default_rng(random_seed + 991),
    )
    support_threshold = float(np.quantile(source_prob[source_prob > 0.0], 0.20))
    source_support = source_prob >= support_threshold
    free_energy = -np.log(np.maximum(source_prob, 1e-300))
    fe_score = np.zeros_like(source_prob, dtype=float)
    if np.any(source_support):
        fe_valid = free_energy[source_support]
        denom = max(float(fe_valid.max() - fe_valid.min()), 1e-12)
        fe_score[source_support] = (fe_valid - fe_valid.min()) / denom
    return Atlas(
        source_xy=source_xy,
        target_xy=target_xy,
        x_edges=x_edges,
        y_edges=y_edges,
        source_hist=source_hist,
        source_prob=source_prob,
        source_support=source_support,
        free_energy_score=fe_score,
        support_threshold=support_threshold,
        map_label=map_label,
        prior_quality=prior_quality,
    )


def build_tica_atlas(cfg: ReplayConfig) -> tuple[Atlas, dict[str, np.ndarray]]:
    artifacts = cfg.repo_root / "outputs" / "artifacts"
    tica_dir = cfg.repo_root / "outputs" / "tica_artifacts_a10"
    a10_features = _load_npy(artifacts / "a10_features.npy")
    a8_features = _load_npy(artifacts / "a8q2_features.npy")
    a8_nhb = _load_npy(artifacts / "a8q2_nhb.npy")
    mean = _load_npy(tica_dir / "a10_tica_mean0.npy")
    ev = _load_npy(tica_dir / "a10_tica_singular_vectors_left.npy")
    if cfg.map_kind == "tica1d":
        src1 = project_tica(a10_features, mean, ev, dim=1)
        tgt1 = project_tica(a8_features, mean, ev, dim=1)
        source_xy = np.column_stack([src1[:, 0], np.zeros(len(src1))])
        target_xy = np.column_stack([tgt1[:, 0], np.zeros(len(tgt1))])
    elif cfg.map_kind == "tica2d":
        source_xy = project_tica(a10_features, mean, ev, dim=2)
        target_xy = project_tica(a8_features, mean, ev, dim=2)
    else:
        raise ValueError(f"Unsupported TICA map_kind: {cfg.map_kind}")
    atlas = _make_atlas(
        source_xy,
        target_xy,
        cfg.n_bins_x,
        cfg.n_bins_y,
        cfg.map_kind,
        cfg.prior_quality,
        cfg.prior_strength,
        cfg.random_seed,
        cfg.bin_edges_mode,
    )
    return atlas, {"a8q2_nhb": a8_nhb}


def _load_spib_model(model_dir: Path, device):
    import torch

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.spib_features import SPIB_FEATURE_DIM
    from src.spib_lib import SPIB

    with open(model_dir / "spib_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    z_dim = int(metadata["z_dim"])
    output_dim = int(metadata["output_dim"])
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
    ckpt = torch.load(str(model_dir / "spib_train_final_cpt.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    mean = np.load(model_dir / "spib_feature_mean.npy")
    std = np.load(model_dir / "spib_feature_std.npy")
    return model, mean, std, z_dim


def _encode_spib(model, features: np.ndarray, mean: np.ndarray, std: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    import torch

    from src.spib_features import apply_spib_standardization

    x = apply_spib_standardization(features, mean, std)
    blocks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            z_mean, _ = model.encode(batch)
            blocks.append(z_mean.cpu().numpy())
    return np.vstack(blocks)


def _compute_a8q2_spib_features(repo_root: Path) -> np.ndarray:
    """Compute cached A8Q2 85-D SPIB features in the same extended+helix order as artifacts."""

    from src.analysis import BURNIN_FRAMES, load_trajectory
    from src.spib_features import compute_spib_features

    sim_dir = repo_root / "outputs" / "btp_equil" / "A8Q2"
    blocks = []
    for lab in ("extended", "helix"):
        dcd = sim_dir / f"A8Q2_{lab}_prod.dcd"
        top = sim_dir / f"A8Q2_{lab}_prod_top.pdb"
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        if traj.n_frames <= BURNIN_FRAMES:
            raise RuntimeError(f"{dcd}: need > {BURNIN_FRAMES} frames, got {traj.n_frames}")
        blocks.append(compute_spib_features(traj[BURNIN_FRAMES:]))
    return np.vstack(blocks)


def _spib_transfer_cache(cfg: ReplayConfig, z_dim: int) -> Path:
    return cfg.repo_root / "outputs" / "compass" / "cache" / f"spib_transfer_z{z_dim}.npz"


def build_spib_atlas(cfg: ReplayConfig) -> tuple[Atlas, dict[str, np.ndarray]]:
    import torch

    z_dim = 1 if cfg.map_kind == "spib1d" else 2
    model_dir = cfg.repo_root / "outputs" / ("spib_a10_model" if z_dim == 1 else "spib_a10_model_z2")
    analysis_dir = cfg.repo_root / "outputs" / ("spib_a10_analysis" if z_dim == 1 else "spib_a10_analysis_z2")
    cache = _spib_transfer_cache(cfg, z_dim)
    target_nhb = _load_npy(cfg.repo_root / "outputs" / "artifacts" / "a8q2_nhb.npy")
    if cache.is_file():
        data = np.load(cache)
        source_z = np.asarray(data["source_z"], dtype=float)
        target_z = np.asarray(data["target_z"], dtype=float)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        source_npz = np.load(analysis_dir / "spib_a10_frame_assignments.npz", allow_pickle=True)
        source_z = np.asarray(source_npz["z"], dtype=float)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, mean, std, _ = _load_spib_model(model_dir, device)
        a8_spib_features = _compute_a8q2_spib_features(cfg.repo_root)
        target_z = _encode_spib(model, a8_spib_features, mean, std, device)
        np.savez_compressed(cache, source_z=source_z, target_z=target_z)
    if z_dim == 1:
        source_xy = np.column_stack([source_z[:, 0], np.zeros(len(source_z))])
        target_xy = np.column_stack([target_z[:, 0], np.zeros(len(target_z))])
    else:
        source_xy = source_z[:, :2]
        target_xy = target_z[:, :2]
    atlas = _make_atlas(
        source_xy,
        target_xy,
        cfg.n_bins_x,
        cfg.n_bins_y,
        cfg.map_kind,
        cfg.prior_quality,
        cfg.prior_strength,
        cfg.random_seed,
        cfg.bin_edges_mode,
    )
    return atlas, {"a8q2_nhb": target_nhb}


def build_atlas(cfg: ReplayConfig) -> tuple[Atlas, dict[str, np.ndarray]]:
    if cfg.map_kind in {"tica1d", "tica2d"}:
        return build_tica_atlas(cfg)
    if cfg.map_kind in {"spib1d", "spib2d"}:
        return build_spib_atlas(cfg)
    if cfg.map_kind in {"hybrid_spib1d_tica2d", "hybrid_spib2d_tica2d"}:
        return build_hybrid_atlas(cfg)
    raise ValueError(f"Unsupported map_kind: {cfg.map_kind}")


def build_hybrid_atlas(cfg: ReplayConfig) -> tuple[Atlas, dict[str, np.ndarray]]:
    """Use TICA2D for density correction and SPIB latent space for discovery."""

    tica_atlas, arrays = build_tica_atlas(replace(cfg, map_kind="tica2d"))
    spib_kind = "spib1d" if cfg.map_kind == "hybrid_spib1d_tica2d" else "spib2d"
    spib_atlas, _ = build_spib_atlas(replace(cfg, map_kind=spib_kind))
    atlas = replace(
        tica_atlas,
        frontier_xy=spib_atlas.target_xy,
        map_label=f"hybrid({spib_kind} frontier + tica2d density)",
    )
    return atlas, arrays


def _chunk_slices(n_frames: int, chunk_frames: int) -> list[slice]:
    n_chunks = n_frames // chunk_frames
    return [slice(i * chunk_frames, (i + 1) * chunk_frames) for i in range(n_chunks)]


def _bin_indices(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bx = np.searchsorted(x_edges, xy[:, 0], side="right") - 1
    by = np.searchsorted(y_edges, xy[:, 1], side="right") - 1
    bx = np.clip(bx, 0, len(x_edges) - 2)
    by = np.clip(by, 0, len(y_edges) - 2)
    return bx.astype(int), by.astype(int)


def _selected_indices(chunks: list[slice], selected: Iterable[int]) -> np.ndarray:
    chosen = list(selected)
    if not chosen:
        return np.asarray([], dtype=int)
    return np.concatenate([np.arange(chunks[i].start, chunks[i].stop) for i in chosen])


def _selected_hist(chunks: list[slice], selected: Iterable[int], xy: np.ndarray, atlas: Atlas) -> np.ndarray:
    idx = _selected_indices(chunks, selected)
    if idx.size == 0:
        return np.zeros_like(atlas.source_hist, dtype=float)
    return _hist2d_xy(xy[idx], atlas.x_edges, atlas.y_edges)


def _important_source_mask(atlas: Atlas, min_bins: int = 20) -> np.ndarray:
    flat = atlas.source_prob.ravel()
    k = min(max(min_bins, flat.size // 20), flat.size)
    cutoff = np.partition(flat, -k)[-k]
    return atlas.source_prob >= cutoff


def _evaluate(
    chunks: list[slice],
    selected: list[int],
    xy: np.ndarray,
    ref_hist: np.ndarray,
    atlas: Atlas,
    nhb: np.ndarray,
    helix_threshold: int,
    open_threshold: int,
) -> dict[str, float]:
    hist = _selected_hist(chunks, selected, xy, atlas)
    idx = _selected_indices(chunks, selected)
    sample_xy = xy[idx] if idx.size else np.empty((0, xy.shape[1]), dtype=float)
    if idx.size:
        f_helix = float(np.mean(nhb[idx] >= helix_threshold))
        f_open = float(np.mean(nhb[idx] <= open_threshold))
    else:
        f_helix = 0.0
        f_open = 0.0
    f_helix_ref = float(np.mean(nhb >= helix_threshold))
    f_open_ref = float(np.mean(nhb <= open_threshold))
    visited = hist > 0.0
    important = _important_source_mask(atlas)
    return {
        "jsd": _jsd(ref_hist, hist),
        "latent_wasserstein": _latent_wasserstein(xy, sample_xy),
        "latent_mmd": _mmd_rbf(xy, sample_xy),
        "coverage": _coverage(ref_hist, hist),
        "important_coverage": float(np.mean(visited[important])) if np.any(important) else 0.0,
        "source_mass_visited": float(atlas.source_prob[visited].sum()),
        "f_helix": f_helix,
        "f_helix_error": abs(f_helix - f_helix_ref),
        "f_open": f_open,
        "f_open_error": abs(f_open - f_open_ref),
    }


def _weights_for_step(variant: str, schedule: str, fraction_done: float) -> dict[str, float]:
    base = {
        "source_deficit": 1.25,
        "source_representative": 0.65,
        "source_boundary_trust": 0.25,
        "frontier_disagreement": 0.35,
        "unvisited": 0.25,
        "novelty": 0.25,
        "redundancy": -0.80,
    }
    presets = {
        "exploratory": {
            "source_deficit": 0.15,
            "source_representative": 0.10,
            "source_boundary_trust": 1.20,
            "frontier_disagreement": 0.90,
            "unvisited": 0.75,
            "novelty": 0.65,
            "redundancy": -0.90,
        },
        "balanced": base,
        "exploit": {
            "source_deficit": 1.45,
            "source_representative": 0.85,
            "source_boundary_trust": 0.15,
            "frontier_disagreement": 0.10,
            "unvisited": 0.15,
            "novelty": 0.10,
            "redundancy": -0.60,
        },
        "source_only": {
            "source_deficit": 1.25,
            "source_representative": 0.65,
            "source_boundary_trust": 0.25,
            "frontier_disagreement": 0.0,
            "unvisited": 0.0,
            "novelty": 0.0,
            "redundancy": -0.40,
        },
        "pure_frontier": {
            "source_deficit": 0.0,
            "source_representative": 0.0,
            "source_boundary_trust": 0.0,
            "frontier_disagreement": 0.80,
            "unvisited": 0.80,
            "novelty": 0.80,
            "redundancy": -0.80,
        },
        "hybrid": {
            "source_deficit": 0.95,
            "source_representative": 0.45,
            "source_boundary_trust": 0.20,
            "frontier_disagreement": 0.85,
            "unvisited": 0.35,
            "novelty": 0.75,
            "redundancy": -0.85,
        },
        "hybrid_discovery": {
            "source_deficit": 0.20,
            "source_representative": 0.10,
            "source_boundary_trust": 0.25,
            "frontier_disagreement": 1.15,
            "unvisited": 0.70,
            "novelty": 0.95,
            "redundancy": -0.90,
        },
        "hybrid_density": {
            "source_deficit": 1.35,
            "source_representative": 0.75,
            "source_boundary_trust": 0.20,
            "frontier_disagreement": 0.35,
            "unvisited": 0.20,
            "novelty": 0.30,
            "redundancy": -0.70,
        },
    }
    if variant not in presets:
        raise ValueError(f"Unknown COMPASS variant: {variant}")
    if schedule == "static":
        return presets[variant]
    if schedule == "two_phase":
        if variant == "hybrid":
            if fraction_done < 0.30:
                return presets["hybrid_discovery"]
            if fraction_done < 0.70:
                return presets["hybrid"]
            return presets["hybrid_density"]
        if fraction_done < 0.30:
            return presets["exploratory"]
        if fraction_done < 0.70:
            return presets["balanced"]
        return presets["exploit"]
    if schedule == "anneal":
        early, late = presets["exploratory"], presets["exploit"]
        t = float(np.clip(fraction_done, 0.0, 1.0))
        return {k: (1.0 - t) * early[k] + t * late[k] for k in early}
    raise ValueError(f"Unknown schedule: {schedule}")


def _source_deficit_grid(atlas: Atlas, bin_attempts: np.ndarray, mode: str) -> np.ndarray:
    observed = bin_attempts.astype(float)
    observed = observed / max(float(observed.sum()), 1.0)
    desired = atlas.source_prob / atlas.source_prob.sum()
    if mode == "raw":
        return np.maximum(desired - observed, 0.0)
    if mode == "information_gain":
        lam = 20.0
        return desired * np.maximum(1.0 - observed / np.maximum(desired, 1e-12), 0.0) / (1.0 + lam * desired)
    raise ValueError(f"Unknown deficit_mode: {mode}")


def _prior_agreement(atlas: Atlas, bin_attempts: np.ndarray) -> float:
    """Agreement between selected start-bin frequencies and the source prior."""

    total = float(bin_attempts.sum())
    if total <= 1.0:
        return 1.0
    observed = bin_attempts.astype(float).ravel() / total
    desired = atlas.source_prob.astype(float).ravel()
    desired = desired / desired.sum()
    support = np.logical_or(observed > 0.0, desired > 0.0)
    divergence = float(jensenshannon(desired[support], observed[support], base=2) ** 2)
    return float(np.clip(1.0 - divergence, 0.0, 1.0))


def _apply_trust_controller(
    weights: dict[str, float],
    cfg: ReplayConfig,
    atlas: Atlas,
    bin_attempts: np.ndarray,
) -> tuple[dict[str, float], float]:
    if cfg.trust_controller == "none":
        return weights, 1.0
    if cfg.trust_controller != "prior_agreement":
        raise ValueError(f"Unknown trust_controller: {cfg.trust_controller}")

    agreement = _prior_agreement(atlas, bin_attempts)
    trust = float(np.clip(agreement / max(cfg.trust_threshold, 1e-6), 0.0, 1.0))
    out = dict(weights)
    for key in ("source_deficit", "source_representative", "source_boundary_trust"):
        out[key] *= trust
    frontier_boost = 1.0 + (1.0 - trust)
    for key in ("frontier_disagreement", "unvisited", "novelty"):
        out[key] *= frontier_boost
    return out, trust


def _compass_scores(
    atlas: Atlas,
    chunk_starts: np.ndarray,
    selected: list[int],
    chunk_bins: tuple[np.ndarray, np.ndarray],
    bin_attempts: np.ndarray,
    cfg: ReplayConfig,
    fraction_done: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    bx, by = chunk_bins
    source_boundary = atlas.free_energy_score[bx, by]
    trust = atlas.source_support[bx, by].astype(float)
    unvisited = 1.0 / (1.0 + bin_attempts[bx, by].astype(float))
    failure_decay = unvisited.copy()
    source_representative = _zscore(atlas.source_prob[bx, by])
    if selected:
        deficit_grid = _source_deficit_grid(atlas, bin_attempts, cfg.deficit_mode)
        source_deficit = _zscore(deficit_grid[bx, by])
    else:
        source_deficit = source_representative.copy()
    acq_source = atlas.target_xy[chunk_starts, :2]
    acq_frontier = (atlas.frontier_xy if atlas.frontier_xy is not None else atlas.target_xy)[chunk_starts, :2]
    if selected:
        selected_frontier = acq_frontier[selected]
        d2 = np.sum((acq_frontier[:, None, :] - selected_frontier[None, :, :]) ** 2, axis=2)
        distance_novelty = _zscore(np.sqrt(np.min(d2, axis=1)))
        center = selected_frontier.mean(axis=0, keepdims=True)
        frontier = _zscore(np.sqrt(np.sum((acq_frontier - center) ** 2, axis=1)))
        redundancy = np.exp(-np.min(d2, axis=1) / 0.02)
        if cfg.novelty_mode == "distance":
            novelty = distance_novelty
        elif cfg.novelty_mode == "epistemic":
            uncertainty_grid = np.sqrt(atlas.source_prob * (1.0 - atlas.source_prob)) / np.sqrt(
                1.0 + bin_attempts.astype(float)
            )
            novelty = _zscore(uncertainty_grid[bx, by]) * unvisited
        else:
            raise ValueError(f"Unknown novelty_mode: {cfg.novelty_mode}")
    else:
        novelty = np.ones(len(chunk_starts), dtype=float)
        frontier = np.ones(len(chunk_starts), dtype=float)
        redundancy = np.zeros(len(chunk_starts), dtype=float)
    frontier_disagreement = frontier * (1.0 - trust)
    components = {
        "source_deficit": source_deficit,
        "source_representative": source_representative,
        "source_boundary_trust": source_boundary * trust,
        "frontier_disagreement": frontier_disagreement,
        "unvisited": unvisited,
        "novelty": novelty,
        "redundancy": redundancy,
        "failure_decay": failure_decay,
        "trust": trust,
        "prior_trust": np.full(len(chunk_starts), _prior_agreement(atlas, bin_attempts), dtype=float),
        "source_boundary": source_boundary,
        "frontier_x": acq_frontier[:, 0],
        "frontier_y": acq_frontier[:, 1],
        "source_x": acq_source[:, 0],
        "source_y": acq_source[:, 1],
    }
    weights = _weights_for_step(cfg.variant, cfg.schedule, fraction_done)
    weights, trust_factor = _apply_trust_controller(weights, cfg, atlas, bin_attempts)
    weighted = {k: weights[k] * components[k] for k in weights}
    raw_score = np.zeros(len(chunk_starts), dtype=float)
    for arr in weighted.values():
        raw_score += arr
    if cfg.restart_cost > 0.0 and selected:
        restart_distance = np.sqrt(np.sum((acq_frontier - acq_frontier[selected[-1]]) ** 2, axis=1))
        components["restart_distance"] = restart_distance
        raw_score -= cfg.restart_cost * _zscore(restart_distance)
    components["controller_trust_factor"] = np.full(len(chunk_starts), trust_factor, dtype=float)
    score = raw_score * failure_decay
    return score, components, weights, weighted


def _softmax_choice(score: np.ndarray, remaining: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int = 5) -> int:
    masked = np.where(remaining, score, -np.inf)
    if temperature <= 0.0:
        return int(np.argmax(masked))
    valid_idx = np.flatnonzero(remaining)
    if valid_idx.size == 0:
        return int(np.argmax(masked))
    top = valid_idx[np.argsort(masked[valid_idx])[-top_k:]]
    sub = masked[top] - masked[top].max()
    p = np.exp(sub / max(temperature, 1e-6))
    p = p / p.sum()
    return int(rng.choice(top, p=p))


def run_compass_replay(
    atlas: Atlas,
    target_nhb: np.ndarray,
    cfg: ReplayConfig,
    start_chunk: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = _chunk_slices(len(target_nhb), cfg.chunk_frames)
    n_chunks = len(chunks)
    n_steps = min(cfg.n_steps, n_chunks)
    chunk_starts = np.array([s.start for s in chunks], dtype=int)
    chunk_xy = atlas.target_xy[chunk_starts]
    chunk_bins = _bin_indices(chunk_xy, atlas.x_edges, atlas.y_edges)
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    rng = rng if rng is not None else np.random.default_rng(cfg.random_seed)
    selected: list[int] = []
    remaining = np.ones(n_chunks, dtype=bool)
    bin_attempts = np.zeros_like(atlas.source_hist, dtype=int)
    metric_rows: list[dict[str, float | int | str]] = []
    decision_rows: list[dict[str, float | int | str]] = []

    def add_metrics(step: int) -> None:
        metrics = _evaluate(
            chunks,
            selected,
            atlas.target_xy,
            ref_hist,
            atlas,
            target_nhb,
            cfg.helix_nhb_threshold,
            cfg.open_nhb_threshold,
        )
        metric_rows.append(
            {
                "policy": "COMPASS",
                "variant": cfg.variant,
                "schedule": cfg.schedule,
                "map_kind": cfg.map_kind,
                "prior_quality": cfg.prior_quality,
                "deficit_mode": cfg.deficit_mode,
                "step": step,
                "selected_chunks": len(selected),
                "sampled_frames": len(selected) * cfg.chunk_frames,
                "sampled_ns": len(selected) * cfg.chunk_frames * FRAME_SPACING_NS,
                **metrics,
            }
        )

    first = int(start_chunk) if 0 <= start_chunk < n_chunks else int(rng.integers(0, n_chunks))
    selected.append(first)
    remaining[first] = False
    bx0, by0 = int(chunk_bins[0][first]), int(chunk_bins[1][first])
    bin_attempts[bx0, by0] += 1
    add_metrics(1)
    decision_rows.append(
        {
            "step": 1,
            "chunk": first,
            "start_frame": int(chunk_starts[first]),
            "start_x": float(chunk_xy[first, 0]),
            "start_y": float(chunk_xy[first, 1]),
            "eval_start_nhb": float(target_nhb[chunk_starts[first]]),
            "bin_x": bx0,
            "bin_y": by0,
            "score": float("nan"),
            "reason": "init_seed",
        }
    )

    for step in range(2, n_steps + 1):
        fraction_done = (step - 1) / max(float(n_steps), 1.0)
        score, comp, weights, weighted = _compass_scores(
            atlas, chunk_starts, selected, chunk_bins, bin_attempts, cfg, fraction_done
        )
        chosen = _softmax_choice(score, remaining, rng, cfg.softmax_temperature)
        selected.append(chosen)
        remaining[chosen] = False
        bx, by = int(chunk_bins[0][chosen]), int(chunk_bins[1][chosen])
        bin_attempts[bx, by] += 1
        contribution = {k: float(v[chosen]) for k, v in weighted.items()}
        positive = {k: v for k, v in contribution.items() if v > 0}
        reason = max(positive.items(), key=lambda kv: kv[1])[0] if positive else "penalty_limited"
        add_metrics(step)
        decision_rows.append(
            {
                "step": step,
                "chunk": chosen,
                "start_frame": int(chunk_starts[chosen]),
                "start_x": float(chunk_xy[chosen, 0]),
                "start_y": float(chunk_xy[chosen, 1]),
                "eval_start_nhb": float(target_nhb[chunk_starts[chosen]]),
                "bin_x": bx,
                "bin_y": by,
                "score": float(score[chosen]),
                "reason": reason,
                **{f"component_{k}": float(v[chosen]) for k, v in comp.items()},
                **{f"weight_{k}": float(v) for k, v in weights.items()},
                **{f"weighted_{k}": float(v) for k, v in contribution.items()},
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(decision_rows)


def run_random_replays(atlas: Atlas, target_nhb: np.ndarray, cfg: ReplayConfig) -> pd.DataFrame:
    chunks = _chunk_slices(len(target_nhb), cfg.chunk_frames)
    n_chunks = len(chunks)
    ref_hist = _hist2d_xy(atlas.target_xy, atlas.x_edges, atlas.y_edges)
    rng = np.random.default_rng(cfg.random_seed)
    rows: list[dict[str, float | int | str]] = []
    n_steps = min(cfg.n_steps, n_chunks)
    for rep in range(cfg.random_repeats):
        if cfg.bootstrap_random:
            order = rng.choice(n_chunks, size=n_steps, replace=True)
        else:
            order = rng.permutation(n_chunks)[:n_steps]
        selected: list[int] = []
        for step, chunk in enumerate(order, start=1):
            selected.append(int(chunk))
            metrics = _evaluate(
                chunks,
                selected,
                atlas.target_xy,
                ref_hist,
                atlas,
                target_nhb,
                cfg.helix_nhb_threshold,
                cfg.open_nhb_threshold,
            )
            rows.append(
                {
                    "policy": "Random",
                    "repeat": rep,
                    "step": step,
                    "selected_chunks": len(selected),
                    "sampled_frames": len(selected) * cfg.chunk_frames,
                    "sampled_ns": len(selected) * cfg.chunk_frames * FRAME_SPACING_NS,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


METRIC_COLS = [
    "jsd",
    "latent_wasserstein",
    "latent_mmd",
    "coverage",
    "important_coverage",
    "source_mass_visited",
    "f_helix",
    "f_helix_error",
    "f_open",
    "f_open_error",
]


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, as_index=False)[METRIC_COLS].agg(["mean", "std"])
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    rename = {f"{c}_": c for c in group_cols}
    return out.rename(columns=rename)


def _summarize_random(random_df: pd.DataFrame) -> pd.DataFrame:
    summary = _summarize(random_df, ["policy", "step", "sampled_ns"])
    for metric in METRIC_COLS:
        qs = random_df.groupby(["step", "sampled_ns"])[metric].quantile([0.025, 0.975]).unstack()
        qs = qs.reset_index().rename(columns={0.025: f"{metric}_lo", 0.975: f"{metric}_hi"})
        summary = summary.merge(qs, on=["step", "sampled_ns"], how="left")
    return summary


def _summarize_compass(compass_df: pd.DataFrame) -> pd.DataFrame:
    return _summarize(compass_df, ["step", "sampled_ns"])


def plot_curves(compass_df: pd.DataFrame, random_summary: pd.DataFrame, output_dir: Path) -> None:
    specs = [
        ("f_helix_error", "Helix-fraction error", "lower is better"),
        ("coverage", "Coverage vs A8Q2 reference", "higher is better"),
        ("important_coverage", "Important-bin coverage", "higher is better"),
        ("jsd", "JSD vs A8Q2 reference", "lower is better"),
        ("source_mass_visited", "Source-prior mass visited", "higher is better"),
        ("f_open_error", "Open-fraction error (NHB <= 1)", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    compass_summary = _summarize_compass(compass_df)
    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        cm = compass_summary[f"{metric}_mean"]
        cs = compass_summary[f"{metric}_std"].fillna(0.0)
        cx = compass_summary["sampled_ns"]
        ax.plot(cx, cm, label="COMPASS", lw=2.6, color=_COLOR_COMPASS)
        if (cs > 0).any():
            ax.fill_between(cx, cm - cs, cm + cs, color=_COLOR_COMPASS, alpha=0.15)
        mean = random_summary[f"{metric}_mean"]
        x = random_summary["sampled_ns"]
        lo_col, hi_col = f"{metric}_lo", f"{metric}_hi"
        lo = random_summary[lo_col] if lo_col in random_summary else mean - random_summary[f"{metric}_std"].fillna(0.0)
        hi = random_summary[hi_col] if hi_col in random_summary else mean + random_summary[f"{metric}_std"].fillna(0.0)
        ax.plot(x, mean, label="Random mean", color=_COLOR_RANDOM_MEAN, lw=2.0)
        ax.fill_between(x, lo, hi, color=_COLOR_RANDOM_BAND, alpha=0.28, label="Random 95% band")
        ax.set_title(title, fontweight="600")
        ax.set_xlabel("Sampled time (ns)")
        ax.set_ylabel(ylabel)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )
    fig.suptitle("Retrospective replay metrics vs sampled budget", fontsize=13, fontweight="700", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.11, wspace=0.28, hspace=0.38)
    fig.savefig(output_dir / "compass_replay_curves.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_map(atlas: Atlas, target_nhb: np.ndarray, decisions: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sc = ax.scatter(
        atlas.target_xy[:, 0],
        atlas.target_xy[:, 1],
        c=target_nhb,
        s=6,
        alpha=0.22,
        cmap="viridis",
        linewidths=0,
    )
    first = decisions.head(40)
    ax.scatter(
        first["start_x"],
        first["start_y"],
        c=first["step"],
        cmap="plasma",
        s=42,
        edgecolor="black",
        linewidth=0.4,
        label="COMPASS first 40 choices",
    )
    for _, row in first.head(12).iterrows():
        ax.text(row["start_x"], row["start_y"], str(int(row["step"])), fontsize=7)
    ax.set_xlabel(f"{atlas.map_label} x")
    ax.set_ylabel(f"{atlas.map_label} y")
    ax.set_title(f"COMPASS choices in {atlas.map_label} space")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="NHB (evaluation only)")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_dir / "compass_selected_probe_map.png", dpi=250)
    plt.close(fig)


def plot_reason_counts(decisions: pd.DataFrame, output_dir: Path) -> None:
    counts = decisions["reason"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    counts.plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_xlabel("selected chunks")
    ax.set_title("Dominant weighted score contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / "compass_reason_counts.png", dpi=250)
    plt.close(fig)


def plot_component_decomposition(decisions: pd.DataFrame, output_dir: Path, max_steps: int = 60) -> None:
    weighted_cols = [c for c in decisions.columns if c.startswith("weighted_")]
    if not weighted_cols:
        return
    sub = decisions[decisions["step"] > 1].head(max_steps).copy()
    if sub.empty:
        return
    x = sub["step"].to_numpy()
    # Colorblind-friendly sequence (extends Okabe–Ito as needed).
    pal = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
        "#000000",
    )
    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    positive_cols = [c for c in weighted_cols if sub[c].max() > 0]
    bottom = np.zeros(len(sub))
    for i, col in enumerate(positive_cols):
        vals = sub[col].clip(lower=0).to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.82,
            label=col.replace("weighted_", ""),
            color=pal[i % len(pal)],
            edgecolor="white",
            linewidth=0.35,
        )
        bottom += vals
    neg = np.zeros(len(sub))
    n_off = len(positive_cols)
    for j, col in enumerate([c for c in weighted_cols if sub[c].min() < 0]):
        vals = sub[col].clip(upper=0).to_numpy()
        ax.bar(
            x,
            vals,
            bottom=neg,
            width=0.82,
            label=col.replace("weighted_", ""),
            color=pal[(n_off + j) % len(pal)],
            edgecolor="white",
            linewidth=0.35,
        )
        neg += vals
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Weighted contribution")
    ax.set_title("COMPASS score decomposition (selected chunks)", fontweight="600")
    ax.legend(
        frameon=True,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fancybox=True,
        framealpha=0.96,
    )
    ax.grid(axis="y", alpha=0.35)
    ax.axhline(0.0, color="#333333", lw=0.8)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_dir / "compass_score_decomposition.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_first_passage(atlas: Atlas, decisions: pd.DataFrame, output_dir: Path, cfg: ReplayConfig) -> None:
    important = _important_source_mask(atlas)
    rows = []
    for bx, by in zip(*np.where(important)):
        hits = decisions[(decisions["bin_x"] == bx) & (decisions["bin_y"] == by)]
        first_step = int(hits["step"].min()) if not hits.empty else -1
        rows.append(
            {
                "bin_x": int(bx),
                "bin_y": int(by),
                "source_prob": float(atlas.source_prob[bx, by]),
                "first_step": first_step,
                "first_ns": float(first_step * cfg.chunk_frames * FRAME_SPACING_NS) if first_step > 0 else np.nan,
            }
        )
    df = pd.DataFrame(rows).sort_values("source_prob", ascending=False)
    df.to_csv(output_dir / "compass_first_passage_bins.csv", index=False)
    reached = df["first_ns"].dropna().sort_values().to_numpy()
    if reached.size:
        y = np.arange(1, reached.size + 1) / len(df)
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.step(reached, y, where="post", color="#D1495B", lw=2.2)
        ax.set_xlabel("sampled ns")
        ax.set_ylabel("fraction of important source bins discovered")
        ax.set_ylim(0, 1.02)
        ax.set_title("First-passage discovery curve")
        ax.grid(alpha=0.25)
        fig.savefig(output_dir / "compass_first_passage_ecdf.png", dpi=240)
        plt.close(fig)


def write_summary(
    cfg: ReplayConfig,
    atlas: Atlas,
    compass_df: pd.DataFrame,
    random_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary_compass = _summarize_compass(compass_df)
    final_compass = summary_compass.iloc[-1].to_dict()
    final_random = random_summary.iloc[-1].to_dict()
    summary = {
        "method": "COMPASS enhanced retrospective chunk acquisition",
        "map": atlas.map_label,
        "chunk_frames": cfg.chunk_frames,
        "variant": cfg.variant,
        "schedule": cfg.schedule,
        "deficit_mode": cfg.deficit_mode,
        "prior_quality": cfg.prior_quality,
        "prior_strength": cfg.prior_strength,
        "trust_controller": cfg.trust_controller,
        "trust_threshold": cfg.trust_threshold,
        "restart_cost": cfg.restart_cost,
        "novelty_mode": cfg.novelty_mode,
        "frame_spacing_ns": FRAME_SPACING_NS,
        "n_steps": int(compass_df["step"].max()),
        "random_repeats": cfg.random_repeats,
        "bootstrap_random": cfg.bootstrap_random,
        "support_threshold": atlas.support_threshold,
        "bin_edges_mode": cfg.bin_edges_mode,
        "final_compass": final_compass,
        "final_random_mean": final_random,
        "notes": [
            "No new MD; replay over cached A8Q2 arrays/chunks.",
            "NHB labels are used only for evaluation, not for selection.",
            "Random baseline uses bootstrap-with-replacement by default.",
            "Bin edges default to source_only (source latent percentiles); target histograms clip to outer bins.",
        ],
    }
    with open(output_dir / "compass_replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    warnings.warn(
        "src.compass.replay_enhanced oracle replay is deprecated as a headline benchmark. "
        "Use src.compass.v2 bench for equivalence-cloud candidates and online-faithful seeds.",
        DeprecationWarning,
        stacklevel=1,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compass/replay_enhanced"))
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--n-bins-x", type=int, default=30)
    parser.add_argument("--n-bins-y", type=int, default=None)
    parser.add_argument(
        "--map-kind",
        choices=["tica2d", "tica1d", "spib1d", "spib2d", "hybrid_spib1d_tica2d", "hybrid_spib2d_tica2d"],
        default="tica2d",
    )
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--helix-nhb-threshold", type=int, default=DEFAULT_HELIX_NHB_THRESHOLD)
    parser.add_argument("--open-nhb-threshold", type=int, default=DEFAULT_OPEN_NHB_THRESHOLD)
    parser.add_argument("--variant", choices=["balanced", "exploratory", "exploit", "source_only", "pure_frontier", "hybrid"], default="balanced")
    parser.add_argument("--schedule", choices=["static", "two_phase", "anneal"], default="static")
    parser.add_argument("--deficit-mode", choices=["raw", "information_gain"], default="information_gain")
    parser.add_argument("--prior-quality", choices=["none", "uniform", "shuffled", "inverted", "noisy"], default="none")
    parser.add_argument("--prior-strength", type=float, default=0.50)
    parser.add_argument("--compass-repeats", type=int, default=1)
    parser.add_argument("--softmax-temperature", type=float, default=0.0)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--trust-controller", choices=["none", "prior_agreement"], default="none")
    parser.add_argument("--trust-threshold", type=float, default=0.65)
    parser.add_argument("--restart-cost", type=float, default=0.0)
    parser.add_argument("--novelty-mode", choices=["distance", "epistemic"], default="distance")
    parser.add_argument("--no-bootstrap-random", action="store_true")
    parser.add_argument(
        "--bin-edges",
        choices=["source_only", "pooled"],
        default="source_only",
        help="source_only: grid from source latent only. pooled: legacy pooled source+target percentiles.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ReplayConfig(
        repo_root=repo_root,
        output_dir=output_dir,
        chunk_frames=args.chunk_frames,
        n_steps=args.n_steps,
        random_repeats=args.random_repeats,
        n_bins_x=args.n_bins_x,
        random_seed=args.random_seed,
        helix_nhb_threshold=args.helix_nhb_threshold,
        open_nhb_threshold=args.open_nhb_threshold,
        variant=args.variant,
        schedule=args.schedule,
        deficit_mode=args.deficit_mode,
        prior_quality=args.prior_quality,
        prior_strength=args.prior_strength,
        compass_repeats=args.compass_repeats,
        softmax_temperature=args.softmax_temperature,
        map_kind=args.map_kind,
        n_bins_y=args.n_bins_y,
        bootstrap_random=not args.no_bootstrap_random,
        trust_controller=args.trust_controller,
        trust_threshold=args.trust_threshold,
        restart_cost=args.restart_cost,
        novelty_mode=args.novelty_mode,
        bin_edges_mode=args.bin_edges,
    )
    atlas, arrays = build_atlas(cfg)
    target_nhb = arrays["a8q2_nhb"]
    if cfg.compass_repeats <= 1:
        compass_df, decisions = run_compass_replay(
            atlas, target_nhb, cfg, start_chunk=args.start_chunk, rng=np.random.default_rng(cfg.random_seed)
        )
    else:
        n_chunks = len(_chunk_slices(len(target_nhb), cfg.chunk_frames))
        compass_dfs = []
        decisions = None
        seed_rng = np.random.default_rng(cfg.random_seed)
        for rep in range(cfg.compass_repeats):
            sub_rng = np.random.default_rng(cfg.random_seed + rep + 1)
            start = int(seed_rng.integers(0, n_chunks))
            df, dec = run_compass_replay(atlas, target_nhb, cfg, start_chunk=start, rng=sub_rng)
            df = df.copy()
            df["repeat"] = rep
            compass_dfs.append(df)
            if rep == 0:
                decisions = dec
        compass_df = pd.concat(compass_dfs, ignore_index=True)
    random_df = run_random_replays(atlas, target_nhb, cfg)
    random_summary = _summarize_random(random_df)
    compass_df.to_csv(output_dir / "compass_metrics.csv", index=False)
    decisions.to_csv(output_dir / "compass_decisions.csv", index=False)
    random_df.to_csv(output_dir / "random_metrics.csv", index=False)
    random_summary.to_csv(output_dir / "random_summary.csv", index=False)
    plot_curves(compass_df, random_summary, output_dir)
    plot_map(atlas, target_nhb, decisions, output_dir)
    plot_reason_counts(decisions, output_dir)
    plot_component_decomposition(decisions, output_dir)
    write_first_passage(atlas, decisions, output_dir, cfg)
    write_summary(cfg, atlas, compass_df, random_summary, output_dir)
    final_c = _summarize_compass(compass_df).iloc[-1]
    final_r = random_summary.iloc[-1]
    print(f"Wrote enhanced COMPASS replay outputs to {output_dir}")
    print(
        "Final @ %.1f ns: COMPASS JSD=%.4f cov=%.3f imp_cov=%.3f helix_err=%.4f open_err=%.4f | "
        "Random JSD=%.4f cov=%.3f imp_cov=%.3f helix_err=%.4f open_err=%.4f"
        % (
            final_c["sampled_ns"],
            final_c["jsd_mean"],
            final_c["coverage_mean"],
            final_c["important_coverage_mean"],
            final_c["f_helix_error_mean"],
            final_c["f_open_error_mean"],
            final_r["jsd_mean"],
            final_r["coverage_mean"],
            final_r["important_coverage_mean"],
            final_r["f_helix_error_mean"],
            final_r["f_open_error_mean"],
        )
    )


if __name__ == "__main__":
    main()
