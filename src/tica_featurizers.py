"""Reusable TICA input features for proteins (backbone torsions, CA geometry, contacts).

All featurizers take an MDTraj protein trajectory (optionally with ACE/NME) and
return float32 arrays of shape (n_frames, n_features) suitable for TICA.
"""

from __future__ import annotations

import numpy as np

from src.analysis import compute_featurization

try:
    import mdtraj as md
except ImportError as exc:  # pragma: no cover
    raise ImportError("mdtraj is required for tica_featurizers") from exc


def _ca_atom_indices(traj: "md.Trajectory") -> np.ndarray:
    idx = traj.topology.select("name CA")
    if len(idx) < 3:
        raise ValueError("Need at least 3 Cα atoms for geometry features.")
    return np.asarray(idx, dtype=int)


def ca_distances_lower_triangle(traj: "md.Trajectory") -> np.ndarray:
    """All unique Cα–Cα distances (i < j), column order (0,1),(0,2),...,(n-2,n-1)."""

    ca = _ca_atom_indices(traj)
    n = len(ca)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((ca[i], ca[j]))
    if not pairs:
        raise ValueError("No CA pairs")
    p = np.asarray(pairs, dtype=int)
    d = md.compute_distances(traj, p)
    return np.asarray(d, dtype=np.float32)


def ca_distances_longrange(traj: "md.Trajectory", min_sep: int = 4) -> np.ndarray:
    """Cα–Cα distances for residue pairs with sequence separation ≥ ``min_sep`` (i < j, j−i ≥ min_sep).

    Order: increasing i, then j (e.g. for n=10, min_sep=4: (0,4)…(0,9), (1,5)…, …).
    """

    ca = _ca_atom_indices(traj)
    n = len(ca)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if j - i >= min_sep:
                pairs.append((ca[i], ca[j]))
    if not pairs:
        raise ValueError("No long-range CA pairs (increase chain length or lower min_sep).")
    p = np.asarray(pairs, dtype=int)
    d = md.compute_distances(traj, p)
    return np.asarray(d, dtype=np.float32)


def inv_r_ca_lower_triangle(traj: "md.Trajectory", r_min_nm: float = 0.08) -> np.ndarray:
    """1 / r for the same Cα pairs as :func:`ca_distances_lower_triangle`."""

    d = ca_distances_lower_triangle(traj).astype(np.float64)
    d = np.maximum(d, r_min_nm)
    return (1.0 / d).astype(np.float32)


def backbone_angle_cos_at_ca(traj: "md.Trajectory") -> np.ndarray:
    """cos(∠(CA_{i-1}, CA_i, CA_{i+1})) for each interior residue i."""

    ca = _ca_atom_indices(traj)
    if len(ca) < 3:
        raise ValueError("Need ≥3 Cα atoms")
    trip = np.stack([ca[:-2], ca[1:-1], ca[2:]], axis=1)
    ang = md.compute_angles(traj, trip)
    return np.cos(ang).astype(np.float32)


def ca_pairwise_distance_velocity(traj: "md.Trajectory", r_min_nm: float = 0.08) -> np.ndarray:
    """Per-pair first difference of Cα–Cα distance: Δd = d(t) - d(t-1); first frame zero-padded.

    Same pair ordering as :func:`ca_distances_lower_triangle`.
    """

    d = ca_distances_lower_triangle(traj).astype(np.float64)
    dv = np.zeros_like(d)
    if d.shape[0] > 1:
        dv[1:] = np.diff(d, axis=0)
    return np.clip(dv, -5.0, 5.0).astype(np.float32)


def featurize_torsions_sc(traj: "md.Trajectory") -> np.ndarray:
    """40D sin/cos φ,ψ (protocol-matched :func:`src.analysis.compute_featurization`)."""

    return compute_featurization(traj)


def featurize_torsions_plus_ca_dist(traj: "md.Trajectory") -> np.ndarray:
    """Concatenate sin/cos torsions and Cα lower-triangle distances."""

    t = featurize_torsions_sc(traj)
    d = ca_distances_lower_triangle(traj)
    if t.shape[0] != d.shape[0]:
        raise ValueError("traj featurization length mismatch")
    return np.hstack([t, d]).astype(np.float32)


# Registry: name -> (function, description)
FEATURE_SET_DOC: dict[str, str] = {
    "torsions_sc": "sin/cos backbone φ,ψ (40D), same as production pipeline",
    "ca_dist_lower": "all Cα–Cα pairwise distances (lower triangle, nm)",
    "ca_dist_longrange": "Cα–Cα distances for sequence separation ≥ 4 (compact tertiary register)",
    "inv_r_ca": "1/r for Cα–Cα pairs (capped, nm⁻¹)",
    "backbone_angle_cos": "cos(θ) for Cα_{i-1}–Cα_i–Cα_{i+1} angles",
    "ca_dist_velocity": "per-pair Δ(distance) per frame; first row zero (nm/frame scale)",
    "torsions_plus_ca": "concat torsions_sc + ca_dist_lower (40 + n_pair)",
    "torsions_ca_zscore_donor": "SPECIAL: z-score torsions and full CA dist using μ,σ from A10 only, then concat",
    "torsions_longrange_zscore_donor": "SPECIAL: z-score torsions + long-range CA dist; μ,σ from A10 only",
}

FEATURIZERS: dict[str, callable] = {
    "torsions_sc": featurize_torsions_sc,
    "ca_dist_lower": ca_distances_lower_triangle,
    "ca_dist_longrange": ca_distances_longrange,
    "inv_r_ca": inv_r_ca_lower_triangle,
    "backbone_angle_cos": backbone_angle_cos_at_ca,
    "ca_dist_velocity": ca_pairwise_distance_velocity,
    "torsions_plus_ca": featurize_torsions_plus_ca_dist,
}

# Keys handled in scripts/tica_feature_set_sweep.py (donor-normalized concat).
SPECIAL_POOL_FEATURE_SETS: frozenset[str] = frozenset(
    {"torsions_ca_zscore_donor", "torsions_longrange_zscore_donor"}
)
