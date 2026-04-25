"""
SPIB featurization: 85-D vectors for Calm decapeptide trajectories.

Per frame:
  - 45 × 1/r : inverse Calpha–Calpha distance for all unordered pairs (10 choose 2)
  - 40 × sin/cos φ,ψ : same backbone torsion encoding as :func:`compute_featurization`

A10 and A8Q2 both have 10 Calpha atoms; pair ordering uses residue order in the
topology (combinations of CA indices), so feature dimension matches across systems.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from src.analysis import compute_featurization

if TYPE_CHECKING:
    import mdtraj as md

# C(10, 2) Calpha–Calpha pairs
N_CA_CA_PAIRS: int = 45
SPIB_FEATURE_DIM: int = N_CA_CA_PAIRS + 40  # 85

# Avoid division by zero when two Cα are extremely close (nm)
INV_R_EPS_NM: float = 1e-6


def ca_ca_pair_indices(topology: "md.Topology") -> np.ndarray:
    """Return shape (45, 2) int array of atom indices for all Cα–Cα pairs (i < j in residue order)."""
    ca = topology.select("name CA")
    if len(ca) != 10:
        raise ValueError(f"Expected 10 Calpha atoms for SPIB features; got {len(ca)}")
    pairs: list[tuple[int, int]] = []
    for i, j in itertools.combinations(range(len(ca)), 2):
        pairs.append((int(ca[i]), int(ca[j])))
    return np.asarray(pairs, dtype=np.int32)


def compute_spib_features(traj_protein: "md.Trajectory") -> np.ndarray:
    """Compute 85-D SPIB features for each frame.

    Parameters
    ----------
    traj_protein : mdtraj.Trajectory
        Protein-only trajectory (same selection as ``load_trajectory(..., protein_only=True)``).

    Returns
    -------
    features : np.ndarray
        Shape ``(n_frames, 85)``, dtype float32.
    """
    import mdtraj as md

    pairs = ca_ca_pair_indices(traj_protein.topology)
    dists_nm = md.compute_distances(traj_protein, pairs, periodic=False)
    inv_r = (1.0 / (dists_nm + INV_R_EPS_NM)).astype(np.float32)
    torsions = compute_featurization(traj_protein)
    return np.hstack([inv_r, torsions.astype(np.float32)])


def apply_spib_standardization(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-score features using saved donor statistics (for frozen transfer).

    Parameters
    ----------
    features : (n_frames, 85) array
    mean, std : (85,) arrays from A10 training
    """
    x = np.asarray(features, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    s = np.asarray(std, dtype=np.float64).reshape(1, -1)
    s_safe = np.where(s < 1e-12, 1.0, s)
    out = (x - m) / s_safe
    return out.astype(np.float32)
