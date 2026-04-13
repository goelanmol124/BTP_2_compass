"""
adaptive_seeding.py — Three-strategy adaptive seeding experiment for A8Q2.

Implements the Phase 8 adaptive seeding protocol from EXPERIMENT_PROTOCOL.md §11.
Three strategies are run in parallel for N_ITERATIONS iterations of N_SEEDS seeds
of SEED_NS nanoseconds each:

  Strategy 1 — Random:
      Uniform random frame selection from the pooled trajectory. No CV.

  Strategy 2 — Transferred TICA:
      A10 TIC-1 projected onto A8Q2 backbone torsion features. Metastable
      least-counts bin selection. Requires pre-trained A10 TICA eigenvectors.

  Strategy 3 — Native TICA:
      A8Q2 TIC-1 re-trained each iteration (deferred to iteration 3). Falls
      back to N_HB as a stand-in CV for iterations 1–2, and on subsequent
      iterations if TICA validation fails (block bootstrap min |cos θ| < 0.90
      or |r(TIC-1, N_HB)| < 0.70). Which CV was actually used is logged.

Initialization (HTMD generator approach, no pilot run):
  Iteration 1: all 8 seeds start from the final NPT equilibration frame
  (A8Q2_extended_npt_final.pdb) with fresh Maxwell-Boltzmann velocities.
  From iteration 2 onward the seed selection algorithm is applied normally.

References
----------
  Hruska et al. J. Chem. Phys. 149, 244119 (2018)  — metastable least-counts
  Doerr & De Fabritiis. J. Chem. Theory Comput. 10, 2064 (2014)  — HTMD seeding
  EXPERIMENT_PROTOCOL.md §11 — full protocol specification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import simulation parameters from the single source of truth
# ---------------------------------------------------------------------------
from src.analysis import (
    histogram_rmsd_nhb_fixed_edges,
    pooled_ground_truth_fes_rmsd_nhb,
)

from src.config import (
    TEMPERATURE_K,
    TIMESTEP_FS,
    DCD_INTERVAL_PS,
    CHK_INTERVAL_PS,
    FRICTION_PS,
    NONBONDED_CUTOFF_NM,
    SWITCH_DISTANCE_NM,
    EWALD_TOLERANCE,
    TICA_MIN_LAG_FRAMES,
    TICA_EPSILON,
    TICA_SCALING,
    TICA_BOOTSTRAP_BLOCK_SIZE,
    TICA_N_BOOTSTRAP,
    SEED_NS as CONFIG_SEED_NS,
    N_SEEDS as CONFIG_N_SEEDS,
    N_ITERATIONS as CONFIG_N_ITERATIONS,
    N_CV_BINS,
    FES_MIN_COUNT,
    N_HB_MAX,
    NHB_THRESHOLD_FRAC,
)

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module can be imported for unit
# testing without a full GPU environment.
# ---------------------------------------------------------------------------
try:
    from src.tica_train import block_bootstrap_tic1_stability  # type: ignore
    _TICA_TRAIN_AVAILABLE = True
except ImportError:
    _TICA_TRAIN_AVAILABLE = False

try:
    from src.metrics import (  # type: ignore
        compute_coverage,
        compute_entropy,
        compute_f_helix_error,
        compute_jsd,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    compute_f_helix_error = None  # type: ignore

# ---------------------------------------------------------------------------
# Module-level constants derived from config
# ---------------------------------------------------------------------------
SEED_NS: float = float(CONFIG_SEED_NS)
"""Duration of each seed run in ns (from src.config)."""

N_SEEDS: int = int(CONFIG_N_SEEDS)
"""Seeds per iteration (from src.config)."""

N_ITERATIONS: int = int(CONFIG_N_ITERATIONS)
"""Total adaptive seeding iterations per strategy (from src.config)."""

SEED_STEPS: int = int(SEED_NS * 1_000_000 / TIMESTEP_FS)
"""1_500_000 steps (30 ns at 2 fs/step)."""

DCD_STEPS: int = int(DCD_INTERVAL_PS * (1_000 / TIMESTEP_FS))
"""DCD frame every 100 ps → 300 frames per 30 ns seed."""

CHK_STEPS: int = int(CHK_INTERVAL_PS * (1_000 / TIMESTEP_FS))
"""Checkpoint every 5000 ps → at most 5 ns lost on crash."""

N_BINS: int = int(N_CV_BINS)
"""Equal-width CV bins for metastable least-counts (from ``src.config.N_CV_BINS``)."""

TICA_COSINE_SIM_THRESHOLD: float = 0.90
"""Block bootstrap cosine similarity threshold for native TICA validation."""

TICA_PEARSON_R_THRESHOLD: float = 0.70
"""Minimum |r(TIC-1, N_HB)| required to accept the native TICA model."""

TICA_DEFER_UNTIL: int = 3
"""Earliest iteration at which native TICA training is attempted (1-indexed)."""

log = logging.getLogger(__name__)


# =============================================================================
# SEED SELECTION — Strategy 1: Random
# =============================================================================


def select_seeds_random(
    pooled_features: np.ndarray,
    n_seeds: int = 8,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """Select ``n_seeds`` frame indices uniformly at random from pooled_features.

    Strategy 1 (Random) baseline: no collective variable is used. Each frame
    in the pooled trajectory has equal probability of being selected.
    Sampling is with replacement to avoid exhausting small pools.

    Parameters
    ----------
    pooled_features : np.ndarray
        Shape (n_frames, n_features). Only the frame count (axis 0) is used;
        feature values are not examined.
    n_seeds : int
        Number of seed frame indices to return.
    rng_seed : int or None
        NumPy random seed for reproducibility. None → non-deterministic.

    Returns
    -------
    np.ndarray
        Shape (n_seeds,), dtype int64. Indices into pooled_features row axis.
    """
    rng = np.random.default_rng(rng_seed)
    n_frames = len(pooled_features)
    return rng.integers(0, n_frames, size=n_seeds)


# =============================================================================
# SEED SELECTION — Strategies 2 & 3: Metastable Least-Counts
# =============================================================================


def select_seeds_least_counts(
    cv_values: np.ndarray,
    n_seeds: int = 8,
    n_bins: int = 10,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """Metastable least-counts seed selection on a 1D collective variable.

    Algorithm (Hruska et al. J. Chem. Phys. 149, 244119, 2018):
    1. Discretise the 1D CV range into ``n_bins`` equal-width bins.
    2. Count frames per bin.
    3. Rank bins by count in ascending order (least-populated first).
    4. For each of the ``n_seeds`` least-populated bins: draw one frame at
       random from that bin.
    5. If the chosen bin is empty: fall back to the closest populated bin
       (scanning outward from the empty bin index in both directions).

    Why coarse (10) bins rather than fine-grained least-counts:
    Fine-bin least-counts wastes seeds on high-free-energy regions that are
    sparsely populated simply because they are energetically unfavourable, not
    because they are undersampled. Ten coarse bins map onto physically
    meaningful regions: deep coil basin, transition region, helix basin, and
    some high-FE extremes. Protocol §11.3.

    Parameters
    ----------
    cv_values : np.ndarray
        Shape (n_frames,). CV value for each pooled frame. Must be 1D and
        finite (no NaN/Inf).
    n_seeds : int
        Number of seed frames to select.
    n_bins : int
        Number of equal-width bins for discretisation. Protocol default: 10.
    rng_seed : int or None
        NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_seeds,), dtype int64. Frame indices into cv_values.

    Raises
    ------
    ValueError
        If cv_values is empty or contains non-finite values.
    """
    cv_values = np.asarray(cv_values, dtype=float)
    if cv_values.ndim != 1 or len(cv_values) == 0:
        raise ValueError("cv_values must be a non-empty 1D array.")
    if not np.all(np.isfinite(cv_values)):
        raise ValueError("cv_values contains NaN or Inf values.")

    rng = np.random.default_rng(rng_seed)
    n_frames = len(cv_values)

    # --- Discretise into equal-width bins ---
    cv_min, cv_max = cv_values.min(), cv_values.max()
    if cv_min == cv_max:
        # Degenerate: all frames in one bin → random selection
        log.warning(
            "CV range is zero (all frames have identical CV value). "
            "Falling back to uniform random selection."
        )
        return rng.integers(0, n_frames, size=n_seeds)

    edges = np.linspace(cv_min, cv_max, n_bins + 1)
    # Ensure the maximum value lands in the last bin, not past it
    bin_ids = np.searchsorted(edges[1:-1], cv_values, side="right")  # shape (n_frames,)

    # Bin → list of frame indices
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for frame_idx, b in enumerate(bin_ids):
        bins[b].append(frame_idx)

    # --- Rank bins by count ascending ---
    counts = np.array([len(b) for b in bins], dtype=int)
    sorted_bin_order = np.argsort(counts, kind="stable")  # ascending count

    selected: list[int] = []
    for rank in range(n_seeds):
        bin_idx = int(sorted_bin_order[rank % n_bins])

        # If this bin is empty, find closest populated bin
        if counts[bin_idx] == 0:
            bin_idx = _find_nearest_populated_bin(bins, bin_idx)

        chosen_frame = int(rng.choice(bins[bin_idx]))
        selected.append(chosen_frame)

    return np.array(selected, dtype=np.int64)


def _find_nearest_populated_bin(bins: list[list[int]], empty_idx: int) -> int:
    """Return the index of the closest non-empty bin to ``empty_idx``.

    Scans outward from ``empty_idx`` in both directions simultaneously until
    a non-empty bin is found. Raises RuntimeError if all bins are empty
    (should never happen when called from select_seeds_least_counts with a
    non-empty trajectory).
    """
    n = len(bins)
    for delta in range(1, n + 1):
        for direction in (-delta, delta):
            candidate = empty_idx + direction
            if 0 <= candidate < n and len(bins[candidate]) > 0:
                return candidate
    raise RuntimeError("All bins are empty — the pooled trajectory has no frames.")


# =============================================================================
# CV PROJECTION
# =============================================================================


def project_cv(
    features: np.ndarray,
    strategy: str,
    cv_model=None,
    nhb: Optional[np.ndarray] = None,
    mean_donor: Optional[np.ndarray] = None,
    eigenvectors_donor: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Compute CV values for each frame in the pooled feature array.

    Dispatches to the appropriate projection based on ``strategy``:

    ``'random'``
        Returns None. Strategy 1 needs no CV.

    ``'transferred'``
        Projects A8Q2 features onto A10 TIC-1 using the donor TICA mean and
        eigenvectors. The projection is:
            z = (features − mean_donor) @ eigenvectors_donor[:, 0]
        This is the TICA projection equation applied cross-system (transfer).
        See EXPERIMENT_PROTOCOL.md §8.2 for the transfer metric derivation.

    ``'native'``
        If ``cv_model`` is a fitted deeptime TICA model: project using its
        ``transform()`` method and return TIC-1 (column 0).
        If ``cv_model`` is None (iterations 1–2, or failed validation): return
        ``nhb`` directly as the CV. N_HB is a physical observable that tracks
        the helix-coil transition without requiring training data.

    Parameters
    ----------
    features : np.ndarray
        Shape (n_frames, 40). Sin/cos backbone torsion features.
    strategy : str
        One of ``'random'``, ``'transferred'``, ``'native'``.
    cv_model : deeptime.decomposition.TICA or None
        Fitted TICA model for the native strategy (deeptime 0.4+).
        Ignored for ``'random'`` and ``'transferred'``.
    nhb : np.ndarray or None
        Shape (n_frames,). Per-frame hydrogen bond count. Used as fallback CV
        for the native strategy when no TICA model is available.
    mean_donor : np.ndarray or None
        Shape (40,). Feature mean from the donor (A10) TICA fit.
        Required for ``'transferred'``.
    eigenvectors_donor : np.ndarray or None
        Shape (40, n_tics). TICA eigenvectors from the donor A10 fit.
        Required for ``'transferred'``.

    Returns
    -------
    np.ndarray of shape (n_frames,), or None for strategy ``'random'``.

    Raises
    ------
    ValueError
        If required arguments are missing for the chosen strategy.
    """
    if strategy == "random":
        return None

    if strategy == "transferred":
        if mean_donor is None or eigenvectors_donor is None:
            raise ValueError(
                "strategy='transferred' requires mean_donor and eigenvectors_donor."
            )
        # Cross-system TICA projection: (X - μ_A10) @ v_A10[:,0]
        centered = features - mean_donor[np.newaxis, :]
        return (centered @ eigenvectors_donor[:, 0]).astype(np.float64)

    if strategy == "native":
        if cv_model is not None:
            # deeptime TICA: transform returns (n_frames, n_tics)
            projected = cv_model.transform(features)
            if projected.ndim == 2:
                return projected[:, 0].astype(np.float64)
            return projected.astype(np.float64)
        else:
            # N_HB fallback for iterations 1–2 (or failed TICA validation)
            if nhb is None:
                raise ValueError(
                    "strategy='native' with cv_model=None requires nhb array "
                    "(N_HB fallback for iterations 1–2)."
                )
            return np.asarray(nhb, dtype=np.float64)

    raise ValueError(
        f"Unknown strategy '{strategy}'. Must be 'random', 'transferred', or 'native'."
    )


# =============================================================================
# FEATURIZATION HELPERS
# =============================================================================


def _featurize_trajectory(
    traj_dcd: str,
    top_pdb: str,
    helix_ref_pdb: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a DCD and compute features, N_HB, and backbone RMSD vs ideal helix.

    Uses the same atom selection and observables as Phase 5 / ``src.analysis``.

    Parameters
    ----------
    traj_dcd : str
        Path to DCD trajectory (full system: protein + water + ions).
    top_pdb : str
        Path to topology PDB written immediately before the DCDReporter.
    helix_ref_pdb : str
        Ideal helix PDB for :func:`src.analysis.compute_backbone_rmsd_to_ideal_helix`.

    Returns
    -------
    features : np.ndarray
        Shape (n_frames, 40), sin/cos backbone torsion features.
    nhb : np.ndarray
        Shape (n_frames,), i→i+4 backbone H-bond counts.
    rmsd : np.ndarray
        Shape (n_frames,), BB RMSD to ideal helix (nm).
    """
    from src.analysis import (
        compute_backbone_rmsd_to_ideal_helix,
        compute_featurization,
        compute_nhb,
        load_trajectory,
    )

    traj_prot = load_trajectory(traj_dcd, top_pdb, protein_only=True)
    features = compute_featurization(traj_prot)
    nhb = compute_nhb(traj_prot).astype(np.float64)
    rmsd = compute_backbone_rmsd_to_ideal_helix(traj_prot, helix_ref_pdb)
    return features, nhb, rmsd


def _pool_trajectories(
    dcd_paths: list[str],
    top_pdb: str,
    helix_ref_pdb: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Featurize and concatenate multiple DCD trajectories.

    Parameters
    ----------
    dcd_paths : list[str]
        Ordered list of DCD file paths to pool (all seeds from all prior
        iterations). Non-existent paths are silently skipped with a warning.
    top_pdb : str
        Common topology PDB (same topology for all trajectories in the pool).
    helix_ref_pdb : str
        Ideal helix PDB for RMSD (same as ground-truth FES definition).

    Returns
    -------
    features_pool : np.ndarray
        Shape (n_total_frames, 40).
    nhb_pool : np.ndarray
        Shape (n_total_frames,).
    rmsd_pool : np.ndarray
        Shape (n_total_frames,), RMSD vs ideal helix (nm).
    segments : list[tuple[str, int]]
        ``(dcd_path, n_frames)`` for each successfully loaded file, in order.
        Maps pooled row indices to the correct DCD for :func:`run_seed`.
    """
    all_features: list[np.ndarray] = []
    all_nhb: list[np.ndarray] = []
    all_rmsd: list[np.ndarray] = []
    segments: list[tuple[str, int]] = []

    for dcd in dcd_paths:
        if not os.path.exists(dcd):
            log.warning("Pooling: DCD not found, skipping: %s", dcd)
            continue
        try:
            feats, nhb, rmsd = _featurize_trajectory(dcd, top_pdb, helix_ref_pdb)
            segments.append((dcd, int(feats.shape[0])))
            all_features.append(feats)
            all_nhb.append(nhb)
            all_rmsd.append(rmsd)
        except Exception as exc:
            log.warning("Pooling: failed to load %s (%s), skipping.", dcd, exc)

    if not all_features:
        raise RuntimeError(
            "No valid trajectories found in pool. Cannot proceed with seed selection."
        )

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_nhb, axis=0),
        np.concatenate(all_rmsd, axis=0),
        segments,
    )


def _pooled_frame_to_dcd_local(
    segments: list[tuple[str, int]],
    pooled_index: int,
) -> tuple[str, int]:
    """Map a concatenated-pool frame index to ``(dcd_path, index_within_dcd)``."""
    off = 0
    for path, n in segments:
        if pooled_index < off + n:
            return path, pooled_index - off
        off += n
    raise IndexError(
        f"Pooled frame index {pooled_index} out of range for pool "
        f"({len(segments)} segments, {off} total frames)."
    )


# =============================================================================
# NATIVE TICA TRAINING AND VALIDATION
# =============================================================================


def _try_train_native_tica(
    features_pool: np.ndarray,
    nhb_pool: np.ndarray,
    iteration: int,
    lag: Optional[int] = None,
    n_components: int = 5,
    rng_seed: Optional[int] = None,
) -> tuple[Optional[object], str]:
    """Attempt to train a native TICA model on pooled A8Q2 data.

    Schedule (EXPERIMENT_PROTOCOL.md §11.3):
    - Iterations 1–2 (``iteration < TICA_DEFER_UNTIL``): return (None, 'nhb').
      N_HB is used as stand-in CV — no TICA training needed.
    - Iteration ≥ TICA_DEFER_UNTIL: attempt TICA training, then validate:
        (a) Block bootstrap cosine similarity > 0.90 for TIC-1.
        (b) |r(TIC-1, N_HB)| > 0.70.
      If both pass → return (model, 'tica').
      If either fails → return (None, 'nhb') and retry at next iteration.

    Parameters
    ----------
    features_pool : np.ndarray
        Shape (n_frames, 40). Full pooled feature array (all iterations so far).
    nhb_pool : np.ndarray
        Shape (n_frames,). Corresponding N_HB values.
    iteration : int
        Current 1-indexed iteration number.
    lag : int or None
        TICA lag in frames (100 ps spacing). If None, uses
        ``config.TICA_MIN_LAG_FRAMES`` (aligned with production policy).
    n_components : int
        Number of TICA components to compute.
    rng_seed : int or None
        Seed for the block bootstrap RNG.

    Returns
    -------
    model : deeptime TICA model or None
        Fitted TICA model if validation passed, None otherwise.
    cv_used : str
        ``'tica'`` if the model is valid, ``'nhb'`` if falling back.
    """
    if iteration < TICA_DEFER_UNTIL:
        log.info(
            "[native TICA] Iteration %d < %d (TICA_DEFER_UNTIL): using N_HB fallback.",
            iteration,
            TICA_DEFER_UNTIL,
        )
        return None, "nhb"

    try:
        from deeptime.decomposition import TICA  # type: ignore
    except ImportError:
        log.warning(
            "[native TICA] deeptime not available; using N_HB fallback."
        )
        return None, "nhb"

    lag_eff = TICA_MIN_LAG_FRAMES if lag is None else lag

    # --- Train TICA (epsilon/scaling match Phase 6 / tica_train defaults) ---
    tica_model = TICA(
        lagtime=lag_eff,
        dim=n_components,
        epsilon=TICA_EPSILON,
        scaling=TICA_SCALING,
    )
    try:
        tica_model = tica_model.fit(features_pool).fetch_model()
    except Exception as exc:
        log.warning("[native TICA] Training failed (%s); using N_HB fallback.", exc)
        return None, "nhb"

    # --- Validation (a): block bootstrap — worst-case min |cos θ| vs threshold ---
    cosine_sim: float = 0.0
    if _TICA_TRAIN_AVAILABLE:
        try:
            boot = block_bootstrap_tic1_stability(
                features_pool,
                lagtime=lag_eff,
                block_size=TICA_BOOTSTRAP_BLOCK_SIZE,
                n_bootstrap=TICA_N_BOOTSTRAP,
                rng_seed=rng_seed if rng_seed is not None else 42,
            )
            cosine_sim = float(boot["min_cos_sim"])
        except Exception as exc:
            log.warning(
                "[native TICA] Block bootstrap failed (%s); treating cosine_sim=0.",
                exc,
            )
    else:
        log.warning(
            "[native TICA] src.tica_train not available; skipping block bootstrap."
        )

    if cosine_sim < TICA_COSINE_SIM_THRESHOLD:
        log.info(
            "[native TICA] Iteration %d: cosine_sim=%.3f < %.2f threshold. "
            "Staying on N_HB.",
            iteration,
            cosine_sim,
            TICA_COSINE_SIM_THRESHOLD,
        )
        return None, "nhb"

    # --- Validation (b): |r(TIC-1, N_HB)| ---
    try:
        tic1_values = tica_model.transform(features_pool)
        if tic1_values.ndim == 2:
            tic1_values = tic1_values[:, 0]
        pearson_r = float(np.corrcoef(tic1_values, nhb_pool)[0, 1])
        abs_r = abs(pearson_r)
    except Exception as exc:
        log.warning(
            "[native TICA] Pearson r computation failed (%s); using N_HB fallback.",
            exc,
        )
        return None, "nhb"

    if abs_r < TICA_PEARSON_R_THRESHOLD:
        log.info(
            "[native TICA] Iteration %d: |r(TIC-1, N_HB)|=%.3f < %.2f threshold. "
            "Staying on N_HB.",
            iteration,
            abs_r,
            TICA_PEARSON_R_THRESHOLD,
        )
        return None, "nhb"

    log.info(
        "[native TICA] Iteration %d: cosine_sim=%.3f, |r|=%.3f — TICA validated, switching to TIC-1.",
        iteration,
        cosine_sim,
        abs_r,
    )
    return tica_model, "tica"


# =============================================================================
# SINGLE SEED MD RUN
# =============================================================================


def run_seed(
    seed_frame_index: int,
    traj_pool_dcd: str,
    prod_top_pdb: str,
    system_xml: str,
    output_prefix: str,
    output_dir: Path,
    seed_id: int,
    iteration: int,
    strategy: str,
    random_seed: int,
    platform,
    seed_steps: int = SEED_STEPS,
) -> dict:
    """Run one 30 ns NVT seed from a frame selected from the pooled trajectory.

    Implementation follows EXPERIMENT_PROTOCOL.md §11.4 exactly:

    1. Load the pooled trajectory DCD against ``prod_top_pdb`` (full system:
       protein + water + ions).
    2. Extract box vectors and positions from the **same** frame (NVT production
       has a fixed box, but this ensures correctness for any future NPT pools).
    3. Set state in the required order: box vectors → positions → constraints →
       velocities. Re-draw velocities from Maxwell-Boltzmann at 330 K with
       ``random_seed`` — this keeps seeds statistically independent from the
       parent trajectory.
    4. No energy minimisation — pooled frames are already valid configurations.
    5. Run SEED_STEPS = 1_500_000 steps (30 ns at 2 fs).

    Checkpoint/resume: if a checkpoint file already exists, the run is resumed
    from that checkpoint rather than restarted. If the seed is already complete
    (checkpoint step ≥ SEED_STEPS), the function returns immediately.

    The topology PDB (``prod_top_pdb``) is written into ``output_dir`` before
    the DCDReporter starts, so that the seed DCD can be loaded later.

    Parameters
    ----------
    seed_frame_index : int
        Index of the selected frame in the pooled trajectory.
    traj_pool_dcd : str
        Path to the pooled trajectory DCD for this strategy/iteration (or the
        generator PDB path for iteration 1, signalled by seed_frame_index=-1).
    prod_top_pdb : str
        Path to ``*_extended_prod_top.pdb`` (matches ``system_xml``). Used for
        iterations ≥ 2 (DCD frames). Iteration 1 ignores this for topology and
        loads the generator PDB via ``simulate._npt_final_topology_positions_box``.
    system_xml : str
        Path to the serialised OpenMM System XML (no barostat — NVT production).
    output_prefix : str
        Filename prefix for seed outputs (e.g. ``"A8Q2_random_iter01_seed03"``).
    output_dir : Path
        Directory where seed DCD, log, and checkpoint files are written.
    seed_id : int
        Zero-indexed seed identifier within this iteration (0–7).
    iteration : int
        One-indexed iteration number (1–5).
    strategy : str
        One of ``'random'``, ``'transferred'``, ``'native'``.
    random_seed : int
        RNG seed for velocity assignment — ensures reproducibility and
        statistical independence of this seed from all others.
    platform : openmm.Platform
        Pre-configured OpenMM Platform (CUDA/OpenCL/CPU).

    Returns
    -------
    dict with keys:
        ``seed_id``, ``iteration``, ``strategy``,
        ``output_dcd``, ``output_log``, ``n_steps_run``.
    """
    import mdtraj as md
    import openmm as mm
    import openmm.app as app
    from openmm import XmlSerializer, unit

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dcd = str(output_dir / f"{output_prefix}.dcd")
    seed_log = str(output_dir / f"{output_prefix}.log")
    seed_chk = str(output_dir / f"{output_prefix}.chk")

    # --- Build simulation context ---
    with open(system_xml) as fh:
        system = XmlSerializer.deserialize(fh.read())

    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PS / unit.picosecond,
        TIMESTEP_FS * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(random_seed)

    n_sys = int(system.getNumParticles())

    # Iteration 1: never use app.PDBFile() on the NPT-final PDB — OpenMM merges
    # duplicate atom names (e.g. two ALA H) so topology/positions diverge from
    # the serialized System XML (same bug Phase 4 avoids via simulate._npt_final_*).
    if seed_frame_index == -1:
        from pathlib import Path as _Path

        from src.simulate import _npt_final_topology_positions_box

        top_omm, positions0, box_vecs = _npt_final_topology_positions_box(
            _Path(traj_pool_dcd), n_sys
        )
        if box_vecs is None:
            raise ValueError(
                f"Generator PDB '{traj_pool_dcd}' has no periodic box vectors. "
                "Re-run Phase 3 to regenerate the NPT final PDB with CRYST1 / unitcell."
            )
        sim = app.Simulation(top_omm, system, integrator, platform)
    else:
        # ``*_prod_top.pdb`` can hit the same duplicate-name merge as NPT-final PDBs
        # when read through ``PDBFile`` (e.g. 2670 topology vs 2671 XML particles).
        # MDTraj preserves one record per ATOM line — match Phase 4 / iteration 1.
        pdb_top = app.PDBFile(prod_top_pdb)
        n_pdbfile = pdb_top.topology.getNumAtoms()
        if n_pdbfile == n_sys:
            omm_top = pdb_top.topology
        else:
            traj_top = md.load_pdb(prod_top_pdb)
            if traj_top.n_atoms != n_sys:
                raise ValueError(
                    f"prod_top_pdb {prod_top_pdb!r}: PDBFile topology has {n_pdbfile} "
                    f"atoms, MDTraj loaded {traj_top.n_atoms}, system XML has {n_sys}. "
                    "Regenerate topology from production or check PDB/XML pairing."
                )
            omm_top = traj_top.topology.to_openmm()
        sim = app.Simulation(omm_top, system, integrator, platform)

    # --- Checkpoint/resume ---
    steps_done = 0
    resume = os.path.exists(seed_chk) and os.path.getsize(seed_chk) > 0
    if resume:
        sim.loadCheckpoint(seed_chk)
        steps_done = sim.currentStep
        steps_remaining = seed_steps - steps_done
        if steps_remaining <= 0:
            log.info(
                "Seed iter%02d/%s/%02d already complete (%d steps). Skipping.",
                iteration, strategy, seed_id, steps_done,
            )
            return {
                "seed_id": seed_id,
                "iteration": iteration,
                "strategy": strategy,
                "output_dcd": seed_dcd,
                "output_log": seed_log,
                "n_steps_run": steps_done,
            }
        log.info(
            "Resuming seed iter%02d/%s/%02d from step %d (%d remaining).",
            iteration, strategy, seed_id, steps_done, steps_remaining,
        )
    else:
        steps_remaining = seed_steps

        # --- Set state from selected frame (or generator PDB for iteration 1) ---
        if seed_frame_index == -1:
            # positions0 / box_vecs already loaded above with MDTraj-safe topology
            sim.context.setPeriodicBoxVectors(*box_vecs)
            sim.context.setPositions(positions0)
        else:
            # Iterations 2+: load only the selected frame from the pooled DCD.
            if not os.path.exists(traj_pool_dcd):
                raise FileNotFoundError(f"Pooled DCD not found: {traj_pool_dcd}")
            try:
                frame = md.load_frame(traj_pool_dcd, int(seed_frame_index), top=prod_top_pdb)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load frame {seed_frame_index} from {traj_pool_dcd}: {exc}"
                ) from exc

            seed_positions = frame.openmm_positions(0)
            # Some MDTraj/DCD combinations have no unitcell vectors (unitcell_vectors=None).
            # In that case we use box vectors from the production topology if present,
            # else keep the current context box vectors from the serialized System.
            if getattr(frame, "unitcell_vectors", None) is not None:
                seed_box_vectors = frame.openmm_boxes(0)
                sim.context.setPeriodicBoxVectors(*seed_box_vectors)
            else:
                top_box = app.PDBFile(prod_top_pdb).topology.getPeriodicBoxVectors()
                if top_box is not None:
                    sim.context.setPeriodicBoxVectors(*top_box)
                else:
                    log.warning(
                        "Frame %d from %s has no unitcell vectors and %s has no CRYST1 box; "
                        "using existing context box vectors.",
                        seed_frame_index, traj_pool_dcd, prod_top_pdb,
                    )
            sim.context.setPositions(seed_positions)

        sim.context.applyConstraints(1e-7)
        # Fresh M-B velocities — keeps seeds statistically independent (§11.4)
        sim.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, random_seed)

        # Write topology PDB before starting the DCDReporter (§6.6 / §11.4)
        seed_top_pdb = str(output_dir / f"{output_prefix}_top.pdb")
        state_for_top = sim.context.getState(getPositions=True)
        with open(seed_top_pdb, "w") as fh:
            app.PDBFile.writeFile(
                sim.topology, state_for_top.getPositions(), fh
            )

    # --- Reporters ---
    log_mode_append = resume and os.path.exists(seed_log)
    dcd_mode_append = resume and os.path.exists(seed_dcd)

    sim.reporters.append(
        app.StateDataReporter(
            seed_log,
            DCD_STEPS,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            totalSteps=seed_steps,
            separator="\t",
            append=log_mode_append,
        )
    )
    sim.reporters.append(app.DCDReporter(seed_dcd, DCD_STEPS, append=dcd_mode_append))
    sim.reporters.append(app.CheckpointReporter(seed_chk, CHK_STEPS))

    # --- Run ---
    t0 = time.time()
    sim.step(steps_remaining)
    elapsed = time.time() - t0
    total_steps = steps_done + steps_remaining
    ns_per_day = (steps_remaining / (1_000 / TIMESTEP_FS) / 1_000) / (elapsed / 86_400)

    log.info(
        "Seed iter%02d/%s/%02d done: %d steps in %.1f min (%.0f ns/day).",
        iteration, strategy, seed_id,
        total_steps, elapsed / 60, ns_per_day,
    )

    return {
        "seed_id": seed_id,
        "iteration": iteration,
        "strategy": strategy,
        "output_dcd": seed_dcd,
        "output_log": seed_log,
        "n_steps_run": total_steps,
    }


# =============================================================================
# ITERATION 1: GENERATOR-BASED INITIALISATION
# =============================================================================


def _run_iteration1_seeds(
    generator_pdb: str,
    system_xml: str,
    prod_top_pdb: str,
    strategy: str,
    output_dir: Path,
    platform,
    n_seeds: int = N_SEEDS,
    seed_steps: int = SEED_STEPS,
) -> list[dict]:
    """Run 8 independent 30 ns seeds from the generator PDB.

    HTMD generator approach: all seeds start from the same equilibrated
    coordinates but with different Maxwell-Boltzmann velocity seeds (seed_id
    + 100). This ensures diversity purely from velocity randomness, which is
    sufficient for the initial iteration. Protocol §11.2.

    Parameters
    ----------
    generator_pdb : str
        Path to A8Q2_extended_npt_final.pdb (final NPT equilibration frame).
    system_xml : str
        Path to serialised NVT system XML.
    prod_top_pdb : str
        Topology PDB matching the system XML topology.
    strategy : str
        Strategy label used for output directory naming.
    output_dir : Path
        Root directory; seed output goes to ``output_dir/iter01/<strategy>/``.
    platform : openmm.Platform
        Pre-configured OpenMM platform.

    Returns
    -------
    list[dict]
        One result dict per seed (from run_seed).
    """
    iter_dir = output_dir / "iter01" / strategy
    results = []

    tag = _system_tag_from_prod_top(prod_top_pdb)
    for seed_id in range(n_seeds):
        prefix = f"{tag}_{strategy}_iter01_seed{seed_id:02d}"
        result_json = iter_dir / f"{prefix}_result.json"

        if result_json.exists():
            with open(result_json) as fh:
                cached = json.load(fh)
            log.info(
                "Iteration 1 seed %d/%s already done (loaded from JSON).", seed_id, strategy
            )
            results.append(cached)
            continue

        random_seed = seed_id + 100  # seeds 100–107 for iteration 1
        result = run_seed(
            seed_frame_index=-1,        # sentinel: use generator PDB
            traj_pool_dcd=generator_pdb,
            prod_top_pdb=prod_top_pdb,
            system_xml=system_xml,
            output_prefix=prefix,
            output_dir=iter_dir,
            seed_id=seed_id,
            iteration=1,
            strategy=strategy,
            random_seed=random_seed,
            platform=platform,
            seed_steps=seed_steps,
        )

        iter_dir.mkdir(parents=True, exist_ok=True)
        with open(result_json, "w") as fh:
            json.dump(result, fh, indent=2)
        results.append(result)

    return results


# =============================================================================
# FULL ITERATION (iterations 2+)
# =============================================================================


def run_iteration(
    iteration: int,
    strategy: str,
    pool_dcds: list[str],
    prod_top_pdb: str,
    system_xml: str,
    ground_truth_fes: dict,
    helix_ref_pdb: str,
    output_dir: Path,
    tica_model_a10=None,
    mean_donor: Optional[np.ndarray] = None,
    eigenvectors_donor: Optional[np.ndarray] = None,
    native_tica_model=None,
    platform=None,
    rng_seed: Optional[int] = None,
    n_seeds: int = N_SEEDS,
    seed_steps: int = SEED_STEPS,
) -> dict:
    """Run one full adaptive seeding iteration.

    Executes the complete pipeline for a single iteration:
    1. Featurize and pool all prior seed trajectories.
    2. Compute CV values and select N_SEEDS seed frames (strategy-dependent).
    3. Run N_SEEDS × 30 ns seed MD simulations.
    4. Pool new seed DCDs with prior pool, recompute features.
    5. Compute 2D FES (RMSD×N_HB) on the fixed reference grid (Phase 5).
    6. Compute JSD, coverage, entropy vs ground truth, and f_helix_error.
    7. For strategy 'native': attempt TICA retraining on expanded pool.
    8. Save iteration summary JSON.

    Parameters
    ----------
    iteration : int
        1-indexed iteration number. Must be ≥ 2 for normal seed selection;
        use :func:`_run_iteration1_seeds` for iteration 1.
    strategy : str
        One of ``'random'``, ``'transferred'``, ``'native'``.
    pool_dcds : list[str]
        All seed DCD paths from previous iterations (chronologically ordered).
        Must be non-empty (iterations 2+ require at least iteration 1 data).
    prod_top_pdb : str
        Topology PDB for the full solvated A8Q2 system.
    system_xml : str
        Serialised NVT OpenMM System XML (no barostat).
    ground_truth_fes : dict
        Output of :func:`src.analysis.pooled_ground_truth_fes_rmsd_nhb`: keys
        ``hist``, ``edges_rmsd``, ``edges_nhb``, ``f_helix`` (reference 2D counts
        and bin edges for JSD / coverage / entropy; helix fraction for error).
    helix_ref_pdb : str
        Ideal helix PDB path (RMSD definition; must match ground truth).
    output_dir : Path
        Root output directory; iteration output goes to
        ``output_dir/iter{iteration:02d}/{strategy}/``.
    tica_model_a10 : deeptime TICA model or None
        Fitted A10 TICA model (unused for 'random' and 'native').
    mean_donor : np.ndarray or None
        Shape (40,). A10 TICA feature mean (required for 'transferred').
    eigenvectors_donor : np.ndarray or None
        Shape (40, n_tics). A10 TICA eigenvectors (required for 'transferred').
    native_tica_model : deeptime TICA model or None
        Current best-known native TICA model. May be None if not yet validated.
        Updated in-place for strategy 'native' if validation passes.
    platform : openmm.Platform or None
        OpenMM platform. If None, :func:`src.utils.get_best_platform` is used.
    rng_seed : int or None
        Master RNG seed; individual seed RNGs are derived as
        ``rng_seed * 1000 + iteration * 100 + seed_id``.

    Returns
    -------
    dict with keys:
        ``iteration``, ``strategy``, ``cv_used``,
        ``seed_paths``,       – list of output DCD paths
        ``jsd``,              – Jensen–Shannon divergence vs ground truth
        ``coverage``,         – fraction of reference bins hit
        ``entropy``,          – normalised distribution entropy
        ``f_helix_error``,    – |f_helix_seeded − f_helix_groundtruth|
        ``native_tica_model`` – updated model (None if strategy != 'native')
    """
    from src.utils import get_best_platform

    if platform is None:
        platform = get_best_platform()

    iter_dir = output_dir / f"iter{iteration:02d}" / strategy
    iter_dir.mkdir(parents=True, exist_ok=True)
    summary_json = iter_dir / "iteration_summary.json"

    # --- Resume: skip if already complete ---
    if summary_json.exists():
        with open(summary_json) as fh:
            cached_summary = json.load(fh)
        log.info(
            "Iteration %d/%s already done (loaded from JSON).", iteration, strategy
        )
        return cached_summary

    # --- Step 1: Pool prior trajectories ---
    log.info("Iteration %d/%s: pooling %d trajectories.", iteration, strategy, len(pool_dcds))
    features_pool, nhb_pool, _rmsd_prior, pool_segments = _pool_trajectories(
        pool_dcds, prod_top_pdb, helix_ref_pdb
    )

    # --- Step 2: Compute CV and select seed frames ---
    cv_used: str = strategy  # default label; overridden for 'native'

    if strategy == "random":
        cv_values = None
        seed_frame_indices = select_seeds_random(
            features_pool,
            n_seeds=n_seeds,
            rng_seed=rng_seed,
        )
    elif strategy == "transferred":
        cv_values = project_cv(
            features_pool,
            strategy="transferred",
            mean_donor=mean_donor,
            eigenvectors_donor=eigenvectors_donor,
        )
        seed_frame_indices = select_seeds_least_counts(
            cv_values,
            n_seeds=n_seeds,
            n_bins=N_BINS,
            rng_seed=rng_seed,
        )
    else:  # native
        updated_model, cv_used = _try_train_native_tica(
            features_pool,
            nhb_pool,
            iteration=iteration,
            rng_seed=rng_seed,
        )
        if updated_model is not None:
            native_tica_model = updated_model

        cv_values = project_cv(
            features_pool,
            strategy="native",
            cv_model=native_tica_model,
            nhb=nhb_pool,
        )
        seed_frame_indices = select_seeds_least_counts(
            cv_values,
            n_seeds=n_seeds,
            n_bins=N_BINS,
            rng_seed=rng_seed,
        )

    # --- Step 3: Run seeds ---
    log.info(
        "Iteration %d/%s (cv=%s): running %d seeds.",
        iteration, strategy, cv_used, n_seeds,
    )
    seed_results: list[dict] = []
    new_dcds: list[str] = []

    tag = _system_tag_from_prod_top(prod_top_pdb)
    for seed_id, frame_idx in enumerate(seed_frame_indices):
        prefix = f"{tag}_{strategy}_iter{iteration:02d}_seed{seed_id:02d}"
        result_json = iter_dir / f"{prefix}_result.json"

        if result_json.exists():
            try:
                with open(result_json) as fh:
                    result = json.load(fh)
                if not isinstance(result, dict) or "output_dcd" not in result:
                    raise ValueError("missing output_dcd")
                out_dcd = str(result["output_dcd"])
                if not os.path.exists(out_dcd):
                    raise ValueError(f"cached output_dcd does not exist: {out_dcd}")
                log.info("Seed %d already done (loaded from JSON).", seed_id)
            except Exception as exc:
                log.warning(
                    "Seed %d cache invalid (%s); re-running this seed.",
                    seed_id, exc,
                )
                result = None
        else:
            result = None

        if result is None:
            # Derive a deterministic per-seed random seed
            per_seed_rng = (
                (rng_seed if rng_seed is not None else 0) * 1_000
                + iteration * 100
                + seed_id
            ) % (2**31 - 1)

            dcd_for_frame, local_frame = _pooled_frame_to_dcd_local(
                pool_segments, int(frame_idx)
            )
            result = run_seed(
                seed_frame_index=int(local_frame),
                traj_pool_dcd=dcd_for_frame,
                prod_top_pdb=prod_top_pdb,
                system_xml=system_xml,
                output_prefix=prefix,
                output_dir=iter_dir,
                seed_id=seed_id,
                iteration=iteration,
                strategy=strategy,
                random_seed=per_seed_rng,
                platform=platform,
                seed_steps=seed_steps,
            )
            with open(result_json, "w") as fh:
                json.dump(result, fh, indent=2)

        seed_results.append(result)
        new_dcds.append(result["output_dcd"])

    # --- Step 4: Pool updated data and compute FES ---
    all_dcds = pool_dcds + new_dcds
    features_updated, nhb_updated, rmsd_updated, _ = _pool_trajectories(
        all_dcds, prod_top_pdb, helix_ref_pdb
    )

    if strategy == "random":
        cv_updated = None
    elif strategy == "transferred":
        cv_updated = project_cv(
            features_updated,
            strategy="transferred",
            mean_donor=mean_donor,
            eigenvectors_donor=eigenvectors_donor,
        )
    else:
        cv_updated = project_cv(
            features_updated,
            strategy="native",
            cv_model=native_tica_model,
            nhb=nhb_updated,
        )

    # --- Step 5: Compute 2D FES and metrics (RMSD × N_HB, Phase 5) ---
    sampled_hist, jsd_val, cov_val, ent_val, f_helix_err = _compute_fes_metrics(
        nhb_updated,
        rmsd_updated,
        ground_truth_fes,
    )

    # --- Step 6: Save iteration summary ---
    summary = {
        "iteration": iteration,
        "strategy": strategy,
        "cv_used": cv_used,
        "seed_paths": new_dcds,
        "jsd": float(jsd_val),
        "coverage": float(cov_val),
        "entropy": float(ent_val),
        "f_helix_error": float(f_helix_err),
        "f_helix_groundtruth": float(ground_truth_fes["f_helix"]),
        "fes_definition": "rmsd_x_nhb",
        "n_frames_pool": int(len(features_updated)),
        "native_tica_model": None,  # not JSON-serialisable; tracked in memory
    }

    with open(summary_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    log.info(
        "Iteration %d/%s complete: JSD=%.4f, coverage=%.3f, entropy=%.3f, "
        "f_helix_err=%.4f, cv_used=%s",
        iteration, strategy,
        jsd_val, cov_val, ent_val, f_helix_err, cv_used,
    )

    summary["native_tica_model"] = native_tica_model
    return summary


# =============================================================================
# FES AND METRICS HELPERS
# =============================================================================


def _compute_fes_metrics(
    nhb_pool: np.ndarray,
    rmsd_pool: np.ndarray,
    ground_truth_fes: dict,
    min_count: int = FES_MIN_COUNT,
) -> tuple[np.ndarray, float, float, float, float]:
    """Compute RMSD×N_HB histogram and metrics vs Phase-5 ground truth.

    Parameters
    ----------
    nhb_pool, rmsd_pool : np.ndarray
        Pooled per-frame observables (same definition as ``compute_2d_fes``).
    ground_truth_fes : dict
        Reference bundle from :func:`src.analysis.pooled_ground_truth_fes_rmsd_nhb`
        with keys ``hist``, ``edges_rmsd``, ``edges_nhb``, ``f_helix``.
    min_count : int
        Masked-domain threshold (``FES_MIN_COUNT``).

    Returns
    -------
    sampled_hist : np.ndarray
        Raw counts on the **same** RMSD/N_HB bin edges as the reference.
    jsd, cov, ent, f_helix_error
        As in protocol §12 / ``src.metrics``.
    """
    ground_truth_hist = np.asarray(ground_truth_fes["hist"], dtype=float)
    edges_rmsd = np.asarray(ground_truth_fes["edges_rmsd"])
    edges_nhb = np.asarray(ground_truth_fes["edges_nhb"])
    f_helix_gt = float(ground_truth_fes["f_helix"])

    sampled_hist = histogram_rmsd_nhb_fixed_edges(
        rmsd_pool,
        nhb_pool,
        edges_rmsd,
        edges_nhb,
    )

    if _METRICS_AVAILABLE:
        jsd_val = float(compute_jsd(ground_truth_hist, sampled_hist, min_count=min_count))
        cov_val = float(
            compute_coverage(ground_truth_hist, sampled_hist, min_count=min_count)
        )
        ent_val = float(
            compute_entropy(sampled_hist, ground_truth_hist, min_count=min_count)
        )
    else:
        jsd_val = _fallback_jsd(ground_truth_hist, sampled_hist, min_count=min_count)
        cov_val = _fallback_coverage(ground_truth_hist, sampled_hist, min_count=min_count)
        ent_val = _normalised_entropy(sampled_hist)

    if np.isfinite(f_helix_gt):
        if _METRICS_AVAILABLE and compute_f_helix_error is not None:
            f_helix_err = float(compute_f_helix_error(nhb_pool, f_helix_gt))
        else:
            thr = float(N_HB_MAX * NHB_THRESHOLD_FRAC)
            f_helix_err = abs(float((nhb_pool >= thr).mean()) - f_helix_gt)
    else:
        f_helix_err = float("nan")

    return sampled_hist, jsd_val, cov_val, ent_val, f_helix_err


def _fallback_jsd(
    hist_ref: np.ndarray,
    hist_sample: np.ndarray,
    min_count: int = 3,
) -> float:
    """JSD computation when src.metrics is not available."""
    from scipy.spatial.distance import jensenshannon

    p = hist_ref.flatten().astype(float)
    q = hist_sample.flatten().astype(float)
    mask = p >= min_count
    p_m = p[mask]
    q_m = q[mask]
    p_m /= p_m.sum()
    q_m_sum = q_m.sum()
    q_m_norm = q_m / q_m_sum if q_m_sum > 0 else q_m
    return float(jensenshannon(p_m, q_m_norm, base=2) ** 2)


def _fallback_coverage(
    hist_ref: np.ndarray,
    hist_sample: np.ndarray,
    min_count: int = 3,
) -> float:
    """Coverage computation when src.metrics is not available."""
    valid = hist_ref >= min_count
    if not valid.any():
        return 0.0
    return float((hist_sample[valid] > 0).mean())


def _normalised_entropy(hist: np.ndarray) -> float:
    """H(q) normalised by log₂(|Ω|) where Ω is the set of populated bins.

    H = −Σ q_i log₂(q_i)  for q_i > 0, normalised to [0, 1].
    """
    counts = hist.flatten().astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    q = counts[counts > 0] / total
    h = -float(np.sum(q * np.log2(q)))
    n_populated = np.count_nonzero(counts)
    if n_populated <= 1:
        return 0.0
    return h / np.log2(n_populated)


# =============================================================================
# FULL ADAPTIVE SEEDING EXPERIMENT
# =============================================================================


def _prod_top_pdb_for_dcd(dcd_path: str) -> str:
    """``A8Q2_helix_prod.dcd`` → ``A8Q2_helix_prod_top.pdb`` (same directory)."""
    p = Path(dcd_path)
    top = p.parent / f"{p.stem}_top.pdb"
    if not top.is_file():
        raise FileNotFoundError(
            f"Expected MDTraj topology next to production DCD: {top} (for {dcd_path})"
        )
    return str(top)


def run_adaptive_seeding(
    generator_pdb: str,
    system_xml: str,
    prod_top_pdb: str,
    ground_truth_dcd_helix: str,
    ground_truth_dcd_extended: str,
    output_dir: Path,
    tica_artifacts_dir: Optional[Path] = None,
    n_iterations: int = 5,
    n_seeds: int = 8,
    seed_ns: float = 30.0,
    ground_truth_top_helix: Optional[str] = None,
    ground_truth_top_extended: Optional[str] = None,
    helix_ref_pdb: Optional[str] = None,
    skip_iter1_consistency_check: bool = False,
) -> dict:
    """Run the complete adaptive seeding experiment for all three strategies.

    Orchestrates all 5 iterations × 3 strategies × 8 seeds × 30 ns = 3600 ns.
    Strategies 1 and 3 start immediately; Strategy 2 requires TICA artifacts
    from A10 (loaded from ``tica_artifacts_dir`` if provided).

    The experiment is resumable: each iteration's result is saved as JSON, and
    already-complete iterations are skipped.

    Parameters
    ----------
    generator_pdb : str
        Path to ``A8Q2_extended_npt_final.pdb`` (Phase 3 output).
    system_xml : str
        Path to serialised NVT OpenMM System XML for A8Q2 (no barostat).
    prod_top_pdb : str
        Path to ``A8Q2_extended_prod_top.pdb`` — topology for **seed** trajectories
        (extended box). Written immediately before extended production DCDReporter.
    ground_truth_top_helix, ground_truth_top_extended : str or None
        MDTraj topologies for each ground-truth DCD. If ``None``, inferred as
        ``{dcd_stem}_top.pdb`` beside each DCD (required when helix and extended
        replicas have different atom counts).
    helix_ref_pdb : str or None
        Ideal helix PDB for RMSD (e.g. ``A8Q2_helix.pdb``). If ``None``, uses
        ``{PREFIX}_helix.pdb`` in the same directory as ``prod_top_pdb``.
    ground_truth_dcd_helix : str
        Path to ``A8Q2_helix_prod.dcd`` (500 ns ground-truth helix replica).
    ground_truth_dcd_extended : str
        Path to ``A8Q2_extended_prod.dcd`` (500 ns ground-truth extended replica).
    output_dir : Path
        Root directory for all seeding outputs. Created if absent.
    tica_artifacts_dir : Path or None
        Directory containing ``a10_tica_singular_vectors_left.npy`` and
        ``a10_tica_mean0.npy`` (saved by ``tica_train``). Legacy names
        ``a10_tica_eigenvectors.npy`` / ``a10_tica_mean.npy`` are also accepted.
        Required for Strategy 2 (transferred TICA). If None or files absent,
        Strategy 2 is skipped with a warning.
    n_iterations : int
        Number of adaptive seeding iterations (default 5).
    n_seeds : int
        Seeds per iteration (default 8; must match N_SEEDS).
    seed_ns : float
        Duration of each seed run in ns (default 30.0; must match SEED_NS).
    skip_iter1_consistency_check : bool
        If True, skip the mandatory cross-strategy iteration-1 initial-topology
        SHA256 gate (not recommended).

    Returns
    -------
    dict
        Nested results keyed by strategy, then by iteration:
        ``results[strategy][iteration] = iteration_summary_dict``
    """
    from src.utils import apply_windows_checkpoint_patch, get_best_platform

    apply_windows_checkpoint_patch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging as early as possible so validation/runtime errors and
    # CLI-level exception hooks write to adaptive_seeding.log as well as stdout.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir / "adaptive_seeding.log"), mode="a"),
        ],
    )

    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    if seed_ns <= 0:
        raise ValueError(f"seed_ns must be > 0, got {seed_ns}")
    seed_steps = int(seed_ns * 1_000_000 / TIMESTEP_FS)

    gt_top_helix_pdb = ground_truth_top_helix or _prod_top_pdb_for_dcd(
        ground_truth_dcd_helix
    )
    gt_top_extended_pdb = ground_truth_top_extended or _prod_top_pdb_for_dcd(
        ground_truth_dcd_extended
    )

    platform = get_best_platform()

    # --- Ground-truth RMSD×N_HB reference (Phase 5) ---
    tag = _system_tag_from_prod_top(prod_top_pdb)
    helix_ref_resolved = (
        helix_ref_pdb
        if helix_ref_pdb
        else str(Path(prod_top_pdb).resolve().parent / f"{tag}_helix.pdb")
    )
    if not os.path.isfile(helix_ref_resolved):
        log.warning(
            "Ideal helix PDB not found at %s — FES metrics need it (place %s_helix.pdb "
            "next to prod top or pass helix_ref_pdb).",
            helix_ref_resolved,
            tag,
        )
    log.info("Loading ground-truth trajectories for FES reference (RMSD×N_HB).")
    gt_fes = pooled_ground_truth_fes_rmsd_nhb(
        [
            (ground_truth_dcd_helix, gt_top_helix_pdb),
            (ground_truth_dcd_extended, gt_top_extended_pdb),
        ],
        helix_ref_resolved,
    )

    # --- Load A10 TICA artifacts for Strategy 2 ---
    mean_donor: Optional[np.ndarray] = None
    eigenvectors_donor: Optional[np.ndarray] = None
    if tica_artifacts_dir is not None:
        tica_dir = Path(tica_artifacts_dir)
        # Canonical names from tica_train.save_tica_artifacts (protocol §9.6)
        ev_path = tica_dir / "a10_tica_singular_vectors_left.npy"
        mean_path = tica_dir / "a10_tica_mean0.npy"
        # Legacy fallbacks (older drafts used PyEMMA-style names)
        if not ev_path.exists():
            legacy_ev = tica_dir / "a10_tica_eigenvectors.npy"
            if legacy_ev.exists():
                ev_path = legacy_ev
        if not mean_path.exists():
            legacy_mean = tica_dir / "a10_tica_mean.npy"
            if legacy_mean.exists():
                mean_path = legacy_mean
        if ev_path.exists() and mean_path.exists():
            eigenvectors_donor = np.load(str(ev_path))
            mean_donor = np.load(str(mean_path))
            log.info(
                "Loaded A10 TICA artifacts: singular_vectors_left %s, mean_0 %s.",
                eigenvectors_donor.shape,
                mean_donor.shape,
            )
        else:
            log.warning(
                "TICA artifacts not found in %s (expected "
                "a10_tica_singular_vectors_left.npy + a10_tica_mean0.npy). "
                "Strategy 'transferred' will be skipped.",
                tica_dir,
            )

    strategies = ["random", "native"]
    if mean_donor is not None and eigenvectors_donor is not None:
        strategies.append("transferred")
    else:
        log.warning("Strategy 'transferred' disabled — A10 TICA artifacts not available.")

    results: dict[str, dict] = {s: {} for s in strategies}
    pool_dcds: dict[str, list[str]] = {s: [] for s in strategies}
    native_tica_models: dict[str, object] = {"native": None}

    # --- Iteration 1: generator-based initialisation ---
    for strategy in strategies:
        iter1_cache = output_dir / "iter01_results.json"
        if iter1_cache.exists():
            with open(iter1_cache) as fh:
                cached = json.load(fh)
            if strategy in cached:
                log.info("Iteration 1/%s loaded from cache.", strategy)
                iter1_results = cached[strategy]
                # iter01_results.json stores a list of per-seed dicts; results[.][1]
                # must match the fresh-run shape (summary dict) for _serialise_results.
                results[strategy][1] = {
                    "iteration": 1,
                    "strategy": strategy,
                    "cv_used": "generator",
                    "seed_paths": [r["output_dcd"] for r in iter1_results],
                    "jsd": None,
                    "coverage": None,
                    "entropy": None,
                    "f_helix_error": None,
                }
                pool_dcds[strategy] = results[strategy][1]["seed_paths"]
                continue

        log.info("=== Running Iteration 1 / Strategy '%s' ===", strategy)
        iter1_seed_results = _run_iteration1_seeds(
            generator_pdb=generator_pdb,
            system_xml=system_xml,
            prod_top_pdb=prod_top_pdb,
            strategy=strategy,
            output_dir=output_dir,
            platform=platform,
            n_seeds=n_seeds,
            seed_steps=seed_steps,
        )
        iter1_summary = {
            "iteration": 1,
            "strategy": strategy,
            "cv_used": "generator",
            "seed_paths": [r["output_dcd"] for r in iter1_seed_results],
            "jsd": None,
            "coverage": None,
            "entropy": None,
            "f_helix_error": None,
        }
        results[strategy][1] = iter1_summary
        pool_dcds[strategy] = iter1_summary["seed_paths"]

        # Cache to disk
        _append_iter1_cache(output_dir / "iter01_results.json", strategy, iter1_seed_results)

    # --- Mandatory iteration-1 cross-strategy gate (initial *_top.pdb) ---
    if skip_iter1_consistency_check:
        log.warning(
            "Iteration-1 cross-strategy consistency check SKIPPED (--no-iter1-consistency-check). "
            "Downstream metrics may not be comparable across strategies."
        )
        v_payload = {
            "passed": None,
            "skipped": True,
            "reason": "user_disabled_skip_iter1_consistency_check",
            "strategies": list(strategies),
        }
        _write_iter1_verification_json(output_dir, v_payload)
    else:
        v_payload = _verify_iter1_initial_tops_consistent(
            output_dir=output_dir,
            strategies=strategies,
            n_seeds=n_seeds,
            prod_top_pdb=prod_top_pdb,
            generator_pdb=generator_pdb,
        )
        _write_iter1_verification_json(output_dir, v_payload)
        if not v_payload.get("passed"):
            bad = [
                r["seed_id"]
                for r in v_payload.get("per_seed", [])
                if not r.get("ok", True)
            ]
            raise RuntimeError(
                "Iteration-1 cross-strategy initial-topology verification FAILED "
                f"(seed_ids with issues: {bad}). "
                "All strategies must share identical Cartesian coordinates in "
                "iter01/*/*_top.pdb for each seed before iteration 2. "
                "Remove outputs/seeding_*/iter01*, iter01_results.json, and "
                "adaptive_seeding_results.json, then re-run seeding in one session. "
                f"Audit file: {output_dir / 'iter01_verification.json'}"
            )

    # --- Iterations 2+ ---
    for iteration in range(2, n_iterations + 1):
        for strategy in strategies:
            log.info("=== Running Iteration %d / Strategy '%s' ===", iteration, strategy)

            rng_seed = 42 * iteration + {"random": 0, "transferred": 1, "native": 2}[strategy]

            iter_summary = run_iteration(
                iteration=iteration,
                strategy=strategy,
                pool_dcds=list(pool_dcds[strategy]),
                prod_top_pdb=prod_top_pdb,
                system_xml=system_xml,
                ground_truth_fes=gt_fes,
                helix_ref_pdb=helix_ref_resolved,
                output_dir=output_dir,
                tica_model_a10=None,
                mean_donor=mean_donor,
                eigenvectors_donor=eigenvectors_donor,
                native_tica_model=native_tica_models.get("native"),
                platform=platform,
                rng_seed=rng_seed,
                n_seeds=n_seeds,
                seed_steps=seed_steps,
            )

            # Update the native TICA model in memory if it was retrained
            if strategy == "native" and iter_summary.get("native_tica_model") is not None:
                native_tica_models["native"] = iter_summary["native_tica_model"]

            results[strategy][iteration] = iter_summary
            pool_dcds[strategy].extend(iter_summary.get("seed_paths", []))

    # --- Save final results ---
    final_results_path = output_dir / "adaptive_seeding_results.json"
    _serialise_results(results, final_results_path)
    log.info("Adaptive seeding complete. Results saved to %s", final_results_path)

    return results


# =============================================================================
# GROUND TRUTH HISTOGRAM
# =============================================================================


def _build_ground_truth_histogram(
    dcd_helix: str,
    dcd_extended: str,
    top_helix_pdb: str,
    top_extended_pdb: str,
    helix_ref_pdb: str,
    nbins: int = 25,
) -> np.ndarray:
    """Return raw RMSD×N_HB ground-truth counts (Phase 5; shape ``nbins × (N_HB_MAX+1)``).

    Thin wrapper around :func:`src.analysis.pooled_ground_truth_fes_rmsd_nhb` for
    tests and smoke scripts that only need the histogram array.
    """
    bundle = pooled_ground_truth_fes_rmsd_nhb(
        [(dcd_helix, top_helix_pdb), (dcd_extended, top_extended_pdb)],
        helix_ref_pdb,
        nbins=nbins,
    )
    return np.asarray(bundle["hist"], dtype=float)


# =============================================================================
# SERIALISATION HELPERS
# =============================================================================


def _serialise_results(results: dict, path: Path) -> None:
    """Serialise the complete results dict to JSON, dropping non-serialisable objects."""
    clean: dict = {}
    for strategy, iters in results.items():
        clean[strategy] = {}
        for it_key, summary in iters.items():
            if not isinstance(summary, dict):
                raise TypeError(
                    f"results[{strategy!r}][{it_key!r}] must be a dict for JSON export, "
                    f"got {type(summary).__name__}. "
                    "If this is iteration 1 after loading iter01_results.json from cache, "
                    "ensure the cache branch wraps the cached seed list in the same summary "
                    "dict shape as a fresh iteration-1 run."
                )
            clean_summary = {
                k: v for k, v in summary.items()
                if k != "native_tica_model"
            }
            clean[strategy][str(it_key)] = clean_summary
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2)


def _append_iter1_cache(
    cache_path: Path,
    strategy: str,
    seed_results: list[dict],
) -> None:
    """Append iteration-1 seed results for a strategy to the shared cache JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if cache_path.exists():
        with open(cache_path) as fh:
            existing = json.load(fh)
    existing[strategy] = seed_results
    with open(cache_path, "w") as fh:
        json.dump(existing, fh, indent=2)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _system_tag_from_prod_top(prod_top_pdb: str) -> str:
    """``A8Q2_extended_prod_top.pdb`` → ``A8Q2`` (must match seed filename prefix)."""
    return Path(prod_top_pdb).name.split("_")[0]


def _verify_iter1_initial_tops_consistent(
    *,
    output_dir: Path,
    strategies: list[str],
    n_seeds: int,
    prod_top_pdb: str,
    generator_pdb: str,
) -> dict:
    """Protocol gate: iteration-1 pre-DCD ``*_top.pdb`` must match across strategies.

    Same generator PDB + same per-seed velocity protocol ⇒ identical Cartesian
    coordinates before the DCD reporter starts. If hashes diverge, later
    iterations pool inconsistent ensembles and downstream analysis is invalid.
    """
    if len(strategies) < 2:
        return {
            "passed": True,
            "skipped": True,
            "reason": "fewer_than_two_strategies",
            "strategies": list(strategies),
        }

    tag = _system_tag_from_prod_top(prod_top_pdb)
    gen_path = Path(generator_pdb)
    gen_sha = _sha256_file(gen_path) if gen_path.is_file() else None

    per_seed: list[dict] = []
    all_ok = True

    for seed_id in range(n_seeds):
        row: dict = {"seed_id": seed_id, "sha256_top": {}}
        hashes: dict[str, str | None] = {}
        missing: list[str] = []

        for s in strategies:
            top = output_dir / "iter01" / s / f"{tag}_{s}_iter01_seed{seed_id:02d}_top.pdb"
            if not top.is_file():
                missing.append(str(top))
                hashes[s] = None
            else:
                h = _sha256_file(top)
                hashes[s] = h
                row["sha256_top"][s] = h

        if missing:
            all_ok = False
            row["ok"] = False
            row["error"] = "missing_initial_top_pdb"
            row["missing_paths"] = missing
            per_seed.append(row)
            continue

        ref = hashes[strategies[0]]
        same = all(hashes[s] == ref for s in strategies)
        row["ok"] = bool(same)
        if not same:
            all_ok = False
        per_seed.append(row)

    out = {
        "passed": all_ok,
        "skipped": False,
        "system_tag": tag,
        "strategies": list(strategies),
        "n_seeds": n_seeds,
        "generator_pdb": str(generator_pdb),
        "generator_pdb_sha256": gen_sha,
        "per_seed": per_seed,
    }
    return out


def _write_iter1_verification_json(output_dir: Path, payload: dict) -> None:
    path = output_dir / "iter01_verification.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Wrote iteration-1 verification audit: %s", path)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main() -> None:
    """Command-line interface for the adaptive seeding experiment.

    Usage example
    -------------
    python -m src.adaptive_seeding \\
        --generator_pdb A8Q2_extended_npt_final.pdb \\
        --system_xml A8Q2_extended_npt_final.xml \\
        --prod_top_pdb A8Q2_extended_prod_top.pdb \\
        --gt_dcd_helix A8Q2_helix_prod.dcd \\
        --gt_dcd_extended A8Q2_extended_prod.dcd \\
        --tica_artifacts_dir tica_artifacts/ \\
        --output_dir seeding_results/ \\
        --n_iterations 5 \\
        --n_seeds 8 \\
        --strategy all
    """
    parser = argparse.ArgumentParser(
        description="Adaptive seeding experiment: 3 strategies × 5 iterations × 8 × 30 ns seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generator_pdb",
        required=True,
        help="Path to A8Q2_extended_npt_final.pdb (Phase 3 output).",
    )
    parser.add_argument(
        "--system_xml",
        required=True,
        help="Serialised NVT OpenMM System XML for A8Q2 (no barostat).",
    )
    parser.add_argument(
        "--prod_top_pdb",
        required=True,
        help="A8Q2_extended_prod_top.pdb — topology for seed trajectories (extended box).",
    )
    parser.add_argument(
        "--gt_dcd_helix",
        required=True,
        help="Path to A8Q2_helix_prod.dcd (500 ns ground-truth).",
    )
    parser.add_argument(
        "--gt_dcd_extended",
        required=True,
        help="Path to A8Q2_extended_prod.dcd (500 ns ground-truth).",
    )
    parser.add_argument(
        "--gt_top_helix",
        default=None,
        help="Topology for helix GT DCD (default: {stem}_top.pdb beside helix DCD).",
    )
    parser.add_argument(
        "--gt_top_extended",
        default=None,
        help="Topology for extended GT DCD (default: {stem}_top.pdb beside extended DCD).",
    )
    parser.add_argument(
        "--helix_ref_pdb",
        default=None,
        help=(
            "Ideal helix PDB for RMSD FES axis (default: {PREFIX}_helix.pdb beside prod_top_pdb)."
        ),
    )
    parser.add_argument(
        "--tica_artifacts_dir",
        default=None,
        help=(
            "Directory with a10_tica_singular_vectors_left.npy and a10_tica_mean0.npy "
            "(from tica_train). Required for strategy 'transferred'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory for all seed trajectories and results.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=N_ITERATIONS,
        help="Number of adaptive seeding iterations per strategy.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=N_SEEDS,
        help="Number of seeds per iteration.",
    )
    parser.add_argument(
        "--seed_ns",
        type=float,
        default=SEED_NS,
        help="Length of each seed trajectory in nanoseconds.",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", "random", "transferred", "native"],
        default="all",
        help=(
            "Which strategy (or strategies) to run. 'all' runs all three. "
            "Individual strategy names run only that strategy."
        ),
    )
    parser.add_argument(
        "--no-iter1-consistency-check",
        action="store_true",
        help=(
            "Disable mandatory cross-strategy iteration-1 initial-topology verification. "
            "Not recommended: invalid iter01 breaks cross-strategy comparisons."
        ),
    )

    args = parser.parse_args()

    # Validate required file paths
    for attr, label in [
        ("generator_pdb", "generator PDB"),
        ("system_xml", "system XML"),
        ("prod_top_pdb", "production topology PDB"),
    ]:
        path = getattr(args, attr)
        if not os.path.exists(path):
            parser.error(f"{label} not found: {path}")

    # For single-strategy runs, we still call run_adaptive_seeding but restrict
    # the TICA artifacts to force a skip of unwanted strategies.
    tica_dir: Optional[Path] = None
    if args.tica_artifacts_dir is not None:
        tica_dir = Path(args.tica_artifacts_dir)

    if args.strategy != "all":
        # Restrict to requested strategy by manipulating artifacts availability
        if args.strategy == "random":
            tica_dir = None  # disables transferred; native runs but won't be used
        elif args.strategy == "native":
            tica_dir = None  # disables transferred

    results = run_adaptive_seeding(
        generator_pdb=args.generator_pdb,
        system_xml=args.system_xml,
        prod_top_pdb=args.prod_top_pdb,
        ground_truth_dcd_helix=args.gt_dcd_helix,
        ground_truth_dcd_extended=args.gt_dcd_extended,
        output_dir=Path(args.output_dir),
        tica_artifacts_dir=tica_dir,
        n_iterations=args.n_iterations,
        n_seeds=args.n_seeds,
        seed_ns=args.seed_ns,
        ground_truth_top_helix=args.gt_top_helix,
        ground_truth_top_extended=args.gt_top_extended,
        helix_ref_pdb=args.helix_ref_pdb,
        skip_iter1_consistency_check=args.no_iter1_consistency_check,
    )

    # Print a brief convergence summary
    print("\n" + "=" * 70)
    print("  ADAPTIVE SEEDING — CONVERGENCE SUMMARY")
    print("=" * 70)
    for strategy in sorted(results):
        print(f"\n  Strategy: {strategy}")
        for it in sorted(results[strategy]):
            s = results[strategy][it]
            if not isinstance(s, dict):
                print(f"    Iter {it!r}: <expected dict summary, got {type(s).__name__}>")
                continue
            jsd = s.get("jsd")
            cov = s.get("coverage")
            cv = s.get("cv_used", "—")
            jsd_str = f"{jsd:.4f}" if jsd is not None else "—"
            cov_str = f"{cov:.3f}" if cov is not None else "—"
            print(
                f"    Iter {it:2d}: JSD={jsd_str}  coverage={cov_str}  cv_used={cv}"
            )
    print()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )
        root.exception(
            "adaptive_seeding CLI: unhandled exception (see stderr if no file handler yet)"
        )
        raise
