"""
utils.py — Shared utility functions for the BTP_2_Calm simulation pipeline.

Functions here are used by two or more notebooks/scripts and must not carry
hard-coded simulation parameters — import those from config.py instead.

Heavy optional dependencies (openmm, mdtraj, pymongo) are guarded by
TYPE_CHECKING or try/except so the module can be imported for unit testing
without a full GPU environment installed.
"""

from __future__ import annotations

import os
import platform as _platform
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import openmm
    import openmm.app
    import mdtraj

# ──────────────────────────────────────────────────────────────────────────────
# PLATFORM SELECTION
# ──────────────────────────────────────────────────────────────────────────────


def get_best_platform() -> "openmm.Platform":
    """Return the fastest available OpenMM platform: CUDA > OpenCL > CPU > Reference.

    For CUDA, explicitly sets mixed precision and device index 0 so the choice
    is reproducible across different OpenMM builds (some default to 'single').
    Mixed precision computes forces in float32 and integrates in float64, which
    is the validated setting for CHARMM36m + TIP3P (OpenMM team recommendation).

    Returns
    -------
    openmm.Platform
        The fastest available platform, configured and ready for use in a
        Simulation constructor.
    """
    import openmm as mm
    from openmm import Platform

    preference = ["CUDA", "OpenCL", "CPU", "Reference"]
    available = {
        Platform.getPlatform(i).getName(): Platform.getPlatform(i)
        for i in range(Platform.getNumPlatforms())
    }
    for name in preference:
        if name in available:
            plat = available[name]
            if name == "CUDA":
                plat.setPropertyDefaultValue("CudaPrecision", "mixed")
                plat.setPropertyDefaultValue("CudaDeviceIndex", "0")
            elif name == "CPU":
                try:
                    threads = plat.getPropertyDefaultValue("CpuThreads")
                    _ = threads  # surfaced via print in calling code if needed
                except Exception:
                    pass
            return plat
    return Platform.getPlatform(0)


# ──────────────────────────────────────────────────────────────────────────────
# DEGREES OF FREEDOM
# ──────────────────────────────────────────────────────────────────────────────


def compute_dof(system: "openmm.System") -> int:
    """Compute the correct number of degrees of freedom for a solvated system.

    The naive 3N overestimates DOF by ~33% for a water-solvated peptide with
    HBonds+rigidWater constraints, causing the instantaneous temperature
    computed from KE = (DOF/2)·k_B·T to be wrong by the same factor.

    Correct formula
    ---------------
    DOF = 3N − N_constraints − 3

    The −3 accounts for the CMMotionRemover force that removes 3 translational
    degrees of freedom from the centre-of-mass motion.

    Parameters
    ----------
    system : openmm.System
        Fully parameterised system (after createSystem; barostat presence does
        not affect this calculation).

    Returns
    -------
    int
        Corrected degrees of freedom.
    """
    n_atoms = system.getNumParticles()
    n_constraints = system.getNumConstraints()
    return 3 * n_atoms - n_constraints - 3


# ──────────────────────────────────────────────────────────────────────────────
# DENSITY CALCULATION
# ──────────────────────────────────────────────────────────────────────────────


def compute_density_kgm3(system: "openmm.System", state) -> float:
    """Compute system density in kg/m³ from total mass and periodic box volume.

    Uses the general triclinic formula V = a · (b × c), which is correct for
    both orthogonal cubic boxes and non-orthogonal (triclinic) boxes that arise
    after NPT pressure coupling.

    Parameters
    ----------
    system : openmm.System
        The simulated system (used to sum particle masses).
    state : openmm.State
        State obtained with getPositions=False, getPeriodicBoxVectors=True.
        Typically: ``sim.context.getState()``.

    Returns
    -------
    float
        Density in kg/m³. For TIP3P water at 330 K with CHARMM36m, expect
        985–1010 kg/m³ at convergence.
    """
    import openmm.unit as unit  # local import — avoids failing without openmm at module level

    total_mass_kg = (
        sum(
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        )
        * 1.66054e-27  # 1 Da in kg
    )
    box = state.getPeriodicBoxVectors()
    a = np.array([v.value_in_unit(unit.meter) for v in box[0]])
    b = np.array([v.value_in_unit(unit.meter) for v in box[1]])
    c = np.array([v.value_in_unit(unit.meter) for v in box[2]])
    vol = abs(np.dot(a, np.cross(b, c)))
    return total_mass_kg / vol


def section(title: str) -> None:
    """Print a formatted section banner for long-running simulation logs.

    Matches the visual style used in the legacy MD notebooks (now ``src.simulate``).
    so console output stays readable during multi-hour GPU runs.
    """
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURIZATION
# ──────────────────────────────────────────────────────────────────────────────


def featurize(traj_protein: "mdtraj.Trajectory") -> np.ndarray:
    """Compute 40-dimensional sin/cos backbone torsion features from an MDTraj trajectory.

    Feature layout (40 columns total)
    ----------------------------------
    Columns  0–9  : sin(φ) for residues 1–10
    Columns 10–19 : cos(φ) for residues 1–10
    Columns 20–29 : sin(ψ) for residues 1–10
    Columns 30–39 : cos(ψ) for residues 1–10

    Why sin/cos encoding
    --------------------
    Raw torsion angles are periodic on [−π, +π) with an artificial discontinuity
    at ±180°. The sin/cos pair embeds each angle as a point on the unit circle in
    ℝ², giving a continuous, bounded representation without discontinuity. This is
    the standard encoding for TICA of backbone torsions (Pérez-Hernández 2013;
    Schwantes & Pande 2013; Sittel & Stock 2016). Protocol §9.1.

    No standardisation
    ------------------
    Sin/cos features are already bounded on [−1, +1] with mean ≈ 0. All 40
    features are dimensionally homogeneous (no unit mixing). Applying z-score
    standardisation would inflate the variance of torsions that barely fluctuate
    (deeply trapped in a basin) and suppress informative features. deeptime's
    TICA mean-centres internally; no additional scaling is needed. Protocol §9.1.

    Parameters
    ----------
    traj_protein : mdtraj.Trajectory
        Protein-only trajectory (water and ions stripped). Must contain exactly
        N_RESIDUES backbone torsion pairs. Load as::

            traj_prot = traj.atom_slice(traj.topology.select("protein"))

    Returns
    -------
    np.ndarray
        Shape (n_frames, 40), dtype float64.

    Raises
    ------
    AssertionError
        If the trajectory does not yield exactly 10 φ or 10 ψ angles, indicating
        a topology mismatch.
    """
    import mdtraj as md

    _, phi = md.compute_phi(traj_protein)   # (n_frames, 9) or (n_frames, 10)
    _, psi = md.compute_psi(traj_protein)

    # 10-residue chain: MDTraj gives 9 φ (res 2–10) and 9 ψ (res 1–9).
    if phi.shape[1] == 9 and psi.shape[1] == 9:
        n = phi.shape[0]
        phi10 = np.zeros((n, 10), dtype=np.float64)
        psi10 = np.zeros((n, 10), dtype=np.float64)
        phi10[:, 1:] = phi
        psi10[:, :-1] = psi
        phi, psi = phi10, psi10
    elif phi.shape[1] != 10 or psi.shape[1] != 10:
        raise ValueError(
            f"Expected 9 (padded to 10) or 10 φ/ψ columns; got phi {phi.shape}, psi {psi.shape}"
        )

    return np.concatenate(
        [np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)],
        axis=1,
    ).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# BURN-IN REMOVAL
# ──────────────────────────────────────────────────────────────────────────────


def remove_burn_in(
    features: np.ndarray,
    frame_spacing_ps: float,
    burn_in_ns: float,
) -> np.ndarray:
    """Remove the initialisation burn-in period from a feature array.

    Discards the first ``burn_in_ns`` of frames to exclude the non-equilibrium
    transient where the system escapes its starting conformation. Keeping the
    burn-in would bias the TICA covariance matrix C(0) toward the starting state
    and produce TIC-1 eigenvectors that represent 'escape from initialisation'
    rather than the equilibrium helix-coil reaction coordinate. Protocol §8.1.

    Parameters
    ----------
    features : np.ndarray
        Shape (n_frames, n_features). Rows ordered chronologically.
    frame_spacing_ps : float
        Time between consecutive frames in picoseconds (e.g. 100.0 for a DCD
        saved every 100 ps).
    burn_in_ns : float
        Length of burn-in period to discard in nanoseconds. From config:
        ``BURN_IN_NS = 25.0`` (5 × τ_eq ≈ 5–8 ns). Protocol §8.1.

    Returns
    -------
    np.ndarray
        Feature array with the first ``n_burn`` rows removed.

    Raises
    ------
    ValueError
        If the burn-in period is longer than the trajectory.
    """
    n_burn = int(burn_in_ns * 1_000.0 / frame_spacing_ps)
    if n_burn >= len(features):
        raise ValueError(
            f"Burn-in ({burn_in_ns} ns = {n_burn} frames) is ≥ trajectory "
            f"length ({len(features)} frames). Reduce burn_in_ns or use a "
            "longer trajectory."
        )
    return features[n_burn:]


# ──────────────────────────────────────────────────────────────────────────────
# WINDOWS CHECKPOINT PATCH
# ──────────────────────────────────────────────────────────────────────────────


def apply_windows_checkpoint_patch() -> None:
    """Patch OpenMM's safesave module for atomic rename on Windows.

    On Windows, ``os.replace()`` over a file that another process (or the same
    process) still has open raises ``PermissionError`` / ``OSError``. OpenMM's
    ``CheckpointReporter`` calls safesave, which in turn calls ``os.replace()``
    to atomically overwrite the previous checkpoint. This fails intermittently
    on Windows with fast checkpointing intervals.

    The patch replaces ``openmm.app.internal.safesave.save`` with a version that:
    1. Writes the data to a temporary file in the same directory as the target
       (ensures the rename is on the same filesystem/volume — required for
       atomic rename semantics).
    2. Retries ``os.replace()`` up to 6 times with exponential back-off
       (0.2 s, 0.4 s, 0.8 s, 1.6 s, 3.2 s) before re-raising the error.

    On Linux/macOS, ``os.replace()`` is already atomic at the kernel level and
    this function is a no-op.

    This patch is idempotent: calling it multiple times has no additional effect.
    """
    if _platform.system() != "Windows":
        return

    import tempfile as _tempfile
    import time as _time
    import openmm.app.internal.safesave as _safesave

    def _patched_save(data: "bytes | str", filename: str) -> None:
        dir_ = os.path.dirname(os.path.abspath(filename))
        fd, tmp = _tempfile.mkstemp(
            dir=dir_, prefix=os.path.basename(filename) + ".tmp"
        )
        try:
            payload = data if isinstance(data, (bytes, bytearray)) else data.encode()
            os.write(fd, payload)
        finally:
            os.close(fd)

        for attempt in range(6):
            try:
                os.replace(tmp, filename)
                return
            except (PermissionError, OSError):
                if attempt == 5:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise
                _time.sleep(0.2 * (2 ** attempt))  # 0.2, 0.4, 0.8, 1.6, 3.2 s

    _safesave.save = _patched_save


# ──────────────────────────────────────────────────────────────────────────────
# STATE SERIALISATION
# ──────────────────────────────────────────────────────────────────────────────


def save_state_xml(simulation: "openmm.app.Simulation", path: str) -> None:
    """Serialise the full simulation state to an XML file.

    The serialised state includes positions, velocities, periodic box vectors,
    and all force parameters, making it a complete and reproducible snapshot of
    the simulation context. It can be reloaded with::

        sim.loadState(path)

    Unlike a checkpoint (``.chk``), the XML state is human-readable and
    platform-independent (checkpoints are platform-specific binary files).

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The simulation whose context state is to be saved.
    path : str
        Output file path (e.g. ``"A10_extended_npt_final.xml"``).
    """
    simulation.saveState(path)


# ──────────────────────────────────────────────────────────────────────────────
# MONGODB HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _get_mongo_client():
    """Return a pymongo MongoClient using the URI from the environment.

    Returns None if pymongo is not installed or the environment variable is not
    set, making all mongo helpers graceful no-ops in non-RunPod environments.
    """
    try:
        import pymongo
    except ImportError:
        return None

    from src.config import MONGO_URI_ENV_VAR

    uri = os.environ.get(MONGO_URI_ENV_VAR)
    if not uri:
        return None
    return pymongo.MongoClient(uri)


def mongo_upload(db_name: str, collection: str, document: dict) -> None:
    """Upload a document to a MongoDB Atlas collection.

    Used to share artifacts (feature arrays serialised as lists, eigenvectors,
    scalar metrics, figure bytes) between RunPod pods via MongoDB Atlas (free M0
    tier). Large trajectory files (.dcd, ~5 GB) are never uploaded — only small
    analysis outputs. Protocol §13.

    No-op if the ``MONGO_URI`` environment variable is not set or if pymongo is
    not installed; a warning is printed in that case.

    Parameters
    ----------
    db_name : str
        MongoDB database name (from config: ``MONGO_DB = "btp_calm"``).
    collection : str
        Collection name within the database (e.g. ``"a10_tica"``).
    document : dict
        JSON-serialisable document to insert. Must not contain NumPy arrays
        directly — convert with ``.tolist()`` before passing.
    """
    client = _get_mongo_client()
    if client is None:
        print(
            f"[mongo_upload] WARNING: MONGO_URI not set or pymongo not installed. "
            f"Skipping upload to {db_name}.{collection}."
        )
        return
    try:
        client[db_name][collection].insert_one(document)
    finally:
        client.close()


def mongo_download(
    db_name: str, collection: str, query: dict
) -> "dict | None":
    """Fetch the first matching document from a MongoDB Atlas collection.

    Returns None if the document is not found, if ``MONGO_URI`` is not set, or
    if pymongo is not installed. Callers must check the return value before use.

    Parameters
    ----------
    db_name : str
        MongoDB database name (from config: ``MONGO_DB = "btp_calm"``).
    collection : str
        Collection name (e.g. ``"a10_tica"``).
    query : dict
        MongoDB filter document (e.g. ``{"system": "A10", "phase": "tica"}``).

    Returns
    -------
    dict or None
        The first matching document (including the ``_id`` field), or None if
        not found.
    """
    client = _get_mongo_client()
    if client is None:
        return None
    try:
        return client[db_name][collection].find_one(query)
    finally:
        client.close()
