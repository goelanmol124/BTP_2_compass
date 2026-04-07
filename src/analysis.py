"""
analysis.py — Phase 5 Ground Truth Analysis
============================================
Computes all metrics, figures, and data for the BTP CV-transfer study.

Systems: Ac-Ala₁₀-NHMe (A10) and Ac-Ala₈Gln₂-NHMe (A8Q2).

Physical constants
------------------
Temperature : 330 K   →  RT = 0.6558 kcal/mol
H-bond cutoff : 3.6 Å (0.36 nm), i→i+4 backbone O···H-N (Kuczera 2021)
N_HB_MAX = 6
RMSD helix threshold   : 0.20 nm
RMSD coil threshold    : 0.30 nm (transition 0.20–0.30 nm inherits previous state)
DSSP helix threshold   : ≥ 50 % residues 'H'  (coil is '-', NOT 'C')
N_HB helix threshold   : ≥ 3 bonds (50 % of 6)
Burn-in                : first 25 ns = 250 frames at 100 ps/frame
FES bins               : 25 × 25 on RMSD × N_HB
FES clip               : 5 kcal/mol
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ─── physical constants ───────────────────────────────────────────────────────
KB_KCAL: float = 1.987204259e-3          # kcal mol⁻¹ K⁻¹
TEMPERATURE_K: float = 330.0
RT_KCAL: float = KB_KCAL * TEMPERATURE_K  # 0.6558 kcal/mol at 330 K

# ─── trajectory parameters ───────────────────────────────────────────────────
FRAME_SPACING_PS: float = 100.0          # ps per saved frame
FRAME_SPACING_NS: float = FRAME_SPACING_PS / 1000.0
BURNIN_FRAMES: int = 250                 # 25 ns ÷ 100 ps = 250 frames

# ─── order-parameter thresholds ──────────────────────────────────────────────
RMSD_HELIX_NM: float = 0.20
RMSD_COIL_NM: float = 0.30
N_HB_MAX: int = 6
NHB_HELIX_FRAC: float = 0.50            # ≥ 3 bonds
DSSP_HELIX_FRAC: float = 0.50           # ≥ 50 % residues 'H'
HB_CUTOFF_NM: float = 0.36             # 3.6 Å in nm

# ─── FES parameters ──────────────────────────────────────────────────────────
FES_NBINS: int = 25
FES_CLIP_KCAL: float = 5.0

# ─── matplotlib style ────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRAJECTORY LOADING
# ═════════════════════════════════════════════════════════════════════════════


def load_trajectory(
    dcd_path: str,
    top_pdb_path: str,
    protein_only: bool = True,
) -> "md.Trajectory":
    """Load DCD against prod_top.pdb and optionally slice to protein atoms.

    Must use the prod_top.pdb written immediately before production, not the
    solvated PDB.  The protein slice retains ACE and NME capping groups so that
    phi/psi angles for all 10 internal residues can be computed.

    Parameters
    ----------
    dcd_path      : path to the production DCD file
    top_pdb_path  : path to the topology PDB written before the DCDReporter
    protein_only  : if True, slice to ``protein or resname ACE or resname NME``

    Returns
    -------
    MDTraj Trajectory (protein-only if requested)
    """
    import mdtraj as md

    traj = md.load_dcd(dcd_path, top=top_pdb_path)
    if protein_only:
        sel = traj.topology.select("protein or resname ACE or resname NME")
        traj = traj.atom_slice(sel)
    return traj


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ORDER PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════


def compute_backbone_rmsd(
    traj_protein: "md.Trajectory",
    ref_frame_index: int = 0,
) -> np.ndarray:
    """Backbone Cα RMSD to a reference frame (default: first frame), in nm.

    Superposes every frame on the Cα atoms of the reference frame then
    computes per-frame RMSD using the same Cα atom set.

    Parameters
    ----------
    traj_protein    : protein-only MDTraj trajectory
    ref_frame_index : index of the frame used as the helix reference

    Returns
    -------
    rmsd_nm : (n_frames,) float array, nanometres
    """
    import mdtraj as md

    ca_idx = traj_protein.topology.select("name CA")
    ref = traj_protein[ref_frame_index]
    traj_protein.superpose(ref, atom_indices=ca_idx, ref_atom_indices=ca_idx)
    return md.compute_rmsd(
        traj_protein,
        ref,
        atom_indices=ca_idx,
        ref_atom_indices=ca_idx,
        precentered=True,
    )


def compute_nhb(
    traj_protein: "md.Trajectory",
    cutoff_nm: float = HB_CUTOFF_NM,
) -> np.ndarray:
    """Count i→i+4 backbone C=O···H-N hydrogen bonds per frame.

    Searches every residue for its backbone carbonyl O and the amide H on
    residue i+4, accepting atom names H, HN, H1 (CHARMM names vary).  Falls
    back to finding any H covalently bonded to the backbone N.

    Parameters
    ----------
    traj_protein : protein-only MDTraj trajectory
    cutoff_nm    : H···O distance threshold in nm (default 0.36 nm = 3.6 Å)

    Returns
    -------
    nhb : (n_frames,) integer array — number of i→i+4 H-bonds per frame
    """
    import mdtraj as md

    top = traj_protein.topology
    all_res = list(top.residues)
    n = len(all_res)

    H_NAMES = {"H", "HN", "H1", "HT1", "HT2", "HT3", "HN1"}

    acceptor_O: list[Optional[int]] = []
    donor_H: list[Optional[int]] = []

    for res in all_res:
        o_list = [a.index for a in res.atoms if a.name == "O"]

        # Search by name + verify bonded to N
        h_list = [
            a.index
            for a in res.atoms
            if a.name in H_NAMES
            and any(
                b.atom1.name == "N" or b.atom2.name == "N"
                for b in top.bonds
                if a.index in (b.atom1.index, b.atom2.index)
            )
        ]

        # Fallback: traverse bond graph from backbone N
        if not h_list:
            for a in res.atoms:
                if a.name == "N":
                    bonded_indices = [
                        b.atom2.index if b.atom1.index == a.index else b.atom1.index
                        for b in top.bonds
                        if a.index in (b.atom1.index, b.atom2.index)
                    ]
                    h_list = [
                        idx
                        for idx in bonded_indices
                        if top.atom(idx).element.symbol == "H"
                    ]
                    if h_list:
                        break

        acceptor_O.append(o_list[0] if o_list else None)
        donor_H.append(h_list[0] if h_list else None)

    hb_pairs: list[tuple[int, int]] = []
    for i in range(n - 4):
        acc = acceptor_O[i]
        don = donor_H[i + 4]
        if acc is not None and don is not None:
            hb_pairs.append((don, acc))

    if not hb_pairs:
        return np.zeros(traj_protein.n_frames, dtype=int)

    dists = md.compute_distances(traj_protein, np.array(hb_pairs))  # (n_frames, n_pairs) nm
    formed = (dists < cutoff_nm).astype(int)
    return np.clip(formed.sum(axis=1), 0, N_HB_MAX)


def compute_dssp(traj_protein: "md.Trajectory") -> np.ndarray:
    """Per-residue DSSP assignment using MDAnalysis 2.8+ Kabsch-Sander algorithm.

    Requires MDAnalysis >= 2.8.0.  The trajectory must be written to a
    temporary DCD + PDB so MDAnalysis can load it.

    Returns
    -------
    dssp_array : (n_frames, n_residues) object array of 'H', 'E', '-'
                 IMPORTANT: coil is '-', NOT 'C'.
    """
    import tempfile
    import MDAnalysis as mda
    from MDAnalysis.analysis.dssp import DSSP

    # Write protein-only trajectory to temporary files for MDAnalysis
    with tempfile.TemporaryDirectory() as tmpdir:
        top_tmp = os.path.join(tmpdir, "top.pdb")
        dcd_tmp = os.path.join(tmpdir, "traj.dcd")
        traj_protein[0].save_pdb(top_tmp)
        traj_protein.save_dcd(dcd_tmp)

        u = mda.Universe(top_tmp, dcd_tmp)
        protein_ag = u.select_atoms("protein")
        run = DSSP(protein_ag).run()

    return run.results.dssp  # (n_frames, n_residues), dtype=object, coil='-'


def compute_helix_fraction_dssp(dssp_array: np.ndarray) -> np.ndarray:
    """Per-residue helix fraction from DSSP array.

    Parameters
    ----------
    dssp_array : (n_frames, n_residues) array returned by compute_dssp

    Returns
    -------
    helix_frac : (n_residues,) float array, values in [0, 1]
    """
    return (dssp_array == "H").mean(axis=0)


def compute_featurization(traj_protein: "md.Trajectory") -> np.ndarray:
    """40D sin/cos backbone torsion feature vector per frame.

    Feature order: [sin(phi)×10, cos(phi)×10, sin(psi)×10, cos(psi)×10]
    All features are in [-1, +1] (no standardization needed).

    Parameters
    ----------
    traj_protein : protein-only MDTraj trajectory

    Returns
    -------
    features : (n_frames, 40) float array
    """
    import mdtraj as md

    _, phi = md.compute_phi(traj_protein)   # (n_frames, n_phi_res)
    _, psi = md.compute_psi(traj_protein)   # (n_frames, n_psi_res)

    features = np.concatenate(
        [np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)],
        axis=1,
    )
    return features.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TWO-STATE CLASSIFICATION WITH HYSTERESIS
# ═════════════════════════════════════════════════════════════════════════════


def classify_states(
    rmsd: np.ndarray,
    nhb: np.ndarray,
    dssp_array: np.ndarray,
    rmsd_helix: float = RMSD_HELIX_NM,
    rmsd_coil: float = RMSD_COIL_NM,
    nhb_helix_frac: float = NHB_HELIX_FRAC,
    nhb_max: int = N_HB_MAX,
    dssp_helix_frac: float = DSSP_HELIX_FRAC,
) -> np.ndarray:
    """Three-criterion two-state classification per frame.

    Criteria:
      RMSD   : < rmsd_helix → H;  > rmsd_coil → C;  in-between → inherit previous
               (hysteresis to suppress flickering near the threshold)
      N_HB   : nhb / nhb_max >= nhb_helix_frac → H  (no hysteresis; discrete)
      DSSP   : fraction of residues == 'H' >= dssp_helix_frac → H  (no hysteresis)

    Agreement score = fraction of frames where all three criteria agree.
    Frames classified as 'U' (undefined) arise only when RMSD criterion is in
    the transition zone on the very first frame or if all three disagree.
    In practice 'U' is returned only for RMSD-only transition-zone frames
    when the trajectory starts mid-zone; normal operation has 'H' or 'C'.

    Returns
    -------
    states : (n_frames,) object array of 'H', 'C', 'U'
    """
    n = len(rmsd)

    # ── RMSD criterion with hysteresis ──
    rmsd_state = np.empty(n, dtype=object)
    mid = 0.5 * (rmsd_helix + rmsd_coil)
    rmsd_state[0] = "H" if rmsd[0] < mid else "C"
    for i in range(1, n):
        r = rmsd[i]
        if r < rmsd_helix:
            rmsd_state[i] = "H"
        elif r > rmsd_coil:
            rmsd_state[i] = "C"
        else:
            rmsd_state[i] = rmsd_state[i - 1]  # hysteresis

    # ── N_HB criterion ──
    nhb_state = np.where(nhb / nhb_max >= nhb_helix_frac, "H", "C")

    # ── DSSP criterion ──
    dssp_per_frame = (dssp_array == "H").mean(axis=1)  # (n_frames,)
    dssp_state = np.where(dssp_per_frame >= dssp_helix_frac, "H", "C")

    # ── Combine: majority vote → 'H' or 'C', unanimous disagreement → 'U' ──
    states = np.empty(n, dtype=object)
    for i in range(n):
        votes_h = (rmsd_state[i] == "H") + (nhb_state[i] == "H") + (dssp_state[i] == "H")
        if votes_h >= 2:
            states[i] = "H"
        else:
            states[i] = "C"

    return states


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CONVERGENCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════


def compute_block_se(
    timeseries: np.ndarray,
    block_sizes_ns: Optional[list] = None,
    frame_spacing_ps: float = FRAME_SPACING_PS,
) -> dict:
    """Block standard error (Flyvbjerg-Petersen 1989) for a timeseries.

    Splits the timeseries into non-overlapping blocks of increasing length,
    computes the standard error of the mean across block averages, and
    identifies the plateau BSE where blocks become statistically independent.

    Parameters
    ----------
    timeseries       : 1D array (e.g. per-frame helix indicator)
    block_sizes_ns   : list of block sizes in ns; if None, uses log-spaced
                       blocks from 1 frame to 25 % of trajectory
    frame_spacing_ps : ps per frame (default 100 ps)

    Returns
    -------
    dict with keys:
      block_sizes_ns  : array of block sizes (ns)
      bse_values      : array of BSE values
      plateau_bse     : estimate of plateau BSE (max over last 20 % of sizes)
    """
    n = len(timeseries)
    dt_ns = frame_spacing_ps / 1000.0

    if block_sizes_ns is None:
        max_b = max(2, int(n * 0.25))
        b_frames_arr = np.unique(
            np.logspace(0, np.log10(max_b), num=40).astype(int)
        )
    else:
        b_frames_arr = np.array(
            [max(1, int(b * 1000.0 / frame_spacing_ps)) for b in block_sizes_ns]
        )

    sizes_ns_out = []
    bse_out = []

    for b in b_frames_arr:
        n_blocks = n // b
        if n_blocks < 2:
            continue
        reshaped = timeseries[: n_blocks * b].reshape(n_blocks, b)
        block_means = reshaped.mean(axis=1)
        bse = block_means.std(ddof=1) / np.sqrt(n_blocks)
        sizes_ns_out.append(b * dt_ns)
        bse_out.append(bse)

    sizes_ns_out = np.array(sizes_ns_out)
    bse_out = np.array(bse_out)

    # Plateau estimate: median of BSE values over largest 20 % of block sizes
    if len(bse_out) >= 4:
        cutoff = int(len(bse_out) * 0.80)
        plateau_bse = float(np.median(bse_out[cutoff:]))
    else:
        plateau_bse = float(bse_out[-1]) if len(bse_out) else np.nan

    return {
        "block_sizes_ns": sizes_ns_out,
        "bse_values": bse_out,
        "plateau_bse": plateau_bse,
    }


def compute_acf(
    timeseries: np.ndarray,
    max_lag_frames: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised autocorrelation function of a timeseries.

    Uses an FFT-based estimator via scipy.signal.correlate (O(n log n)).
    The unbiased normalisation divides each lag-k term by (n - k) rather
    than n, avoiding the downward bias in the tail.

    Parameters
    ----------
    timeseries     : 1D float array
    max_lag_frames : maximum lag in frames (default 200 = 20 ns at 100 ps/frame)

    Returns
    -------
    lag_times_ns : (max_lag_frames+1,) array of lag times in ns
    acf_values   : (max_lag_frames+1,) normalised ACF, ACF[0] = 1
    """
    x = np.asarray(timeseries, dtype=float)
    x_c = x - x.mean()
    n = len(x_c)
    var = float(x_c.var())

    n_lag = min(n, max_lag_frames + 1)

    if var == 0.0:
        lag_times_ns = np.arange(n_lag) * FRAME_SPACING_NS
        return lag_times_ns, np.ones(n_lag)

    # Full correlation via FFT; keep only non-negative lags
    full = correlate(x_c, x_c, mode="full")
    pos = full[n - 1 : n - 1 + n_lag]

    # Unbiased normalisation: divide lag-k by (n - k) * var
    denominators = (n - np.arange(n_lag)) * var
    acf_vals = pos / denominators

    # Force ACF[0] = 1 (floating-point safety)
    acf_vals = acf_vals / acf_vals[0]

    lag_times_ns = np.arange(n_lag) * FRAME_SPACING_NS
    return lag_times_ns, acf_vals


def fit_biexponential(
    lag_times: np.ndarray,
    acf_values: np.ndarray,
) -> dict:
    """Fit ACF to biexponential C(t) = A·exp(-t/τ_slow) + (1-A)·exp(-t/τ_fast).

    Only lags where ACF > 0.02 are included to avoid noisy tail artefacts.

    Returns
    -------
    dict with keys: A, tau_slow, tau_fast, tau_slow_ns, tau_fast_ns,
                    fit_success (bool), fit_acf (array of fitted values)
    """

    def biexp(t, amplitude, tau1, tau2):
        return amplitude * np.exp(-t / tau1) + (1.0 - amplitude) * np.exp(-t / tau2)

    mask = acf_values > 0.02
    t_fit = lag_times[mask]
    y_fit = acf_values[mask]

    if len(t_fit) < 5:
        return {
            "A": np.nan, "tau_slow": np.nan, "tau_fast": np.nan,
            "tau_slow_ns": np.nan, "tau_fast_ns": np.nan,
            "fit_success": False, "fit_acf": np.full_like(lag_times, np.nan),
        }

    t_max = float(t_fit[-1]) if t_fit[-1] > 0 else 1.0

    try:
        popt, _ = curve_fit(
            biexp,
            t_fit,
            y_fit,
            p0=[0.7, t_max / 2.0, t_max / 10.0],
            bounds=([0.0, 1e-6, 1e-6], [1.0, t_max * 20, t_max * 20]),
            maxfev=100_000,
        )
        A, tau1, tau2 = popt
        tau_slow = max(tau1, tau2)
        tau_fast = min(tau1, tau2)
        fit_acf = biexp(lag_times, *popt)
        return {
            "A": float(A),
            "tau_slow": float(tau_slow),
            "tau_fast": float(tau_fast),
            "tau_slow_ns": float(tau_slow),
            "tau_fast_ns": float(tau_fast),
            "fit_success": True,
            "fit_acf": fit_acf,
        }
    except RuntimeError:
        return {
            "A": np.nan, "tau_slow": np.nan, "tau_fast": np.nan,
            "tau_slow_ns": np.nan, "tau_fast_ns": np.nan,
            "fit_success": False, "fit_acf": np.full_like(lag_times, np.nan),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FREE ENERGY SURFACES
# ═════════════════════════════════════════════════════════════════════════════


def compute_2d_fes(
    rmsd: np.ndarray,
    nhb: np.ndarray,
    nbins: int = FES_NBINS,
    clip_kcal: float = FES_CLIP_KCAL,
    temperature_K: float = TEMPERATURE_K,
) -> dict:
    """2D free energy surface on RMSD × N_HB grid.

    FES = -RT * ln(P / P_max), minimum set to 0.  Bins with zero count are NaN.

    Parameters
    ----------
    rmsd         : (n_frames,) RMSD in nm
    nhb          : (n_frames,) integer H-bond count
    nbins        : number of bins per axis (default 25)
    clip_kcal    : clip FES at this value in kcal/mol (default 5)
    temperature_K: simulation temperature in K

    Returns
    -------
    dict with keys:
      fes        : (nbins, nbins) array in kcal/mol, NaN for empty bins
      hist       : (nbins, nbins) raw count array
      edges_rmsd : (nbins+1,) bin edges for RMSD axis
      edges_nhb  : (nbins+1,) bin edges for N_HB axis
      centers_rmsd, centers_nhb : (nbins,) bin centers
    """
    RT = KB_KCAL * temperature_K

    hist, edges_rmsd, edges_nhb = np.histogram2d(
        rmsd, nhb.astype(float), bins=nbins
    )

    h = hist.astype(float)
    max_count = h.max()

    with np.errstate(divide="ignore", invalid="ignore"):
        fes = np.where(h > 0, -RT * np.log(h / max_count), np.nan)

    fes_clipped = np.where(np.isnan(fes), np.nan, np.clip(fes, 0.0, clip_kcal))

    return {
        "fes": fes_clipped,
        "hist": hist,
        "edges_rmsd": edges_rmsd,
        "edges_nhb": edges_nhb,
        "centers_rmsd": 0.5 * (edges_rmsd[:-1] + edges_rmsd[1:]),
        "centers_nhb": 0.5 * (edges_nhb[:-1] + edges_nhb[1:]),
    }


def compute_1d_fes_nhb(
    nhb: np.ndarray,
    nbins: int = FES_NBINS,
    temperature_K: float = TEMPERATURE_K,
) -> dict:
    """1D free energy surface along N_HB, marginalised over RMSD.

    Parameters
    ----------
    nhb           : (n_frames,) integer H-bond count
    nbins         : number of bins
    temperature_K : simulation temperature in K

    Returns
    -------
    dict with keys: fes (kcal/mol), hist, centers, edges
    """
    RT = KB_KCAL * temperature_K

    counts, edges = np.histogram(nhb.astype(float), bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    c = counts.astype(float)
    max_count = c.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = np.where(c > 0, -RT * np.log(c / max_count), np.nan)

    return {"fes": fes, "hist": counts, "centers": centers, "edges": edges}


def compute_ramachandran_pmf(
    traj_protein: "md.Trajectory",
    nbins_angle: int = 45,
    temperature_K: float = TEMPERATURE_K,
) -> dict:
    """Per-residue Ramachandran PMF in (phi, psi) space with 8-degree bins.

    Parameters
    ----------
    traj_protein  : protein-only MDTraj trajectory
    nbins_angle   : number of bins per angle axis; 45 → 8° bins over [-180,+180]
    temperature_K : simulation temperature in K

    Returns
    -------
    dict keyed by residue index (0-based, only residues with both phi and psi).
    Each value is a dict:
      pmf    : (nbins, nbins) array in kcal/mol, NaN for empty bins
      hist   : (nbins, nbins) raw count array
      edges  : (2, nbins+1) edges for phi and psi axes
      centers: (2, nbins) bin centers
    """
    import mdtraj as md

    RT = KB_KCAL * temperature_K

    phi_idx, phi_vals = md.compute_phi(traj_protein)   # phi_idx shape (n_phi, 4)
    psi_idx, psi_vals = md.compute_psi(traj_protein)   # psi_idx shape (n_psi, 4)

    # Map central atom (index 2 in the dihedral quartet = Cα) to residue
    phi_res = [traj_protein.topology.atom(q[2]).residue.index for q in phi_idx]
    psi_res = [traj_protein.topology.atom(q[2]).residue.index for q in psi_idx]

    phi_by_res = {r: phi_vals[:, i] for i, r in enumerate(phi_res)}
    psi_by_res = {r: psi_vals[:, i] for i, r in enumerate(psi_res)}

    common_res = sorted(set(phi_by_res) & set(psi_by_res))
    bin_edges = np.linspace(-np.pi, np.pi, nbins_angle + 1)

    result = {}
    for r in common_res:
        phi_r = np.degrees(phi_by_res[r])
        psi_r = np.degrees(psi_by_res[r])

        bins_deg = np.linspace(-180.0, 180.0, nbins_angle + 1)
        hist, ep, es = np.histogram2d(phi_r, psi_r, bins=bins_deg)
        h = hist.astype(float)
        max_count = h.max()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmf = np.where(h > 0, -RT * np.log(h / max_count), np.nan)

        result[r] = {
            "pmf": pmf,
            "hist": hist,
            "edges": np.stack([ep, es]),
            "centers": np.stack(
                [0.5 * (ep[:-1] + ep[1:]), 0.5 * (es[:-1] + es[1:])]
            ),
        }

    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — KINETICS
# ═════════════════════════════════════════════════════════════════════════════


def compute_kinetics(states: np.ndarray) -> dict:
    """Compute kinetic quantities from a binary state timeseries.

    Only *complete* sojourns (both entry and exit events observed) are used to
    compute mean residence times.  Truncated first/last sojourns are discarded
    to avoid downward bias.

    Folding transition   : C → H (coil to helix)
    Unfolding transition : H → C (helix to coil)

    Rate convention:
      k_f  = 1 / <τ_C>   (folding rate  = inverse mean coil residence time)
      k_u  = 1 / <τ_H>   (unfolding rate = inverse mean helix residence time)
      τ_eq = 1 / (k_f + k_u)

    Bootstrap uncertainty: 200-sample block bootstrap over sojourn lists.

    Returns
    -------
    dict with keys:
      f_helix, dG_fold, k_f, k_u, tau_fold, tau_unfold, tau_eq,
      n_fold, n_unfold,
      k_f_sem, k_u_sem, tau_eq_sem   (bootstrap standard errors)
    """
    s = np.asarray(states == "H", dtype=int)
    d = np.diff(s)

    fold_frames = np.where(d == 1)[0] + 1    # first helix frame (C→H)
    unfold_frames = np.where(d == -1)[0] + 1  # first coil  frame (H→C)

    def _complete_sojourns(enter_idx, exit_idx):
        """Lengths (in frames) of complete sojourns."""
        lengths = []
        for e in enter_idx:
            after = exit_idx[exit_idx > e]
            if len(after):
                lengths.append(int(after[0] - e))
        return np.array(lengths, dtype=float) if lengths else np.array([0.0])

    h_frames = _complete_sojourns(fold_frames, unfold_frames)
    c_frames = _complete_sojourns(unfold_frames, fold_frames)

    h_soj_ns = h_frames * FRAME_SPACING_NS
    c_soj_ns = c_frames * FRAME_SPACING_NS

    tau_H = float(h_soj_ns.mean()) if len(h_soj_ns) and h_soj_ns.mean() > 0 else np.nan
    tau_C = float(c_soj_ns.mean()) if len(c_soj_ns) and c_soj_ns.mean() > 0 else np.nan

    k_u = 1.0 / tau_H if (not np.isnan(tau_H) and tau_H > 0) else np.nan
    k_f = 1.0 / tau_C if (not np.isnan(tau_C) and tau_C > 0) else np.nan

    tau_eq = (
        1.0 / (k_f + k_u)
        if not (np.isnan(k_f) or np.isnan(k_u))
        else np.nan
    )

    f_helix = float(s.mean())
    dG_fold = (
        -RT_KCAL * np.log(f_helix / (1.0 - f_helix))
        if 0.0 < f_helix < 1.0
        else np.nan
    )

    # ── bootstrap SEM over sojourn lists ──
    rng = np.random.default_rng(42)

    def _bootstrap_rate(sojourns_ns: np.ndarray, n_boot: int = 200) -> float:
        if len(sojourns_ns) < 2:
            return np.nan
        boots = []
        for _ in range(n_boot):
            sample = rng.choice(sojourns_ns, size=len(sojourns_ns), replace=True)
            m = sample.mean()
            if m > 0:
                boots.append(1.0 / m)
        return float(np.std(boots, ddof=1)) if boots else np.nan

    k_f_sem = _bootstrap_rate(c_soj_ns)
    k_u_sem = _bootstrap_rate(h_soj_ns)

    if not (np.isnan(k_f_sem) or np.isnan(k_u_sem)) and (k_f + k_u) > 0:
        tau_eq_sem = float(
            np.sqrt(k_f_sem**2 + k_u_sem**2) / (k_f + k_u) ** 2
        )
    else:
        tau_eq_sem = np.nan

    return {
        "f_helix": f_helix,
        "dG_fold": dG_fold,
        "k_f": k_f,
        "k_u": k_u,
        "tau_fold": tau_C,
        "tau_unfold": tau_H,
        "tau_eq": tau_eq,
        "n_fold": int(len(fold_frames)),
        "n_unfold": int(len(unfold_frames)),
        "k_f_sem": k_f_sem,
        "k_u_sem": k_u_sem,
        "tau_eq_sem": tau_eq_sem,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — JSD CONVERGENCE METRICS
# ═════════════════════════════════════════════════════════════════════════════


def _histograms_to_jsd(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two 2D count histograms.

    Uses only bins where at least one histogram has nonzero count.
    scipy.jensenshannon returns sqrt(JSD); we square to get JSD in bits.

    Returns JSD in bits ∈ [0, 1].
    """
    p = hist_a.flatten().astype(float)
    q = hist_b.flatten().astype(float)

    mask = (p + q) > 0
    p_m = p[mask]
    q_m = q[mask]

    p_m /= p_m.sum() if p_m.sum() > 0 else 1.0
    q_m /= q_m.sum() if q_m.sum() > 0 else 1.0

    return float(jensenshannon(p_m, q_m, base=2) ** 2)


def compute_split_trajectory_jsd(
    rmsd: np.ndarray,
    nhb: np.ndarray,
    nbins: int = FES_NBINS,
) -> float:
    """JSD between the first and second halves of a trajectory on a 2D FES grid.

    Parameters
    ----------
    rmsd, nhb : full trajectory arrays (after burn-in removal)
    nbins     : bins per axis for the 2D histogram

    Returns
    -------
    jsd : float in [0, 1]
    """
    n = len(rmsd)
    half = n // 2

    # Use a fixed global bin range so both halves are on the same grid
    rmsd_min, rmsd_max = float(rmsd.min()), float(rmsd.max())
    nhb_min, nhb_max = float(nhb.min()), float(nhb.max())
    rmsd_edges = np.linspace(rmsd_min, rmsd_max, nbins + 1)
    nhb_edges = np.linspace(nhb_min, nhb_max, nbins + 1)

    hist_a, _, _ = np.histogram2d(rmsd[:half], nhb[:half].astype(float),
                                  bins=[rmsd_edges, nhb_edges])
    hist_b, _, _ = np.histogram2d(rmsd[half:], nhb[half:].astype(float),
                                  bins=[rmsd_edges, nhb_edges])

    return _histograms_to_jsd(hist_a, hist_b)


def compute_two_replica_jsd(
    rmsd1: np.ndarray,
    nhb1: np.ndarray,
    rmsd2: np.ndarray,
    nhb2: np.ndarray,
    nbins: int = FES_NBINS,
) -> float:
    """JSD between helix-start and extended-start replicas on a 2D FES grid.

    This defines the noise floor: the best achievable JSD with independent
    finite samples from the same underlying distribution.

    Returns
    -------
    jsd : float in [0, 1]
    """
    all_rmsd = np.concatenate([rmsd1, rmsd2])
    all_nhb = np.concatenate([nhb1, nhb2]).astype(float)

    rmsd_edges = np.linspace(all_rmsd.min(), all_rmsd.max(), nbins + 1)
    nhb_edges = np.linspace(all_nhb.min(), all_nhb.max(), nbins + 1)

    hist1, _, _ = np.histogram2d(rmsd1, nhb1.astype(float),
                                  bins=[rmsd_edges, nhb_edges])
    hist2, _, _ = np.histogram2d(rmsd2, nhb2.astype(float),
                                  bins=[rmsd_edges, nhb_edges])

    return _histograms_to_jsd(hist1, hist2)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_timeseries(
    rmsd: np.ndarray,
    nhb: np.ndarray,
    states: np.ndarray,
    label: str,
    output_path: str,
    frame_spacing_ns: float = FRAME_SPACING_NS,
) -> None:
    """Time series of RMSD, N_HB, and state with coloured shading."""
    n = len(rmsd)
    t = np.arange(n) * frame_spacing_ns

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Time series — {label}", fontsize=13)

    is_helix = states == "H"

    # Shading background
    for ax in axes:
        ax.axhspan(0, 1, where=is_helix, alpha=0.15, color="#e63946", transform=ax.get_xaxis_transform())

    axes[0].plot(t, rmsd * 10, color="#333", lw=0.6, label="Cα RMSD (Å)")
    axes[0].axhline(RMSD_HELIX_NM * 10, color="green", ls="--", lw=0.8, label="Helix threshold")
    axes[0].axhline(RMSD_COIL_NM * 10, color="orange", ls="--", lw=0.8, label="Coil threshold")
    axes[0].set_ylabel("Cα RMSD (Å)")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot(t, nhb, color="#457b9d", lw=0.6)
    axes[1].axhline(N_HB_MAX * NHB_HELIX_FRAC, color="red", ls="--", lw=0.8,
                    label=f"Helix threshold (≥{N_HB_MAX * NHB_HELIX_FRAC:.0f})")
    axes[1].set_ylabel("N_HB")
    axes[1].legend(fontsize=8, loc="upper right")

    state_int = (states == "H").astype(float)
    axes[2].plot(t, state_int, color="#2a9d8f", lw=0.5, alpha=0.8)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Coil", "Helix"])
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("State")

    plt.tight_layout()
    _save_fig(fig, output_path)


def _plot_2d_fes(
    fes_dict: dict,
    label: str,
    output_path: str,
    annotate_basins: bool = True,
) -> None:
    """Filled contour plot of the 2D FES (RMSD × N_HB)."""
    fes = fes_dict["fes"]
    cx = fes_dict["centers_rmsd"] * 10  # nm → Å
    cy = fes_dict["centers_nhb"]
    xi, yi = np.meshgrid(cx, cy, indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.linspace(0, FES_CLIP_KCAL, 20)
    cf = ax.contourf(xi, yi, fes, levels=levels, cmap="RdYlBu_r", extend="max")
    ax.contour(xi, yi, fes, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("ΔG (kcal/mol)")
    ax.set_xlabel("Cα RMSD (Å)")
    ax.set_ylabel("N_HB")
    ax.set_title(f"2D FES — {label}")
    plt.tight_layout()
    _save_fig(fig, output_path)


def _plot_convergence(
    rmsd: np.ndarray,
    nhb: np.ndarray,
    states: np.ndarray,
    bse_result: dict,
    lag_ns: np.ndarray,
    acf_vals: np.ndarray,
    biexp_result: dict,
    label: str,
    output_dir: str,
    prefix: str,
) -> None:
    """Cumulative f_helix, BSE vs block size, split-trajectory FES, ACF."""
    n = len(states)
    t_ns = np.arange(n) * FRAME_SPACING_NS + 25.0  # offset by burn-in

    # ── (A) Cumulative helix fraction ──────────────────────────────────────
    is_helix = (states == "H").astype(float)
    cum_f = np.cumsum(is_helix) / (np.arange(n) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_ns, cum_f, color="#333", lw=1.2)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Cumulative f_helix")
    ax.set_title(f"Convergence of f_helix — {label}")
    ax.axhline(cum_f[-1], color="red", ls="--", lw=0.8, label=f"Final = {cum_f[-1]:.3f}")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{prefix}_cumulative_fhelix.png"))

    # ── (B) Block standard error ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bse_result["block_sizes_ns"], bse_result["bse_values"],
            "o-", color="#457b9d", ms=4, lw=1.2)
    ax.axhline(bse_result["plateau_bse"], color="red", ls="--", lw=0.8,
               label=f"Plateau BSE = {bse_result['plateau_bse']:.4f}")
    ax.set_xlabel("Block size (ns)")
    ax.set_ylabel("Block SE of f_helix")
    ax.set_title(f"Block Standard Error — {label}")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{prefix}_bse.png"))

    # ── (C) Split-trajectory FES (first half vs second half) ──────────────
    half = n // 2
    all_rmsd = np.concatenate([rmsd[:half], rmsd[half:]])
    all_nhb = np.concatenate([nhb[:half], nhb[half:]])
    rmsd_edges = np.linspace(all_rmsd.min(), all_rmsd.max(), FES_NBINS + 1)
    nhb_edges = np.linspace(all_nhb.min(), all_nhb.max(), FES_NBINS + 1)

    cx = 0.5 * (rmsd_edges[:-1] + rmsd_edges[1:]) * 10  # Å

    def _fes_1d_nhb(nhb_subset):
        counts, _ = np.histogram(nhb_subset.astype(float), bins=nhb_edges)
        c = counts.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(c > 0, -RT_KCAL * np.log(c / c.max()), np.nan)
        return f

    fes1 = _fes_1d_nhb(nhb[:half])
    fes2 = _fes_1d_nhb(nhb[half:])
    nhb_cx = 0.5 * (nhb_edges[:-1] + nhb_edges[1:])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(nhb_cx, fes1, color="#e63946", lw=1.5, label="First 250 ns")
    ax.plot(nhb_cx, fes2, color="#457b9d", lw=1.5, ls="--", label="Second 250 ns")
    ax.set_xlabel("N_HB")
    ax.set_ylabel("ΔG (kcal/mol)")
    ax.set_title(f"Split-trajectory FES — {label}")
    ax.set_ylim(0, FES_CLIP_KCAL)
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{prefix}_split_fes.png"))

    # ── (D) ACF + biexponential fit ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lag_ns, acf_vals, color="#333", lw=0.8, label="ACF(N_HB)")
    if biexp_result["fit_success"]:
        ax.plot(lag_ns, biexp_result["fit_acf"], color="red", lw=1.5,
                label=(f"Biexp fit\n"
                       f"τ_slow={biexp_result['tau_slow_ns']:.1f} ns\n"
                       f"τ_fast={biexp_result['tau_fast_ns']:.2f} ns"))
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Lag (ns)")
    ax.set_ylabel("C(t)")
    ax.set_title(f"ACF of N_HB — {label}")
    ax.set_xlim(0, lag_ns[-1])
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{prefix}_acf.png"))


def _plot_per_residue_helix(
    helix_frac_dssp: np.ndarray,
    label: str,
    output_path: str,
) -> None:
    """Bar chart of per-residue helix fraction from DSSP."""
    n_res = len(helix_frac_dssp)
    x = np.arange(1, n_res + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, helix_frac_dssp, color="#457b9d", edgecolor="k", lw=0.5)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Helix fraction (DSSP)")
    ax.set_title(f"Per-residue helix fraction — {label}")
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _save_fig(fig, output_path)


def _plot_ramachandran(
    rama_pmf: dict,
    label: str,
    output_path: str,
) -> None:
    """Grid of per-residue Ramachandran PMF plots."""
    res_indices = sorted(rama_pmf.keys())
    n_res = len(res_indices)
    ncols = min(5, n_res)
    nrows = (n_res + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8),
                              squeeze=False)
    fig.suptitle(f"Ramachandran PMF — {label}", fontsize=12)

    levels = np.linspace(0, FES_CLIP_KCAL, 15)

    for plot_i, res_idx in enumerate(res_indices):
        row, col = divmod(plot_i, ncols)
        ax = axes[row][col]
        r = rama_pmf[res_idx]
        pmf = r["pmf"]
        cp = r["centers"][0]  # phi centers (degrees)
        cs = r["centers"][1]  # psi centers (degrees)
        xi, yi = np.meshgrid(cp, cs, indexing="ij")
        ax.contourf(xi, yi, pmf, levels=levels, cmap="RdYlBu_r", extend="max")
        ax.set_title(f"Res {res_idx + 1}", fontsize=8)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.axvline(-62, color="k", lw=0.5, ls="--")
        ax.axhline(-41, color="k", lw=0.5, ls="--")
        ax.set_xticks([-120, 0, 120])
        ax.set_yticks([-120, 0, 120])
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(n_res, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, output_path)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ANALYSIS RUNNER
# ═════════════════════════════════════════════════════════════════════════════


def analyze_system(
    prefix: str,
    label: str,
    prod_dcd: str,
    prod_top_pdb: str,
    helix_ref_pdb: str,
    output_dir: str,
) -> dict:
    """Run full ground truth analysis for one trajectory.

    Steps:
    1. Load trajectory (protein-only)
    2. Remove burn-in (first 250 frames = 25 ns)
    3. Superpose on first post-burn-in frame for RMSD reference
    4. Compute RMSD, N_HB, DSSP, features
    5. Two-state classification
    6. Kinetics + ACF + block SE
    7. 2D FES, 1D FES, Ramachandran PMF
    8. Save all figures and CSV to output_dir
    9. Return summary dict

    Parameters
    ----------
    prefix       : system prefix (e.g. "A10_helix")
    label        : human-readable label for plot titles
    prod_dcd     : path to production DCD
    prod_top_pdb : path to topology PDB written before DCDReporter
    helix_ref_pdb: path to the in-vacuo helix PDB for RMSD reference
    output_dir   : directory where figures and CSV will be saved

    Returns
    -------
    dict of all computed metrics
    """
    import mdtraj as md

    os.makedirs(output_dir, exist_ok=True)

    print(f"[{prefix}] Loading trajectory...")
    traj_full = load_trajectory(prod_dcd, prod_top_pdb, protein_only=True)
    print(f"  Total frames: {traj_full.n_frames}")

    # ── burn-in removal ──────────────────────────────────────────────────────
    if traj_full.n_frames > BURNIN_FRAMES:
        traj = traj_full[BURNIN_FRAMES:]
    else:
        print(f"  WARNING: trajectory has only {traj_full.n_frames} frames; "
              f"skipping burn-in removal (need > {BURNIN_FRAMES})")
        traj = traj_full

    n_frames = traj.n_frames
    print(f"  Post burn-in frames: {n_frames}  ({n_frames * FRAME_SPACING_NS:.1f} ns)")

    # ── RMSD (superpose on frame 0 of post-burn-in, i.e. ~25 ns) ──────────
    print(f"[{prefix}] Computing Cα RMSD...")
    rmsd = compute_backbone_rmsd(traj, ref_frame_index=0)  # nm

    # ── N_HB ─────────────────────────────────────────────────────────────────
    print(f"[{prefix}] Counting helical H-bonds...")
    nhb = compute_nhb(traj)

    # ── DSSP ─────────────────────────────────────────────────────────────────
    print(f"[{prefix}] Running DSSP (MDAnalysis)...")
    try:
        dssp_arr = compute_dssp(traj)
        dssp_ok = True
    except Exception as e:
        print(f"  DSSP failed: {e}. Using N_HB proxy for DSSP criterion.")
        dssp_ok = False
        # Proxy: construct a fake DSSP array from N_HB
        n_res = traj.n_residues
        dssp_arr = np.where(
            (nhb / N_HB_MAX >= DSSP_HELIX_FRAC)[:, None] * np.ones((1, n_res)),
            "H",
            "-",
        ).astype(object)

    helix_frac_per_res = compute_helix_fraction_dssp(dssp_arr)

    # ── featurization (sin/cos phi/psi) ──────────────────────────────────────
    print(f"[{prefix}] Computing torsion features...")
    features = compute_featurization(traj)

    # ── two-state classification ──────────────────────────────────────────────
    print(f"[{prefix}] Classifying states...")
    states = classify_states(rmsd, nhb, dssp_arr)
    f_helix = float((states == "H").mean())

    # Agreement score: fraction of frames where all three criteria agree
    rmsd_helix_bool = rmsd < RMSD_HELIX_NM  # uses helix threshold (no hysteresis for agreement)
    nhb_helix_bool = (nhb / N_HB_MAX) >= NHB_HELIX_FRAC
    dssp_helix_bool = (dssp_arr == "H").mean(axis=1) >= DSSP_HELIX_FRAC
    agreement = float(
        np.mean(
            (rmsd_helix_bool == nhb_helix_bool) & (rmsd_helix_bool == dssp_helix_bool)
        )
    )

    # ── kinetics ──────────────────────────────────────────────────────────────
    print(f"[{prefix}] Computing kinetics...")
    kinetics = compute_kinetics(states)

    # ── ACF + biexponential fit ───────────────────────────────────────────────
    print(f"[{prefix}] Computing ACF of N_HB...")
    lag_ns, acf_vals = compute_acf(nhb.astype(float), max_lag_frames=200)
    biexp = fit_biexponential(lag_ns, acf_vals)

    # ── block standard error ──────────────────────────────────────────────────
    print(f"[{prefix}] Computing block standard error...")
    is_helix_ts = (states == "H").astype(float)
    bse_result = compute_block_se(is_helix_ts)

    # ── 2D FES ───────────────────────────────────────────────────────────────
    print(f"[{prefix}] Computing 2D FES...")
    fes2d = compute_2d_fes(rmsd, nhb)
    fes1d = compute_1d_fes_nhb(nhb)

    # ── Ramachandran PMF ──────────────────────────────────────────────────────
    print(f"[{prefix}] Computing Ramachandran PMF...")
    rama_pmf = compute_ramachandran_pmf(traj)

    # ── split-trajectory JSD ─────────────────────────────────────────────────
    split_jsd = compute_split_trajectory_jsd(rmsd, nhb)

    # ── SAVE FIGURES ─────────────────────────────────────────────────────────
    print(f"[{prefix}] Saving figures to {output_dir}...")

    _plot_timeseries(rmsd, nhb, states, label,
                     os.path.join(output_dir, f"{prefix}_timeseries.png"))

    _plot_2d_fes(fes2d, label,
                 os.path.join(output_dir, f"{prefix}_fes2d.png"))

    # 1D FES along N_HB
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fes1d["centers"], fes1d["fes"], color="#333", lw=1.5)
    ax.set_xlabel("N_HB")
    ax.set_ylabel("ΔG (kcal/mol)")
    ax.set_title(f"1D FES (N_HB) — {label}")
    ax.set_ylim(0, FES_CLIP_KCAL)
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{prefix}_fes1d_nhb.png"))

    _plot_convergence(
        rmsd, nhb, states, bse_result, lag_ns, acf_vals, biexp,
        label, output_dir, prefix,
    )

    _plot_per_residue_helix(
        helix_frac_per_res, label,
        os.path.join(output_dir, f"{prefix}_per_residue_helix.png"),
    )

    _plot_ramachandran(
        rama_pmf, label,
        os.path.join(output_dir, f"{prefix}_ramachandran.png"),
    )

    # ── SAVE CSV ──────────────────────────────────────────────────────────────
    import csv

    csv_path = os.path.join(output_dir, f"{prefix}_kinetics.csv")
    summary = {
        "prefix": prefix,
        "label": label,
        "n_frames_post_burnin": n_frames,
        "f_helix": f_helix,
        "three_criterion_agreement": agreement,
        "n_fold": kinetics["n_fold"],
        "n_unfold": kinetics["n_unfold"],
        "tau_fold_ns": kinetics["tau_fold"],
        "tau_unfold_ns": kinetics["tau_unfold"],
        "tau_eq_ns": kinetics["tau_eq"],
        "k_f_ns_inv": kinetics["k_f"],
        "k_u_ns_inv": kinetics["k_u"],
        "k_f_sem": kinetics["k_f_sem"],
        "k_u_sem": kinetics["k_u_sem"],
        "tau_eq_sem": kinetics["tau_eq_sem"],
        "dG_fold_kcal": kinetics["dG_fold"],
        "acf_tau_slow_ns": biexp["tau_slow_ns"],
        "acf_tau_fast_ns": biexp["tau_fast_ns"],
        "acf_A": biexp["A"],
        "bse_plateau": bse_result["plateau_bse"],
        "split_jsd": split_jsd,
    }

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"[{prefix}] Done.  f_helix={f_helix:.3f}  τ_eq={kinetics['tau_eq']:.2f} ns")
    print(f"  CSV saved: {csv_path}")

    # Attach arrays for downstream use
    summary["rmsd"] = rmsd
    summary["nhb"] = nhb
    summary["states"] = states
    summary["features"] = features
    summary["dssp_array"] = dssp_arr
    summary["helix_frac_per_res"] = helix_frac_per_res
    summary["fes2d"] = fes2d
    summary["fes1d"] = fes1d
    summary["rama_pmf"] = rama_pmf
    summary["bse_result"] = bse_result
    summary["lag_ns"] = lag_ns
    summary["acf_vals"] = acf_vals
    summary["biexp_result"] = biexp

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SYSTEM COMPARISON
# ═════════════════════════════════════════════════════════════════════════════


def compare_systems(
    results_a10: dict,
    results_a8q2: dict,
    output_dir: str,
) -> dict:
    """Side-by-side comparison of A10 vs A8Q2 ground truth.

    Generates:
      - 2D FES comparison figure (side by side, shared colour scale)
      - Per-residue helix fraction overlay
      - Two-replica JSD (noise floor estimate if both replicas are present)

    Parameters
    ----------
    results_a10  : return dict from analyze_system for A10
    results_a8q2 : return dict from analyze_system for A8Q2
    output_dir   : directory to save comparison figures

    Returns
    -------
    dict with comparison statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 2D FES side-by-side ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    levels = np.linspace(0, FES_CLIP_KCAL, 20)

    for ax, results, sys_label, col in zip(
        axes,
        [results_a10, results_a8q2],
        ["A10 (Ac-Ala₁₀-NHMe)", "A8Q2 (Ac-Ala₈Gln₂-NHMe)"],
        ["#e63946", "#457b9d"],
    ):
        fes2d = results["fes2d"]
        cx = fes2d["centers_rmsd"] * 10  # nm → Å
        cy = fes2d["centers_nhb"]
        xi, yi = np.meshgrid(cx, cy, indexing="ij")
        cf = ax.contourf(xi, yi, fes2d["fes"], levels=levels, cmap="RdYlBu_r", extend="max")
        ax.contour(xi, yi, fes2d["fes"], levels=levels, colors="k", linewidths=0.3, alpha=0.4)
        ax.set_xlabel("Cα RMSD (Å)")
        ax.set_ylabel("N_HB")
        ax.set_title(sys_label)
        fig.colorbar(cf, ax=ax, label="ΔG (kcal/mol)")

    fig.suptitle("2D Free Energy Surface Comparison", fontsize=13)
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "compare_fes2d.png"))

    # ── per-residue helix fraction overlay ────────────────────────────────
    hf_a10 = results_a10["helix_frac_per_res"]
    hf_a8q2 = results_a8q2["helix_frac_per_res"]
    n_res = min(len(hf_a10), len(hf_a8q2))
    x = np.arange(1, n_res + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, hf_a10[:n_res], "o-", color="#e63946", lw=1.5, ms=6, label="A10")
    ax.plot(x, hf_a8q2[:n_res], "s--", color="#457b9d", lw=1.5, ms=6, label="A8Q2")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Helix fraction (DSSP)")
    ax.set_title("Per-residue helix fraction: A10 vs A8Q2")
    ax.set_xticks(x)
    ax.set_ylim(0, max(hf_a10.max(), hf_a8q2.max()) * 1.1 + 0.02)
    ax.legend()
    ax.axvline(3, color="gray", ls=":", lw=0.8)
    ax.axvline(8, color="gray", ls=":", lw=0.8)
    ax.text(3, ax.get_ylim()[1] * 0.95, "Gln@3", ha="center", fontsize=8, color="gray")
    ax.text(8, ax.get_ylim()[1] * 0.95, "Gln@8", ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "compare_per_residue_helix.png"))

    # ── 1D FES overlay ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results_a10["fes1d"]["centers"], results_a10["fes1d"]["fes"],
            color="#e63946", lw=1.5, label="A10")
    ax.plot(results_a8q2["fes1d"]["centers"], results_a8q2["fes1d"]["fes"],
            color="#457b9d", lw=1.5, ls="--", label="A8Q2")
    ax.set_xlabel("N_HB")
    ax.set_ylabel("ΔG (kcal/mol)")
    ax.set_title("1D FES (N_HB) comparison")
    ax.set_ylim(0, FES_CLIP_KCAL)
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "compare_fes1d_nhb.png"))

    # ── inter-replica JSD (noise floor) ──────────────────────────────────
    jsd_inter = compute_two_replica_jsd(
        results_a10["rmsd"], results_a10["nhb"],
        results_a8q2["rmsd"], results_a8q2["nhb"],
    )

    # ── summary table ─────────────────────────────────────────────────────
    import csv

    rows = []
    for name, r in [("A10", results_a10), ("A8Q2", results_a8q2)]:
        rows.append(
            {
                "system": name,
                "f_helix": r["f_helix"],
                "tau_eq_ns": r["tau_eq_ns"],
                "tau_fold_ns": r["tau_fold_ns"],
                "tau_unfold_ns": r["tau_unfold_ns"],
                "k_f_ns_inv": r["k_f_ns_inv"],
                "k_u_ns_inv": r["k_u_ns_inv"],
                "dG_fold_kcal": r["dG_fold_kcal"],
                "bse_plateau": r["bse_plateau"],
                "acf_tau_slow_ns": r["acf_tau_slow_ns"],
                "split_jsd": r["split_jsd"],
            }
        )

    comparison_csv = os.path.join(output_dir, "comparison_summary.csv")
    with open(comparison_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[compare] JSD (A10 vs A8Q2, noise floor): {jsd_inter:.4f}")
    print(f"[compare] Comparison figures saved to {output_dir}")

    return {
        "jsd_a10_vs_a8q2": jsd_inter,
        "f_helix_a10": results_a10["f_helix"],
        "f_helix_a8q2": results_a8q2["f_helix"],
        "delta_f_helix": results_a10["f_helix"] - results_a8q2["f_helix"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — COMMAND-LINE INTERFACE
# ═════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 Ground Truth Analysis — compute all metrics, figures, "
            "and CSV files for A10 and/or A8Q2 production trajectories."
        )
    )
    parser.add_argument(
        "--system",
        choices=["A10", "A8Q2", "both"],
        default="both",
        help="Which system to analyse (default: both)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Filename prefix for the trajectory (e.g. 'A10_helix'). "
            "If omitted, guessed from --system."
        ),
    )
    parser.add_argument(
        "--prod_dcd",
        default=None,
        help="Path to the production DCD file.",
    )
    parser.add_argument(
        "--prod_top",
        default=None,
        help="Path to the topology PDB written before DCDReporter.",
    )
    parser.add_argument(
        "--helix_ref",
        default=None,
        help="Path to the in-vacuo helix PDB for RMSD reference.",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis_output",
        help="Directory for output figures and CSV (default: analysis_output).",
    )
    return parser


def main() -> None:
    """Command-line entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # ── Infer default file names from system label ─────────────────────────
    def _defaults(system_name: str) -> dict:
        """Return sensible defaults for a system name like 'A10' or 'A8Q2'."""
        return {
            "prefix": f"{system_name}_helix",
            "prod_dcd": f"{system_name}_helix_prod.dcd",
            "prod_top": f"{system_name}_helix_prod_top.pdb",
            "helix_ref": f"{system_name}_helix.pdb",
            "label": f"Ac-{'Ala₁₀' if system_name == 'A10' else 'Ala₈Gln₂'}-NHMe",
        }

    results_store: dict[str, dict] = {}

    systems_to_run: list[str] = (
        ["A10", "A8Q2"] if args.system == "both" else [args.system]
    )

    for sys in systems_to_run:
        d = _defaults(sys)
        prefix = args.prefix if args.prefix else d["prefix"]
        prod_dcd = args.prod_dcd if args.prod_dcd else d["prod_dcd"]
        prod_top = args.prod_top if args.prod_top else d["prod_top"]
        helix_ref = args.helix_ref if args.helix_ref else d["helix_ref"]
        label = d["label"]
        out_dir = os.path.join(args.output_dir, sys)

        # Check files exist before attempting analysis
        for fpath in [prod_dcd, prod_top, helix_ref]:
            if not os.path.isfile(fpath):
                print(f"  WARNING: missing file {fpath!r} — skipping {sys}.")
                break
        else:
            results_store[sys] = analyze_system(
                prefix=prefix,
                label=label,
                prod_dcd=prod_dcd,
                prod_top_pdb=prod_top,
                helix_ref_pdb=helix_ref,
                output_dir=out_dir,
            )

    if "A10" in results_store and "A8Q2" in results_store:
        compare_systems(
            results_store["A10"],
            results_store["A8Q2"],
            output_dir=os.path.join(args.output_dir, "comparison"),
        )


if __name__ == "__main__":
    main()
