"""
simulate.py — Phase 1–4 MD simulations for the BTP CV-transfer study.

**This module is the only supported driver for A10/A8Q2 setup → production MD**
(legacy ``A10_Prod_Run.ipynb`` / ``A8Q2_Prod_Run.ipynb`` were retired after parity
review against this file and ``src/config.py``).

Handles all MD simulation phases for both peptide systems:
  Phase 1 — Structure preparation (ACE/CT3 capped peptide building)
  Phase 2 — Parameterization and solvation (CHARMM36m + mTIP3P)
  Phase 3 — Equilibration (minimize → NVT 1 ns → NPT 10 ns → re-solvate → NPT 2 ns)
  Phase 4 — Ground-truth production MD (NVT 500 ns × 2 replicas)

Systems:
  A10  : Ac-Ala₁₀-NHMe (deca-alanine)
  A8Q2 : Ac-Ala₈Gln₂-NHMe (Gln at positions 3 and 8)

Usage
-----
Run as a script (default ``--output_dir`` is ``outputs/btp_equil``; each system
writes under ``outputs/btp_equil/A10`` or ``.../A8Q2``)::

    python -m src.simulate --system A10  --phase all
    python -m src.simulate --system A8Q2 --phase equil
    python -m src.simulate --system both --phase all

Or import for programmatic use::

    from src.simulate import run_system
    run_system(prefix="A10", sequence=["ALA"]*10, output_dir=Path("outputs/btp_equil/A10"))

Critical design notes
---------------------
CHARMM36m cap implementation:
    CHARMM36m has NO standalone ACE or NME residue templates. Capping is done
    via PATCHES that add atoms CY/OY/CAY/HY1/HY2/HY3 (ACE) and NT/HNT/CAT/
    HT1/HT2/HT3 (CT3) to the terminal residues. The in-memory OpenMM Topology
    (never re-read via app.PDBFile) must be passed to createSystem because the
    PDB reader renames HT1→H, HT2→H2, HT3→H3 globally, destroying CT3 atom names.

PeptideBuilder GLN bug:
    Geometry.geometry("GLN") returns GlyGeo (glycine) instead of GlnGeo.
    Always use Geometry.GlnGeo() directly for glutamine residues.

prod_top.pdb:
    Written from sim_prod.topology immediately before DCDReporter starts. This
    is the ONLY safe MDTraj topology for the production DCD. Pre-equilibration
    PDBs have wrong atom counts after re-solvation and must NOT be used.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# OpenMM imports — deferred to function bodies when possible so the module can
# be imported for documentation/testing without OpenMM installed.
try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit, Vec3, XmlSerializer, Platform
    from pdbfixer import PDBFixer
    from PeptideBuilder import Geometry
    import PeptideBuilder
    import Bio.PDB
    _OPENMM_AVAILABLE = True
except ImportError as _e:
    _OPENMM_AVAILABLE = False
    _IMPORT_ERROR = _e

try:
    import mdtraj as md
    _MDTRAJ_AVAILABLE = True
except ImportError:
    _MDTRAJ_AVAILABLE = False

from src.config import (
    SEQUENCES,
    CONFORMATIONS,
    TEMPERATURE_K,
    PRESSURE_BAR,
    FRICTION_PS,
    TIMESTEP_FS,
    NVT_NS, NPT_NS, NPT_RESOLVATE_NS, PROD_US,
    LOG_INTERVAL_PS, DCD_INTERVAL_PS, CHK_INTERVAL_PS,
    RESTRAINT_K, PADDING_NM, RESOLVATION_PADDING_NM, IONIC_STRENGTH_M,
    NONBONDED_CUTOFF_NM, SWITCH_DISTANCE_NM, EWALD_TOLERANCE,
    MIN_MAX_ITER, MIN_TOLERANCE_KJ,
    STEPS_PER_PS, NVT_STEPS, NPT_STEPS, NPT_RESOLVATE_STEPS, PROD_STEPS,
    LOG_STEPS, DCD_STEPS, CHK_STEPS,
    FF_PROTEIN,
    FF_WATER,
)
from src.utils import (
    apply_windows_checkpoint_patch,
    get_best_platform,
    compute_dof,
    compute_density_kgm3,
    section,
)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — STRUCTURE BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def make_geo(resname: str, phi: float, psi: float):
    """
    Return a PeptideBuilder Geometry object for *resname* with phi/psi angles set.

    This is a thin wrapper that avoids the known PeptideBuilder bug where
    ``Geometry.geometry("GLN")`` returns ``GlyGeo`` (glycine) instead of
    ``GlnGeo`` (glutamine). For all residues not explicitly mapped here,
    a ValueError is raised immediately rather than silently producing wrong geometry.

    The geometry object controls the bond lengths, valence angles, and dihedral
    preferences that PeptideBuilder uses when adding each residue to the chain.
    Setting ``g.phi`` and ``g.psi_im1`` overrides the default ideal-geometry
    values with the user-specified backbone dihedral angles.

    Parameters
    ----------
    resname : str
        Three-letter residue code, e.g. "ALA" or "GLN".
    phi : float
        Backbone φ dihedral angle in degrees.
    psi : float
        Backbone ψ dihedral angle in degrees (stored as psi_im1 in PeptideBuilder).

    Returns
    -------
    PeptideBuilder.Geometry.*Geo
        A geometry object with phi and psi_im1 set.

    Raises
    ------
    ValueError
        If *resname* is not in the supported set {ALA, GLN}.
    """
    geo_map = {
        "ALA": Geometry.AlaGeo,
        "GLN": Geometry.GlnGeo,  # bypass broken Geometry.geometry("GLN") lookup
    }
    if resname not in geo_map:
        raise ValueError(
            f"Unsupported residue '{resname}'. "
            f"Supported: {list(geo_map)}. "
            "Add a mapping in make_geo() for other residues."
        )
    g = geo_map[resname]()
    g.phi = phi
    g.psi_im1 = psi
    return g


def build_peptide(
    sequence: list[str],
    phi: float,
    psi: float,
    output_path: Path,
    label: str,
    prefix: str,
) -> tuple:
    """
    Build a capped peptide using CHARMM36m's native ACE/CT3 patch scheme.

    Returns the in-memory OpenMM Topology and positions in nanometres.
    The topology is the *only* correct object to pass to ``prepare_system``
    because ``app.PDBFile()`` re-reading would corrupt CHARMM atom names.

    Scientific context — why capping is mandatory
    ----------------------------------------------
    An uncapped peptide has NH₃⁺ (N-terminus, +1) and COO⁻ (C-terminus, -1),
    creating an artificial ~10–30 Debye dipole along the peptide axis. This
    dipole interacts with the α-helix macrodipole and artificially stabilises
    helix by ~1–3 kcal/mol — completely overwhelming the true ΔG_fold of only
    ~0.3–1.0 kcal/mol. Every published deca-alanine study (Kuczera 2021/2024,
    Zhuang 2021) uses ACE/NME caps.

    How CHARMM36m implements caps (the patch mechanism)
    ----------------------------------------------------
    CHARMM36m has NO standalone ACE or NME residue templates (verified by
    inspection of charmm36_2024.xml). Capping is done via PATCHES:
      - ACE_0 patch: adds CY, OY, CAY, HY1, HY2, HY3 to the N-terminal residue
      - CT3_0 patch: adds NT, HNT, CAT, HT1, HT2, HT3 to the C-terminal residue
    The terminal residues retain their own name (ALA or GLN) but carry extra atoms.
    OpenMM's ForceField.createSystem() detects these extra atoms and automatically
    matches the patched template.

    Why the topology is never re-read through app.PDBFile()
    -------------------------------------------------------
    OpenMM's PDB reader applies global atom-name aliases: HT1→H, HT2→H2,
    HT3→H3. This renames the CT3 methyl hydrogens to look like extra backbone
    amide protons, causing ``No template found`` from createSystem. The only
    safe approach is to keep the topology in memory and never round-trip it
    through a PDB file for force-field purposes (writing a PDB for visual
    inspection is fine; re-reading it is not).

    5-step implementation
    ---------------------
    1. PeptideBuilder builds the backbone with add_terminal_OXT (required so
       PDBFixer recognises a complete C-terminus).
    2. PDBFixer completes all-atom structure and adds H at pH 7.0.
       This gives NH₃⁺/COO⁻ termini with H, H2, H3 on N-term and OXT on C-term.
    3. New OpenMM Topology is built, copying all PDBFixer atoms EXCEPT:
       H2, H3 from first residue (extra NH₃⁺ protons — conflict with ACE patch)
       OXT, HXT from last residue (carboxylate oxygen — conflict with CT3 patch)
    4a. ACE patch atom positions (CY, OY, CAY, HY1, HY2, HY3) are computed
        geometrically from N-terminal residue backbone vectors.
    4b. CT3 patch atom positions (NT, HNT, CAT, HT1, HT2, HT3) are computed
        geometrically from C-terminal residue backbone vectors.
    5. Patch atoms are added to the topology with correct bonds, and positions
       are converted to nm. An inspection PDB is written (atom names preserved
       because writeFile uses the in-memory topology directly).

    Parameters
    ----------
    sequence : list[str]
        List of three-letter residue codes (N→C order, no caps), e.g.
        ["ALA"]*10 for A10 or ["ALA","ALA","GLN",...] for A8Q2.
    phi : float
        Backbone φ angle in degrees (applied to all residues).
    psi : float
        Backbone ψ angle in degrees (applied to all residues).
    output_path : Path
        Where to write the inspection PDB (NOT the topology used for simulation).
    label : str
        Human-readable label for progress messages, e.g. "extended".
    prefix : str
        System prefix for file naming, e.g. "A10" or "A8Q2".

    Returns
    -------
    tuple[app.Topology, openmm.unit.Quantity]
        ``(topology, positions_nm)`` where positions_nm is a Quantity in nanometres.
        Pass both directly to ``prepare_system``; do NOT re-read the PDB.
    """
    import openmm.app.element as elem_mod

    raw_pdb = str(output_path).replace(".pdb", "_raw.pdb")

    # ── Step 1: PeptideBuilder backbone ──────────────────────────────────
    geo0 = make_geo(sequence[0], phi, psi)
    structure = PeptideBuilder.initialize_res(geo0)
    for resname in sequence[1:]:
        PeptideBuilder.add_residue(structure, make_geo(resname, phi, psi))
    PeptideBuilder.add_terminal_OXT(structure)  # required for PDBFixer template matching

    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(raw_pdb)

    # ── Step 2: PDBFixer — all-atom completion ────────────────────────────
    fixer = PDBFixer(filename=raw_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    src_top = fixer.topology
    src_pos = list(fixer.positions)
    res_list = list(src_top.residues())
    n_res = len(res_list)

    # ── Step 3: Build new topology, removing conflicting terminal atoms ───
    # H2, H3 on first residue: extra NH₃⁺ protons — ACE patch provides the cap instead
    # OXT, HXT on last residue: carboxylate O/H — CT3 patch provides the cap instead
    del_first = {"H2", "H3"}
    del_last = {"OXT", "HXT"}

    def adict_of(res):
        """Extract {atom_name: coord_angstrom_array} for one residue."""
        return {
            a.name: np.array(src_pos[a.index].value_in_unit(unit.angstrom))
            for a in res.atoms()
        }

    af = adict_of(res_list[0])   # first residue atom positions (Å)
    al = adict_of(res_list[-1])  # last residue atom positions (Å)

    # ── Step 4a: ACE patch atom positions ────────────────────────────────
    # ACE_0 adds: CY (carbonyl C), OY (carbonyl O), CAY (methyl C),
    #             HY1, HY2, HY3 (methyl H). Bond: CY–N (peptide bond).
    N_f = af["N"]
    CA_f = af["CA"]
    O_f = af["O"]
    C_f = af["C"]
    v_f = (N_f - CA_f) / np.linalg.norm(N_f - CA_f)   # unit vector CA→N
    # Peptide-plane perpendicular for OY placement (carbonyl lies in peptide plane)
    c1 = np.cross(v_f, O_f - C_f)
    c1 /= np.linalg.norm(c1)
    perp_f = np.cross(c1, v_f)
    perp_f /= np.linalg.norm(perp_f)

    CY = N_f + 1.33 * v_f         # C–N peptide bond = 1.33 Å
    OY = CY + 1.23 * perp_f       # C=O carbonyl = 1.23 Å
    CAY = CY + 1.52 * v_f         # C–CH₃ single bond = 1.52 Å
    r_h = 1.09                     # C–H bond length (Å)
    a2 = np.radians(70.5)          # half tetrahedral angle
    HY1 = CAY + r_h * np.array([np.sin(a2), -np.cos(a2), 0.0])
    HY2 = CAY + r_h * np.array([-np.sin(a2) * 0.5, -np.cos(a2),  np.sin(a2) * 0.866])
    HY3 = CAY + r_h * np.array([-np.sin(a2) * 0.5, -np.cos(a2), -np.sin(a2) * 0.866])
    ace_extra = [
        ("CY",  elem_mod.carbon,   CY),
        ("OY",  elem_mod.oxygen,   OY),
        ("CAY", elem_mod.carbon,  CAY),
        ("HY1", elem_mod.hydrogen, HY1),
        ("HY2", elem_mod.hydrogen, HY2),
        ("HY3", elem_mod.hydrogen, HY3),
    ]

    # ── Step 4b: CT3 patch atom positions ────────────────────────────────
    # CT3_0 adds: NT (methylamide N), HNT (amide H), CAT (methyl C),
    #             HT1, HT2, HT3 (methyl H). Bond: C_last–NT (peptide bond).
    C_l = al["C"]
    CA_l = al["CA"]
    v_l = (C_l - CA_l) / np.linalg.norm(C_l - CA_l)   # unit vector CA→C

    NT = C_l + 1.33 * v_l
    HNT = NT + np.array([0.0, 1.01, 0.0])     # amide H, ~1.01 Å N–H
    CAT = NT + 1.47 * v_l                      # N–CH₃ = 1.47 Å
    HT1 = CAT + r_h * np.array([np.sin(a2), -np.cos(a2), 0.0])
    HT2 = CAT + r_h * np.array([-np.sin(a2) * 0.5, -np.cos(a2),  np.sin(a2) * 0.866])
    HT3 = CAT + r_h * np.array([-np.sin(a2) * 0.5, -np.cos(a2), -np.sin(a2) * 0.866])
    ct3_extra = [
        ("NT",  elem_mod.nitrogen,  NT),
        ("HNT", elem_mod.hydrogen, HNT),
        ("CAT", elem_mod.carbon,   CAT),
        ("HT1", elem_mod.hydrogen, HT1),
        ("HT2", elem_mod.hydrogen, HT2),
        ("HT3", elem_mod.hydrogen, HT3),
    ]

    # ── Build new topology ────────────────────────────────────────────────
    new_top = app.Topology()
    new_chain = new_top.addChain()
    new_pos = []          # list of np.array([x,y,z]) in Å
    old_to_new = {}       # old atom index → new Atom object
    ace_atoms = {}        # patch atom name → new Atom
    ct3_atoms = {}        # patch atom name → new Atom

    for i, res in enumerate(res_list):
        skip = (del_first if i == 0 else set()) | (del_last if i == n_res - 1 else set())
        new_res = new_top.addResidue(res.name, new_chain)

        for atom in res.atoms():
            if atom.name in skip:
                continue
            new_atom = new_top.addAtom(atom.name, atom.element, new_res)
            old_to_new[atom.index] = new_atom
            new_pos.append(np.array(src_pos[atom.index].value_in_unit(unit.angstrom)))

        if i == 0:
            for aname, aelem, coord in ace_extra:
                new_atom = new_top.addAtom(aname, aelem, new_res)
                ace_atoms[aname] = new_atom
                new_pos.append(coord)

        if i == n_res - 1:
            for aname, aelem, coord in ct3_extra:
                new_atom = new_top.addAtom(aname, aelem, new_res)
                ct3_atoms[aname] = new_atom
                new_pos.append(coord)

    # Copy bonds from PDBFixer for kept atoms
    for bond in src_top.bonds():
        a1, a2_ = bond[0], bond[1]
        if a1.index in old_to_new and a2_.index in old_to_new:
            new_top.addBond(old_to_new[a1.index], old_to_new[a2_.index])

    # Wire ACE patch bonds
    def get_atom(res, name):
        return next(a for a in res.atoms() if a.name == name)

    first_res = list(new_top.residues())[0]
    last_res = list(new_top.residues())[-1]
    N_first = get_atom(first_res, "N")
    C_last_a = get_atom(last_res, "C")

    new_top.addBond(ace_atoms["CY"],  N_first)           # ACE carbonyl C – ALA/GLN N
    new_top.addBond(ace_atoms["CY"],  ace_atoms["OY"])   # C=O
    new_top.addBond(ace_atoms["CY"],  ace_atoms["CAY"])  # C–methyl
    new_top.addBond(ace_atoms["CAY"], ace_atoms["HY1"])
    new_top.addBond(ace_atoms["CAY"], ace_atoms["HY2"])
    new_top.addBond(ace_atoms["CAY"], ace_atoms["HY3"])

    # Wire CT3 patch bonds
    new_top.addBond(C_last_a,          ct3_atoms["NT"])   # ALA/GLN C – methylamide N
    new_top.addBond(ct3_atoms["NT"],   ct3_atoms["HNT"])
    new_top.addBond(ct3_atoms["NT"],   ct3_atoms["CAT"])
    new_top.addBond(ct3_atoms["CAT"],  ct3_atoms["HT1"])
    new_top.addBond(ct3_atoms["CAT"],  ct3_atoms["HT2"])
    new_top.addBond(ct3_atoms["CAT"],  ct3_atoms["HT3"])

    # ── Step 5: Convert to nm Quantity and write inspection PDB ──────────
    positions_nm = [Vec3(*(p / 10.0)) for p in new_pos] * unit.nanometer

    # writeFile uses the in-memory topology directly → atom names preserved.
    # This PDB is for visual inspection ONLY. Never re-read via app.PDBFile()
    # for force-field purposes (would corrupt HT1/HT2/HT3 names).
    with open(output_path, "w") as f:
        app.PDBFile.writeFile(new_top, positions_nm, f)

    # Capping verification
    first_r = list(new_top.residues())[0]
    last_r = list(new_top.residues())[-1]
    has_ace = any(a.name in {"CY", "OY", "CAY"} for a in first_r.atoms())
    has_ct3 = any(a.name in {"NT", "CAT"} for a in last_r.atoms())
    res_names = [r.name for r in new_top.residues()]
    atoms_all = list(new_top.atoms())

    print(f"  [{label}] → {output_path}")
    print(f"    Residues : {len(res_names)}  ({', '.join(res_names)})")
    print(f"    Atoms    : {len(atoms_all)}  (all-atom including H)")
    print(f"    φ / ψ    : {phi:.1f}° / {psi:.1f}°")
    if has_ace and has_ct3:
        print(f"    Capping  : ✓ ACE patch (CY/OY/CAY on first {res_names[0]}) "
              f"+ CT3 patch (NT/CAT on last {res_names[-1]})")
        print(f"               Ac-{''.join(sequence)}-NHMe — neutral termini")
    else:
        print(f"    WARN CAPPING CHECK FAILED: has_ace={has_ace}  has_ct3={has_ct3}")
    print()

    return new_top, positions_nm


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — PARAMETERIZATION AND SOLVATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_system(
    peptide_topology,
    peptide_positions,
    solvated_pdb: Path,
    xml_out: Path,
    label: str,
    padding_nm: float = PADDING_NM,
) -> tuple:
    """
    Solvate a capped peptide and create the full OpenMM System.

    This function:
    1. Passes the in-memory topology and positions directly to ``app.Modeller``
       (never through ``app.PDBFile()``).
    2. Calls ``modeller.addSolvent`` with 0.15 M NaCl in a cubic periodic box
       with *padding_nm* padding on all sides.
    3. Creates the System with PME electrostatics, 1.2 nm cutoff, 1.0 nm switch,
       HBonds constraints, and rigid water.
    4. Serializes the System to XML for safe roundtrip to Phase 3.

    Force field choice: CHARMM36m
    --------------------------------
    ``charmm36_2024.xml`` (built into OpenMM 8.x, no extra package required)
    uses CHARMM36m backbone torsion CMAP corrections that improve helix-coil
    balance in short peptides (Huang et al. 2017, Nat. Methods).

    Water model: modified TIP3P (mTIP3P)
    ----------------------------------------
    ``charmm36_2024/water.xml`` implements CHARMM's modified TIP3P, which adds
    Lennard-Jones terms on the hydrogen atoms (ε_H = 0.046 kcal/mol). CHARMM36m
    was parameterised and validated exclusively with mTIP3P. Standard TIP3P
    (no H LJ terms) is incompatible and gives poorer helix-coil balance.
    The ``model="tip3p"`` argument to addSolvent is correct because the
    charmm36_2024/water.xml file overrides the tip3p template with mTIP3P.

    Nonbonded settings rationale
    ----------------------------
    - cutoff 1.2 nm: mandatory for CHARMM36m; parameterised at this exact value
    - switchDistance 1.0 nm: CHARMM LJ force-switch from 10 → 12 Å; OpenMM uses
      potential-based switching (unavoidable limitation, accepted by community)
    - PME ewaldErrorTolerance 0.0005: CHARMM36m recommended (Kim et al. 2015)
    - HBonds + rigidWater: enables 2 fs timestep; SETTLE for TIP3P is exact + fast

    System serialization
    --------------------
    The XML contains the complete potential-energy function. It is re-loaded at
    every subsequent phase, ensuring reproducibility. Barostat is NOT included
    in this XML; it is added fresh in run_equilibration's NPT phase so that
    NPT and NVT phases always start from a clean system.

    Parameters
    ----------
    peptide_topology : app.Topology
        In-memory topology from ``build_peptide`` (correct CHARMM atom names).
    peptide_positions : openmm.unit.Quantity
        Corresponding positions in nm from ``build_peptide``.
    solvated_pdb : Path
        Where to write the solvated PDB (for inspection, not simulation use).
    xml_out : Path
        Where to write the serialized System XML.
    label : str
        Human-readable label for progress messages.
    padding_nm : float
        Solvation box padding in nm (default: PADDING_NM = 1.5 nm).

    Returns
    -------
    tuple[mm.System, app.Topology, openmm.unit.Quantity, int]
        ``(system, topology, positions, dof)``
    """
    print(f"  Processing [{label}]")

    forcefield = app.ForceField(FF_PROTEIN, FF_WATER)
    # Pass topology/positions directly — do NOT re-read the PDB via app.PDBFile()
    # (would rename HT1→H, HT2→H2, HT3→H3, destroying CT3 patch atom names)
    modeller = app.Modeller(peptide_topology, peptide_positions)

    modeller.addSolvent(
        forcefield,
        model=         "tip3p",                          # mTIP3P via charmm36_2024/water.xml
        padding=       padding_nm * unit.nanometers,
        ionicStrength= IONIC_STRENGTH_M * unit.molar,
        neutralize=    True,                              # mandatory: PME diverges for net charge
    )

    n_atoms = modeller.topology.getNumAtoms()
    n_waters = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
    n_na = sum(1 for r in modeller.topology.residues() if r.name == "NA")
    n_cl = sum(1 for r in modeller.topology.residues() if r.name == "CL")
    print(f"    Padding  : {padding_nm} nm")
    print(f"    Atoms    : {n_atoms:,}")
    print(f"    Waters   : {n_waters:,}  |  Na⁺: {n_na}  |  Cl⁻: {n_cl}")

    with open(solvated_pdb, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print(f"    Saved    : {solvated_pdb}")

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=     app.PME,
        nonbondedCutoff=     NONBONDED_CUTOFF_NM * unit.nanometers,
        switchDistance=      SWITCH_DISTANCE_NM  * unit.nanometers,
        constraints=         app.HBonds,
        rigidWater=          True,
        ewaldErrorTolerance= EWALD_TOLERANCE,
    )

    n_constr = system.getNumConstraints()
    dof = 3 * n_atoms - n_constr - 3
    print(f"    DOF      : {dof:,}   (naive 3N = {3*n_atoms:,})")
    print(f"    Forces   : {[type(f).__name__ for f in system.getForces()]}")

    with open(xml_out, "w") as f:
        f.write(XmlSerializer.serialize(system))
    print(f"    Saved    : {xml_out}\n")

    return system, modeller.topology, modeller.positions, dof


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL RESTRAINTS
# ─────────────────────────────────────────────────────────────────────────────

def add_positional_restraints(system, topology, positions, k: float) -> int:
    """
    Add harmonic positional restraints to all non-solvent, non-hydrogen atoms.

    Uses ``CustomExternalForce("k * periodicdistance(...)")`` rather than a
    simple Euclidean distance because ``periodicdistance`` correctly handles
    periodic boundary crossing: if a restrained atom crosses the box boundary,
    the force is computed from the minimum-image distance to the reference
    position, preventing force blow-up artefacts near the box edge.

    Applied to: all atoms where (a) residue is not HOH, NA, or CL, and
    (b) element symbol is not H. This restrains all peptide backbone and
    side-chain heavy atoms.

    Restraint constant k = 1000 kJ/mol/nm² means a 0.1 nm (1 Å) displacement
    costs 10 kJ/mol >> k_BT ≈ 2.75 kJ/mol at 330 K → peptide effectively frozen.

    The restraints are added to *system* in-place (mutation). This system is
    used for minimization and NVT. For NPT, the original XML is re-read fresh
    (no restraints), ensuring the barostat-free system is clean.

    Parameters
    ----------
    system : mm.System
        System to modify in-place.
    topology : app.Topology
        Topology matching the system.
    positions : openmm.unit.Quantity
        Reference positions to restrain toward (usually the solvated PDB positions).
    k : float
        Force constant in kJ/mol/nm².

    Returns
    -------
    int
        Number of atoms that received restraints.
    """
    restraint = mm.CustomExternalForce(
        "k * periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    restraint.addGlobalParameter("k", k * unit.kilojoules_per_mole / unit.nanometers**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    n_restrained = 0
    for atom in topology.atoms():
        if (
            atom.residue.name not in ("HOH", "NA", "CL")
            and atom.element is not None
            and atom.element.symbol != "H"
        ):
            pos = positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
            n_restrained += 1

    system.addForce(restraint)
    return n_restrained


def _n_atom_records_in_pdb(pdb_path: Path) -> int:
    """Count ATOM/HETATM lines (file order) without using PDBFile."""
    n = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                n += 1
    return n


def _first_n_atom_xyz_nm_from_pdb(pdb_path: Path, n: int) -> np.ndarray:
    """
    First ``n`` ATOM/HETATM Cartesian coordinates in **file order** (nm).

    OpenMM's ``PDBFile`` can merge/rename duplicate atom names (e.g. two ``H``
    on one ALA), shifting the coordinate list so it no longer matches the
    solvated topology atom order — then re-solvation + MD hits NaNs.  Reading
    raw PDB columns avoids that.
    """
    coords: list[list[float]] = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coords.append([x, y, z])
                if len(coords) >= n:
                    break
    if len(coords) < n:
        raise ValueError(
            f"PDB {pdb_path}: expected ≥{n} ATOM/HETATM records, found {len(coords)}"
        )
    # PDB uses Å; OpenMM uses nm
    return np.ascontiguousarray(coords, dtype=np.float64) / 10.0


def _npt_final_topology_positions_box(
    pdb_path: Path,
    n_particles: int,
) -> tuple:
    """Load NPT-final solvated PDB for Phase 4 cold start.

    OpenMM's ``PDBFile`` merges atoms that share a residue + name (e.g. two
    backbone ``H`` on CHARMM ALA), producing **fewer** atoms than the
    serialized ``System`` XML (4830 vs 4829) and breaking ``setPositions``.
    MDTraj keeps every ATOM record and matches the XML particle count.

    Returns
    -------
    topology : openmm.app.Topology
    positions : openmm.unit.Quantity
    box_vecs : tuple[Vec3, Vec3, Vec3] | None
        Periodic box edge vectors in nm, or None if not found (caller errors).
    """
    from openmm import Vec3, unit

    pdb_path = Path(pdb_path)

    if _MDTRAJ_AVAILABLE:
        traj = md.load_pdb(str(pdb_path))
        if traj.n_atoms != n_particles:
            raise ValueError(
                f"{pdb_path}: MDTraj loaded {traj.n_atoms} atoms but production "
                f"system XML has {n_particles} particles — check PDB/XML pairing."
            )
        omm_top = traj.topology.to_openmm()
        positions = traj.openmm_positions(0)
        box_vecs = None
        if traj.unitcell_vectors is not None:
            M = traj.unitcell_vectors[0]
            box_vecs = (
                Vec3(float(M[0, 0]), float(M[0, 1]), float(M[0, 2])),
                Vec3(float(M[1, 0]), float(M[1, 1]), float(M[1, 2])),
                Vec3(float(M[2, 0]), float(M[2, 1]), float(M[2, 2])),
            )
        if box_vecs is None:
            pdb_box = app.PDBFile(str(pdb_path))
            box_vecs = pdb_box.topology.getPeriodicBoxVectors()
        return omm_top, positions, box_vecs

    pdb = app.PDBFile(str(pdb_path))
    omm_top = pdb.topology
    if omm_top.getNumAtoms() != n_particles:
        raise ValueError(
            f"{pdb_path}: PDBFile topology has {omm_top.getNumAtoms()} atoms but "
            f"system XML has {n_particles} particles (duplicate atom names in PDB). "
            "Install mdtraj, or regenerate the NPT-final PDB with unique atom names."
        )
    coords_nm = _first_n_atom_xyz_nm_from_pdb(pdb_path, n_particles)
    positions = unit.Quantity(coords_nm, unit.nanometer)
    box_vecs = omm_top.getPeriodicBoxVectors()
    return omm_top, positions, box_vecs


def _extended_resolv_marker(output_dir: Path, prefix: str) -> Path:
    """Written only after extended re-solvate + 2 ns NPT completes successfully."""
    return output_dir / f"{prefix}_extended_resolv_complete"


def equil_phase3_complete(output_dir: Path, prefix: str, label: str) -> bool:
    """
    True when Phase 3 artifacts exist. Extended additionally requires the
    re-solvate marker *or* a compact-box ``npt_final`` (atom count ≪ large
    solvated box) so we do not skip after a crash between large-box NPT and
    compact-box re-solvate, while still accepting legacy runs without a marker.
    """
    fp = output_dir / f"{prefix}_{label}_npt_final.pdb"
    fx = output_dir / f"{prefix}_{label}_npt_final.xml"
    if not fp.exists() or not fx.exists():
        return False
    if label == "extended":
        if _extended_resolv_marker(output_dir, prefix).exists():
            return True
        # Legacy: equil finished before marker file existed — compact box ≈ 5–8k atoms.
        try:
            n_at = _n_atom_records_in_pdb(fp)
        except OSError:
            return False
        return n_at < 12_000
    return True


def _extended_resolvation_pipeline(
    peptide_top_orig,
    peptide_positions,
    prefix: str,
    label: str,
    output_dir: Path,
    platform,
    final_pdb: Path,
    final_xml: Path,
) -> None:
    """
    Extract peptide → re-solvate (1.2 nm) → NPT_RESOLVATE_NS NPT;
    overwrite ``final_pdb`` / ``final_xml`` with compact-box results.
    """
    section(f"[{label}] Re-solvation — compact box for production")
    print("  Extracting equilibrated peptide → re-solvating with 1.2 nm padding.")
    print("  Follows Kuczera et al. 2021 methodology exactly.\n")

    n_pep = peptide_top_orig.getNumAtoms()
    peptide_only_pdb = output_dir / f"{prefix}_{label}_peptide_only.pdb"
    with open(peptide_only_pdb, "w") as f:
        app.PDBFile.writeFile(peptide_top_orig, peptide_positions, f)
    print(f"  Peptide-only PDB : {peptide_only_pdb}  ({n_pep} atoms, correct CHARMM names)")

    forcefield_rs = app.ForceField(FF_PROTEIN, FF_WATER)
    modeller_rs = app.Modeller(peptide_top_orig, peptide_positions)
    modeller_rs.addSolvent(
        forcefield_rs,
        model=         "tip3p",
        padding=       RESOLVATION_PADDING_NM * unit.nanometers,
        ionicStrength= IONIC_STRENGTH_M * unit.molar,
        neutralize=    True,
    )

    rs_solvated_pdb = output_dir / f"{prefix}_{label}_resolvated.pdb"
    rs_xml = output_dir / f"{prefix}_{label}_resolvated.xml"
    with open(rs_solvated_pdb, "w") as f:
        app.PDBFile.writeFile(modeller_rs.topology, modeller_rs.positions, f)
    print(f"  Re-solvated PDB : {rs_solvated_pdb}")

    system_rs = forcefield_rs.createSystem(
        modeller_rs.topology,
        nonbondedMethod=     app.PME,
        nonbondedCutoff=     NONBONDED_CUTOFF_NM * unit.nanometers,
        switchDistance=      SWITCH_DISTANCE_NM  * unit.nanometers,
        constraints=         app.HBonds,
        rigidWater=          True,
        ewaldErrorTolerance= EWALD_TOLERANCE,
    )

    n_rs = modeller_rs.topology.getNumAtoms()
    n_rs_wat = sum(1 for r in modeller_rs.topology.residues() if r.name == "HOH")
    print(f"  Re-solvated atoms   : {n_rs:,}  ({n_rs_wat:,} waters)")

    with open(rs_xml, "w") as f:
        f.write(XmlSerializer.serialize(system_rs))

    print(f"  Running {NPT_RESOLVATE_NS} ns NPT on re-solvated system...")

    system_rs_npt = XmlSerializer.deserialize(open(rs_xml).read())
    system_rs_npt.addForce(mm.MonteCarloBarostat(
        PRESSURE_BAR  * unit.bar,
        TEMPERATURE_K * unit.kelvin,
        25,
    ))
    integrator_rs = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PS   / unit.picosecond,
        TIMESTEP_FS   * unit.femtoseconds,
    )
    integrator_rs.setRandomNumberSeed(42)
    sim_rs = app.Simulation(modeller_rs.topology, system_rs_npt, integrator_rs, platform)

    rs_log = output_dir / f"{prefix}_{label}_resolvate_npt.log"
    rs_dcd = output_dir / f"{prefix}_{label}_resolvate_npt.dcd"
    rs_chk = output_dir / f"{prefix}_{label}_resolvate_npt.chk"

    rs_resume = rs_chk.exists() and rs_chk.stat().st_size > 0
    if rs_resume:
        print(f"  Resuming re-solvation NPT from checkpoint: {rs_chk}")
        sim_rs.loadCheckpoint(str(rs_chk))
        rs_steps_done = sim_rs.currentStep
        rs_remaining = NPT_RESOLVATE_STEPS - rs_steps_done
        if rs_remaining <= 0:
            print("  Re-solvation NPT already complete.")
            rs_remaining = 0
        else:
            print(f"  Steps done: {rs_steps_done:,}  Remaining: {rs_remaining:,}")
    else:
        sim_rs.context.setPositions(modeller_rs.positions)
        # Fresh solvated box often has vdW clashes; minimize before NPT + random velocities.
        section(f"[{label}] Re-solvated box — pre-NPT minimization")
        t_min_rs = time.time()
        sim_rs.minimizeEnergy(
            tolerance=MIN_TOLERANCE_KJ * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=MIN_MAX_ITER,
        )
        print(f"  Wall time : {time.time() - t_min_rs:.1f} s\n")
        sim_rs.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 42)
        rs_remaining = NPT_RESOLVATE_STEPS

    if rs_remaining > 0:
        rs_log_mode = "a" if rs_resume else "w"
        sim_rs.reporters.append(app.StateDataReporter(
            str(rs_log), LOG_STEPS,
            step=True, time=True, temperature=True, density=True,
            progress=True, remainingTime=True, totalSteps=NPT_RESOLVATE_STEPS,
            separator="\t", append=(rs_log_mode == "a"),
        ))
        sim_rs.reporters.append(app.StateDataReporter(
            sys.stdout, LOG_STEPS * 20,
            step=True, time=True, temperature=True, density=True,
            progress=True, remainingTime=True, totalSteps=NPT_RESOLVATE_STEPS,
            separator="\t",
        ))
        sim_rs.reporters.append(app.DCDReporter(
            str(rs_dcd), DCD_STEPS,
            append=(rs_resume and rs_dcd.exists()),
        ))
        sim_rs.reporters.append(app.CheckpointReporter(str(rs_chk), CHK_STEPS))
        sim_rs.step(rs_remaining)

    rs_end = sim_rs.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    rs_density = compute_density_kgm3(system_rs_npt, sim_rs.context.getState())
    print(f"  Re-solvated density : {rs_density:.1f} kg/m³ (target: ~990–1010)")

    with open(final_pdb, "w") as f:
        app.PDBFile.writeFile(modeller_rs.topology, rs_end.getPositions(), f)
    with open(rs_xml) as f_in, open(final_xml, "w") as f_out:
        f_out.write(f_in.read())

    print(f"  Re-solvated final PDB : {final_pdb}")
    print(f"  Re-solvated system XML: {final_xml}")

    _extended_resolv_marker(output_dir, prefix).write_text("ok\n", encoding="ascii")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — EQUILIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_equilibration(conf: dict, output_dir: Path, platform) -> None:
    """
    Full equilibration pipeline for one conformation: minimize → NVT → NPT →
    (for extended conformation only) re-solvate → NPT 2 ns.

    Each sub-phase has checkpoint/resume: if a `.chk` file exists and is
    non-empty, the phase resumes from where it left off. If steps_remaining ≤ 0
    the sub-phase is skipped entirely.

    The final output is two files:
      ``{PREFIX}_{label}_npt_final.pdb`` — positions + box vectors from NPT endpoint
      ``{PREFIX}_{label}_npt_final.xml`` — serialized System without barostat

    These are the sole inputs to ``run_production``.

    Sub-phase details
    -----------------
    Step 1 — Energy minimization with positional restraints (k = 1000 kJ/mol/nm²)
        L-BFGS minimizer resolves initial Van der Waals clashes in the freshly
        solvated structure (typically a drop of ~185,000 kJ/mol). Restraints
        prevent the peptide from drifting during minimization.

    Step 2 — NVT 1 ns at TEMPERATURE_K with restraints
        Allows water molecules to diffuse and pack around the restrained peptide
        backbone. At 330 K TIP3P water diffuses ~7 × 10⁻⁹ m²/s; a water molecule
        traverses the peptide diameter (~1.5 nm) in ~0.05 ns. 1 ns provides 20×
        that timescale, ensuring complete hydration-shell reorganization.

    Step 3 — NPT 10 ns at TEMPERATURE_K / PRESSURE_BAR, no restraints
        Fresh system deserialization (critically: NO restraints, no barostat yet
        from the XML). Adds MonteCarloBarostat (25-step volume moves). Converges
        box density to ~990–1010 kg/m³ for CHARMM36m + mTIP3P at 330 K.

    Step 4 (extended only) — Re-solvation + NPT 2 ns
        The extended conformation's initial box holds ~16,000 atoms (1.5 nm
        padding around a 3.4 nm peptide). After NPT equilibration the box is
        still 3× larger than needed. Following Kuczera 2021: extract peptide
        coordinates from NPT final state, re-solvate with 1.2 nm padding, run
        2 ns NPT on the compact box. The original in-memory peptide topology
        (never re-read from PDB) is used to preserve HT1/HT2/HT3 atom names.
        Overwrites final_pdb and final_xml with the re-solvated results.

    Parameters
    ----------
    conf : dict
        Conformation configuration with keys:
          ``label``   : "extended" or "helix"
          ``pdb``     : path to solvated PDB (used for minimization/NVT/NPT topology)
          ``xml``     : path to serialized System XML
          ``pep_top`` : in-memory peptide topology (correct CHARMM atom names)
          ``pep_pos`` : in-memory peptide positions (nm Quantity)
    output_dir : Path
        Directory where all output files are written.
    platform : openmm.Platform
        OpenMM computation platform (from ``get_best_platform()``).
    """
    label = conf["label"]
    pdb_path = Path(conf["pdb"])
    xml_path = Path(conf["xml"])
    peptide_top_orig = conf.get("pep_top")
    peptide_pos_orig = conf.get("pep_pos")
    prefix = conf.get("prefix", "SYS")

    section(f"{prefix} [{label.upper()}] — Phase 3 Equilibration")
    print(f"  Input PDB : {pdb_path}")
    print(f"  Input XML : {xml_path}")
    print(f"  NVT       : {NVT_NS} ns  ({NVT_STEPS:,} steps)")
    print(f"  NPT       : {NPT_NS} ns  ({NPT_STEPS:,} steps)")
    print(f"  Timestep  : {TIMESTEP_FS} fs")

    for path in [pdb_path, xml_path]:
        if not path.exists():
            print(f"\nERROR: {path} not found. Run Phase 2 first.")
            return

    # Large-box NPT finished but re-solvate crashed / was skipped: finish compact box only.
    resolv_marker = _extended_resolv_marker(output_dir, prefix)
    npt_final_early = output_dir / f"{prefix}_{label}_npt_final.pdb"
    npt_final_xml_early = output_dir / f"{prefix}_{label}_npt_final.xml"
    if (
        label == "extended"
        and peptide_top_orig is not None
        and npt_final_early.exists()
        and npt_final_xml_early.exists()
        and not resolv_marker.exists()
    ):
        section(f"{prefix} [EXTENDED] — Resume re-solvation only (large-box NPT already done)")
        n_pep_rs = peptide_top_orig.getNumAtoms()
        arr_rs = _first_n_atom_xyz_nm_from_pdb(npt_final_early, n_pep_rs)
        pep_pos_rs = unit.Quantity(arr_rs, unit.nanometer)
        _extended_resolvation_pipeline(
            peptide_top_orig,
            pep_pos_rs,
            prefix,
            label,
            output_dir,
            platform,
            npt_final_early,
            npt_final_xml_early,
        )
        print(f"  [{label.upper()}] Phase 3 complete.\n")
        return

    pdb = app.PDBFile(str(pdb_path))
    with open(xml_path) as f:
        system_nvt = XmlSerializer.deserialize(f.read())

    N = system_nvt.getNumParticles()
    DOF = compute_dof(system_nvt)
    kB = 8.314e-3  # kJ/mol/K

    print(f"\n  Atoms      : {N:,}")
    print(f"  DOF        : {DOF:,}  (naive 3N = {3*N:,}, difference = {3*N - DOF:,})")

    # ── Step 1: Energy minimization ───────────────────────────────────────
    section(f"[{label}] Step 1: Energy Minimization")

    # Add restraints BEFORE building any Simulation — they persist through NVT.
    n_restrained = add_positional_restraints(
        system_nvt, pdb.topology, pdb.positions, RESTRAINT_K
    )
    print(f"  Restraints applied to {n_restrained} heavy atoms (k = {RESTRAINT_K} kJ/mol/nm²)")
    print(f"  Restraints will persist through NVT, then released for NPT.\n")

    # LangevinMiddleIntegrator: BAOAB splitting applies thermostat in the middle
    # of the velocity update, giving better configurational sampling vs the older
    # Langevin integrator at the same timestep.
    integrator_min = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PS   / unit.picosecond,
        TIMESTEP_FS   * unit.femtoseconds,
    )
    sim_min = app.Simulation(pdb.topology, system_nvt, integrator_min, platform)
    sim_min.context.setPositions(pdb.positions)

    state_pre = sim_min.context.getState(getEnergy=True)
    pe_pre = state_pre.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"  Pre-minimization PE  : {pe_pre:>12.1f} kJ/mol")

    t0 = time.time()
    sim_min.minimizeEnergy(
        tolerance=     MIN_TOLERANCE_KJ * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations= MIN_MAX_ITER,
    )
    t_min = time.time() - t0

    state_post = sim_min.context.getState(getEnergy=True, getPositions=True)
    pe_post = state_post.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"  Post-minimization PE : {pe_post:>12.1f} kJ/mol")
    print(f"  Energy drop          : {pe_pre - pe_post:>12.1f} kJ/mol")
    print(f"  Wall time            : {t_min:.1f} s")

    min_pdb = output_dir / f"{prefix}_{label}_minimized.pdb"
    with open(min_pdb, "w") as f:
        app.PDBFile.writeFile(pdb.topology, state_post.getPositions(), f)
    print(f"  Saved                : {min_pdb}")

    # ── Step 2: NVT equilibration ─────────────────────────────────────────
    section(f"[{label}] Step 2: NVT — {NVT_NS} ns at {TEMPERATURE_K} K")
    print(f"  Peptide positional restraints remain active during NVT.")
    print(f"  Water relaxes freely around the restrained backbone.\n")

    integrator_nvt = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PS   / unit.picosecond,
        TIMESTEP_FS   * unit.femtoseconds,
    )
    integrator_nvt.setRandomNumberSeed(42)
    sim_nvt = app.Simulation(pdb.topology, system_nvt, integrator_nvt, platform)

    nvt_log = output_dir / f"{prefix}_{label}_nvt.log"
    nvt_dcd = output_dir / f"{prefix}_{label}_nvt.dcd"
    nvt_chk = output_dir / f"{prefix}_{label}_nvt.chk"

    # Remove stale temp files from previous crashed checkpoint writes
    for tmp_f in [str(nvt_chk) + ".0.tmp", str(nvt_chk) + ".1.tmp"]:
        if os.path.exists(tmp_f):
            os.remove(tmp_f)
            print(f"  Removed stale temp file: {tmp_f}")

    nvt_resume = False
    steps_done = 0
    steps_remaining = NVT_STEPS

    if nvt_chk.exists() and nvt_chk.stat().st_size > 0:
        sim_nvt.loadCheckpoint(str(nvt_chk))
        steps_done = sim_nvt.currentStep
        steps_remaining = NVT_STEPS - steps_done
        nvt_resume = True
        if steps_remaining <= 0:
            print(f"  NVT already complete ({steps_done:,} steps in checkpoint). Skipping.")
            nvt_end_state = sim_nvt.context.getState(
                getEnergy=True, getPositions=True, getVelocities=True
            )
        else:
            print(
                f"  Resuming NVT from checkpoint at step {steps_done:,} "
                f"({steps_done / STEPS_PER_PS / 1000:.3f} ns). "
                f"Remaining: {steps_remaining:,} steps "
                f"({steps_remaining / STEPS_PER_PS / 1000:.3f} ns)."
            )
    else:
        # Fresh start: seed positions from minimized state
        sim_nvt.context.setPositions(state_post.getPositions())
        sim_nvt.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 42)
        print(f"  Starting NVT from scratch ({NVT_STEPS:,} steps).")

    if not (nvt_resume and steps_remaining <= 0):
        log_mode = "a" if (nvt_resume and nvt_log.exists()) else "w"
        sim_nvt.reporters.append(app.StateDataReporter(
            str(nvt_log), LOG_STEPS,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, progress=True, remainingTime=True,
            totalSteps=NVT_STEPS, separator="\t", append=(log_mode == "a"),
        ))
        sim_nvt.reporters.append(app.StateDataReporter(
            sys.stdout, LOG_STEPS * 10,
            step=True, time=True, temperature=True, progress=True,
            remainingTime=True, totalSteps=NVT_STEPS, separator="\t",
        ))
        sim_nvt.reporters.append(app.DCDReporter(
            str(nvt_dcd), DCD_STEPS,
            append=(nvt_resume and nvt_dcd.exists()),
        ))
        sim_nvt.reporters.append(app.CheckpointReporter(str(nvt_chk), CHK_STEPS))

        print(f"  Log → {nvt_log}  |  Traj → {nvt_dcd}  |  Ckpt → {nvt_chk}")
        print(f"  Running {steps_remaining:,} steps...\n")

        t0 = time.time()
        sim_nvt.step(steps_remaining)
        t_nvt = time.time() - t0

        nvt_end_state = sim_nvt.context.getState(
            getEnergy=True, getPositions=True, getVelocities=True
        )
        print(f"\n  NVT complete in {t_nvt/60:.1f} min")

    ke_nvt = nvt_end_state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    T_nvt = 2 * ke_nvt / (DOF * kB)
    print(f"  Final instantaneous T : {T_nvt:.1f} K  (target: {TEMPERATURE_K} K)")
    if abs(T_nvt - TEMPERATURE_K) > 20:
        print("  WARNING: T deviates >20 K — consider extending NVT or checking thermostat.")

    # ── Step 3: NPT equilibration ─────────────────────────────────────────
    section(f"[{label}] Step 3: NPT — {NPT_NS} ns at {TEMPERATURE_K} K / {PRESSURE_BAR} bar")
    print("  Deserializing fresh system for NPT (no positional restraints).")
    print("  Adding Monte Carlo Barostat (volume moves every 25 steps).\n")

    # CRITICAL: always deserialize a fresh system for NPT.
    # system_nvt has an extra CustomExternalForce (positional restraints).
    # Re-using it for NPT would keep those restraints active.
    # XmlSerializer round-trips the System exactly — same forces, same parameters,
    # same atom order — but as a new Python object with no shared memory.
    with open(xml_path) as f:
        system_npt = XmlSerializer.deserialize(f.read())

    # MonteCarloBarostat: isotropic volume moves every 25 steps, Metropolis accept.
    # Correct for liquid-phase peptide-in-water; no piston mass to tune.
    system_npt.addForce(mm.MonteCarloBarostat(
        PRESSURE_BAR  * unit.bar,
        TEMPERATURE_K * unit.kelvin,
        25,
    ))

    DOF_npt = compute_dof(system_npt)

    integrator_npt = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PS   / unit.picosecond,
        TIMESTEP_FS   * unit.femtoseconds,
    )
    integrator_npt.setRandomNumberSeed(42)
    sim_npt = app.Simulation(pdb.topology, system_npt, integrator_npt, platform)

    npt_log = output_dir / f"{prefix}_{label}_npt.log"
    npt_dcd = output_dir / f"{prefix}_{label}_npt.dcd"
    npt_chk = output_dir / f"{prefix}_{label}_npt.chk"

    for tmp_f in [str(npt_chk) + ".0.tmp", str(npt_chk) + ".1.tmp"]:
        if os.path.exists(tmp_f):
            os.remove(tmp_f)
            print(f"  Removed stale temp file: {tmp_f}")

    npt_resume = False
    npt_steps_done = 0
    npt_steps_remaining = NPT_STEPS

    if npt_chk.exists() and npt_chk.stat().st_size > 0:
        sim_npt.loadCheckpoint(str(npt_chk))
        npt_steps_done = sim_npt.currentStep
        npt_steps_remaining = NPT_STEPS - npt_steps_done
        npt_resume = True
        if npt_steps_remaining <= 0:
            print(f"  NPT already complete ({npt_steps_done:,} steps in checkpoint). Skipping step().")
        else:
            print(
                f"  Resuming NPT from checkpoint at step {npt_steps_done:,} "
                f"({npt_steps_done / STEPS_PER_PS / 1000:.3f} ns). "
                f"Remaining: {npt_steps_remaining:,} steps."
            )
    else:
        # Seed from end of NVT
        sim_npt.context.setPositions(nvt_end_state.getPositions())
        sim_npt.context.setVelocities(nvt_end_state.getVelocities())
        sim_npt.context.setPeriodicBoxVectors(*nvt_end_state.getPeriodicBoxVectors())

    npt_log_mode = "a" if (npt_resume and npt_log.exists()) else "w"
    sim_npt.reporters.append(app.StateDataReporter(
        str(npt_log), LOG_STEPS,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, density=True, progress=True, remainingTime=True,
        totalSteps=NPT_STEPS, separator="\t", append=(npt_log_mode == "a"),
    ))
    sim_npt.reporters.append(app.StateDataReporter(
        sys.stdout, LOG_STEPS * 10,
        step=True, time=True, temperature=True, density=True,
        progress=True, remainingTime=True, totalSteps=NPT_STEPS, separator="\t",
    ))
    sim_npt.reporters.append(app.DCDReporter(
        str(npt_dcd), DCD_STEPS,
        append=(npt_resume and npt_dcd.exists()),
    ))
    sim_npt.reporters.append(app.CheckpointReporter(str(npt_chk), CHK_STEPS))

    print(f"  Log → {npt_log}  |  Traj → {npt_dcd}  |  Ckpt → {npt_chk}")
    print(f"  Running {npt_steps_remaining:,} steps...\n")

    t0 = time.time()
    if npt_steps_remaining > 0:
        sim_npt.step(npt_steps_remaining)
    t_npt = time.time() - t0

    # ── Convergence checks ────────────────────────────────────────────────
    section(f"[{label}] Convergence Checks")
    npt_end = sim_npt.context.getState(getEnergy=True, getPositions=True)
    ke_npt = npt_end.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    T_final = 2 * ke_npt / (DOF_npt * kB)
    density = compute_density_kgm3(system_npt, npt_end)

    print(f"\n  NPT complete in {t_npt/60:.1f} min")
    print(f"  {'Metric':<38} {'Value':>10}   {'Target':>10}   Status")
    print(f"  {'-'*68}")

    t_ok = abs(T_final - TEMPERATURE_K) < 15
    print(
        f"  {'Final instantaneous T (K)':<38} {T_final:>10.1f}   {TEMPERATURE_K:>10.1f}   "
        f"{'OK' if t_ok else 'WARN extend NVT/NPT'}"
    )
    d_ok = 970 < density < 1030
    print(
        f"  {'Final density (kg/m³)':<38} {density:>10.1f}   {'~1000':>10}   "
        f"{'OK' if d_ok else 'WARN extend NPT'}"
    )

    try:
        temps = []
        with open(npt_log) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                t_key = [k for k in row if "Temperature" in k]
                if t_key:
                    temps.append(float(row[t_key[0]]))
        if temps:
            mean_T = float(np.mean(temps))
            std_T = float(np.std(temps))
            mt_ok = abs(mean_T - TEMPERATURE_K) < 5
            print(
                f"  {'Mean NPT temperature (K)':<38} {mean_T:>10.1f}   {TEMPERATURE_K:>10.1f}   "
                f"{'OK' if mt_ok else 'WARN'}"
            )
            print(
                f"  {'NPT temperature std dev (K)':<38} {std_T:>10.1f}   {'< 15':>10}   "
                f"{'OK' if std_T < 15 else 'WARN'}"
            )
    except Exception as e:
        print(f"  (Log parse skipped: {e})")

    if _MDTRAJ_AVAILABLE:
        try:
            traj = md.load(str(npt_dcd), top=str(pdb_path))
            bb_idx = traj.topology.select("protein and backbone")
            traj_bb = traj.atom_slice(bb_idx)
            rmsd = md.rmsd(traj_bb, traj_bb, 0)
            r_ok = float(rmsd[-1]) < 0.5
            print(
                f"  {'Backbone RMSD vs start, final (nm)':<38} {float(rmsd[-1]):>10.3f}   {'< 0.5':>10}   "
                f"{'OK' if r_ok else 'WARN large drift'}"
            )
        except Exception as e:
            print(f"  (RMSD check skipped: {e})")
    else:
        print("  (RMSD check skipped: mdtraj not available)")

    # ── Save final NPT state ──────────────────────────────────────────────
    final_pdb = output_dir / f"{prefix}_{label}_npt_final.pdb"
    final_xml = output_dir / f"{prefix}_{label}_npt_final.xml"

    # enforcePeriodicBox=True wraps molecules back into the primary box.
    final_state = sim_npt.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    with open(final_pdb, "w") as f:
        app.PDBFile.writeFile(pdb.topology, final_state.getPositions(), f)

    # Serialize the system WITHOUT barostat for Phase 4 NVT production.
    # Read fresh from the original xml_path (no barostat), serialize again.
    with open(xml_path) as f_in:
        system_for_prod = XmlSerializer.deserialize(f_in.read())
    with open(final_xml, "w") as f_out:
        f_out.write(XmlSerializer.serialize(system_for_prod))

    print(f"\n  NPT final PDB : {final_pdb}")
    print(f"  NVT prod XML  : {final_xml}  (no barostat — for Phase 4)")

    # ── Re-solvation (extended conformation only, Kuczera 2021) ──────────
    if label == "extended":
        # Peptide atoms are first in the solvated topology (Modeller order).
        n_pep = peptide_top_orig.getNumAtoms()
        all_nm = final_state.getPositions(asNumpy=True)
        peptide_positions = unit.Quantity(
            np.ascontiguousarray(all_nm[:n_pep], dtype=np.float64),
            unit.nanometer,
        )
        _extended_resolvation_pipeline(
            peptide_top_orig,
            peptide_positions,
            prefix,
            label,
            output_dir,
            platform,
            final_pdb,
            final_xml,
        )

    print(f"  [{label.upper()}] Phase 3 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — PRODUCTION MD
# ─────────────────────────────────────────────────────────────────────────────

def run_production(conf: dict, output_dir: Path, platform) -> None:
    """
    NVT production run: 500 ns per conformation, checkpoint/resume enabled.

    Reads ``{PREFIX}_{label}_npt_final.pdb`` and ``{PREFIX}_{label}_npt_final.xml``
    from *output_dir* and runs PROD_STEPS (= 250,000,000) steps of NVT dynamics
    at TEMPERATURE_K = 330 K.

    Why NVT (not NPT) for production
    ---------------------------------
    After NPT equilibration the box volume is at its equilibrium value. Switching
    to NVT:
    (1) removes barostat overhead (~5% per 25 steps);
    (2) eliminates box-volume fluctuations that add noise to structural observables;
    (3) is the correct ensemble for computing Helmholtz free energy ΔA_fold
        (PΔV correction ≈ 0.001 kcal/mol, negligible);
    (4) matches Kuczera 2021 methodology explicitly ("NVT conditions for
        simplicity and efficiency").

    prod_top.pdb — the critical topology file
    ------------------------------------------
    Before the DCDReporter is created, a topology PDB is written from
    ``sim_prod.topology`` and the live context positions. This file reflects
    the **exact** atom ordering of the DCD being written. It is the ONLY safe
    MDTraj topology for this production DCD:
    - Pre-resolvation PDBs have wrong atom counts (re-solvated box differs).
    - Re-reading via ``app.PDBFile()`` gives the solvated-box topology, which
      may have a different atom count if re-solvation was performed.
    - Using the wrong topology with MDTraj gives either an immediate
      ``ValueError`` (atom count mismatch) or silently wrong torsion angles
      (atom order mismatch with same count — the dangerous case).

    On resume, the topology PDB is NOT re-written because the existing file
    already matches the DCD that has been partially written.

    Checkpoint/resume logic
    -----------------------
    1. Check if ``{PREFIX}_{label}_prod.chk`` exists and is non-empty.
    2. If yes: load checkpoint, read ``currentStep``, compute remaining steps.
    3. If remaining ≤ 0: print "already complete" and return.
    4. If no checkpoint: load topology + positions from NPT final PDB via
       **MDTraj** when available (avoids ``PDBFile`` merging duplicate H names
       vs the XML particle count), draw velocities from Maxwell-Boltzmann at
       TEMPERATURE_K, set box vectors from the PDB unit cell.

    Parameters
    ----------
    conf : dict
        Must contain ``label`` key and optionally ``prefix`` key.
        The PDB and XML paths are constructed from output_dir and these keys.
    output_dir : Path
        Directory containing the NPT final files and where production output
        files will be written.
    platform : openmm.Platform
        OpenMM computation platform (from ``get_best_platform()``).
    """
    label = conf["label"]
    prefix = conf.get("prefix", "SYS")
    pdb_path = output_dir / f"{prefix}_{label}_npt_final.pdb"
    xml_path = output_dir / f"{prefix}_{label}_npt_final.xml"

    section(f"{prefix} [{label.upper()}] — Phase 4: NVT Production ({PROD_US} µs)")
    print(f"  Input PDB : {pdb_path}")
    print(f"  Input XML : {xml_path}")
    print(f"  Duration  : {PROD_US} µs  ({PROD_STEPS:,} steps at {TIMESTEP_FS} fs)")
    print(f"  Temp      : {TEMPERATURE_K} K  (NVT — volume fixed from NPT equilibration)")
    print(f"  DCD frame : every {DCD_INTERVAL_PS} ps  ({DCD_STEPS:,} steps)")

    for path in [pdb_path, xml_path]:
        if not path.exists():
            print(f"\n  ERROR: {path} not found. Run Phase 3 first.")
            return

    with open(xml_path) as f:
        system_prod = XmlSerializer.deserialize(f.read())

    N = system_prod.getNumParticles()
    prod_top, positions0, box_vecs = _npt_final_topology_positions_box(pdb_path, N)
    DOF = compute_dof(system_prod)
    kB = 8.314e-3

    integrator_prod = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PS   / unit.picosecond,
        TIMESTEP_FS   * unit.femtoseconds,
    )
    # Different seed from equilibration to decorrelate random noise sequences
    integrator_prod.setRandomNumberSeed(99)

    sim_prod = app.Simulation(prod_top, system_prod, integrator_prod, platform)

    prod_log = output_dir / f"{prefix}_{label}_prod.log"
    prod_dcd = output_dir / f"{prefix}_{label}_prod.dcd"
    prod_chk = output_dir / f"{prefix}_{label}_prod.chk"

    # Remove stale temp files
    for tmp_f in [str(prod_chk) + ".0.tmp", str(prod_chk) + ".1.tmp"]:
        if os.path.exists(tmp_f):
            os.remove(tmp_f)

    prod_resume = False
    steps_done = 0
    steps_remaining = PROD_STEPS

    if prod_chk.exists() and prod_chk.stat().st_size > 0:
        sim_prod.loadCheckpoint(str(prod_chk))
        steps_done = sim_prod.currentStep
        steps_remaining = PROD_STEPS - steps_done
        prod_resume = True
        if steps_remaining <= 0:
            print(f"\n  Production already complete ({steps_done:,} steps). Skipping.")
            return
        print(
            f"\n  Resuming from checkpoint: step {steps_done:,} "
            f"({steps_done / STEPS_PER_PS / 1e6:.4f} µs). "
            f"Remaining: {steps_remaining:,} steps "
            f"({steps_remaining / STEPS_PER_PS / 1e6:.4f} µs)."
        )
    else:
        # Seed from Phase 3 NPT final state (topology/positions via MDTraj when
        # available — avoids PDBFile dropping duplicate-named hydrogens).
        sim_prod.context.setPositions(positions0)
        sim_prod.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 99)
        if box_vecs is None:
            raise ValueError(
                f"No periodic box vectors found in {pdb_path}. "
                "Re-run Phase 3 to regenerate the NPT final PDB with a CRYST1 record."
            )
        sim_prod.context.setPeriodicBoxVectors(*box_vecs)
        print(f"\n  Starting production from scratch ({PROD_STEPS:,} steps).")

    # ── WRITE TOPOLOGY PDB BEFORE STARTING THE DCD REPORTER ──────────────
    # This must happen BEFORE adding reporters. The topology PDB reflects the
    # exact atom ordering of sim_prod.topology at the moment DCDReporter starts.
    # On resume, the existing file already matches the DCD.
    prod_top_pdb = output_dir / f"{prefix}_{label}_prod_top.pdb"
    if not prod_resume:
        with open(prod_top_pdb, "w") as _f:
            app.PDBFile.writeFile(
                sim_prod.topology,
                sim_prod.context.getState(getPositions=True).getPositions(),
                _f,
            )
        print(f"  Topology PDB : {prod_top_pdb}")
        print(f"  (Use this — and ONLY this — as the MDTraj topology for the production DCD)")

    log_mode = "a" if (prod_resume and prod_log.exists()) else "w"
    sim_prod.reporters.append(app.StateDataReporter(
        str(prod_log), LOG_STEPS,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, progress=True, remainingTime=True,
        totalSteps=PROD_STEPS, separator="\t", append=(log_mode == "a"),
    ))
    sim_prod.reporters.append(app.StateDataReporter(
        sys.stdout, LOG_STEPS * 50,   # print to screen every 500 ps
        step=True, time=True, temperature=True, progress=True,
        remainingTime=True, totalSteps=PROD_STEPS, separator="\t",
    ))
    sim_prod.reporters.append(app.DCDReporter(
        str(prod_dcd), DCD_STEPS,
        append=(prod_resume and prod_dcd.exists()),
    ))
    sim_prod.reporters.append(app.CheckpointReporter(str(prod_chk), CHK_STEPS))

    print(f"  Log  : {prod_log}")
    print(f"  Traj : {prod_dcd}  (frame every {DCD_INTERVAL_PS} ps)")
    print(f"  Ckpt : {prod_chk}  (every {CHK_INTERVAL_PS} ps)")
    print(f"  Running {steps_remaining:,} steps...\n")

    t0 = time.time()
    sim_prod.step(steps_remaining)
    t_prod = time.time() - t0

    prod_end = sim_prod.context.getState(getEnergy=True)
    ke_prod = prod_end.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    T_final = 2 * ke_prod / (DOF * kB)
    ns_run = steps_remaining / STEPS_PER_PS / 1000
    ns_per_day = ns_run / (t_prod / 86400)

    print(f"\n  Production complete in {t_prod/3600:.2f} h  ({ns_per_day:.0f} ns/day)")
    print(f"  Final instantaneous T : {T_final:.1f} K  (target: {TEMPERATURE_K} K)")
    print(f"  [{label.upper()}] Phase 4 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_system(
    prefix: str,
    sequence: list[str],
    output_dir: Path,
    phases: str = "all",
) -> None:
    """
    Orchestrate the complete simulation pipeline for one peptide system.

    Runs all requested phases in order: build → solvate → equilibrate → production.
    Each phase checks for existing output files and skips completed phases
    (idempotent: safe to re-run after interruption).

    Two starting conformations are simulated:
      - extended: φ = −120°, ψ = +140° (maximally extended β-strand / PPII)
      - helix:    φ = −57°,  ψ = −47°  (ideal α-helix)

    Running from both conformations and checking that the final FES (helix fraction,
    basin locations) converges from both starting points is the standard convergence
    test for helix-coil simulations (Kuczera 2021 dual-start methodology).

    Idempotency details
    -------------------
    - Phase 1 (build): skips if ``{prefix}_{label}.pdb`` already exists.
    - Phase 2 (solvate): skips if ``{prefix}_{label}.xml`` already exists.
    - Phase 3 (equil): skips both conformations if both ``_npt_final.pdb`` and
      ``_npt_final.xml`` exist. Within a conformation, each sub-phase has its
      own checkpoint; ``run_equilibration`` handles per-step resume.
    - Phase 4 (prod): per-conformation checkpoint; ``run_production`` handles
      per-step resume.

    Parameters
    ----------
    prefix : str
        System name used as filename prefix, e.g. "A10" or "A8Q2".
    sequence : list[str]
        List of three-letter residue codes (N→C, no caps), e.g. ["ALA"]*10.
    output_dir : Path
        Root directory for all output files. Created if it does not exist.
    phases : str
        Which phases to run: "build", "equil", "prod", or "all".
        "all" runs all four phases in order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_windows_checkpoint_patch()

    section(f"SYSTEM: {prefix}  |  Sequence: {'-'.join(sequence)}")
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Phases requested : {phases}")
    print(f"  Conformations    : extended (φ=−120°, ψ=+140°)  +  helix (φ=−57°, ψ=−47°)")
    print(f"  Temperature      : {TEMPERATURE_K} K")
    print(
        f"  Production       : {int(PROD_US * 1000)} ns NVT per replica "
        f"({PROD_US:g} µs; {PROD_STEPS:,} steps)"
    )

    # φ/ψ / filenames match ``src.config.CONFORMATIONS`` (single source of truth).
    conformation_defs = [(phi, psi, lab) for phi, psi, lab in CONFORMATIONS]

    # ── Phase 1: Build ────────────────────────────────────────────────────
    if phases in ("build", "all"):
        section(f"{prefix} — PHASE 1: Build Structures (ACE/CT3 capped)")
        print(f"  Force field : CHARMM36m — ACE patch (N-term) + CT3 patch (C-term)")
        print(f"  PeptideBuilder GLN bug: bypassed via GlnGeo() direct instantiation\n")

    pep_tops: dict[str, object] = {}
    pep_pos: dict[str, object] = {}

    for phi, psi, label in conformation_defs:
        out_pdb = output_dir / f"{prefix}_{label}.pdb"

        if phases in ("build", "all"):
            if out_pdb.exists():
                print(f"  [{label}] PDB already exists at {out_pdb}, skipping build.")
                # Rebuild topology in memory (we still need it for prepare_system)
                top, pos = build_peptide(sequence, phi, psi, out_pdb, label, prefix)
            else:
                top, pos = build_peptide(sequence, phi, psi, out_pdb, label, prefix)
        else:
            # Not building — create topology anyway for downstream phases
            top, pos = build_peptide(sequence, phi, psi, out_pdb, label, prefix)

        pep_tops[label] = top
        pep_pos[label] = pos

    if phases in ("build",):
        print(f"\n[{prefix}] Phase 1 complete. Exiting as requested (--phase build).")
        return

    # ── Phase 2: Solvate ──────────────────────────────────────────────────
    if phases in ("equil", "all"):
        section(f"{prefix} — PHASE 2: Parameterize & Solvate")
        print(f"  Force field : charmm36_2024.xml (CHARMM36m backbone torsions)")
        print(f"  Water model : charmm36_2024/water.xml (modified TIP3P with H LJ terms)")
        print(f"  Padding     : {PADDING_NM} nm  |  Ionic strength: {IONIC_STRENGTH_M} M NaCl\n")

    systems: dict[str, object] = {}
    tops: dict[str, object] = {}
    pos_dict: dict[str, object] = {}

    for phi, psi, label in conformation_defs:
        solvated_pdb = output_dir / f"{prefix}_{label}_solvated.pdb"
        xml_out = output_dir / f"{prefix}_{label}.xml"

        if phases in ("equil", "all"):
            if xml_out.exists() and solvated_pdb.exists():
                print(f"  [{label}] Solvated XML already exists at {xml_out}, skipping solvation.")
                # Load topology for equilibration (use PDBFile — solvated PDB is safe to re-read
                # for topology, we just need the top; actual createSystem uses in-memory peptide top)
                _pdb_obj = app.PDBFile(str(solvated_pdb))
                tops[label] = _pdb_obj.topology
                pos_dict[label] = _pdb_obj.positions
            else:
                sys_obj, top_solv, pos_solv, dof_solv = prepare_system(
                    pep_tops[label], pep_pos[label],
                    solvated_pdb, xml_out, label, PADDING_NM,
                )
                systems[label] = sys_obj
                tops[label] = top_solv
                pos_dict[label] = pos_solv

    if phases in ("equil", "all"):
        print(f"[{prefix}] Phase 2 complete.\n")

    # ── Phase 3: Equilibration ────────────────────────────────────────────
    if phases in ("equil", "all"):
        all_equil_done = all(
            equil_phase3_complete(output_dir, prefix, lab)
            for _, _, lab in conformation_defs
        )

        if all_equil_done:
            section(f"{prefix} — PHASE 3: EQUILIBRATION (skipped; cached results found)")
            print(f"  All Phase 3 final PDB/XML outputs already exist in {output_dir}.")
            print(
                f"  Extended: also requires `{prefix}_extended_resolv_complete` after re-solvate.\n"
                f"  Delete *_npt_final.* and/or that marker to force re-run.\n"
            )
        else:
            section(f"{prefix} — PHASE 3: EQUILIBRATION")
            print(f"  Timestep    : {TIMESTEP_FS} fs")
            print(f"  Temperature : {TEMPERATURE_K} K")
            print(f"  Pressure    : {PRESSURE_BAR} bar")
            print(f"  NVT         : {NVT_NS} ns  (restrained)")
            print(f"  NPT         : {NPT_NS} ns  (unrestrained, barostat on)")
            print(f"  Re-solvate  : extended conformation → {NPT_RESOLVATE_NS} ns NPT after re-solvation")
            print(f"  Restraint k : {RESTRAINT_K} kJ/mol/nm²  (minimization + NVT only)\n")

            platform = get_best_platform()

            for phi, psi, label in conformation_defs:
                if equil_phase3_complete(output_dir, prefix, label):
                    print(f"  [{label}] Phase 3 already complete (files exist). Skipping.")
                    continue

                conf = {
                    "label":   label,
                    "pdb":     str(output_dir / f"{prefix}_{label}_solvated.pdb"),
                    "xml":     str(output_dir / f"{prefix}_{label}.xml"),
                    "pep_top": pep_tops[label],
                    "pep_pos": pep_pos[label],
                    "prefix":  prefix,
                }
                run_equilibration(conf, output_dir, platform)

            print(f"\n[{prefix}] Phase 3 complete.\n")

    # ── Phase 4: Production ───────────────────────────────────────────────
    if phases in ("prod", "all"):
        section(f"{prefix} — PHASE 4: NVT PRODUCTION RUNS")
        print(f"  Duration per conformation : {PROD_US} µs  ({PROD_STEPS:,} steps)")
        print(f"  Ensemble                  : NVT at {TEMPERATURE_K} K (volume fixed)")
        print(f"  DCD output                : every {DCD_INTERVAL_PS} ps")
        print(f"  Checkpoint                : every {CHK_INTERVAL_PS} ps (resume-safe)\n")

        platform = get_best_platform()

        wall_t0 = time.time()
        for phi, psi, label in conformation_defs:
            conf = {"label": label, "prefix": prefix}
            run_production(conf, output_dir, platform)

        total_h = (time.time() - wall_t0) / 3600
        section(f"{prefix} ALL PHASES COMPLETE")
        print(f"  Total Phase 4 wall time : {total_h:.2f} h")
        print(f"\n  Output trajectories:")
        for _, _, label in conformation_defs:
            dcd = output_dir / f"{prefix}_{label}_prod.dcd"
            top_pdb = output_dir / f"{prefix}_{label}_prod_top.pdb"
            print(f"    {dcd.name}  ({int(PROD_US*1000)} ns NVT at {TEMPERATURE_K} K)")
            print(f"    MDTraj topology: {top_pdb.name}  ← use ONLY this file")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

_ALL_SENTINEL = "__ALL__"  # internal: single run_system(..., phases="all") pass


def parse_phase_chain(phase_spec: str) -> list[str]:
    """Parse ``--phase`` string into an ordered list of steps.

    - ``all`` → ``[_ALL_SENTINEL]`` (one efficient ``run_system(..., "all")`` call).
    - ``equil,prod`` → ``["equil", "prod"]`` (equilibration then production in one process).
    - ``build,equil,prod`` → ``[_ALL_SENTINEL]`` (equivalent to ``all``).

    Raises
    ------
    ValueError
        On unknown tokens or invalid combination (e.g. ``all`` mixed with others).
    """
    raw = (phase_spec or "").strip()
    if not raw:
        return [_ALL_SENTINEL]

    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = {"build", "equil", "prod", "all"}
    for p in parts:
        if p not in valid:
            raise ValueError(
                f"Unknown phase '{p}'.  Valid: build, equil, prod, all — or comma-separated "
                f"chains such as equil,prod."
            )

    if "all" in parts:
        if len(parts) > 1:
            raise ValueError(
                "'all' cannot be combined with other phases; use --phase all alone."
            )
        return [_ALL_SENTINEL]

    # Preserve order, drop duplicates
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)

    if out == ["build", "equil", "prod"]:
        return [_ALL_SENTINEL]
    return out


def main() -> None:
    """
    Command-line interface for the simulation pipeline.

    Examples
    --------
    Run all phases for A10::

        python -m src.simulate --system A10 --phase all

    Run only equilibration for A8Q2 into a custom directory::

        python -m src.simulate --system A8Q2 --phase equil --output_dir /scratch/A8Q2

    Run both systems through the complete pipeline::

        python -m src.simulate --system both --phase all --output_dir /scratch/btp

    Chain equilibration then production in one process (no second command)::

        python -u -m src.simulate --system both --phase equil,prod --output_dir runs/btp
    """
    if not _OPENMM_AVAILABLE:
        print(f"ERROR: Required packages not available: {_IMPORT_ERROR}")
        print("Install with: pip install openmm pdbfixer PeptideBuilder biopython")
        sys.exit(1)

    # Windows log redirection (Start-Process, pipes) often uses cp1252 and crashes
    # on Greek letters (phi/psi) and other Unicode in print(). Force UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass

    parser = argparse.ArgumentParser(
        description="btp_2_compass MD simulation pipeline (Phases 1–4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.simulate --system A10  --phase all
  python -m src.simulate --system A8Q2 --phase equil --output_dir /scratch/A8Q2
  python -m src.simulate --system both --phase all   --output_dir /scratch/btp
  python -u -m src.simulate --system both --phase equil,prod --output_dir ./runs
        (equil then prod automatically; idempotent skips)
        """,
    )
    parser.add_argument(
        "--system",
        choices=["A10", "A8Q2", "both"],
        required=True,
        help="Which peptide system to simulate.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        metavar="PHASES",
        help=(
            "One of: build, equil, prod, all — OR a comma-separated chain, e.g. "
            "equil,prod to run equilibration and then production in one process "
            "(picks up automatically after equil; re-run safe)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/btp_equil",
        help=(
            "Root output directory (created if needed). "
            "Per-system runs write under ``<output_dir>/A10`` and ``.../A8Q2``. "
            "Default matches downstream scripts that expect outputs/btp_equil/."
        ),
    )
    args = parser.parse_args()

    try:
        phase_chain = parse_phase_chain(args.phase)
    except ValueError as exc:
        parser.error(str(exc))

    output_root = Path(args.output_dir)

    systems_to_run = (
        ["A10", "A8Q2"] if args.system == "both" else [args.system]
    )

    for sys_name in systems_to_run:
        sequence = SEQUENCES[sys_name]
        sys_output = output_root / sys_name

        if phase_chain == [_ALL_SENTINEL]:
            run_system(
                prefix=sys_name,
                sequence=sequence,
                output_dir=sys_output,
                phases="all",
            )
        else:
            for i, step in enumerate(phase_chain, start=1):
                print(
                    f"\n{'=' * 60}\n"
                    f"  CHAINED PIPELINE  {sys_name}  |  step {i}/{len(phase_chain)}: {step}\n"
                    f"{'=' * 60}\n"
                )
                run_system(
                    prefix=sys_name,
                    sequence=sequence,
                    output_dir=sys_output,
                    phases=step,
                )


if __name__ == "__main__":
    main()
