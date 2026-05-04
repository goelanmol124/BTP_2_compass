# Complete Experiment Protocol: TICA Collective Variable Transfer Between Helix-Coil Peptides

**Project:** BTP (Bachelor's Thesis Project) — Semester project  
**Version:** 1.0 — April 2026  
**Systems:** Ac-Ala₁₀-NHMe (A10, deca-alanine) and Ac-Ala₈Gln₂-NHMe (A8Q2)  
**Central scientific question:** Does a TICA collective variable trained on A10 correctly identify the helix-coil reaction coordinate of A8Q2, and does it accelerate convergence of the A8Q2 free energy surface beyond random sampling?

---

## Table of Contents

1. [Scientific Background and Motivation](#1-scientific-background-and-motivation)
2. [Systems and Rationale](#2-systems-and-rationale)
3. [Computational Infrastructure](#3-computational-infrastructure)
4. [Phase 1 — Structure Preparation](#4-phase-1--structure-preparation)
5. [Phase 2 — Parameterization and Solvation](#5-phase-2--parameterization-and-solvation)
6. [Phase 3 — Equilibration Protocol](#6-phase-3--equilibration-protocol)
7. [Phase 4 — Ground Truth Production MD](#7-phase-4--ground-truth-production-md)
8. [Phase 5 — Ground Truth Analysis](#8-phase-5--ground-truth-analysis)
9. [Phase 6 — TICA Training on A10](#9-phase-6--tica-training-on-a10)
10. [Phase 7 — TICA Transferability Assessment](#10-phase-7--tica-transferability-assessment)
11. [Phase 8 — Adaptive Seeding Experiment](#11-phase-8--adaptive-seeding-experiment)
12. [Phase 9 — Acceleration Metrics and Final Analysis](#12-phase-9--acceleration-metrics-and-final-analysis)
13. [Distributed Execution on RunPod](#13-distributed-execution-on-runpod)
14. [Known Limitations and Caveats](#14-known-limitations-and-caveats)
15. [Expected Results and Scientific Claims](#15-expected-results-and-scientific-claims)
16. [Complete Reference List](#16-complete-reference-list)

---

## 1. Scientific Background and Motivation

### 1.1 The Helix-Coil Transition

The α-helix ↔ random coil transition in alanine-rich peptides is one of the most fundamental processes in protein biophysics. It involves:

- Formation/breaking of i→i+4 backbone C=O···H-N hydrogen bonds (α-helical register requires 6 such bonds for a 10-residue peptide)
- Cooperative conformational change across multiple residues simultaneously
- A characteristic phi/psi torsion signature: helix state at φ ≈ −62°, ψ ≈ −41°; coil state spread across the Ramachandran upper-left quadrant (φ ≈ −120° to −60°, ψ ≈ +100° to +160°) with significant PPII population

The helix-coil transition in short alanine-based peptides is remarkably slow. At 300 K, the global relaxation time for a 10-residue alanine peptide is approximately 20–30 ns (interpolated from Kuczera 2021 data: ALA8 τ ≈ 13.2 ns, ALA15 τ ≈ 107 ns). Folding times are τ_fold ≈ 200–350 ns and unfolding times τ_unfold ≈ 25–80 ns. The equilibrium strongly favors the coil state: helix content for capped Ac-Ala₁₀-NHMe at 300 K is approximately 15–20%.

### 1.2 The Collective Variable Transfer Problem

When designing enhanced sampling methods (metadynamics, seeded adaptive MD, OPES), a central requirement is an accurate collective variable (CV) that distinguishes the relevant slow degrees of freedom. For helix-coil peptides, TICA (Time-lagged Independent Component Analysis) trained on unbiased MD trajectories reliably identifies the slow helix-coil coordinate as its first component (TIC-1).

The scientific question is: **when a peptide is mutated by 1–3 residues, does the CV (TICA eigenvector) remain valid?** If yes, one can transfer the CV directly without re-training, dramatically reducing the cost of parameterizing acceleration methods for new sequences. If no, the mutation has changed the reaction coordinate itself — which is itself an important scientific result.

### 1.3 Why This Experiment

The experiment tests CV transferability through a concrete, measurable protocol:
1. Train TICA on 500 ns unbiased MD of A10 (source)
2. Apply the same TICA eigenvectors to A8Q2 (target, 2-residue mutation)
3. Measure whether the transferred CV still identifies the helix-coil coordinate on A8Q2
4. Use the transferred CV in an adaptive seeding loop and measure whether it accelerates convergence of the A8Q2 free energy surface compared to random seeding

The result feeds directly into the one-sentence scientific claim: "A TICA CV trained on Ac-Ala₁₀-NHMe transfers to Ac-Ala₈Gln₂-NHMe with [quantified] geometric fidelity and accelerates FES convergence by [X]× relative to random seeding."

---

## 2. Systems and Rationale

### 2.1 Source System: Ac-Ala₁₀-NHMe (A10, Deca-Alanine)

**Full name:** Acetyl-Alanine-Alanine-Alanine-Alanine-Alanine-Alanine-Alanine-Alanine-Alanine-Alanine-N-methylamide  
**Sequence:** ACE - ALA - ALA - ALA - ALA - ALA - ALA - ALA - ALA - ALA - ALA - NME  
**Total residues including caps:** 12 (ACE + 10 ALA + NME)  
**Total atoms (in vacuo):** 112

**Why A10 is the training system:**
- Most extensively studied polyalanine system in the computational literature
- Gold-standard kinetics data available: Kuczera et al. (2021, 2024) provide k_f, k_u, τ_eq at 300 K from multi-µs CHARMM36m simulations with the same force field we use
- 10 residues is large enough to form a stable α-helix (requires ≥6 i→i+4 H-bonds) but small enough for feasible simulation
- The source CV is expected to be clean: homopolymer with identical residues at every position means TIC-1 should straightforwardly represent the global helix-coil coordinate

### 2.2 Target System: Ac-Ala₈Gln₂-NHMe (A8Q2)

**Full name:** Acetyl-Alanine-Alanine-Glutamine-Alanine-Alanine-Alanine-Alanine-Glutamine-Alanine-Alanine-N-methylamide  
**Sequence:** ACE - ALA - ALA - GLN - ALA - ALA - ALA - ALA - GLN - ALA - ALA - NME  
**Position of mutations:** Residues 3 and 8 (Ala → Gln), symmetric placement  
**Total residues including caps:** 12  

**Why A8Q2 is the target system:**

The ideal test case for CV transfer satisfies: (a) few mutations from the source (1–3), (b) the system still undergoes helix-coil transitions (so a comparison is meaningful), (c) experimental data exist for validation (AAQAA peptides), and (d) the chemistry is standard (no exotic residues requiring special parameterization).

A8Q2 satisfies all four:
- **2 mutations from A10** (Ala3→Gln, Ala8→Gln): minimal perturbation to backbone geometry
- **Glutamine is a helix-compatible residue** with slightly lower helix propensity than alanine; the peptide still undergoes helix-coil transitions
- **AAQAA benchmark:** The canonical reference system (AAQAA)₃ (Ac-(Ala-Ala-Gln-Ala-Ala)₃-NH₂) has been extensively studied both experimentally (Shalongo & Stellwagen NMR: f_helix ≈ 20–30% at 300 K) and computationally (Dokainish & Sugita 2022 CHARMM36m REMD: f_helix ≈ 17 ± 1%). Our A8Q2 is a 10-mer version of this motif.
- **Standard CHARMM36m residue:** Gln requires no extra parameterization

**Gln placement rationale:** Positions 3 and 8 are symmetric (3rd from N-terminus, 3rd from C-terminus). This avoids terminus-capping artifacts and tests the CV at internal positions where the helix is most stable.

**Known CHARMM36m artifact for Gln:** Dokainish & Sugita (2022) documented that CHARMM36m systematically underestimates helix fraction at Gln positions by ~5% compared to neighboring Ala residues, likely due to incomplete side-chain parametrization of the Gln amide. This is a known and reported artifact; our per-residue helix fraction analysis will quantify this.

---

## 3. Computational Infrastructure

### 3.1 Software Stack

| Component | Package | Version | Purpose |
|-----------|---------|---------|---------|
| MD Engine | OpenMM | 8.x | All simulations |
| Force Field | OpenMM 8.x built-in | 8.x | CHARMM36m XML files (`charmm36_2024.xml`) |
| Structure building | PeptideBuilder | latest | Initial peptide geometry |
| Structure fixing | PDBFixer | latest | Topology correction and solvation |
| Trajectory analysis | MDTraj | 1.9.x | RMSD, torsion angles, distances |
| Secondary structure | MDAnalysis | ≥2.8.0 | DSSP (Kabsch-Sander algorithm) |
| TICA | deeptime | latest | Generalized eigenvalue problem (PyEMMA is archived since 2023) |
| Database | MongoDB Atlas | free tier (M0) | Job coordination between pods |
| Numerical | NumPy, SciPy | latest | All analysis |

### 3.2 Hardware

**GPU:** RunPod RTX 4090 (24 GB VRAM, GDDR6X)  
**Why RTX 4090 over A100:**

Published OpenMM 8 community benchmarks (July 2023, openmm/openmm-org#94) show:
- RTX 4090: ~2250 ns/day on `pme` test (~24k atom explicit solvent PME)
- A100 PCIe: ~1276 ns/day on same test
- RTX 4090 at $0.34/hr on RunPod vs A100 at $1.19/hr

RTX 4090 is 76% faster and 3.5× cheaper per hour for this system size. Memory bandwidth (A100 HBM2 vs RTX 4090 GDDR6X) matters less than raw CUDA core throughput for systems of 5,000–16,000 atoms that fit easily in GPU L2 cache. VRAM usage for our system is well below 2 GB; the 24 GB RTX 4090 VRAM is irrelevant for this system size.

**Mixed precision:** OpenMM default (mixed) is used. Forces are computed in single precision; integration is in double precision. This is the validated setting for CHARMM36m + TIP3P by the OpenMM team. Full double precision is only needed for force field development work.

### 3.3 CUDA Setup

Fresh pod setup (Ubuntu container):
```bash
pip install "openmm[cuda12]"   # brings CUDA 12 runtime — no toolkit install needed
python -m openmm.testInstallation  # must show CUDA platform available
```

CHARMM36m force field files are loaded as:

```python
forcefield = app.ForceField("charmm36_2024.xml", "charmm36_2024/water.xml")
```

These XML files ship directly with **OpenMM 8.x core** (added in PR #4916, merged May 2025). They do **not** require the separate `openmmforcefields` package. Both notebooks have been tested and run successfully with this exact path on OpenMM 8.x. The `openmmforcefields` package is not needed for this experiment (it is an alternative route that uses `charmm/charmm36.xml` path prefix after `import openmmforcefields`, but the core OpenMM files are what the notebooks actually use).

### 3.4 Pod Architecture

Two pods, each self-contained. No new pods are spun up mid-experiment.

- **Pod 1:** A10 production → TICA training → transfer metrics → shutdown (~14 hrs)
- **Pod 2:** A8Q2 production → seeding experiment → shutdown (~28 hrs)

MongoDB Atlas (free M0 cluster) serves as the communication channel. Small artifacts (feature arrays, eigenvectors, figures, scores) are uploaded to MongoDB. Large trajectory files (~5 GB each DCD) stay on the pod and are discarded at shutdown.

---

## 4. Phase 1 — Structure Preparation

### 4.1 Capping Rationale

Both peptides are built as Ac-Peptide-NHMe (acetyl N-terminus, N-methylamide C-terminus). This is mandatory.

**Physics of the charged terminus artifact:**

An uncapped peptide has NH₃⁺ (N-terminus, formal charge +1) and COO⁻ (C-terminus, formal charge −1). These create a ~10–30 Debye dipole along the peptide axis. This dipole interacts favorably with the α-helix macrodipole (which points N→C). The result: helix is artificially stabilized by ~1–3 kcal/mol for 10-residue peptides. Since the free energy of helix formation at 300 K is only +0.3–1.0 kcal/mol (weakly unfavorable), a 1–3 kcal/mol artifact completely changes the helix-coil equilibrium.

All serious literature uses capped termini: Kuczera 2021/2024 uses Ac-Ala_n-NH₂; Zhuang 2021 explicitly states "N-terminus and C-terminus were neutralized using the usual capping procedure — acetylated and amidated, respectively"; Angelo Rossi NAMD tutorial PSFgen script uses `first ACE; last CT3`.

Our implementation: We use PeptideBuilder to build the raw backbone, then apply OpenMM's built-in ACE and CT3 residue patches via the CHARMM36m force field templates. The in-memory topology carries CHARMM-correct atom names: CY/OY/CAY on the ACE cap, NT/CAT on the NME cap.

**Critical known issue in PeptideBuilder:** `Geometry.geometry('GLN')` returns `GlyGeo` (glycine geometry) instead of `GlnGeo` (glutamine geometry). This is a documented bug. For A8Q2, positions 3 and 8 are instantiated by calling `Geometry.GlnGeo()` directly, bypassing the broken lookup function. This ensures the correct Gln side-chain bond lengths and angles (CB-CG-CD-OE1/NE2) are used analytically by PeptideBuilder, rather than relying on PDBFixer energy minimization to fix a geometrically wrong starting structure.

### 4.2 Starting Conformations

Two independent starting conformations are used for each system:

**Helix conformation:** φ = −57°, ψ = −47° for all residues  
This corresponds to the standard ideal α-helix geometry. End-to-end distance: ~1.5 nm. After solvation: ~4,851 atoms (helix box).

**Extended conformation:** φ = −120°, ψ = +140° for all residues  
This is the maximally extended β-strand-like structure. End-to-end distance: ~3.4 nm. After solvation: ~16,043 atoms (extended box — much larger due to 1.5 nm padding on a 3.4 nm peptide).

**Why two starting conformations:**

Starting from helix and extended independently and converging to the same FES is the strongest available test of simulation convergence. If a single trajectory starting from the helix gives the same FES as one starting from the extended state, the ergodic hypothesis is satisfied on this timescale. This is the methodology of Kuczera et al. 2021, who used this exact dual-start approach and found that ALA21 was NOT converged even at 20 µs (the two replicas disagreed), while ALA8 showed good agreement at 10 µs.

At our accelerated temperature of 330 K (see Section 7), we expect the two replicas for A10 and A8Q2 to agree by ~200–300 ns, providing the convergence check within the 500 ns production budget.

### 4.3 Output Files Per System

After Phase 1, for each system {A10, A8Q2} and each conformation {helix, extended}:

```
{PREFIX}_{label}_raw.pdb          — raw PeptideBuilder output (no caps, no H)
{PREFIX}_{label}.pdb              — capped structure with full CHARMM topology
```

Atom counts (A10): 112 atoms in vacuo (10 ALA + ACE + NME, all-atom including H).  
Atom counts (A8Q2): slightly more due to Gln side chains (~120 atoms in vacuo).

---

## 5. Phase 2 — Parameterization and Solvation

### 5.1 Force Field Choice

**Force field:** CHARMM36m (`charmm36_2024.xml`, built into OpenMM 8.x)  
**Water model:** CHARMM modified TIP3P (`charmm36_2024/water.xml`, built into OpenMM 8.x)

**Why CHARMM36m:**

CHARMM36m (Huang et al. 2017, Nat. Methods 14:71) is an improvement over CHARMM36 specifically targeting intrinsically disordered proteins (IDPs) and disordered-to-ordered transitions. The "m" modification adds corrections to the CMAP backbone torsion potential that improve the balance between helical and extended conformations. For helix-coil equilibrium studies, it is currently the best-validated all-atom force field:
- Dokainish & Sugita 2022 (CHARMM36m REMD of (AAQAA)₃): f_helix = 17 ± 1% vs experimental 20–30%
- Kuczera 2021: τ_fold, τ_unfold match experimental NMR relaxation times within a factor of 2

**Why CHARMM modified TIP3P (mTIP3P) not standard TIP3P:**

`charmm36_2024/water.xml` (built into OpenMM 8.x) implements CHARMM's modified TIP3P which adds Lennard-Jones terms on the hydrogen atoms (ε_H = 0.046 kcal/mol, r_min_H = 0.449 Å). Standard TIP3P (sTIP3P, Jorgensen 1983) has no LJ on hydrogens. The CHARMM36m force field was parameterized and validated exclusively with mTIP3P (enforced by CHARMM-GUI defaults; Kim et al. 2015). Using standard TIP3P (sTIP3P, Jorgensen 1983), which lacks LJ terms on hydrogens, is inconsistent with the parameterization and known to give poorer helix-coil balance (Best & Hummer 2009 discuss water model sensitivity for helix-coil systems). **Never substitute sTIP3P for mTIP3P with CHARMM36m.**

### 5.2 Nonbonded Parameters

| Parameter | Value | Scientific Basis |
|-----------|-------|-----------------|
| Nonbonded cutoff | 1.2 nm | CHARMM36m parameterization requirement (Kim et al. 2015) |
| Switch start | 1.0 nm | Force-based LJ switching 10→12 Å (CHARMM standard) |
| Electrostatics | PME | Mandatory for periodic charged systems |
| Ewald error tolerance | 0.0005 | Standard; controls PME grid resolution |
| 1-4 scaling | 1.0 (full) | CHARMM standard; NOT AMBER's 0.833 |
| Constraints | HBonds + rigidWater | Enables 2 fs timestep |

**Why 1.2 nm cutoff is mandatory (not optional):**

CHARMM36m was parameterized in CHARMM/NAMD using a **force-based LJ switching function** (FSWITCH) from 10 Å to 12 Å. Force-switching smooths the first derivative of the LJ potential (the force) to zero at the cutoff, eliminating discontinuities in forces that can cause energy drift.

**Important limitation of OpenMM:** OpenMM supports only **potential-based (energy) switching** — it multiplies the LJ energy by a quintic smoothing polynomial `S(r)`. This is fundamentally different from CHARMM's FSWITCH, which modifies the force expression itself. The `CustomNonbondedForce` in `charmm36_2024.xml` handles **NBFIX pair-specific LJ overrides**, not custom switching — it also uses OpenMM's potential-based switching when `switchDistance` is set. There is no turnkey implementation of true CHARMM FSWITCH in standard OpenMM.

**Practical consequence:** Running CHARMM36m in OpenMM with `switchDistance=1.0 nm` uses potential-based switching rather than the original FSWITCH. This introduces a small but systematic deviation from the original parameterization context. Kim et al. (2015) documented that switching function choice can affect phase behavior for lipids (gel-phase artifacts). For helix-coil peptides in water, the effect is expected to be sub-kcal/mol (smaller than for lipid bilayers), but it is a known and unresolvable limitation of the OpenMM platform. We accept it here because: (a) the community standard for CHARMM36m OpenMM simulations uses exactly these settings, (b) the comparison between A10 and A8Q2 is performed with identical settings so systematic errors cancel for relative comparisons, and (c) no published CHARMM36m OpenMM helix-coil study has used a workaround.

### 5.3 Solvation Protocol

**Box geometry:** Cubic periodic box (isotropic NPT equilibration)

**Box padding:** 1.5 nm on all sides

**Why 1.5 nm (not 1.0 nm):**

In the extended conformation the peptide spans ~3.4 nm end-to-end. The periodic image of the chain appears at a distance equal to the box edge length minus the peptide extent. With 1.5 nm padding, the box edge is ~3.4 + 2×1.5 = 6.4 nm, leaving ~1.5 nm of solvent between the peptide and its nearest periodic image. The PME cutoff is 1.2 nm. A safe rule is that the solvent gap must exceed the cutoff at all times during the simulation. With 1.5 nm padding: clearance = 1.5 nm > 1.2 nm. Reducing to 1.0 nm padding would give clearance = 1.0 nm, which is equal to the LJ switch start (1.0 nm), meaning extended conformations would routinely interact with their periodic images through the switch region. This would create spurious self-interaction artifacts that bias the helix-coil equilibrium toward artificially compact states. Best & Hummer (2009) used ~3.5 nm nearest-face spacing for their 15-residue AAQAA system, consistent with our ~1.5 nm padding choice.

**Ions:**
- Ionic strength: 0.15 M NaCl
- Ion type: Na⁺ and Cl⁻
- Neutralize: yes (add one counterion type to neutralize net charge if any)
- Algorithm: OpenMM Modeller.addSolvent with ionicStrength parameter

Actual ion counts (from implemented runs):
- Extended box (A10): 5,301 waters, 14 Na⁺, 14 Cl⁻ → 16,043 total atoms
- Helix box (A10): 1,577 waters, 4 Na⁺, 4 Cl⁻ → 4,851 total atoms

**Re-solvation of the extended conformation (Kuczera 2021 methodology):**

After NPT equilibration of the extended conformation, the box is still much larger than the helix box (~16,000 vs ~5,000 atoms). Running production MD at this size would cost ~3× more per step than necessary and make the two replicas (helix and extended) incomparable in computational cost. Following Kuczera et al. (2021) exactly:

1. Extract the equilibrated peptide coordinates from the final NPT frame
2. Retain only the `n_pep` atom positions (peptide atoms are always first in the topology, as they were added to Modeller before addSolvent)
3. Use the original in-memory topology (not the re-read PDB, which normalizes CHARMM atom names HT1/HT2/HT3 to H/H2/H3 — breaking force field template matching)
4. Re-solvate with 1.2 nm padding (smaller box, comparable to helix box)
5. Run 2 ns NPT to re-converge density on the new box

This produces a post-re-solvation box of ~5,000–7,000 atoms for the extended starting conformation, making it computationally comparable to the helix-start production run.

### 5.4 System Serialization

After solvation and system creation, the OpenMM System object (containing all forces and parameters) is serialized to XML using `XmlSerializer.serialize()`. This XML file is the complete, self-contained description of the potential energy function. It is re-loaded at every subsequent phase, ensuring reproducibility: the same system object is used from Phase 3 through Phase 4 without re-creating it from the force field files.

---

## 6. Phase 3 — Equilibration Protocol

### 6.1 Integrator

**Integrator:** LangevinMiddleIntegrator (LangevinMiddle scheme)  
**Timestep:** 2.0 fs  
**Friction coefficient:** 1.0 ps⁻¹ (Langevin thermostat)

**Why LangevinMiddle:** The LangevinMiddle integrator uses a "middle" scheme where the stochastic force is applied in the middle of each integration step. This scheme has better energy conservation properties than the standard Langevin (before/after) scheme, especially at large timesteps. Tested in Kim et al. (2015) for CHARMM36 OpenMM simulations; friction coefficient of 1 ps⁻¹ was found equivalent to 10 ps⁻¹ for equilibrium sampling.

**Why 2 fs (not 4 fs):** CHARMM36m uses CMAP backbone torsion corrections that involve cross-terms between φ and ψ. At 4 fs with HBonds constraints only, these torsion terms can become a source of energy drift. 2 fs is the universal standard in CHARMM36m literature (all papers in our review use 2 fs).

### 6.2 Energy Minimization

**Method:** OpenMM built-in L-BFGS minimizer (default)  
**Convergence criterion:** default (runs until gradient is negligible)  
**Restraints during minimization:** 1000 kJ/mol/nm² harmonic positional restraints on all heavy atoms (55 atoms for A10)

Pre-minimization potential energy: approximately −95,000 kJ/mol (from PeptideBuilder raw structure with some clashes)  
Post-minimization potential energy: approximately −280,000 kJ/mol  
Energy drop: ~185,000 kJ/mol — indicates the initial structure had significant Van der Waals overlaps resolved by minimization

### 6.3 NVT Equilibration

**Duration:** 1.0 ns (500,000 steps at 2 fs)  
**Temperature:** 330 K (see Section 7 for rationale)  
**Ensemble:** NVT (no barostat)  
**Restraints:** 1000 kJ/mol/nm² on all peptide heavy atoms throughout

**Purpose:** Allow water molecules to relax and pack around the restrained peptide backbone. The peptide does not move; only the water rearranges. At the start of a solvated simulation, water molecules are placed randomly around the peptide and have unrealistic local geometries. The NVT phase with restraints prevents the peptide from moving into even worse clashes while water finds its preferred arrangement.

**Trajectory and log outputs:** DCD at 100 ps intervals, log at 100 ps intervals (temperature monitoring), checkpoint every 5000 ps.

Observed temperature stability: approaches target of 330 K within first ~100 ps (thermostat coupling is fast at 1 ps⁻¹); oscillates ±5–10 K around target thereafter (normal for small system).

### 6.4 NPT Equilibration

**Duration:** 10.0 ns (5,000,000 steps at 2 fs)  
**Temperature:** 330 K  
**Pressure:** 1.0 bar  
**Barostat:** MonteCarloBarostat, coupling frequency = 25 steps  
**Restraints:** NONE (all restraints released at the start of NPT)

**Purpose:** Converge box volume and density. After NVT equilibration, the box is not at the correct thermodynamic volume for the current temperature and force field. NPT allows the box to shrink or expand until mechanical equilibrium is reached. For CHARMM36m + TIP3P at 330 K, the target liquid water density is approximately 985–1000 kg/m³.

**Why 10 ns (not shorter):** Kuczera 2021 uses 10 ns NPT as the standard. Schwartz 2022 uses 250 ns NPT (exceptionally conservative). Our 10 ns is the minimum accepted in CHARMM36m literature. Box density typically converges within 2–5 ns; the additional time ensures full convergence.

**Why Monte Carlo Barostat (not Parrinello-Rahman or Nosé-Hoover-Langevin):** OpenMM's MC barostat is the recommended choice for isotropic liquids in OpenMM (Kim et al. 2015). It makes random trial moves on the box volume every 25 steps, accepting or rejecting based on Metropolis criterion. This is thermodynamically rigorous and does not require an additional mass or time constant to tune, unlike Nosé-Hoover-Langevin piston.

**Density monitoring:** `StateDataReporter` with `density=True`. Final NPT density should be 990–1010 kg/m³. Values outside this range indicate force field or solvation errors.

### 6.5 Re-solvation NPT (Extended Conformation Only)

After the 10 ns NPT of the extended conformation:
1. Extract peptide coordinates from the final state
2. Re-solvate with 1.2 nm padding (produces ~5,000–7,000 atom system)
3. Run 2.0 ns NPT to converge density of the new box

This step is only performed for the extended starting conformation. The helix starting conformation already has a small, well-equilibrated box after the initial NPT.

### 6.6 Topology PDB for Production

**Critical step:** Immediately before starting the production `DCDReporter`, write a topology PDB:

```python
with open(f"{PREFIX}_{label}_prod_top.pdb", "w") as f:
    app.PDBFile.writeFile(
        sim_prod.topology,
        sim_prod.context.getState(getPositions=True).getPositions(), f)
```

**Why this is critical:** MDTraj loads DCD files against a topology file. If the topology file atom count or order does not exactly match the DCD, MDTraj will either:
- Raise `ValueError` immediately (if atom counts differ — caught early, no silent failure)
- Silently compute incorrect torsion angles (if atom ordering is wrong with same count — worst case, no error)

The topology PDB written here reflects the exact atom ordering of `sim_prod.topology`, which is the ordering the DCD was written in. This file — and only this file — is used as the MDTraj topology for this specific DCD. The pre-equilibration PDBs (`A10_extended_solvated.pdb` etc.) have different atom counts and must NOT be used as MDTraj topology for the production DCD.

---

## 7. Phase 4 — Ground Truth Production MD

### 7.1 Temperature Choice: 330 K

**Why 330 K (not 300 K):**

At 300 K, the expected relaxation time for A10 is τ_eq ≈ 20–30 ns (Kuczera 2021 Table 1 interpolated). The folding time is τ_fold ≈ 200–350 ns. To observe 50–100 helix-coil transitions in a 500 ns simulation, we need τ_fold ≲ 5–10 ns. This requires a temperature increase.

Using the Arrhenius equation with activation energy E_a ≈ 10–15 kcal/mol (this range is drawn from CHARMM-family folding rate studies such as Best & Hummer 2009 and Zhuang et al. 2021 — Kuczera 2021 itself simulates only at 300 K and does not report E_a; the value cannot be extracted from that paper alone):

- Speedup 300 K → 330 K: exp(E_a/R × (1/300 − 1/330)) ≈ 3–5×
- Expected τ_fold at 330 K: 200–350 ns / (3–5) ≈ 40–115 ns
- Expected τ_unfold at 330 K: 25–80 ns / (3–5) ≈ 5–27 ns
- Expected τ_eq at 330 K: τ_fold × τ_unfold / (τ_fold + τ_unfold) ≈ 5–8 ns
- Expected complete fold/unfold cycles in 500 ns: 500 ns / (τ_fold + τ_unfold) ≈ 500 / (45–142 ns) ≈ **3–11 cycles**

**Note on transition counting:** τ_eq = 1/(k_f + k_u) is the *relaxation time* — the time for a population perturbation to decay. It is NOT the mean round-trip time. The correct estimate for the number of complete helix↔coil cycles is T / (τ_fold + τ_unfold), which is smaller than T / τ_eq by a factor of (k_f + k_u)² / (k_f × k_u) ≥ 4. With these rate estimates, 500 ns provides 3–11 complete cycles. This is modest but sufficient for the FES topology and a ground-truth reference; it is not sufficient for high-precision kinetics.

**Why not 340 K:** Higher temperatures reduce helix content significantly (helix-coil is temperature-dependent with an enthalpy of ~1 kcal/mol per residue). At 340 K the helix fraction of A10 may drop to <5%, making the helix basin sparsely sampled and the FES comparison less informative. 330 K provides enough kinetic acceleration while maintaining reasonable helix fraction (~8–15% at 330 K for capped A10).

**Why not 300 K:** Insufficient cycles. At 300 K, τ_fold ≈ 200–350 ns and τ_unfold ≈ 25–80 ns, so τ_fold + τ_unfold ≈ 225–430 ns. In 500 ns: 500 / (225–430) ≈ 1–2 complete fold/unfold cycles — too few to characterize the FES reliably. At least 3–5 cycles are required for the FES basin locations to be meaningful.

**Known limitation:** All force fields including CHARMM36m show weaker-than-experimental temperature dependence of helix fraction (Best & Hummer 2009). The enthalpy and entropy of helix formation are both ~50% smaller in simulation than in experiment. We mitigate this by comparing A10 vs A8Q2 at the SAME temperature, so this systematic error cancels out in the relative comparison.

### 7.2 Production Ensemble: NVT

**Why NVT (not NPT):**

After NPT equilibration, the box volume is at its equilibrium value for this temperature and force field. Switching to NVT:
1. Removes the need for a barostat (saves ~5% compute overhead)
2. Eliminates box volume fluctuations that add noise to structural observables
3. Is thermodynamically correct: the helix-coil free energy is a Helmholtz free energy (NVT), not Gibbs (NPT). The PΔV correction is ~0.001 kcal/mol and negligible, but NVT is the more natural ensemble
4. Matches the Kuczera 2021 methodology explicitly stated as "NVT conditions for simplicity and efficiency"

The box vectors from the final NPT state are loaded into the production simulation context and kept fixed throughout. The density is therefore that of the NPT-equilibrated system.

### 7.3 Production Run Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Duration | 500 ns per replica | ~3–11 complete fold/unfold cycles at 330 K (see Section 7.1) |
| Replicas | 2 (helix-start + extended-start) | Convergence verification |
| Timestep | 2.0 fs | Standard CHARMM36m |
| Temperature | 330 K | 3–5× kinetic speedup vs 300 K |
| Integrator | LangevinMiddleIntegrator | |
| Friction | 1.0 ps⁻¹ | |
| Random seed | 99 (different from equilibration) | Decorrelate from Phase 3 |
| Ensemble | NVT | Volume fixed from NPT equilibration |
| DCD frame interval | 100 ps | 5000 frames total per replica |
| Log interval | 10 ps | Energy, temperature, progress |
| Checkpoint interval | 5000 ps | Resume support |

**Frame count analysis:**
- 500 ns / 100 ps = 5000 frames per replica
- 2 replicas = 10,000 frames total per system
- At τ_eq ≈ 5–8 ns and 100 ps spacing: ~50–80 frames per correlation time → ~60–100 statistically independent frames (this is a frame count for TICA/FES statistical power, not a transition count)

**Why 100 ps save interval (not 500 ps):**
- 100 ps gives 5000 frames per replica — substantially better TICA covariance matrix statistics and finer diagnostic resolution for the ACF lag scan
- Storage: ~48 KB/frame × 5000 frames ≈ 240 MB DCD per replica — manageable on a laptop
- 500 ps would give only 1000 frames (10 frames per τ_eq), which is the absolute minimum and leaves no margin for diagnostic analysis or lag-time sensitivity tests
- The 100 ps interval is also much safer for catching checkpoint/DCD alignment artefacts during resume

### 7.4 Checkpoint/Resume Logic

Production runs include checkpoint-resume capability. At startup:
1. Check if `{PREFIX}_{label}_prod.chk` exists and is non-empty
2. If yes: load checkpoint, read current step count, compute remaining steps
3. If remaining steps ≤ 0: print "already complete" and skip
4. If no checkpoint: initialize positions from equilibrated NPT final PDB, set velocities from Maxwell-Boltzmann at 330 K

This allows safe interruption (RunPod pod termination) and restart without losing progress.

### 7.5 Expected Output Files Per System

```
{PREFIX}_{label}_prod.dcd           — 500 ns trajectory (stays on pod)
{PREFIX}_{label}_prod.log           — energy, temperature, progress every 10 ps
{PREFIX}_{label}_prod.chk           — checkpoint for resume
{PREFIX}_{label}_prod_top.pdb       — topology PDB for MDTraj loading (critical)
```

---

## 8. Phase 5 — Ground Truth Analysis

This analysis is performed on the pod that ran production, using the trajectory files and the `prod_top.pdb` topology. All output figures and CSV files are uploaded to MongoDB Atlas. The large DCD files are discarded at pod shutdown.

### 8.1 Trajectory Loading

```python
import mdtraj as md

# Load full trajectory against the topology written immediately before production
traj = md.load_dcd(f"{PREFIX}_{label}_prod.dcd",
                   top=f"{PREFIX}_{label}_prod_top.pdb")

# Select protein atoms only (drops water and ions)
# Use "protein" not "backbone" — phi/psi computation needs adjacent residue atoms (ACE/NME)
traj_prot = traj.atom_slice(traj.topology.select("protein"))

# MDTraj returns (n_frames, 9) φ and ψ for a 10-residue linear chain (no φ on
# residue 1, no ψ on residue 10).  Downstream TICA uses 40 sin/cos features:
# pad to 10 angles per type (zeros for undefined φ₁ and ψ₁₀) — see ``src.analysis``
# / ``src.utils``.  Do **not** assert phi.shape[1] == 10 without padding.
_, phi = md.compute_phi(traj_prot)
_, psi = md.compute_psi(traj_prot)
```

**Burn-in removal for TICA:** Discard the first **25 ns** = **250 frames** at **100 ps/frame** (≈3–5× τ_eq at 330 K) before any TICA computation. *(Earlier drafts incorrectly wrote “50 frames”; that would be only 5 ns.)* This removes the transient where the system relaxes from its starting conformation. Keeping the burn-in would bias C(0) toward the initial state and produce TIC-1 eigenvectors that represent “escape from initialization” rather than the equilibrium helix–coil coordinate.

### 8.2 Order Parameters

**Backbone RMSD to ideal helix:**

- Reference structure: helix starting conformation (φ = −57°, ψ = −47°), in vacuo/capped PDB from Phase 1
- Procedure: superpose each frame onto the reference using **Cα** atoms, then compute RMSD (backbone heavy atoms **N, Cα, C, O** vs the same atoms in the reference is preferred for consistency with Kuczera-style geometry; Cα-only RMSD to reference is a valid simpler proxy)
- Implementation note: use MDTraj **`md.rmsd`** with **`precentered=True`** after **`superpose`**, not the removed `compute_rmsd` name
- Unit: **nanometres** in code; thresholds in §8.5 are **0.20 / 0.30 nm** (equivalent to 2.0 / 3.0 Å)
- Physical meaning: global shape deviation from the ideal helix geometry (not deviation from the first frame of the same trajectory)

**Number of helical H-bonds (N_HB):**

- Criterion: i→i+4 backbone C=O···H-N hydrogen bonds with O···N distance < 3.6 Å
- Threshold from Kuczera 2021 (direct quote from methods: "O–N distance criterion of 3.6 Å for the backbone i→i+4 hydrogen bonds")
- For a perfect 10-mer helix: N_HB_max = 6 (bonds: 1→5, 2→6, 3→7, 4→8, 5→9, 6→10)
- Robust atom name search: searches for H, HN, H1 (CHARMM names vary by system)
- Physical meaning: direct count of the specific bonds that define the helix; directly related to the IR amide I band and NMR chemical shifts

**DSSP per-residue assignment (Kabsch-Sander):**

- Algorithm: MDAnalysis 2.8+ DSSP implementation (pure Python, no external binary required)
- Method: electrostatic H-bond energy criterion (E < −0.5 kcal/mol, Kabsch & Sander 1983)
- Outputs per residue per frame: `'H'` (helix — covers α, 3₁₀, π), `'E'` (β-sheet), `'-'` (loop/coil). **Note: the coil character is `'-'` (hyphen), NOT `'C'`.** Using `== 'C'` will always return False.
- Access pattern: `run.results.dssp` is shape `(n_frames, n_residues)`, dtype object. Per-residue helix fraction: `(run.results.dssp == 'H').mean(axis=0)`.
- Physical meaning: secondary structure assignment using a physically grounded H-bond energy criterion fixed by the algorithm, not tuned per-system.

### 8.3 Convergence Analysis

**Cumulative helix fraction f(t):**
- Plot f_helix vs time (running average)
- A plateau in the last 200 ns is required for the trajectory to be considered converged
- If the trace is still drifting at 500 ns, the simulation is not converged

**Block Standard Error (BSE):**
- Method: Flyvbjerg & Petersen (1989, J. Chem. Phys. 91:461)
- Compute: split trajectory into blocks of increasing length (5 ns → 250 ns), compute SE of mean f_helix across blocks
- A plateau in BSE vs block length indicates blocks are statistically independent; the plateau value is the true statistical uncertainty
- Target: BSE plateau < 0.01 (1% absolute uncertainty in helix fraction)
- Block size for plateau: should be >> τ_eq ≈ 5–8 ns; expect plateau at ~20–40 ns blocks

**Split-trajectory FES:**
- Compute FES from first 250 ns and second 250 ns separately
- Overlay on the same 2D plot; visual comparison and JSD between the two halves
- If JSD (first half vs second half) > JSD (replica 1 vs replica 2), the trajectory is not converged

**Two-replica agreement:**
- Compute FES from helix-start and extended-start trajectories separately (after pooling across 2 replicas, each 500 ns, **5000** frames at **100 ps** save interval)
- JSD between the two replicas defines the "noise floor": a lower bound on how well any finite sample can agree with the ground truth
- The seeding experiment's JSD must drop below this noise floor to claim "converged"

### 8.4 Free Energy Surface

**2D FES (RMSD × N_HB):**
- Bin grid: 25 × 25 bins
- FES = −k_B T ln(P(RMSD, N_HB)) where P is normalized 2D histogram
- Gauge convention: minimum of FES set to 0 kcal/mol
- Clip at 5 kcal/mol (bins above this are <0.04% population, effectively empty)
- Expected topology: helix basin at low RMSD + high N_HB; coil basin at high RMSD + low N_HB; transition state at intermediate RMSD and N_HB

**1D FES along N_HB:**
- Integrate 2D FES marginally over RMSD: P(N_HB) = Σ_RMSD P(RMSD, N_HB)
- FES = −k_B T ln(P(N_HB))
- Should show two wells: coil at N_HB ≈ 0–1 and helix at N_HB ≈ 4–6

**Per-residue Ramachandran PMF:**
- 2D PMF in (φ, ψ) space per residue, with 8° bins
- PMF = −k_B T ln(P(φ, ψ))
- Expected: Ala residues should show strong helical basin (φ ≈ −62°, ψ ≈ −41°); Gln residues at positions 3 and 8 in A8Q2 should show weaker helical basin (CHARMM36m artifact from Dokainish 2022)

### 8.5 Two-State Classification (Triple Criterion)

Three independent criteria classify each frame as Helix (H) or Coil (C):

| Criterion | Helix | Coil | Transition zone |
|-----------|-------|------|----------------|
| RMSD (nm) | < 0.20 | > 0.30 | 0.20–0.30 (inherit previous state) |
| N_HB fraction | ≥ 50% of N_HB_max (≥3 bonds) | < 3 bonds | — |
| DSSP | ≥ 50% residues assigned H | < 50% residues H | — |

**Hysteresis on RMSD criterion:** Frames in the transition zone (0.20–0.30 nm) inherit the state label of the previous frame. Without hysteresis, frames oscillating near the threshold generate spurious "flickering" transitions that inflate the transition count by factors of 2–10.

**Agreement score:** Fraction of frames where all three criteria agree. Target: > 90%. If agreement is lower, the three-angstrom regime between helix and coil is overpopulated (intermediate states) or the thresholds need adjustment.

### 8.6 Kinetics

**Transition counting:**

- Folding transition: consecutive C→H state change (Coil → Helix)
- Unfolding transition: consecutive H→C state change (Helix → Coil)
- Mean first passage time: τ_H = average time spent in Helix state before transitioning to Coil (= 1/k_u); τ_C = average time in Coil before transitioning to Helix (= 1/k_f)
- Rates: k_f = 1/τ_C (rate of leaving coil = folding rate); k_u = 1/τ_H (rate of leaving helix = unfolding rate)
- Equilibrium: τ_eq = 1/(k_f + k_u)
- Helix fraction: f_helix = k_f / (k_f + k_u) = τ_H / (τ_H + τ_C)
- Free energy: ΔG_fold = −RT ln(f_helix / (1 − f_helix))

**Literature comparison (Kuczera 2021 Table 2, ALA8, 300 K):**
- k_u = (7.0–7.9) × 10⁻² ns⁻¹ (τ_unfold = 12.7–14.3 ns)
- k_f = (3.6–7.8) × 10⁻³ ns⁻¹ (τ_fold = 128–278 ns)
- τ_eq = 12–13 ns
- ΔG_fold = 1.3–1.8 kcal/mol

Arrhenius correction from 300 K to 330 K with E_a = 10 kcal/mol: speedup ≈ 3.5×. Expected values at 330 K for ALA10: τ_eq ≈ 3.7–8.6 ns, τ_fold ≈ 37–100 ns, τ_unfold ≈ 3.6–8.2 ns.

**Autocorrelation function (ACF) of N_HB:**

Biexponential fit: C(t) = A·exp(−t/τ_slow) + (1−A)·exp(−t/τ_fast)

- τ_fast ≈ 0.1–1 ns: fast local backbone fluctuations within a state
- τ_slow ≈ τ_eq ≈ 3–8 ns: slow global helix-coil transition
- Cross-validation: τ_slow from ACF should match τ_eq from transition counting within factor of 2

### 8.7 A10 vs A8Q2 Comparison

**Side-by-side 2D FES:** A10 and A8Q2 on identical bin grids (RMSD × N_HB). Both pooled from helix and extended replicas after confirming both replicas are consistent.

**Per-residue helix fraction overlay:** For each residue position 1–10, plot DSSP helix fraction for A10 (Ala at all positions) and A8Q2 (Gln at positions 3 and 8). Expected: A8Q2 shows reduced helix fraction at positions 3 and 8 (Gln is less helicogenic than Ala; CHARMM36m systematically underestimates Gln helix fraction by ~5% per Dokainish 2022).

**Backbone torsion covariance eigenvector overlap:** See Section 9 (TICA); this is the same computation reframed as a direct eigenvector comparison rather than requiring TICA.

---

## 9. Phase 6 — TICA Training on A10

### 9.1 Featurization

**Feature vector per frame:**

```python
_, phi = md.compute_phi(traj_prot)   # (n_frames, 9) for 10-residue chain
_, psi = md.compute_psi(traj_prot)   # (n_frames, 9)
# Pad to 10 columns each (undefined φ on residue 1, undefined ψ on residue 10):
# phi10[:, 1:] = phi; psi10[:, :-1] = psi — see ``src.analysis.compute_featurization``
features = np.concatenate([
    np.sin(phi10), np.cos(phi10),
    np.sin(psi10), np.cos(psi10),
], axis=1)   # shape (n_frames, 40)
```

**Total features:** 40 (4 × 10 backbone torsion channels after padding; MDTraj supplies 9 physical φ and 9 ψ per frame)

**Why sin/cos encoding:** Torsion angles are periodic on [−π, +π). Raw angles have an artificial discontinuity at ±180°. The sin/cos pair produces a continuous, bounded representation in ℝ² without discontinuity. This is the standard encoding for TICA of backbone torsions (Pérez-Hernández 2013, Schwantes & Pande 2013, Sittel & Stock 2016).

**Why these features and not others:**
- Distances: redundant for 10-residue helix; torsions capture the same information with fewer features
- Side-chain torsions: Ala methyl χ₁ relaxes on ~ps timescales, contributing only noise to the slow subspace. For A8Q2, Gln χ₁/χ₂ relax on ~100 ps timescales, still much faster than helix-coil τ_eq ~ 5 ns.
- Cα-Cα distances: sometimes used in MSM literature but for this system contain redundant information already in the torsion vector

**No standardization:** Sin/cos features are already bounded on [−1, +1] with mean ~0. All features are dimensionally homogeneous (no mixture of units). Standardizing (mean=0, std=1) would inflate the variance of torsion angles that barely fluctuate (deeply in a basin) and suppress informative features. PyEMMA/deeptime mean-center internally; no further scaling is needed.

**Burn-in removal:** Discard first **250 frames** (**25 ns** at 100 ps/frame) before computing features for TICA. Same window as Phase 5 burn-in.

**After burn-in removal (500 ns production, 100 ps frames):** 5000 − 250 = **4750** frames per replica; pooling helix + extended replicas → **9500** frames for TICA training on A10 (if both replicas are pooled). Adjust counts if `PROD_US` or `DCD_INTERVAL_PS` change.

### 9.2 Frame Count Adequacy

**Statistical argument:**

TICA solves the 40×40 generalized eigenvalue problem: C(τ)w = λC(0)w. Reliable estimation of a 40×40 matrix requires effective sample size n_eff >> 40 (Wishart-type noise). 

- Raw frame count (example): **9500** frames after burn-in when **two** 500 ns replicas (5000 frames each minus 250 burn-in) are pooled for TICA  
- Correlation time in features: τ_eq ≈ 5 ns; frame spacing: **0.1 ns** (100 ps) → **~50** correlated frames per τ_eq  
- Effective independent samples: n_eff ≈ 9500 / 50 ≈ **190**
- Effective sample ratio: n_eff / p = 190 / 40 = 4.75

This is borderline adequate for TIC-1 (the dominant mode, with the largest eigenvalue gap) but insufficient for higher TICs. We use the following mitigations:
- Block bootstrap on TIC-1 eigenvector: draw 100 bootstrap samples with block size **≥ ~50 frames** (≈**5 ns** at 100 ps, comparable to τ_eq) so each block is nearly decorrelated; verify TIC-1 direction is stable (cosine similarity > 0.95). See implementation below.
- Regularization: `epsilon=1e-6` in deeptime TICA to prevent ill-conditioned C(0)
- Validation: Pearson r(TIC-1, N_HB) ≥ 0.70 required to proceed; if not met, fallback protocol activates

**Block bootstrap implementation:**

```python
import numpy as np
from deeptime.decomposition import TICA

def block_bootstrap_tic1_stability(features, lagtime, block_size=50,
                                   n_bootstrap=100, threshold=0.95, rng_seed=42):
    """
    Verify TIC-1 eigenvector stability via block bootstrap.
    At 100 ps/frame, block_size=50 → 5 ns per block (order of τ_eq); adjust if DT changes.
    
    IMPORTANT: use scaling=None here to get raw unit eigenvectors for cosine comparison.
    scaling="kinetic_map" is only for the final production projection.
    """
    rng = np.random.default_rng(rng_seed)
    T = len(features)
    n_blocks_total = T // block_size

    # Reference TIC-1 from full trajectory (scaling=None for geometric comparison)
    ref_model = TICA(lagtime=lagtime, dim=5, epsilon=1e-6,
                     scaling=None).fit(features).fetch_model()
    ref_v1 = ref_model.singular_vectors_left[:, 0]  # shape (40,), unit norm

    cosine_sims = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n_blocks_total, size=n_blocks_total) * block_size
        boot_features = np.concatenate([features[s:s+block_size] for s in starts])
        boot_model = TICA(lagtime=lagtime, dim=5, epsilon=1e-6,
                          scaling=None).fit(boot_features).fetch_model()
        boot_v1 = boot_model.singular_vectors_left[:, 0]
        # |cos θ| accounts for sign ambiguity (eigenvectors are defined up to sign flip)
        cosine_sims[i] = abs(np.dot(ref_v1, boot_v1))

    return {
        "mean_cos_sim": float(np.mean(cosine_sims)),
        "min_cos_sim":  float(np.min(cosine_sims)),
        "pass_fraction": float(np.mean(cosine_sims > threshold)),
        "stable": bool(np.all(cosine_sims > threshold)),
    }

# Usage
results = block_bootstrap_tic1_stability(features_a10, lagtime=chosen_lag_frames)
print(f"TIC-1 stable: {results['stable']} (min cos sim = {results['min_cos_sim']:.3f})")
```

### 9.3 Lag Time Selection

TICA solves: C(τ)w = λC(0)w, where the lag time τ is a hyperparameter.

**Physical constraint on lag time** (assume **100 ps** per saved frame, **0.1 ns**):
- Must be >> τ_fast (local backbone fluctuations, ~0.1–0.5 ns): use lag **≥ ~3–5 frames** (0.3–0.5 ns), not a single frame, so the lag skips purely local jitter.
- Must be << τ_slow (helix–coil, ~5 ns): lag **≪ ~50 frames** (5 ns) so the model does not stride past the slow mode of interest.
- Sweet spot (typical): **τ_lag ≈ 1–4 ns** → **10–40 frames** at 100 ps spacing.

**Implied timescale scan:**

For each candidate lag τ, fit TICA and compute TIC-1 implied timescale:
t₁(τ) = −τ / ln|λ₁(τ)|

Scan lags (in frames), example: [5, 10, 15, 20, 30, 40, 50] → **[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] ns**

Look for: a plateau in t₁(τ) vs τ. A plateau indicates the lag is in the Markovian regime — the slow mode is correctly captured and the choice of τ is not affecting the identified timescale.

**If no clear plateau:** Use a secondary **numerical** rule on the same ITS curve (for example: among lags with L ≥ L_min, pick the consecutive pair with the smallest relative change |Δt₁|/t₁ and take the lower lag). **Do not** rank **different** lag times by comparing raw VAMP-2 scores to each other: deeptime’s VAMP documentation states that models at different τ approximate different Koopman operators, so **VAMP scores are not comparable across lag times** (they remain useful as a per-τ diagnostic when printed alongside each fit).

**Chosen lag:** Often falls in the **~10–40 frame** range (**~1–4 ns**) when τ_eq ≈ 5 ns at **330 K** and Δt_save = 100 ps. The implementation (`src/tica_train.py`) enforces a minimum lag (default **10 frames = 1.0 ns**) and selects from the ITS scan as above; VAMP-2 is logged per lag for inspection only.

### 9.4 TICA Computation (deeptime)

```python
from deeptime.decomposition import TICA

# Fit TICA on A10 features (after burn-in removal)
tica = TICA(lagtime=chosen_lag_frames, dim=5, epsilon=1e-6,
            scaling="kinetic_map")
model = tica.fit(features_a10).fetch_model()

# Extract and save — deeptime uses VAMP/Koopman naming, not legacy PyEMMA names
singular_values  = model.singular_values           # may be length 40; use [:dim] for the fitted rank
eigenvectors     = model.singular_vectors_left     # shape (40, dim); columns = TIC directions
mean_a10         = model.mean_0                    # shape (40,) — instantaneous mean

# Implied timescale from chosen lag
t_impl = -chosen_lag_ns / np.log(np.abs(singular_values[0]))
print(f"TIC-1 implied timescale: {t_impl:.2f} ns")
# Alternatively: model.timescales(lagtime=chosen_lag_frames)[0] * frame_spacing_ns
```

**deeptime naming note:** The fitted model is a `CovarianceKoopmanModel`. It uses VAMP/Koopman terminology: `singular_values` (not `.eigenvalues`), `singular_vectors_left` (not `.eigenvectors`), `mean_0` (not `.mean`). These are functionally equivalent to the legacy PyEMMA names. `model.transform(X)` for projecting new data is unchanged.

The `scaling="kinetic_map"` option multiplies each TIC projection by its singular value (λᵢ), giving kinetic map coordinates where Euclidean distance approximates the true kinetic distance between states.

### 9.5 TIC-1 Validation

**Pearson r(TIC-1, N_HB):**

```python
from scipy.stats import pearsonr

# model.transform() applies kinetic_map scaling internally — correct for projection
tic1_a10 = model.transform(features_a10)[:, 0]
r, pval = pearsonr(tic1_a10, nhb_a10)
print(f"|r| = {abs(r):.3f}")
```

Interpretation:
- |r| ≥ 0.85: TIC-1 cleanly captures helix-coil coordinate → proceed
- 0.70 ≤ |r| < 0.85: moderate correlation → proceed with caution, report value
- |r| < 0.70: TIC-1 does NOT represent helix-coil coordinate → log warning, use N_HB as fallback CV for seeding

**Free energy along TIC-1:** Should show two wells (coil and helix basins) separated by a barrier. Single-well FES = the CV is not discriminating states.

**Implied timescale consistency:** t₁_impl from TICA should equal τ_eq from transition counting within a factor of 2.

### 9.6 Upload to MongoDB

After training: upload `a10_tica_singular_vectors_left.npy` (eigenvectors), `a10_tica_mean0.npy` (mean), `a10_tica_singular_values.npy` (eigenvalues), the lag scan plot, and the validation plot to MongoDB Atlas. These are needed by Pod 2 for transferred seeding. (File names reflect deeptime attribute names to avoid confusion.)

---

## 10. Phase 7 — TICA Transferability Assessment

This phase runs on Pod 1 after downloading `a8q2_features.npy` from MongoDB (uploaded by Pod 2 after A8Q2 production finishes).

### 10.1 Transfer Protocol

```python
# Project A8Q2 features onto A10 TICA
# CRITICAL: subtract A10 mean (not A8Q2 mean) — keeps the same reference frame as training
# mean_a10 = model_a10.mean_0  (deeptime attribute name)
# eigenvectors_a10 = model_a10.singular_vectors_left  (deeptime attribute name)
X_a8q2_centered = features_a8q2 - mean_a10           # mean_a10 shape: (40,)
Z_a8q2 = X_a8q2_centered @ eigenvectors_a10[:, :5]   # eigenvectors_a10 shape: (40, 5)
tic1_transferred = Z_a8q2[:, 0]                       # shape: (n_frames_a8q2,)
```

**Why use A10 mean and not A8Q2 mean:**

The TICA eigenvector was learned in the coordinate frame centered at the A10 mean feature vector. The projection `z = (x − μ_A10) · w` places the A10 trajectory origin at z=0. When projecting A8Q2, we are asking: "given the A10 helix-coil coordinate, where does A8Q2 fall on it?" This requires the same reference frame. Using the A8Q2 mean would change the linear functional (it is no longer the same TIC-1 direction in the transferred sense) and would require a documented convention if adopted.

**Implementation (Phase 8):** `project_cv(..., strategy="transferred")` in `src/adaptive_seeding.py` always subtracts **μ_A10** (`mean_donor`). Optional centering with μ_A8Q2 is not implemented.

### 10.2 Moffett-Shukla Transfer Metrics

From Moffett & Shukla (arXiv:1710.00443) — the primary reference for quantifying TICA transferability between related MD systems.

**D₀ — Decorrelation score:**

```python
C0_B = (X_a8q2_centered.T @ X_a8q2_centered) / len(X_a8q2_centered)
C_V = eigenvectors_a10[:, :K].T @ C0_B @ eigenvectors_a10[:, :K]
D0 = np.linalg.norm(C_V - np.eye(K), 'fro')
```

D₀ measures how well the donor TICs remain mutually decorrelated on the acceptor system. If D₀ ≈ 0, the TICs are still orthonormal on A8Q2's covariance structure → coordinate system is geometrically compatible. If D₀ >> 0, the TICs are no longer orthogonal on A8Q2 → the mutation changed the correlation structure of backbone torsions.

**D_τ — Kinetic structure match:**

```python
C_tau_B = (X_a8q2_centered[:-lag].T @ X_a8q2_centered[lag:]) / (len(X_a8q2_centered) - lag)
C_tau_B = 0.5 * (C_tau_B + C_tau_B.T)  # symmetrize
C_V_tau = eigenvectors_a10[:, :K].T @ C_tau_B @ eigenvectors_a10[:, :K]
Lambda_B = np.diag(native_eigenvalues_a8q2[:K])
D_tau = np.linalg.norm(C_V_tau - Lambda_B, 'fro')
```

D_τ measures whether the A10 TICs preserve the slow dynamics of A8Q2. If the A10 TIC basis represents the same slow modes on A8Q2, C_V_tau should be diagonal with entries equal to the native A8Q2 eigenvalues.

**D_{KM} — Subspace error (K=2 donor TICs vs M=1 native TIC):**

```python
# Compute native A8Q2 TIC-1 (U_M)
tica_a8q2 = TICA(lagtime=chosen_lag_frames, dim=1, epsilon=1e-6)
native_model = tica_a8q2.fit(features_a8q2).fetch_model()
U_M = native_model.singular_vectors_left  # shape (40, 1) — deeptime attribute name

V_K = eigenvectors_a10[:, :2]  # top 2 A10 TICs
X = np.linalg.pinv(V_K) @ U_M
D_KM = np.linalg.norm(V_K @ X - U_M, 'fro')
```

D_{21} (K=2, M=1) asks: "can TIC-1 of A8Q2 be reconstructed from the top 2 TICs of A10?" This is the most physically relevant metric for seeding: if D_{21} is small, then directing exploration in the A10 TIC-1 direction is equivalent to directing it in the A8Q2 TIC-1 direction.

**Pearson r(TIC-1_transferred, N_HB_A8Q2):**

Direct physical interpretability check. If |r| > 0.75, the transferred TIC-1 still tracks the primary helix content observable on A8Q2. If |r| < 0.6, the transferred coordinate has lost physical meaning on A8Q2 for helix-coil.

### 10.3 Transfer FES Comparison

Project A8Q2 trajectory onto (a) transferred A10 TIC-1 and (b) native A8Q2 TIC-1. Compute 1D FES along each. If the two-well structure is preserved in the transferred coordinate, the transfer succeeds geometrically even if the well positions are shifted.

---

## 11. Phase 8 — Adaptive Seeding Experiment

### 11.1 Overview

Three parallel seeding strategies are run on A8Q2, each for 5 iterations of 8 seeds of 30 ns each. The goal is to compare how quickly each strategy converges the A8Q2 FES to the 500 ns ground truth.

| Strategy | CV Used | Starts When | CV Source |
|----------|---------|-------------|-----------|
| 1. Random | None (random frame selection) | Immediately after production | None |
| 2. Transferred TICA | A10 TIC-1 | After eigenvectors in MongoDB | Pod 1 |
| 3. Native TICA | A8Q2 TIC-1 (re-trained each iteration) | Immediately after production | A8Q2 data so far |

Total simulation: 3 strategies × 5 iterations × 8 seeds × 30 ns = 3600 ns = 3.6 µs. But since strategies run sequentially on one GPU (in time order: start strategy 1 and 3 first, strategy 2 after eigenvectors arrive), the wall time is longer than if parallel.

### 11.2 Initialization: Generator-Based (No Pilot Run)

Following standard HTMD practice (Doerr & De Fabritiis 2014), iteration 1 uses the **final equilibration frame** as the single generator structure — not a separate pilot trajectory.

**Why no pilot run:**
A 5 ns pilot at 330 K with τ_fold ≈ 40–115 ns has an ~88–96% chance of zero helix-coil transitions. The 50 frames it produces (at 100 ps save interval) are essentially independent samples within the coil basin, differing only in fast local backbone rattling — not in any slow-mode coordinate. Selecting 8 seeds from 10 frames within a single basin provides no conformational diversity, and the three selection strategies produce indistinguishable iteration 1 results. The 5 ns pilot wastes compute and provides no scientific value.

**Initialization procedure:**

```python
# Use the final equilibrated extended-conformation frame as the generator
# This is already available from Phase 3 NPT output
generator_positions = load_final_npt_frame("A8Q2_extended_npt_final.pdb")
generator_box_vectors = load_box_vectors("A8Q2_extended_npt_final.pdb")

# Iteration 1: run 8 independent 30 ns seeds from this single generator,
# each with a different Maxwell-Boltzmann velocity seed
for i, seed_id in enumerate(range(8)):
    sim.context.setPeriodicBoxVectors(*generator_box_vectors)
    sim.context.setPositions(generator_positions)
    sim.context.applyConstraints(1e-7)
    sim.context.setVelocitiesToTemperature(330.0 * unit.kelvin, seed_id + 100)
    sim.step(iteration1_steps)  # 30 ns
```

The bin-selection logic (Section 11.3) is not applied in iteration 1 because all 8 seeds start from the same coordinates. From iteration 2 onward, pooled data from iteration 1 (8 × 30 ns = 240 ns) is used as the selection pool normally.

### 11.3 Seed Selection Criterion (Metastable Least-Counts)

Based on Hruska et al. (J. Chem. Phys. 149, 244119, 2018) which systematically compared adaptive sampling strategies and found metastable-state allocation outperforms fine-bin allocation for slow inter-basin processes.

**Algorithm per iteration, for Strategies 2 and 3:**

1. Compute the CV value for each frame in the pooled trajectory (all previous seeds concatenated)
2. Discretize the 1D CV range into 10 equal-width bins
3. Count frames in each bin
4. Rank bins by count in ascending order
5. For each of the 8 least-populated bins: select one frame at random from any bin with count > 0, prioritizing the least-populated bin
6. If a bin has zero frames: select from the adjacent bin or the closest populated bin

**Why metastable bins, not fine-bin least-counts:**
- Fine-bin least-counts can waste seeds on high-free-energy (sparsely populated) regions of the landscape that are empty simply because they are unfavorable, not because they are undersampled
- Metastable (coarse) bins at 10 bins ensures seeds are drawn from physically meaningful regions of the CV space
- For a two-state (helix-coil) system, 10 bins captures: deep coil basin, transition region, helix basin, and some high-free-energy extremes

**Strategy 1 (Random):** Select 8 frames uniformly at random from the pooled trajectory. No CV needed.

**Strategy 3 (Native TICA) — Deferred to iteration 3:**

Reliable TICA on a 40-feature system requires n_eff/p ≥ 2.5 (minimum for TIC-1 stability), where p = 40 and n_eff = total_frames / (τ_eq / frame_spacing). With τ_eq ≈ 6 ns and 100 ps spacing, each frame is decorrelated every ~60 frames. Pooled frame counts and their adequacy:

| After iteration | Pooled data | Frames (8 seeds × 300 frames/30 ns) | n_eff ≈ frames/60 | n_eff / 40 | Assessment |
|---|---|---|---|---|---|
| 1 | 240 ns | 2400 | ~40 | ~1.0 | TIC-1 unreliable — C(0) near-singular |
| 2 | 480 ns | 4800 | ~80 | ~2.0 | TIC-1 marginal, bootstrap may fail |
| 3 | 720 ns | 7200 | ~120 | ~3.0 | TIC-1 becoming reliable |
| 4+ | ≥960 ns | ≥9600 | ≥160 | ≥4.0 | Reliable, comparable to A10 TICA |

Additionally, seeded data from iterations 1–2 is heavily coil-biased (starts from coil, τ_fold ≫ 30 ns), so TICA would capture within-coil fast modes, not the helix-coil slow mode.

**Updated schedule for Strategy 3:**
- Iterations 1–2: use N_HB as stand-in CV for bin discretization (physical observable, no training data needed). **Definition:** same as Phase 5 — `analysis.compute_nhb` (i→i+4 backbone C=O···H–N), not Baker–Hubbard counts.
- Iteration 3: attempt TICA re-training on pooled data. Native TICA uses `epsilon` and `scaling` from `src.config` (same as production TICA) and default lag `TICA_MIN_LAG_FRAMES` unless overridden. Block bootstrap validation calls `block_bootstrap_tic1_stability` with `lagtime`, `block_size`, and `n_bootstrap` from config; the acceptance statistic is the **minimum** bootstrap |cos θ| (worst case), compared to 0.90. Also require |r(TIC-1, N_HB)| > 0.70. If both pass, switch to TICA. If either fails, remain on N_HB and try again at iteration 4.
- Log which CV was used each iteration in the output summary.

This schedule matches the behavior of Hruska et al. 2018 and HTMD, which treat early iterations as data accumulation with a heuristic CV, and only deploy the learned CV once enough diverse data exists.

### 11.4 Seed MD Runs

For each selected seed frame:

```python
import mdtraj as md
from openmm import unit

# Load the production trajectory (full system: protein + water + ions)
traj = md.load_dcd(production_dcd, top=prod_top_pdb)
frame = traj[seed_frame_index]

# Box vectors and positions MUST come from the SAME frame
# In NVT production the box is fixed, so all frames have identical box vectors.
# If restarting from an NPT trajectory, use frame-specific box vectors.
seed_box_vectors = frame.openmm_boxes(0)     # consistent with positions
seed_positions   = frame.openmm_positions(0) # ALL atoms: protein + water + ions

# Set state in the required order:
# 1. Box vectors first (positions may be expressed relative to the box)
sim.context.setPeriodicBoxVectors(*seed_box_vectors)

# 2. Positions for ALL atoms (topology.getNumAtoms() must match trajectory atom count)
sim.context.setPositions(seed_positions)

# 3. Apply constraints (required for h-bond constrained systems)
sim.context.applyConstraints(1e-7)

# 4. Re-draw velocities AFTER positions are set (constraint application needs geometry)
# CRITICAL: do NOT use velocities from the seed frame — keeps seeds statistically independent
sim.context.setVelocitiesToTemperature(330.0 * unit.kelvin, random_seed)

# NO energy minimization — trajectory frames are already valid configurations
sim.step(seed_steps)  # 30 ns
```

**Why re-draw velocities:** The seed frame carries correlated momentum from its parent trajectory; reusing it makes the seed a direct continuation of that trajectory. Fresh Maxwell-Boltzmann velocities make each seed statistically independent of the parent run.

**Why all atoms:** `setPositions()` requires exactly `topology.getNumAtoms()` positions — one for every atom in the topology, including water and ions. You cannot set protein-only positions. The production DCD and the `prod_top.pdb` topology must have the same atom count (guaranteed by writing `prod_top.pdb` immediately before the DCDReporter in Section 6.6).

**Box vectors:** In our NVT production runs, the box is fixed (no barostat), so all frames have the same box vectors. Using either `frame.openmm_boxes(0)` or the original equilibration box is equivalent. The practice of using the frame-specific box is shown for correctness and generality.

**Why 30 ns:** At 330 K with τ_fold ≈ 40–115 ns, a 30 ns seed run has a ~23–52% probability of undergoing at least one helix-coil transition. This is enough to escape a basin if near the transition state. Seeds starting in the deep coil basin may not transition, but that is acceptable — even non-transitioning seeds expand the FES sampling of the coil basin.

**8 seeds per iteration:** Balances exploration (multiple independent starting points) with convergence (each seed is 30 ns, enough to do something meaningful). Running more seeds per iteration would reduce the number of iterations in the same total budget. 8 is a common choice in HTMD-style adaptive sampling (Doerr & De Fabritiis 2014 uses 10–15 short trajectories per epoch).

### 11.5 Pooling and FES Update

After each iteration:
1. Concatenate all seed trajectories from this iteration with all previous iterations
2. Re-featurize (sin/cos φ/ψ), project onto CV
3. Compute 2D FES on the fixed reference bin grid
4. Compute JSD, exploration coverage, distribution entropy vs ground truth
5. For Strategy 3: re-train TICA on the expanded pooled A8Q2 data

---

## 12. Phase 9 — Acceleration Metrics and Final Analysis

### 12.1 Reference FES

The ground truth reference is the 2D FES computed from the full 500 ns + 500 ns (helix + extended) A8Q2 production trajectories. At 100 ps saved-frame spacing, each branch yields ~5000 frames before burn-in; discarding the first **250** frames (25 ns) per branch leaves **~4750** frames each, so **~9500** frames total after pooling (not 2000). The bin grid is fixed: 25 × 25 bins over the RMSD × N_HB space.

The reference is defined by the masked domain:
Ω = {bins (i,j) where ground truth count ≥ 3}

Bins with fewer than 3 counts in the ground truth are excluded from all comparisons; they are too noisy to serve as reliable reference.

### 12.2 Jensen-Shannon Divergence

For each seeding strategy at each iteration:

```python
from scipy.spatial.distance import jensenshannon
import numpy as np

def compute_jsd(hist_ref, hist_sample, min_count=3):
    """
    JSD between sampled and reference 2D FES histograms.
    Both are raw count arrays of the same shape.
    Returns JSD in bits (0 = identical distributions, 1 = completely disjoint).
    """
    # Flatten to 1D
    p = hist_ref.flatten().astype(float)
    q = hist_sample.flatten().astype(float)

    # Masked domain: only bins where reference has >= min_count hits
    mask = p >= min_count
    p_m = p[mask]
    q_m = q[mask]

    # Normalize on masked domain
    p_m /= p_m.sum()
    q_m_norm = q_m / q_m.sum() if q_m.sum() > 0 else q_m

    # scipy.jensenshannon returns sqrt(JS); square to get JSD
    return jensenshannon(p_m, q_m_norm, base=2) ** 2
```

**Why JSD over KL divergence:** KL divergence is infinite when the sample has zero counts in bins where the reference has nonzero counts. This is guaranteed to happen in early iterations when seeding is far from complete. JSD uses the mixture distribution M = (p + q)/2 as the pivot, keeping it finite even with zero sample counts in the reference domain.

**What JSD means:** JSD = 0 means the sampled distribution is identical to the ground truth on the masked domain. JSD = 1 (theoretical maximum) means they are completely disjoint. Values of JSD = 0.1–0.3 are typical for partially-converged FES comparisons.

**Noise floor:** Compute JSD between the two independent A8Q2 replicas (helix-start vs extended-start). This is the best achievable JSD with 500 ns of sampling. Seeding JSD must drop below this value to claim "equivalent to ground truth."

**Equilibrium caveat:** Pooled trajectories from adaptive seeding are **not** independent samples from a single Boltzmann distribution unless reweighted (cf. adaptive sampling and MSM construction literature). Treat JSD, coverage, entropy, and |f_helix error| versus the long production reference as **practical** overlap / convergence diagnostics under this protocol, not as proof that the seeded ensemble equals π(x) without further analysis.

### 12.3 Exploration Coverage

```python
def coverage(hist_ref, hist_sample, min_count=3):
    """Fraction of reference bins with at least 1 hit in the sample."""
    valid_bins = hist_ref >= min_count
    return (hist_sample[valid_bins] > 0).mean()
```

Coverage grows from ~10% (first iteration, only explored near the starting point) toward ~90–100% as the seeding covers the full landscape. The transferred TICA strategy should achieve higher coverage earlier than random sampling because it specifically targets undersampled regions.

### 12.4 Distribution Entropy

H(q) = −Σ q_i log₂(q_i) over bins with q_i > 0, normalized by log₂(|Ω|) to be in [0,1].

Higher entropy = more uniform sampling = less biased. Compare entropy of seeded distribution to entropy of ground truth distribution. If seeded entropy < ground truth entropy, there are still undersampled regions.

### 12.5 Scalar Validation: f_helix Error

|f_helix_seeded − f_helix_groundtruth|

This is the most interpretable single number. f_helix_groundtruth ≈ 10–20% for A8Q2 at 330 K (lower than A10 due to less helicogenic Gln residues). If seeding converges f_helix to within 2 percentage points of ground truth by iteration 3, the CV is doing its job.

### 12.6 Final Figure Set

Produced by `analyze.py` running locally after downloading all artifacts from MongoDB:

1. **Convergence plots** for A10 and A8Q2: cumulative f_helix, BSE, split-trajectory FES
2. **Time series** for A10 and A8Q2: RMSD, N_HB, DSSP with state shading
3. **2D FES comparison**: A10 vs A8Q2 side-by-side on same scale
4. **Per-residue helix fraction**: A10 vs A8Q2 overlay (shows Gln effect at positions 3, 8)
5. **Ramachandran PMF** per residue for both systems
6. **TICA lag scan**: implied timescale vs lag time
7. **TICA validation**: TIC-1 vs N_HB scatter with Pearson r
8. **FES along transferred TIC-1 vs native TIC-1** for A8Q2
9. **JSD convergence curves**: 3 strategies, JSD vs cumulative ns
10. **Exploration coverage curves**: 3 strategies, coverage vs cumulative ns
11. **Transferability scores**: D₀, D_τ, D_{KM} as bar chart; Pearson r as number

---

## 13. Distributed Execution on RunPod

> **Note on execution environment:** The protocol below describes the original 2-pod RunPod RTX 4090 architecture. The experiment is currently planned to run on a **local laptop with an RTX 4050 Ti**. The software stack and all phases are identical; only the timing estimates and the pod coordination sections are affected. On a laptop RTX 4050 Ti, expected throughput is ~150–250 ns/day for the solvated systems, giving approximately 4–7 days for 500 ns × 2 replicas per system. MongoDB Atlas coordination and the 2-pod split are not needed for laptop execution — everything runs sequentially: A10 ground truth → TICA → A8Q2 ground truth → seeding. The RunPod sections are retained here for reference and for potential future scale-up.

### 13.1 Bootstrap Script

Run once on each fresh pod before any Python script:

```bash
#!/bin/bash
set -e
apt-get update -qq && apt-get install -y -qq git
git clone https://github.com/yourrepo/BTP_2_Calm.git
cd BTP_2_Calm
pip install "openmm[cuda12]" mdtraj numpy scipy \
    deeptime pymongo matplotlib pandas PeptideBuilder biopython pdbfixer \
    MDAnalysis
python -m openmm.testInstallation   # must print: CUDA platform available
echo "Bootstrap complete."
```

**Note:** `pip install "openmm[cuda12]"` downloads a CUDA 12 runtime bundled with OpenMM. No separate CUDA toolkit installation is needed. The RunPod container must have NVIDIA drivers installed (standard for GPU pods).

### 13.2 Environment Variables

Set `MONGO_URI` as a RunPod secret before launching pods. This persists across all pods without being hardcoded in source code.

```bash
export MONGO_URI="mongodb+srv://user:password@cluster.mongodb.net/btp_experiment"
```

### 13.3 Pod 1 Sequence

```bash
# After bootstrap.sh completes:
python src/run_simulations.py --system A10   # ~13 hrs: build + equil + 500ns x2 + featurize + analyze
python src/train_tica.py                      # ~20 min: lag scan + TICA + transfer metrics
# Pod 1 is now done. Wait for watch.py to confirm all artifacts uploaded, then stop from RunPod dashboard.
```

`run_simulations.py --system A10` internally:
1. Phase 1: build A10 helix and extended structures
2. Phase 2: solvate, parameterize
3. Phase 3: equilibrate (minimize → NVT → NPT → re-solvation NPT)
4. Phase 4: 500 ns production, helix-start replica
5. Phase 4: 500 ns production, extended-start replica
6. Write `prod_top.pdb` for each replica
7. Featurize: compute sin/cos φ/ψ, strip burn-in 25 ns, upload `a10_features.npy` to MongoDB
8. Analyze: full analysis pipeline, upload PNGs + `kinetics_summary.csv`
9. Update MongoDB job status: `a10_ground_truth_done = true`

`train_tica.py` internally:
1. Load `a10_features.npy` from disk (already written by run_simulations.py)
2. Run lag scan (frames [1,2,3,5,8,10,15,20])
3. Fit TICA at chosen lag, validate r(TIC-1, N_HB)
4. Poll MongoDB every 60 s for `a8q2_features.npy` (uploaded by Pod 2)
5. Once available: compute D₀, D_τ, D_{KM}, Pearson r
6. Upload all results to MongoDB

### 13.4 Pod 2 Sequence

**Implemented entry points (this repo):** Phase 1–4 are ``python -u -m src.simulate --system A8Q2 --phase all --output_dir outputs/btp_equil`` (writes under ``outputs/btp_equil/A8Q2/``). After both 500 ns replicas finish, run Phase 5 + feature export + adaptive seeding in one guarded pass:

```bash
# Optional: ensure repo is a git clone so pull works; copied trees have no .git
cd /workspace/BTP_2_Calm && git pull --ff-only

# While simulate is still running OR after it exits — waits for complete logs + DCDs, then continues
nohup python -u scripts/pod2_orchestrate.py \
  --repo-root /workspace/BTP_2_Calm \
  --sim-output-dir /workspace/BTP_2_Calm/outputs/btp_equil/A8Q2 \
  --tica-artifacts-dir /path/to/dir/with/a10_tica_singular_vectors_left.npy \
  > pod2_orchestrate.log 2>&1 &
```

Or: ``bash scripts/runpod_pod2_pipeline.sh`` with ``BTP_REPO``, ``BTP_SIM_DIR``, and optional ``BTP_TICA_ARTIFACTS_DIR`` set. Omit ``--tica-artifacts-dir`` if A10 TICA files are not on the pod yet (transferred seeding strategy is skipped automatically).

**Private GitHub + live MD:** Initialise git in the repo root, add ``origin``, and use a [PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) in the remote URL *once* (``git remote set-url origin https://<TOKEN>@github.com/USER/REPO.git``), then ``git pull --ff-only``. Do not ``git reset --hard`` while a long MD job is writing outputs.

**Orchestrator defaults (``scripts/pod2_orchestrate.py``):** ``git pull`` is **off** unless you pass ``--git-pull`` or set ``BTP_ENABLE_GIT_PULL=1`` (avoids auth errors on pods that use scp/rsync). Set ``BTP_SKIP_GIT_PULL=1`` to force-disable pull even if ``--git-pull`` were passed. A **flock** lock file ``.pod2_orchestrate.lock`` prevents two orchestrators on the same repo (use ``--no-lock`` only for debugging). Before each wait, **``src.config`` is reloaded** from disk; **``src.analysis`` is reloaded** immediately before Phase 5 so long runs pick up updated analysis code.

Legacy protocol names below map to the same stages:

```bash
python src/run_simulations.py --system A8Q2  # ~13 hrs: same as Pod 1 but for A8Q2
python src/run_seeding.py                     # ~14 hrs: 3-strategy seeding loop
```

`run_seeding.py` internally:
1. Load `a8q2_features.npy` and ground-truth 2D FES histogram from disk
2. Start seeding iteration loop
3. Strategy 1 (random) and Strategy 3 (native TICA): start immediately
4. Strategy 2 (transferred TICA): poll MongoDB for eigenvectors from Pod 1; start as soon as they arrive
5. Per iteration: select seeds → run 8 × 30 ns MD → pool → compute metrics → update MongoDB progress
6. Upload JSD curves, histograms, and figures to MongoDB after each strategy completes

### 13.5 MongoDB Schema

**`jobs` collection:**
```json
{
  "job_id": "pod1_a10_ground_truth",
  "system": "A10",
  "task": "ground_truth_md",
  "status": "running",
  "current_phase": "replica_helix_200ns",
  "progress_ns": 200.5,
  "total_ns": 500.0,
  "started_at": "2026-04-08T10:00:00Z",
  "finished_at": null,
  "error": null
}
```

**`artifacts` collection:**
```json
{
  "name": "a10_tica_eigenvectors",
  "produced_by": "pod1",
  "shape": [40, 5],
  "data_b64": "<base64-encoded npy bytes>",
  "created_at": "2026-04-08T23:45:00Z"
}
```

**Artifacts uploaded to MongoDB (complete list):**

| Artifact | Size | Produced by | Used by |
|----------|------|-------------|---------|
| `a10_features.npy` | ~320 KB | Pod 1 | Pod 1 (TICA) |
| `a8q2_features.npy` | ~320 KB | Pod 2 | Pod 1 (transfer), Pod 2 (seeding) |
| `a10_tica_eigenvectors.npy` | <10 KB | Pod 1 | Pod 2 (strategy 2) |
| `a10_tica_mean.npy` | <1 KB | Pod 1 | Pod 2 (strategy 2) |
| `a10_tica_eigenvalues.npy` | <1 KB | Pod 1 | Pod 2 (strategy 2) |
| `transferability_scores.json` | <5 KB | Pod 1 | Local analyze.py |
| `a10_kinetics_summary.csv` | <5 KB | Pod 1 | Local analyze.py |
| `a8q2_kinetics_summary.csv` | <5 KB | Pod 2 | Local analyze.py |
| `jsd_curves_random.json` | <50 KB | Pod 2 | Local analyze.py |
| `jsd_curves_transferred.json` | <50 KB | Pod 2 | Local analyze.py |
| `jsd_curves_native.json` | <50 KB | Pod 2 | Local analyze.py |
| All analysis PNGs (~20 files) | ~5 MB total | Pod 1 + Pod 2 | Local analyze.py |

**Not in MongoDB (stays on pod, discarded at shutdown):**

- `A10_helix_prod.dcd` (~2.5 GB), `A10_extended_prod.dcd` (~2.5 GB)
- `A8Q2_helix_prod.dcd` (~2.5 GB), `A8Q2_extended_prod.dcd` (~2.5 GB)
- All seeding trajectory DCDs (~3.6 µs × ~5 MB/ns → ~18 GB total)

### 13.6 Timing and Cost Summary

| Phase | Wall Time | Pod | Cost |
|-------|-----------|-----|------|
| Build + equil (A10, 2 conformations) | ~1 hr | Pod 1 | $0.34 |
| 500 ns production × 2 replicas (A10) | ~12 hrs | Pod 1 | $4.08 |
| Featurize + analyze (A10) | ~30 min | Pod 1 | $0.17 |
| TICA training + transfer metrics | ~20 min | Pod 1 | $0.11 |
| **Pod 1 total** | **~14 hrs** | | **~$5** |
| Build + equil (A8Q2, 2 conformations) | ~1 hr | Pod 2 | $0.34 |
| 500 ns production × 2 replicas (A8Q2) | ~12 hrs | Pod 2 | $4.08 |
| Featurize + analyze (A8Q2) | ~30 min | Pod 2 | $0.17 |
| Seeding 3 strategies × 5 iter × 8 seeds × 30 ns | ~14 hrs | Pod 2 | $4.76 |
| **Pod 2 total** | **~28 hrs** | | **~$10** |
| **Grand total** | **~28 hrs wall clock** | **2 pods** | **~$15** |

Both pods run simultaneously, so the experiment completes in ~28 hours of wall clock time.

---

## 14. Known Limitations and Caveats

### 14.1 Force Field Limitations

**Temperature dependence:** CHARMM36m underestimates the temperature dependence of helix fraction by ~50% (Best & Hummer 2009). The experimental melting curve is steeper than simulation. Comparing A10 vs A8Q2 at the SAME temperature cancels this systematic error for relative comparisons.

**Gln backbone:** CHARMM36m systematically underestimates helicity at Gln positions by ~5% relative to neighbors (Dokainish & Sugita 2022). This is a known artifact; our per-residue analysis will document it.

**mTIP3P:** Modified TIP3P does not reproduce experimental water self-diffusion constant as well as TIP4P-Ew. However, it is the correct water model for CHARMM36m and produces correct helix-coil balance.

### 14.2 Simulation Length vs Convergence

**Strict statistical requirement:** For fully converged thermodynamics of A10 at 300 K, Kuczera 2021 used 10 µs. Our 500 ns at 330 K provides approximately 3–11 complete helix↔coil cycles (see Section 7.1 for the correct calculation). This is adequate for establishing FES topology and comparing strategies but not for publishing helix fraction to 1% precision or kinetic rates to <20% error. All reported quantities should include bootstrap uncertainty estimates.

**What 500 ns is sufficient for:**
- Computing the 2D FES landscape topology (basin locations and approximate barrier heights to ±1 kcal/mol)
- Computing f_helix to ±5 percentage points
- Computing τ_eq to within a factor of 2
- Providing a reference for seeding convergence comparisons

**What 500 ns is insufficient for:**
- Publishing absolute kinetic rates to <10% error
- Precise barrier heights to <0.2 kcal/mol
- Convergence of rare conformational states

### 14.3 TICA Statistical Limitations

With ~190 effective independent samples and 40 features, TIC-1 is reliable but TIC-2 through TIC-5 may be noisy. The block bootstrap validation (Section 9.2) will quantify this. Higher TICs (TIC-2+) are not used in the seeding experiment; only TIC-1 is used as the CV.

The lag time choice is based on the implied timescale plot (with a numerical fallback on |Δt₁|/t₁ between consecutive scan points). VAMP-2 is not used to rank different lag times against each other (see §9.3).

### 14.4 Seeding Experiment Statistical Limitations

**5 iterations × 8 seeds × 30 ns = 1200 ns per strategy.** Whether this is sufficient to observe statistically meaningful differences between strategies depends on:
- How much A8Q2 FES is explored by random sampling alone (if random is already fast, strategies 2 and 3 cannot show improvement)
- Whether the TICA transfer is successful (if r < 0.70 for transferred TIC-1, strategy 2 may be no better than random)

The experiment is designed to produce meaningful results in either case: if transfer fails, that is the scientific finding (2 mutations are enough to change the reaction coordinate). If transfer succeeds, the acceleration is quantified.

Metrics in §12 (JSD, f_helix, etc.) measure agreement with the equilibrium reference trajectory; they do **not** by themselves establish that the adaptively grown ensemble is an unbiased equilibrium sample (see §12.2 equilibrium caveat).

### 14.5 MDTraj/OpenMM Topology Bookkeeping

The re-solvation step changes the atom count of the extended conformation box. If the wrong topology PDB is used for MDTraj loading, the result is:
- If counts differ: immediate `ValueError` (safe, caught early)
- If counts match but ordering differs: silently wrong torsion angles (dangerous)

This is mitigated by writing `prod_top.pdb` from `sim_prod.topology` immediately before starting the DCD reporter, and using only that file for all MDTraj operations on the production DCD.

### 14.6 PeptideBuilder GLN Bug

The `Geometry.geometry('GLN')` function in PeptideBuilder returns `GlyGeo` (glycine geometry) instead of `GlnGeo` (glutamine geometry). This is a confirmed bug in the installed version. For A8Q2, we bypass this by directly instantiating `Geometry.GlnGeo()` at residue positions 3 and 8.

---

## 15. Expected Results and Scientific Claims

### 15.1 Ground Truth MD (A10 and A8Q2)

| Observable | Expected A10 | Expected A8Q2 | Basis |
|-----------|-------------|--------------|-------|
| f_helix (%) | 8–15% | 5–12% | CHARMM36m literature, Arrhenius-corrected to 330 K |
| τ_eq (ns) | 5–8 ns | 6–12 ns | Kuczera 2021 (300 K values) scaled by estimated 3–5× Arrhenius speedup to 330 K; Gln slightly slows folding |
| Dominant state | Coil/PPII | Coil/PPII | Both peptides are majority coil at 330 K |
| Gln effect | N/A | Reduced f_helix at positions 3 and 8 | Dokainish 2022; Gln less helicogenic than Ala |
| 2D FES topology | Two-basin (helix + coil) | Two-basin, coil basin deeper | Standard helix-coil landscape |

### 15.2 TICA on A10

| Observable | Expected | Basis |
|-----------|---------|-------|
| TIC-1 Pearson r vs N_HB | ≥0.85 | Standard for well-parameterized helix-coil TICA |
| TIC-1 implied timescale | 5–8 ns | Should match τ_eq from transition counting |
| TIC-1 FES shape | Two-well | Helix and coil basins separated by barrier |
| Lag time from plateau | 1–2 ns | τ_fast < τ_lag < τ_eq |

### 15.3 Transferability (A10 TICA → A8Q2)

Three outcome scenarios, each with a scientific interpretation:

**Scenario A (60–70% probability): Partial success**
- D₀ < 0.3, D_τ moderate, D_{21} < 0.5, Pearson r ≈ 0.75–0.90
- Interpretation: 2 Ala→Gln mutations change the thermodynamics (helix fraction drops) but not the geometric nature of the reaction coordinate. The helix-coil CV transfers because both peptides fold/unfold through the same backbone torsion changes; only the free energy surface shape shifts.

**Scenario B (20% probability): Clean success**
- D₀ ≈ 0, D_τ ≈ 0, D_{21} ≈ 0, Pearson r > 0.90
- Interpretation: The mutation is effectively invisible to the CV. The helix-coil coordinate is robustly transferable for mutations at internal positions that preserve backbone φ/ψ geometry.

**Scenario C (20% probability): Failure**
- D₀ > 0.5, Pearson r < 0.60, seeding strategy 2 ≈ random
- Interpretation: The two Gln substitutions alter the helix-coil reaction coordinate itself (not just the thermodynamics). This would suggest that Gln at positions 3 and 8 introduces a new slow mode (possibly related to side-chain conformational changes or altered H-bond network) that does not exist in A10 and is not captured by A10 TIC-1.

### 15.4 Seeding Acceleration

If transferability succeeds (Scenario A or B):
- Strategy 2 (transferred TICA) should show JSD decreasing faster than Strategy 1 (random) but may lag Strategy 3 (native TICA)
- By iteration 3–4 out of 5, the transferred strategy is expected to approach the native TICA performance
- Quantitative target: Strategy 2 achieves 50% of Strategy 3's JSD reduction in the same cumulative ns

If transferability fails (Scenario C):
- Strategy 2 performs similarly to Strategy 1 (random): no improvement from using the transferred CV
- Strategy 3 (native TICA) still outperforms random as expected
- This is a valid and publishable negative result

### 15.5 The One-Sentence Scientific Claim

Filled in after running the experiment:

> "A TICA collective variable trained on Ac-Ala₁₀-NHMe (deca-alanine) transfers to Ac-Ala₈Gln₂-NHMe with [D₀=X, Pearson r=Y] fidelity. The transferred CV is [verified/not] to represent the helix-coil reaction coordinate on the target peptide, and adaptive seeding with this CV converges the target free energy surface [Z%] faster than random seeding at equivalent cumulative simulation time."

---

## 16. Complete Reference List

1. **Kuczera K., Szoszkiewicz R., He J.Y., Jas G.S. (2021)** — "Length Dependent Folding Kinetics of Alanine-Based Helical Peptides from Optimal Dimensionality Reduction." *Life* 11(5):385. DOI: 10.3390/life11050385. PMC8170890.  
   *Primary kinetics benchmark. ALA8 at 300 K: k_f = 3.6–7.8 × 10⁻³ ns⁻¹, k_u = 7.0–7.9 × 10⁻² ns⁻¹, τ_eq = 12–13 ns. Two-trajectory methodology.*

2. **Kuczera K., Jas G.S., Szoszkiewicz R. (2024)** — "Helix Formation from Hydrogen Bond Kinetics in Alanine Homopeptides." *Crystals* 14(6):532. DOI: 10.3390/cryst14060532.  
   *Follow-up: ALA8 helix fraction ~6%, τ_eq = 13.2 ns. H-bond kinetics: bottleneck is NHB=4→6 transition.*

3. **Huang J. et al. (2017)** — "CHARMM36m: An improved force field for folded and intrinsically disordered proteins." *Nat. Methods* 14:71–73. DOI: 10.1038/nmeth.4067.  
   *CHARMM36m force field. Modified CMAP for disordered proteins. Validated on AAQAA-type helices.*

4. **Kim S. et al. (2015)** — "CHARMM-GUI Input Generator for NAMD, GROMACS, AMBER, OpenMM, and CHARMM/OpenMM Simulations Using the CHARMM36 Additive Force Field." *J. Chem. Theory Comput.* 12(1):405–413. DOI: 10.1021/acs.jctc.5b00935. PMC4712441.  
   *Official OpenMM protocol for CHARMM36: cutoff 1.2 nm, switch 1.0 nm, force-based switching via CustomNonbondedForce.*

5. **Dokainish H.M., Sugita Y. (2022)** — "Efficient Conformational Sampling of Intrinsically Disordered Proteins." *Molecules* 27(19):6452. PMC9457699.  
   *CHARMM36m REMD of (AAQAA)₃ at 300 K: f_helix = 0.17 ± 0.01. Gln underestimation artifact documented.*

6. **Best R.B., Hummer G. (2009)** — "Optimized molecular dynamics force fields applied to the helix-coil transition of polypeptides." *J. Phys. Chem. B* 113(26):9004–9015. PMC3115786.  
   *Force field comparison for helix-coil. Per-residue helix fraction definition from φ/ψ torsion angles.*

7. **Zhuang Y. et al. (2021)** — "Energetics and structure of alanine-rich α-helices via adaptive steered molecular dynamics." *Biophys. J.* 120(10):2009–2018. PMC8204395.  
   *ASMD of Ala6 through Ala50. 3₁₀-helix intermediate during unfolding. PPII as unfolded state. Helix unwinds from ends.*

8. **Pérez-Hernández G. et al. (2013)** — "Identification of slow molecular order parameters for Markov model construction." *J. Chem. Phys.* 139:015102.  
   *TICA theory and application. Sin/cos torsion features as inputs. Generalized eigenvalue problem formulation.*

9. **Schwantes C.R., Pande V.S. (2013)** — "Improvements in Markov State Model Construction Reveal Many Non-Native Structures in the Folding of NTL9." *J. Chem. Theory Comput.* 9:2000–2009.  
   *TICA for MSM construction. Torsion features for helix-coil systems.*

10. **Moffett A.S., Shukla D. (2017)** — "On the transferability of time-lagged independent components between similar molecular dynamics systems." arXiv:1710.00443.  
    *Transfer metrics: D₀, D_τ, D_{KM}. Framework for quantifying TICA CV transfer between related MD systems.*

11. **Sultan M.M., Pande V.S. (2018)** — "Transferable Neural Networks for Enhanced Sampling of Protein Dynamics." *J. Chem. Theory Comput.* 14:1887–1901.  
    *Transfer of slow modes from WT to mutants. Direct motivation for A10→A8Q2 transfer experiment.*

12. **Hruska E. et al. (2018)** — "Quantitative comparison of adaptive sampling methods for protein dynamics." *J. Chem. Phys.* 149:244119.  
    *Systematic comparison of adaptive sampling strategies. Metastable allocation outperforms fine-bin for slow processes.*

13. **Doerr S., De Fabritiis G. (2014)** — "On-the-fly learning and sampling of ligand binding by high-throughput molecular simulations." *J. Chem. Theory Comput.* 10:2064–2069.  
    *HTMD adaptive sampling. MSM-based seed selection protocol. 8–15 seeds per epoch with 30–100 ns each.*

14. **Grossfield A., Zuckerman D.M. (2009)** — "Quantifying uncertainty and sampling quality in biomolecular simulations." *Annu. Rep. Comput. Chem.* 5:23–48. PMC2865156.  
    *Block Standard Error methodology. 20 independent samples minimum rule. Convergence can only be disproven, not proven.*

15. **Flyvbjerg H., Petersen H.G. (1989)** — "Error estimates on averages of correlated data." *J. Chem. Phys.* 91:461–466.  
    *Block error analysis. Plateau in BSE vs block length gives true statistical uncertainty.*

16. **Kabsch W., Sander C. (1983)** — "Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features." *Biopolymers* 22:2577–2637.  
    *DSSP algorithm. H-bond energy criterion: E = 0.084 × (1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN) × 332 kcal/mol; threshold E < −0.5 kcal/mol.*

17. **Jorgensen W.L. et al. (1983)** — "Comparison of simple potential functions for simulating liquid water." *J. Chem. Phys.* 79:926–935.  
    *TIP3P water model.*

18. **MacKerell A.D. Jr. et al. / CHARMM development group** — The mTIP3P requirement is documented in the CHARMM force field parameter files (`toppar_water_ions.str`) and enforced by the CHARMM-GUI output defaults (Kim et al. 2015). The effect of substituting sTIP3P is discussed qualitatively in several CHARMM36m validation papers; Best & Hummer 2009 (ref. 6) discuss the water model sensitivity. *Note: the original reference 18 as written ("van der Spoel 2016, CHARMM27 with Modified TIP3P") could not be verified as a real publication. The underlying claim about mTIP3P being required with CHARMM36m remains correct and is substantiated by refs. 3 and 4.*

19. **Tiberti M. et al. (2015)** — "ENCORE: Software for Quantitative Ensemble Comparison." *PLOS Comput. Biol.* 11(10):e1004415.  
    *JSD and ensemble comparison methodology. Bootstrap-based uncertainty in JSD estimates.*

20. **Shamsi Z., Cheng K.J., Shukla D. (2018)** — "Reinforcement Learning Based Adaptive Sampling: REAPing Rewards By Exploring Protein Conformational Landscapes." *J. Phys. Chem. B* 122:8386–8395.  
    *Reinforcement learning adaptive sampling. Comparison to least-counts heuristic.*

---

*Document version: 1.0 | April 2026 | Generated from experimental protocol, MD driver ``src/simulate.py``, analysis notebook ``Analysis_FES_Transitions.ipynb``, literature review (A10_MD_Literature_Analysis.md), and distributed architecture planning.*
