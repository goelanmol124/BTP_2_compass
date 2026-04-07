"""
config.py — Single source of truth for all simulation constants.

All values match exactly the validated A10 and A8Q2 notebooks. Importing
this module in simulate.py, analyze.py, and seed.py guarantees every phase
uses the same parameters with no risk of silent divergence.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Sequence lists (one entry per residue, N→C order, no caps)
SEQUENCES = {
    "A10":  ["ALA"] * 10,
    "A8Q2": ["ALA", "ALA", "GLN", "ALA", "ALA", "ALA", "ALA", "GLN", "ALA", "ALA"],
}

# Starting conformations: (phi_deg, psi_deg, label)
CONFORMATIONS = [
    (-120.0, 140.0, "extended"),
    (-57.0, -47.0, "helix"),
]

# ─────────────────────────────────────────────────────────────────────────────
# THERMODYNAMIC / INTEGRATOR PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TEMPERATURE_K: float = 330.0
"""
330 K chosen over 300 K for ~3–5× Arrhenius acceleration of helix-coil kinetics
(Ea ≈ 10 kcal/mol), giving τ_fold ≈ 40–115 ns and 3–11 complete helix↔coil
cycles in 500 ns. 330 K maintains measurable helix content (~8–15% for A10);
340 K would drop it below 5%, making the helix basin too sparse.
Reference: Kuczera et al. Life 2021 (300 K kinetics), Arrhenius-extrapolated.
"""

PRESSURE_BAR: float = 1.0
"""Standard atmospheric pressure; used only during NPT equilibration."""

FRICTION_PS: float = 1.0
"""
Langevin friction coefficient γ = 1 ps⁻¹ (CHARMM36m recommended).
Fast enough to control temperature, slow enough not to overdamp backbone dynamics.
"""

TIMESTEP_FS: float = 2.0
"""
2 fs timestep with HBonds + rigidWater constraints.
Universal standard in CHARMM36m helix-coil literature (Kuczera, Kim 2015).
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION DURATIONS
# ─────────────────────────────────────────────────────────────────────────────

NVT_NS: float = 1.0
"""
1 ns NVT with positional restraints. Allows water shell to equilibrate around
the frozen peptide backbone. TIP3P water diffusion across the peptide diameter
(~1.5 nm) takes ~0.05 ns; 1 ns provides 20× margin.
"""

NPT_NS: float = 10.0
"""
10 ns NPT without restraints. Converges box density to within ±5 kg/m³ of
1000 kg/m³. Minimum validated by Kim et al. 2015 and Kuczera 2021.
"""

NPT_RESOLVATE_NS: float = 2.0
"""
2 ns NPT on re-solvated extended box (Kuczera 2021 methodology).
Box is already small after re-solvation; 2 ns is sufficient to re-converge density.
"""

PROD_US: float = 0.5
"""
500 ns production per replica. At 330 K τ_fold ≈ 40–115 ns, giving 3–11 complete
helix↔coil cycles — sufficient for FES topology and ground-truth reference.
Known limitation: not sufficient for high-precision kinetics (<10% error).
"""

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

LOG_INTERVAL_PS: float = 10.0
"""Log T, PE, KE every 10 ps → 50,000 data points per 500 ns."""

DCD_INTERVAL_PS: float = 100.0
"""
Save DCD frame every 100 ps → 5000 frames per 500 ns replica.
Gives 50 frames per τ_eq ≈ 5–8 ns — adequate for TICA covariance estimation.
Storage: ~240 MB per DCD at ~48 KB/frame for ~5000 atoms.
"""

CHK_INTERVAL_PS: float = 1000.0
"""
Checkpoint every 1000 ps (1 ns). Matches ``A10_Prod_Run.ipynb`` and
``A8Q2_Prod_Run.ipynb`` (lose at most ~1 ns on crash). Some protocol drafts
used 5000 ps for remote pods; override here if you prefer rarer, larger checkpoints.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FORCE FIELD FILES
# ─────────────────────────────────────────────────────────────────────────────

FF_PROTEIN: str = "charmm36_2024.xml"
"""
CHARMM36m protein force field. Built into OpenMM 8.x core — no extra package.
Includes CHARMM36m backbone torsion corrections (improved PPII/helix balance
vs CHARMM36). The ACE/CT3 capping patches are part of this file.
"""

FF_WATER: str = "charmm36_2024/water.xml"
"""
CHARMM modified TIP3P (mTIP3P). Built into OpenMM 8.x core.
Adds Lennard-Jones terms on hydrogen atoms (ε_H=0.046 kcal/mol, r_min_H=0.449 Å).
CHARMM36m was parameterized exclusively with mTIP3P — do NOT substitute sTIP3P.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FORCE FIELD / NONBONDED PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

NONBONDED_CUTOFF_NM: float = 1.2
"""
CHARMM36m parameterization cutoff. Mandatory; do not reduce to 1.0 nm.
Reference: Kim et al. 2015, J. Chem. Theory Comput.
"""

SWITCH_DISTANCE_NM: float = 1.0
"""
LJ switch onset distance. CHARMM force-based LJ switch from 10 Å to 12 Å.
OpenMM uses potential-based switching (unavoidable limitation), but 1.0→1.2 nm
is still the community standard for CHARMM36m + OpenMM.
"""

EWALD_TOLERANCE: float = 0.0005
"""PME grid density relative RMS error. CHARMM36m recommended value."""

# ─────────────────────────────────────────────────────────────────────────────
# SOLVATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

PADDING_NM: float = 1.5
"""
Initial solvation box padding. Extended A10/A8Q2 is ~3.4 nm end-to-end;
1.5 nm padding ensures ≥1.5 nm solvent gap between periodic images (> cutoff).
"""

RESOLVATION_PADDING_NM: float = 1.2
"""
Re-solvation padding for extended conformation (Kuczera 2021 methodology).
Produces a smaller box (~5000–7000 atoms) comparable to the helix-start box.
"""

IONIC_STRENGTH_M: float = 0.15
"""Physiological NaCl concentration (0.15 M). Screens electrostatics."""

# ─────────────────────────────────────────────────────────────────────────────
# RESTRAINT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

RESTRAINT_K: float = 1000.0
"""
Harmonic positional restraint force constant: 1000 kJ/mol/nm².
Applied to all peptide heavy atoms during minimization + NVT only.
At 330 K thermal energy ≈ 2.75 kJ/mol; k=1000 makes a 1 Å displacement cost
10 kJ/mol >> kBT → peptide effectively frozen.
Released at the start of NPT (fresh system deserialization).
"""

# ─────────────────────────────────────────────────────────────────────────────
# MINIMIZATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

MIN_MAX_ITER: int = 5000
"""Maximum L-BFGS minimization iterations. 100–500 is typical; 5000 is a safe limit."""

MIN_TOLERANCE_KJ: float = 10.0
"""Force convergence threshold: 10 kJ/mol/nm = 1 kcal/mol/Å (CHARMM standard)."""

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED STEP COUNTS (computed from the above — do not edit directly)
# ─────────────────────────────────────────────────────────────────────────────

STEPS_PER_PS: int = int(1000 / TIMESTEP_FS)           # 500 at 2 fs

NVT_STEPS: int           = int(NVT_NS           * 1000     * STEPS_PER_PS)
NPT_STEPS: int           = int(NPT_NS           * 1000     * STEPS_PER_PS)
NPT_RESOLVATE_STEPS: int = int(NPT_RESOLVATE_NS * 1000     * STEPS_PER_PS)
PROD_STEPS: int          = int(PROD_US          * 1_000_000 * STEPS_PER_PS)

LOG_STEPS: int = int(LOG_INTERVAL_PS * STEPS_PER_PS)
DCD_STEPS: int = int(DCD_INTERVAL_PS * STEPS_PER_PS)
CHK_STEPS: int = int(CHK_INTERVAL_PS * STEPS_PER_PS)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS / METRICS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

N_HB_MAX: int = 6
"""
Maximum possible α-helical backbone H-bonds for a 10-residue peptide.
An α-helix requires i→i+4 hydrogen bonds; for 10 residues the maximum is 6.
Used as denominator in helix-fraction calculation: f_helix threshold = nhb >= 3.
"""

FES_NBINS: int = 25
"""
Number of bins per axis for the 2D FES histogram (RMSD × N_HB space).
Grid is 25×25 = 625 total bins. Chosen to balance resolution vs. statistical
noise at the sampling density available from 500 ns × 2 replicas (2000 frames).
"""

FES_MIN_COUNT: int = 3
"""
Minimum number of counts in a reference bin for it to be included in the
masked domain Ω used for JSD, coverage, and entropy calculations. Bins with
fewer than 3 counts are too noisy to serve as reliable reference points.
"""

NHB_THRESHOLD_FRAC: float = 0.5
"""
Fraction of N_HB_MAX used as the helix threshold.
f_helix = fraction of frames where nhb >= N_HB_MAX * NHB_THRESHOLD_FRAC = 3.
"""

SEEDING_N_ITER: int = 5
"""Number of adaptive seeding iterations per strategy."""

SEEDING_N_SEEDS: int = 8
"""Number of seed simulations launched per iteration."""

SEEDING_SEED_NS: float = 30.0
"""Duration of each individual seed simulation in nanoseconds."""

STRATEGY_COLORS: dict = {
    "Random":           "#1f77b4",
    "Transferred TICA": "#ff7f0e",
    "Native TICA":      "#2ca02c",
}
"""
Colorblind-friendly palette for the three seeding strategies.
Taken from the matplotlib default 'tab10' cycle (blue, orange, green).
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM / SEQUENCE ALIASES (flat names for easy import)
# ─────────────────────────────────────────────────────────────────────────────

PREFIX_A10: str = "A10"
"""File prefix for Ac-Ala₁₀-NHMe (deca-alanine, source system)."""

PREFIX_A8Q2: str = "A8Q2"
"""File prefix for Ac-Ala₈Gln₂-NHMe (target system, 2 mutations from A10)."""

SEQUENCE_A10: list = ["ALA"] * 10
"""10-residue all-alanine sequence (no caps — caps are added programmatically)."""

SEQUENCE_A8Q2: list = ["ALA","ALA","GLN","ALA","ALA","ALA","ALA","GLN","ALA","ALA"]
"""Ala₈Gln₂: Gln at positions 3 and 8 (1-indexed), symmetric interior placement."""

N_RESIDUES: int = 10
"""Both systems are 10-mer peptides → same TICA feature dimension (40D)."""

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

BURN_IN_NS: float = 25.0
"""
Discard first 25 ns (= 5×τ_eq) before TICA/FES analysis.
Removes initialization bias where the system escapes its starting conformation.
"""

HB_DISTANCE_NM: float = 0.36
"""
i→i+4 backbone H-bond O···N distance cutoff: 3.6 Å = 0.36 nm.
Source: Kuczera et al. 2021 (direct quote from methods).
"""

RMSD_HELIX_NM: float = 0.20
"""Backbone Cα RMSD < 0.20 nm from ideal helix → classified as Helix state."""

RMSD_COIL_NM: float = 0.30
"""Backbone Cα RMSD > 0.30 nm from ideal helix → classified as Coil state.
Frames in the 0.20–0.30 nm transition zone inherit the previous state (hysteresis)."""

HELIX_NHB_FRACTION: float = 0.5
"""H-bond helix criterion: nhb >= N_HB_MAX * 0.5 = 3 bonds → Helix."""

HELIX_DSSP_FRACTION: float = 0.5
"""DSSP helix criterion: fraction of residues assigned 'H' >= 0.5 → Helix."""

FES_CLIP_KCAL: float = 5.0
"""Clip 2D FES at 5 kcal/mol. Bins above this are <0.04% population (effectively empty)."""

RT_KCAL: float = 0.6558
"""RT at 330 K in kcal/mol. R = 1.9872e-3 kcal/mol/K; RT = 1.9872e-3 × 330 = 0.6558."""

# ─────────────────────────────────────────────────────────────────────────────
# TICA PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TICA_LAGTIME_FRAMES_SCAN: list = [1, 2, 3, 5, 8, 10, 15, 20]
"""
Lag-time scan values in frames (100 ps/frame).
In ns: [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0].
Physical sweet spot: τ >> τ_fast (~0.5 ns) but τ << τ_slow (~5 ns).
Expected plateau at 2–4 frames (0.2–0.4 ns).
"""

TICA_DIM: int = 5
"""Number of TICA components to compute. Only TIC-1 is used; extra components
for diagnostic checks and future seeding strategies."""

TICA_EPSILON: float = 1e-6
"""Regularization for TICA generalized eigenvalue problem (ill-conditioning guard)."""

TICA_SCALING: str = "kinetic_map"
"""kinetic_map scaling multiplies TIC projections by their singular values λᵢ,
making Euclidean distance approximate true kinetic distance between states."""

TICA_N_BOOTSTRAP: int = 100
"""Number of block bootstrap resamples for TIC-1 stability verification."""

TICA_BOOTSTRAP_BLOCK_SIZE: int = 20
"""Block size for bootstrap: 20 frames = 10 ns >> τ_eq ≈ 6 ns → approximately independent blocks."""

TICA_BOOTSTRAP_COS_THRESHOLD: float = 0.95
"""Minimum cosine similarity between bootstrap TIC-1 and reference TIC-1.
All 100 bootstrap samples must exceed this threshold for TIC-1 to be 'stable'."""

TICA_PEARSON_THRESHOLD: float = 0.70
"""Minimum |r(TIC-1, N_HB)| for TIC-1 to be considered a valid helix-coil CV.
Below 0.70: fall back to N_HB as CV for adaptive seeding."""

# ─────────────────────────────────────────────────────────────────────────────
# SEEDING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SEED_NS: float = 30.0
"""Duration of each seed simulation. At 330 K τ_fold ≈ 40–115 ns,
so 30 ns has ~23–52% probability of at least one helix-coil transition."""

N_SEEDS: int = 8
"""Seeds per iteration. Balances exploration (diverse starting points) vs
depth (enough time per seed to leave a basin). HTMD typically uses 10–15."""

N_ITERATIONS: int = 5
"""Total adaptive seeding iterations per strategy.
Budget: 3 strategies × 5 iter × 8 seeds × 30 ns = 3600 ns total."""

N_STRATEGIES: int = 3
"""Number of seeding strategies: Random, Transferred TICA, Native TICA."""

N_CV_BINS: int = 10
"""Discretization bins for metastable least-counts selection.
10 bins captures: coil basin, transition region, helix basin, and extremes."""

# ─────────────────────────────────────────────────────────────────────────────
# MONGODB (for RunPod distributed execution — optional)
# ─────────────────────────────────────────────────────────────────────────────

MONGO_URI_ENV_VAR: str = "MONGO_URI"
"""Environment variable name holding the MongoDB Atlas connection URI.
The URI itself is never hardcoded here. Set MONGO_URI in the shell or pod env."""

MONGO_DB: str = "btp_calm"
"""MongoDB database name for this experiment."""
