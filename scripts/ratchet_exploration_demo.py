import os
import sys
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
import mdtraj as md
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

# Import necessary functions from the codebase
from src.adaptive_seeding import project_cv, select_seeds_least_counts
from src.utils import get_best_platform
from src.analysis import (
    pooled_ground_truth_fes_rmsd_nhb, 
    compute_backbone_rmsd_to_ideal_helix, 
    compute_nhb, 
    compute_featurization, 
    load_trajectory
)
from openmm import XmlSerializer

# Set up paths
A8Q2_DIR = Path("outputs/btp_equil/A8Q2")
GENERATOR_PDB = A8Q2_DIR / "A8Q2_extended_npt_final.pdb"
SYSTEM_XML = A8Q2_DIR / "A8Q2_extended_npt_final.xml"
PROD_TOP_PDB = A8Q2_DIR / "A8Q2_extended_prod_top.pdb" 

GT_DCD_HELIX = A8Q2_DIR / "A8Q2_helix_prod.dcd"
GT_DCD_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod.dcd"
GT_TOP_HELIX = A8Q2_DIR / "A8Q2_helix_prod_top.pdb"
GT_TOP_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod_top.pdb"
HELIX_REF_PDB = A8Q2_DIR / "A8Q2_helix.pdb"

TICA_DIR = Path("outputs/btp_equil/A10/tica_validation_run_v2")
TICA_EV = TICA_DIR / "a10_tica_singular_vectors_left.npy"
TICA_MEAN = TICA_DIR / "a10_tica_mean0.npy"

OUTPUT_DIR = Path("outputs/ratchet_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
TEMPERATURE = 330.0 * unit.kelvin
FRICTION = 1.0 / unit.picosecond
TIMESTEP = 2.0 * unit.femtoseconds

ROUNDS = 10
SEEDS_PER_ROUND = 5
SEED_NS = 0.2
DCD_INTERVAL_PS = 10.0 # Save every 10 ps to have 20 frames per seed
STEPS_PER_PS = int(1000 / 2.0)
SEED_STEPS = int(SEED_NS * 1000 * STEPS_PER_PS)
DCD_STEPS = int(DCD_INTERVAL_PS * STEPS_PER_PS)

def run_seed_simulation(seed_frame_coords, box_vectors, out_dcd, steps, platform, rng_seed):
    with open(SYSTEM_XML) as f:
        system = XmlSerializer.deserialize(f.read())
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
    integrator.setRandomNumberSeed(rng_seed)
    
    # Load topology safely
    pdb_top = app.PDBFile(str(PROD_TOP_PDB))
    n_sys = system.getNumParticles()
    if pdb_top.topology.getNumAtoms() == n_sys:
        omm_top = pdb_top.topology
    else:
        traj_top = md.load_pdb(str(PROD_TOP_PDB))
        omm_top = traj_top.topology.to_openmm()
    
    sim = app.Simulation(omm_top, system, integrator, platform)
    
    if len(box_vectors) == 3:
        sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setPositions(seed_frame_coords)
    sim.context.applyConstraints(1e-7)
    sim.context.setVelocitiesToTemperature(TEMPERATURE, rng_seed)
    
    sim.reporters.append(app.DCDReporter(out_dcd, DCD_STEPS))
    sim.step(steps)

def get_frame_data(dcd_path, frame_idx):
    traj = md.load_frame(str(dcd_path), frame_idx, top=str(PROD_TOP_PDB))
    return traj.openmm_positions(0)

def main():
    platform = get_best_platform()
    
    # Load generator positions
    print("Loading generator positions...")
    from src.simulate import _npt_final_topology_positions_box
    with open(SYSTEM_XML) as f:
        system = XmlSerializer.deserialize(f.read())
    _, pos0, box0 = _npt_final_topology_positions_box(GENERATOR_PDB, system.getNumParticles())

    # 1. Standard MD run -> Slice from existing 500ns GT DCD
    print("=== Extracting first 10 ns of Standard MD ===")
    traj_std_prot = load_trajectory(str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED), protein_only=True)
    # GT DCD was saved every 100 ps, so 10 ns = 100 frames
    traj_std_10ns = traj_std_prot[:100]
    std_nhb = compute_nhb(traj_std_10ns)
    std_rmsd = compute_backbone_rmsd_to_ideal_helix(traj_std_10ns, str(HELIX_REF_PDB))

    # 2. Ratchet Exploration
    print("=== Running Ratchet Exploration ===")
    eigenvectors_donor = np.load(TICA_EV)
    mean_donor = np.load(TICA_MEAN)
    
    pool_dcds = []
    
    for r in range(1, ROUNDS + 1):
        round_dcds = []
        if r == 1:
            print(f"Round 1: spawning {SEEDS_PER_ROUND} seeds from generator...")
            for s in range(SEEDS_PER_ROUND):
                out_dcd = OUTPUT_DIR / f"ratchet_r{r}_s{s}.dcd"
                round_dcds.append(out_dcd)
                if not out_dcd.exists():
                    run_seed_simulation(pos0, box0, str(out_dcd), SEED_STEPS, platform, rng_seed=100*r+s)
        else:
            print(f"Round {r}: picking seeds and spawning {SEEDS_PER_ROUND} seeds...")
            all_features = []
            frame_map = [] 
            
            for dcd in pool_dcds:
                traj_prot = load_trajectory(str(dcd), str(PROD_TOP_PDB), protein_only=True)
                feats = compute_featurization(traj_prot)
                all_features.append(feats)
                for f_idx in range(len(traj_prot)):
                    frame_map.append((str(dcd), f_idx))
                    
            pooled_features = np.concatenate(all_features, axis=0)
            cv_values = project_cv(pooled_features, strategy="transferred", mean_donor=mean_donor, eigenvectors_donor=eigenvectors_donor)
            seed_indices = select_seeds_least_counts(cv_values, n_seeds=SEEDS_PER_ROUND, n_bins=10, rng_seed=r)
            
            for s, pooled_idx in enumerate(seed_indices):
                out_dcd = OUTPUT_DIR / f"ratchet_r{r}_s{s}.dcd"
                round_dcds.append(out_dcd)
                if not out_dcd.exists():
                    src_dcd, src_frame_idx = frame_map[pooled_idx]
                    pos = get_frame_data(src_dcd, src_frame_idx)
                    run_seed_simulation(pos, box0, str(out_dcd), SEED_STEPS, platform, rng_seed=100*r+s)
                    
        pool_dcds.extend(round_dcds)

    print("=== Analysis & Plotting ===")
    ratchet_rmsd = []
    ratchet_nhb = []
    for dcd in pool_dcds:
        traj_prot = load_trajectory(str(dcd), str(PROD_TOP_PDB), protein_only=True)
        ratchet_nhb.extend(compute_nhb(traj_prot))
        ratchet_rmsd.extend(compute_backbone_rmsd_to_ideal_helix(traj_prot, str(HELIX_REF_PDB)))
    
    print("Loading Ground Truth FES...")
    gt_fes = pooled_ground_truth_fes_rmsd_nhb(
        [(str(GT_DCD_HELIX), str(GT_TOP_HELIX)), (str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED))],
        str(HELIX_REF_PDB)
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    hist = np.asarray(gt_fes["hist"])
    hist[hist == 0] = np.nan
    X, Y = np.meshgrid(gt_fes["edges_rmsd"], gt_fes["edges_nhb"])
    
    for i, ax in enumerate(axes):
        ax.pcolormesh(X, Y, hist.T, norm=LogNorm(vmin=1, vmax=np.nanmax(hist)), cmap='Greys', alpha=0.5)
        ax.set_xlabel("Backbone RMSD (nm)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Number of H-bonds", fontsize=12)
            
    axes[0].plot(std_rmsd, std_nhb, marker='o', linestyle='-', color='blue', markersize=4, alpha=0.6, linewidth=0.8, label='Standard 10 ns')
    axes[0].set_title("Standard MD (first 10 ns from ground-truth)", fontsize=14)
    axes[0].legend()
    
    axes[1].plot(ratchet_rmsd, ratchet_nhb, marker='o', linestyle='', color='red', markersize=3, alpha=0.4, label='Ratchet (10 ns total)')
    axes[1].set_title("Directed Ratchet Exploration\n(10 rounds x 5 seeds x 0.2 ns)", fontsize=14)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ratchet_comparison.png", dpi=300)
    print(f"Plot saved to {OUTPUT_DIR / 'ratchet_comparison.png'}")

if __name__ == "__main__":
    main()