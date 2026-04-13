import numpy as np
import mdtraj as md
from pathlib import Path
import sys

# Append parent dir to path so src imports work
sys.path.append(".")

from src.analysis import (
    load_trajectory,
    compute_nhb,
    compute_backbone_rmsd_to_ideal_helix,
    pooled_ground_truth_fes_rmsd_nhb,
    histogram_rmsd_nhb_fixed_edges
)
from src.metrics import compute_coverage

# Paths
A8Q2_DIR = Path("outputs/btp_equil/A8Q2")
PROD_TOP_PDB = A8Q2_DIR / "A8Q2_extended_prod_top.pdb" 
GT_DCD_HELIX = A8Q2_DIR / "A8Q2_helix_prod.dcd"
GT_DCD_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod.dcd"
GT_TOP_HELIX = A8Q2_DIR / "A8Q2_helix_prod_top.pdb"
GT_TOP_EXTENDED = A8Q2_DIR / "A8Q2_extended_prod_top.pdb"
HELIX_REF_PDB = A8Q2_DIR / "A8Q2_helix.pdb"

OUTPUT_DIR = Path("outputs/ratchet_demo")

def main():
    print("Loading Ground Truth FES...")
    gt_fes = pooled_ground_truth_fes_rmsd_nhb(
        [(str(GT_DCD_HELIX), str(GT_TOP_HELIX)), (str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED))],
        str(HELIX_REF_PDB)
    )
    ground_truth_hist = np.asarray(gt_fes["hist"])
    edges_rmsd = np.asarray(gt_fes["edges_rmsd"])
    edges_nhb = np.asarray(gt_fes["edges_nhb"])
    
    # 1. Standard MD
    print("Extracting first 10 ns of Standard MD...")
    traj_std_prot = load_trajectory(str(GT_DCD_EXTENDED), str(GT_TOP_EXTENDED), protein_only=True)
    traj_std_10ns = traj_std_prot[:100]
    std_nhb = compute_nhb(traj_std_10ns)
    std_rmsd = compute_backbone_rmsd_to_ideal_helix(traj_std_10ns, str(HELIX_REF_PDB))
    
    std_hist = histogram_rmsd_nhb_fixed_edges(std_rmsd, std_nhb, edges_rmsd, edges_nhb)
    std_cov = compute_coverage(ground_truth_hist, std_hist, min_count=3)
    
    # 2. Ratchet Exploration
    print("Loading Ratchet DCDs...")
    pool_dcds = sorted(list(OUTPUT_DIR.glob("ratchet_r*_s*.dcd")))
    ratchet_rmsd = []
    ratchet_nhb = []
    for dcd in pool_dcds:
        traj_prot = load_trajectory(str(dcd), str(PROD_TOP_PDB), protein_only=True)
        ratchet_nhb.extend(compute_nhb(traj_prot))
        ratchet_rmsd.extend(compute_backbone_rmsd_to_ideal_helix(traj_prot, str(HELIX_REF_PDB)))
        
    ratchet_rmsd = np.array(ratchet_rmsd)
    ratchet_nhb = np.array(ratchet_nhb)
    
    ratchet_hist = histogram_rmsd_nhb_fixed_edges(ratchet_rmsd, ratchet_nhb, edges_rmsd, edges_nhb)
    ratchet_cov = compute_coverage(ground_truth_hist, ratchet_hist, min_count=3)
    
    # Calculate state populations
    # Simple bounds: Coil is RMSD > 0.3 & NHB < 3. Helix is RMSD < 0.20 & NHB >= 3.
    def get_states(rmsd, nhb):
        helix = np.sum((rmsd < 0.20) & (nhb >= 3)) / len(rmsd)
        coil = np.sum((rmsd > 0.30) & (nhb < 3)) / len(rmsd)
        transition = 1.0 - (helix + coil)
        return helix, coil, transition

    std_h, std_c, std_t = get_states(std_rmsd, std_nhb)
    ratch_h, ratch_c, ratch_t = get_states(ratchet_rmsd, ratchet_nhb)
    
    print("\n--- METRICS ---")
    print(f"Total simulated time per protocol: 10 ns")
    print(f"Total frames per protocol: 100 (Standard) vs {len(ratchet_rmsd)} (Ratchet, includes branches)")
    print(f"\nCoverage vs Ground Truth (min_count=3):")
    print(f"  Standard 10 ns: {std_cov:.4f} ({std_cov*100:.1f}%)")
    print(f"  Ratchet 10 ns:  {ratchet_cov:.4f} ({ratchet_cov*100:.1f}%)")
    
    print(f"\nExtremes Explored:")
    print(f"  Standard 10 ns: Min RMSD = {np.min(std_rmsd):.4f} nm, Max NHB = {np.max(std_nhb)}")
    print(f"  Ratchet 10 ns:  Min RMSD = {np.min(ratchet_rmsd):.4f} nm, Max NHB = {np.max(ratchet_nhb)}")
    
    print(f"\nState Distributions:")
    print(f"  Standard 10 ns: Helix={std_h*100:.1f}%, Coil={std_c*100:.1f}%, Transition={std_t*100:.1f}%")
    print(f"  Ratchet 10 ns:  Helix={ratch_h*100:.1f}%, Coil={ratch_c*100:.1f}%, Transition={ratch_t*100:.1f}%")

if __name__ == '__main__':
    main()