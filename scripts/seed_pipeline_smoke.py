#!/usr/bin/env python3
"""
Local smoke test: TICA (transferred) least-counts seed selection + short NVT MD + DCD merge.

Uses A10 extended production only (one DCD/topology so frame indices are unambiguous).
Does not run the full adaptive_seeding driver (30 ns x 8 seeds); validates the idea:
  features -> CV -> select_seeds_least_counts -> extract frames -> OpenMM -> pool DCDs.

Usage (from repo root)::

    python scripts/seed_pipeline_smoke.py
    python scripts/seed_pipeline_smoke.py --n_seeds 4 --n_steps 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _ensure_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def main() -> int:
    _ensure_path()

    parser = argparse.ArgumentParser(description="TICA seeding + short MD smoke test (A10 extended).")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of least-counts seeds (default 3).")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2500,
        help="MD steps per seed at 2 fs (default 2500 = 5 ps).",
    )
    parser.add_argument(
        "--dcd_every_steps",
        type=int,
        default=500,
        help="Write DCD frame every this many steps (default 500 = 1 ps).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO / "outputs" / "seed_smoke_a10",
        help="Output directory for seed DCDs and summary.",
    )
    args = parser.parse_args()

    from openmm import XmlSerializer, unit
    import openmm as mm
    import openmm.app as app
    import mdtraj as md

    from src.adaptive_seeding import project_cv, select_seeds_least_counts
    from src.analysis import BURNIN_FRAMES, compute_featurization, compute_nhb
    from src.config import FRICTION_PS, TEMPERATURE_K, TIMESTEP_FS
    from src.utils import apply_windows_checkpoint_patch, get_best_platform

    apply_windows_checkpoint_patch()

    a10 = REPO / "outputs" / "btp_equil" / "A10"
    prod_dcd = a10 / "A10_extended_prod.dcd"
    prod_top = a10 / "A10_extended_prod_top.pdb"
    system_xml = a10 / "A10_extended_npt_final.xml"
    # OpenMM PDBFile mis-parses prod_top (duplicate H serial → 4169 atoms); System XML
    # has 4170 particles. NPT final PDB matches the serialized system for Simulation.
    top_openmm = a10 / "A10_extended_npt_final.pdb"
    tica_dir = a10 / "tica_validation_run_v2"

    for p in (prod_dcd, prod_top, system_xml, top_openmm):
        if not p.is_file():
            print(f"ERROR: missing required file: {p}")
            return 1

    mean_path = tica_dir / "a10_tica_mean0.npy"
    vec_path = tica_dir / "a10_tica_singular_vectors_left.npy"
    if not mean_path.is_file() or not vec_path.is_file():
        print(
            "ERROR: TICA artefacts not found. Expected:\n"
            f"  {mean_path}\n  {vec_path}\n"
            "Run: python -m src.tica_train ... (see docs/EXPERIMENT_PROTOCOL.md Phase 6)."
        )
        return 1

    mean0 = np.load(mean_path)
    sv_left = np.load(vec_path)
    if mean0.shape != (40,) or sv_left.shape[0] != 40:
        raise ValueError(f"Unexpected TICA shapes: mean {mean0.shape}, vectors {sv_left.shape}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full system trajectory (same atom order as OpenMM / prod_top).
    print(f"[smoke] Loading {prod_dcd.name} …")
    traj_sys = md.load_dcd(str(prod_dcd), top=str(prod_top))
    if traj_sys.n_frames <= BURNIN_FRAMES:
        print("ERROR: trajectory shorter than burn-in.")
        return 1
    traj_post = traj_sys[BURNIN_FRAMES:]
    sel = traj_post.topology.select("protein or resname ACE or resname NME")
    traj_prot = traj_post.atom_slice(sel)

    features = np.asarray(compute_featurization(traj_prot), dtype=np.float64)
    nhb = compute_nhb(traj_prot)
    n_frames = features.shape[0]
    print(f"[smoke] Post burn-in: {n_frames} frames x {features.shape[1]} features")

    cv = project_cv(
        features,
        "transferred",
        mean_donor=mean0,
        eigenvectors_donor=sv_left,
    )
    assert cv is not None

    seed_indices = select_seeds_least_counts(
        cv,
        n_seeds=args.n_seeds,
        n_bins=10,
        rng_seed=42,
    )
    print(f"[smoke] Least-counts seed frame indices (post-burn-in): {seed_indices.tolist()}")
    print(f"[smoke] CV at seeds: {cv[seed_indices].tolist()}")

    with open(system_xml) as fh:
        system = XmlSerializer.deserialize(fh.read())

    platform = get_best_platform()
    pdb = app.PDBFile(str(top_openmm))

    dcd_paths: list[str] = []
    summary_seeds: list[dict] = []

    for i, seed_local in enumerate(seed_indices):
        seed_local = int(seed_local)
        prefix = f"A10_ext_smoke_seed{i:02d}_f{seed_local}"
        dcd_path = str(out_dir / f"{prefix}.dcd")
        log_path = str(out_dir / f"{prefix}.log")

        frame = traj_post[seed_local]
        box = frame.openmm_boxes(0)
        pos = frame.openmm_positions(0)

        integrator = mm.LangevinMiddleIntegrator(
            TEMPERATURE_K * unit.kelvin,
            FRICTION_PS / unit.picosecond,
            TIMESTEP_FS * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(1000 + i)
        sim = app.Simulation(pdb.topology, system, integrator, platform)
        sim.context.setPeriodicBoxVectors(*box)
        sim.context.setPositions(pos)
        sim.context.applyConstraints(1e-7)
        sim.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 1000 + i)

        top_seed = out_dir / f"{prefix}_top.pdb"
        st0 = sim.context.getState(getPositions=True)
        with open(top_seed, "w") as fh:
            app.PDBFile.writeFile(sim.topology, st0.getPositions(), fh)

        sim.reporters.clear()
        sim.reporters.append(
            app.StateDataReporter(
                log_path,
                max(1, args.dcd_every_steps),
                step=True,
                time=True,
                temperature=True,
                progress=True,
                totalSteps=args.n_steps,
                separator="\t",
            )
        )
        sim.reporters.append(app.DCDReporter(dcd_path, args.dcd_every_steps))

        print(f"[smoke] Running seed {i} (frame {seed_local}): {args.n_steps} steps …")
        sim.step(args.n_steps)
        # DCDReporter keeps the file handle open until the reporter is collected;
        # flush/close before mdtraj reads (avoids premature EOF on Windows).
        for rep in sim.reporters:
            out = getattr(rep, "_out", None)
            if out is not None and not out.closed:
                out.flush()
                out.close()
        sim.reporters.clear()
        del sim
        gc.collect()
        dcd_paths.append(dcd_path)
        summary_seeds.append(
            {
                "seed_index": i,
                "frame_post_burnin": seed_local,
                "cv_value": float(cv[seed_local]),
                "dcd": dcd_path,
                "n_steps": args.n_steps,
            }
        )

    # Combine DCDs along time (same topology).
    print("[smoke] Merging trajectories …")
    top0 = str(Path(dcd_paths[0]).with_name(Path(dcd_paths[0]).stem + "_top.pdb"))
    trajs = [md.load_dcd(str(Path(p).resolve()), top=str(Path(top0).resolve())) for p in dcd_paths]
    merged = md.join(trajs)
    combined = out_dir / "combined_smoke.dcd"
    merged.save_dcd(str(combined))
    print(f"[smoke] Combined: {combined}  ({merged.n_frames} frames)")

    meta = {
        "n_seeds": args.n_seeds,
        "n_steps_per_seed": args.n_steps,
        "dcd_every_steps": args.dcd_every_steps,
        "burn_in_frames_skipped": BURNIN_FRAMES,
        "prod_dcd": str(prod_dcd),
        "tica_dir": str(tica_dir),
        "combined_dcd": str(combined),
        "seeds": summary_seeds,
    }
    meta_path = out_dir / "seed_smoke_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[smoke] Wrote {meta_path}")
    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
