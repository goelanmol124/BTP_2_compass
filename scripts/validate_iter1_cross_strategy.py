#!/usr/bin/env python3
"""
Validate that adaptive seeding iteration-1 trajectories match across strategies
for the same seed_id (random / native / transferred).

Protocol: _run_iteration1_seeds uses the same generator PDB and
``random_seed = seed_id + 100`` for every strategy, so DCDs for a given seed
should be bitwise-identical on a fixed platform. If hashes differ, optional
RMSD checks catch numerical drift or mixed-up files.

Usage (after syncing ``archive/outputs_legacy/seeding_a8q2/iter01/...``):

  python scripts/validate_iter1_cross_strategy.py \\
    --seeding-dir archive/outputs_legacy/seeding_a8q2 \\
    --prefix A8Q2 \\
    --n-seeds 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STRATEGIES = ("random", "native", "transferred")


def _dcd_path(seeding_dir: Path, prefix: str, strategy: str, seed_id: int) -> Path:
    return seeding_dir / "iter01" / strategy / f"{prefix}_{strategy}_iter01_seed{seed_id:02d}.dcd"


def _top_path(seeding_dir: Path, prefix: str, strategy: str, seed_id: int) -> Path:
    return seeding_dir / "iter01" / strategy / f"{prefix}_{strategy}_iter01_seed{seed_id:02d}_top.pdb"


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _max_aligned_rmsd_nm(ref, other, protein_atom_indices: np.ndarray) -> float:
    """Max per-frame RMSD (nm) of ``other`` vs ``ref`` after superposing on protein."""
    import mdtraj as md

    if ref.n_frames != other.n_frames:
        return float("nan")
    ref_p = ref.atom_slice(protein_atom_indices, inplace=False)
    oth_p = other.atom_slice(protein_atom_indices, inplace=False)
    rmsds = np.empty(ref_p.n_frames, dtype=np.float64)
    for i in range(ref_p.n_frames):
        t = oth_p.slice(i, copy=True)
        t.superpose(ref_p, frame=i)
        rmsds[i] = float(md.rmsd(t, ref_p, frame=i)[0])
    return float(np.max(rmsds))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--seeding-dir", type=Path, default=Path("archive/outputs_legacy/seeding_a8q2"))
    p.add_argument("--prefix", default="A8Q2")
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument(
        "--rmsd-threshold-nm",
        type=float,
        default=1.0e-6,
        help="If SHA256 differs, fail unless max aligned protein RMSD is below this (nm).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable report.",
    )
    p.add_argument(
        "--skip-rmsd",
        action="store_true",
        help="On DCD hash mismatch, record initial-top SHA256 only (no trajectory load; fast).",
    )
    args = p.parse_args()

    repo = args.repo_root.resolve()
    seeding_dir = (repo / args.seeding_dir).resolve()

    per_seed: list[dict] = []
    all_ok = True
    missing_any = False

    for seed_id in range(args.n_seeds):
        paths: dict[str, Path] = {}
        for s in STRATEGIES:
            paths[s] = _dcd_path(seeding_dir, args.prefix, s, seed_id)

        present = {s: paths[s].is_file() for s in STRATEGIES}
        if not any(present.values()):
            continue
        if not all(present.values()):
            missing_any = True
            all_ok = False
            per_seed.append(
                {
                    "seed_id": seed_id,
                    "status": "missing_dcd",
                    "paths": {k: str(v) for k, v in paths.items()},
                    "present": present,
                }
            )
            continue

        hashes = {s: _file_sha256(paths[s]) for s in STRATEGIES}
        ref_h = hashes["random"]
        same_hash = all(hashes[s] == ref_h for s in STRATEGIES)
        tops = {s: _top_path(seeding_dir, args.prefix, s, seed_id) for s in STRATEGIES}
        top_hashes = {
            s: (_file_sha256(tops[s]) if tops[s].is_file() else None) for s in STRATEGIES
        }
        initial_tops_match_rn = (
            top_hashes["random"] is not None
            and top_hashes["random"] == top_hashes["native"]
        )
        initial_tops_match_rt = (
            top_hashes["random"] is not None
            and top_hashes["random"] == top_hashes["transferred"]
        )

        entry: dict = {
            "seed_id": seed_id,
            "status": "ok",
            "sha256": hashes,
            "byte_identical_across_strategies": same_hash,
            "sizes_bytes": {s: paths[s].stat().st_size for s in STRATEGIES},
            "initial_top_sha256": top_hashes,
            "initial_top_random_vs_native": initial_tops_match_rn,
            "initial_top_random_vs_transferred": initial_tops_match_rt,
        }

        if not same_hash:
            if args.skip_rmsd:
                entry["status"] = "hash_mismatch_skip_rmsd"
                all_ok = False
            elif not all(tops[s].is_file() for s in STRATEGIES):
                entry["status"] = "hash_mismatch_no_top_for_rmsd"
                entry["rmsd_nm"] = None
                all_ok = False
            else:
                import mdtraj as md

                ref = md.load(str(paths["random"]), top=str(tops["random"]))
                prot = ref.topology.select("protein")
                if prot.size == 0:
                    entry["status"] = "no_protein_atoms"
                    all_ok = False
                else:
                    max_r: dict[str, float] = {}
                    ok_rmsd = True
                    for s in ("native", "transferred"):
                        otr = md.load(str(paths[s]), top=str(tops[s]))
                        m = _max_aligned_rmsd_nm(ref, otr, prot)
                        max_r[s] = m
                        if not np.isfinite(m) or m > args.rmsd_threshold_nm:
                            ok_rmsd = False
                    entry["rmsd_nm_vs_random"] = max_r
                    entry["rmsd_within_threshold"] = ok_rmsd
                    if not ok_rmsd:
                        entry["status"] = "hash_mismatch_rmsd_fail"
                        all_ok = False
                    else:
                        entry["status"] = "ok_equivalent_within_rmsd"
        per_seed.append(entry)

    if not per_seed:
        print(
            f"No DCDs found under {seeding_dir / 'iter01'} "
            f"(expected {args.prefix}_*_iter01_seed*.dcd). Sync seeding outputs first.",
            file=sys.stderr,
        )
        sys.exit(2)

    report = {
        "seeding_dir": str(seeding_dir),
        "prefix": args.prefix,
        "n_seeds_requested": args.n_seeds,
        "all_checks_passed": all_ok and not missing_any,
        "missing_partial_set": missing_any,
        "per_seed": per_seed,
    }

    if args.output_json is not None:
        out = (repo / args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(str(out))

    # Human-readable summary
    for row in per_seed:
        sid = row["seed_id"]
        st = row["status"]
        if st == "ok":
            bi = row.get("byte_identical_across_strategies", False)
            print(f"seed {sid:02d}: OK  byte_identical={bi}")
        elif st == "ok_equivalent_within_rmsd":
            print(f"seed {sid:02d}: OK  (hashes differ; RMSD within threshold)")
        elif st == "hash_mismatch_skip_rmsd":
            rn = row.get("initial_top_random_vs_native")
            rt = row.get("initial_top_random_vs_transferred")
            print(f"seed {sid:02d}: FAIL  {st}  (initial tops R==N: {rn}, R==T: {rt})")
        else:
            print(f"seed {sid:02d}: FAIL  {st}")
            if "sha256" in row:
                h = row["sha256"]
                print(f"         sha256 random={h['random'][:16]}... native={h['native'][:16]}...")

    sys.exit(0 if report["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
