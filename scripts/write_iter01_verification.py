#!/usr/bin/env python3
"""Offline: write outputs/.../iter01_verification.json using adaptive_seeding checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.adaptive_seeding import (  # noqa: E402
    _verify_iter1_initial_tops_consistent,
    _write_iter1_verification_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=REPO)
    p.add_argument("--seeding-dir", type=Path, default=Path("archive/outputs_legacy/seeding_a8q2"))
    p.add_argument("--prod-top", type=Path, default=Path("outputs/btp_equil/A8Q2/A8Q2_extended_prod_top.pdb"))
    p.add_argument("--generator-pdb", type=Path, default=Path("outputs/btp_equil/A8Q2/A8Q2_extended_npt_final.pdb"))
    p.add_argument("--n-seeds", type=int, default=6)
    args = p.parse_args()

    repo = args.repo_root.resolve()
    out = (repo / args.seeding_dir).resolve()
    prod = (repo / args.prod_top).resolve()
    gen = (repo / args.generator_pdb).resolve()

    strategies = ["random", "native", "transferred"]
    v = _verify_iter1_initial_tops_consistent(
        output_dir=out,
        strategies=strategies,
        n_seeds=args.n_seeds,
        prod_top_pdb=str(prod),
        generator_pdb=str(gen),
    )
    _write_iter1_verification_json(out, v)
    print(json.dumps(v, indent=2))
    sys.exit(0 if v.get("passed") else 1)


if __name__ == "__main__":
    main()
