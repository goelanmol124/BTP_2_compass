#!/usr/bin/env python3
"""Run COMPASS v2 planned experiment blocks (A–E) via ``compass_v2_bench``.

Writes one output directory per run under ``outputs/v2/plan/`` and a ``manifest.json``.
See docs/COMPASS_V2_EXPERIMENT_MATRIX.md.

Examples::

    python scripts/compass_v2_experiment_suite.py --quick
    python scripts/compass_v2_experiment_suite.py --blocks a,b --random-repeats 36

The plan specifies oracle artifacts under ``outputs/v2/oracle/``; this script can
call ``--build-eq-only`` for M3 k=40 when indexes are missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "scripts" / "compass_v2_bench.py"
ORACLE = REPO_ROOT / "outputs" / "v2" / "oracle"
PLAN_ROOT = REPO_ROOT / "outputs" / "v2" / "plan"
SPIB_ASSIGNMENTS = REPO_ROOT / "outputs" / "spib_a10_analysis" / "spib_a10_frame_assignments.npz"


def spib_atlas_ready() -> bool:
    return SPIB_ASSIGNMENTS.is_file()


@dataclass
class RunSpec:
    name: str
    argv: list[str]
    description: str = ""


def _py() -> str:
    return sys.executable


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_eq_index(eq_method: str, eq_k: int, *, macro_n_clusters: int = 16) -> None:
    path = ORACLE / f"eq_{eq_method}_k{eq_k}.npz"
    if path.is_file():
        return
    print(f"Building missing equivalence index: {path.name}", flush=True)
    _run(
        [
            _py(),
            str(BENCH),
            "--repo-root",
            str(REPO_ROOT),
            "--output-dir",
            str(PLAN_ROOT / "_eq_build_tmp"),
            "--build-eq-only",
            "--eq-method",
            eq_method,
            "--eq-k",
            str(eq_k),
            "--macro-n-clusters",
            str(macro_n_clusters),
        ]
    )


def build_manifest_entries(
    *,
    quick: bool,
    cold_start: int,
    eq_k: int,
    eq_methods: list[str],
    random_repeats: int,
    random_seed: int,
    include_e: bool,
) -> dict[str, list[RunSpec]]:
    n_steps = 8 if quick else 80
    n_rep = min(random_repeats, 4) if quick else random_repeats
    chunk = 10
    # same 80 ns budget: chunk 20 -> n_steps 40
    chunk_c = 20
    n_steps_c = 40 if not quick else 8
    n_steps_long = 20 if quick else 200

    common_tica = [
        "--repo-root",
        str(REPO_ROOT),
        "--map-kind",
        "tica2d",
        "--schedule",
        "two_phase",
        "--deficit-mode",
        "information_gain",
        "--variant",
        "balanced",
        "--chunk-frames",
        str(chunk),
        "--n-steps",
        str(n_steps),
        "--random-seed",
        str(random_seed),
        "--eq-k",
        str(eq_k),
        "--cold-start-frame",
        str(cold_start),
        "--random-repeats",
        str(n_rep),
    ]

    def bench_args(
        policy: str,
        out_rel: str,
        run_rs: bool,
        eq_m: str,
        *,
        map_kind: str = "tica2d",
        cold: int | None = None,
        seed: int | None = None,
    ) -> list[str]:
        base = list(common_tica)
        out_base: list[str] = []
        i = 0
        while i < len(base):
            if base[i] == "--map-kind" and i + 1 < len(base):
                out_base.extend(["--map-kind", map_kind])
                i += 2
                continue
            if cold is not None and base[i] == "--cold-start-frame" and i + 1 < len(base):
                out_base.extend(["--cold-start-frame", str(cold)])
                i += 2
                continue
            if seed is not None and base[i] == "--random-seed" and i + 1 < len(base):
                out_base.extend(["--random-seed", str(seed)])
                i += 2
                continue
            out_base.append(base[i])
            i += 1
        a = [
            str(BENCH),
            *out_base,
            "--output-dir",
            str(REPO_ROOT / out_rel),
            "--policy",
            policy,
            "--eq-method",
            eq_m,
        ]
        if run_rs:
            a.append("--run-random-summary")
        return a

    block_a: list[RunSpec] = []
    for pol, sub, rs in [
        ("compass", "block_a/compass_tica2d", True),
        ("random", "block_a/random_tica2d", True),
        ("uniform_prior", "block_a/uniform_prior_tica2d", True),
        ("chrono", "block_a/chrono_tica2d", False),
    ]:
        block_a.append(
            RunSpec(
                name=f"a_{pol}",
                argv=bench_args(pol, f"outputs/v2/plan/{sub}", rs, "M3"),
                description=f"Block A {pol} TICA2D",
            )
        )

    block_b = [
        RunSpec(
            "b_spib_compass",
            bench_args("compass", "outputs/v2/plan/block_b/compass_spib1d", True, "M3", map_kind="spib1d"),
            "Block B SPIB1D compass + random summary",
        )
    ]

    block_c = [
        RunSpec(
            "c_chunk20",
            [
                str(BENCH),
                "--repo-root",
                str(REPO_ROOT),
                "--map-kind",
                "tica2d",
                "--schedule",
                "two_phase",
                "--deficit-mode",
                "information_gain",
                "--variant",
                "balanced",
                "--chunk-frames",
                str(chunk_c),
                "--n-steps",
                str(n_steps_c),
                "--random-seed",
                str(random_seed),
                "--eq-k",
                str(eq_k),
                "--cold-start-frame",
                str(cold_start),
                "--random-repeats",
                str(n_rep),
                "--output-dir",
                str(REPO_ROOT / "outputs/v2/plan/block_c/compass_chunk20"),
                "--policy",
                "compass",
                "--eq-method",
                "M3",
                "--run-random-summary",
            ],
            "Block C chunk 20 / steps 40 (80 ns budget)",
        )
    ]

    block_d: list[RunSpec] = []
    for eq_alt in eq_methods:
        if eq_alt == "M3":
            continue
        block_d.append(
            RunSpec(
                f"d_compass_{eq_alt}",
                bench_args("compass", f"outputs/v2/plan/block_d/compass_{eq_alt}", True, eq_alt),
                f"Block D compass with {eq_alt}",
            )
        )

    block_e = []
    if include_e:
        block_e.append(
            RunSpec(
                "e_long",
                [
                    str(BENCH),
                    "--repo-root",
                    str(REPO_ROOT),
                    "--map-kind",
                    "tica2d",
                    "--schedule",
                    "two_phase",
                    "--deficit-mode",
                    "information_gain",
                    "--variant",
                    "balanced",
                    "--chunk-frames",
                    str(chunk),
                    "--n-steps",
                    str(n_steps_long),
                    "--random-seed",
                    str(random_seed),
                    "--eq-k",
                    str(eq_k),
                    "--cold-start-frame",
                    str(cold_start),
                    "--random-repeats",
                    str(n_rep),
                    "--output-dir",
                    str(REPO_ROOT / "outputs/v2/plan/block_e/compass_long"),
                    "--policy",
                    "compass",
                    "--eq-method",
                    "M3",
                    "--run-random-summary",
                ],
                "Block E long budget TICA2D compass",
            )
        )

    cold_start_b = RunSpec(
        "a_cold_alt",
        bench_args(
            "compass",
            "outputs/v2/plan/block_a/compass_tica2d_cs4500",
            True,
            "M3",
            cold=4500,
            seed=202,
        ),
        "Optional cold-start sensitivity (4500 / seed 202)",
    )

    return {
        "block_a": block_a,
        "block_b": block_b,
        "block_c": block_c,
        "block_d": block_d,
        "block_e": block_e,
        "cold_start_optional": [cold_start_b],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--blocks",
        type=str,
        default="a,b,c,d,e",
        help="Comma list: a,b,c,d,e,cold — or 'all' (default includes e and cold)",
    )
    p.add_argument("--quick", action="store_true", help="Tiny n_steps / repeats for smoke tests")
    p.add_argument("--random-repeats", type=int, default=36)
    p.add_argument("--random-seed", type=int, default=101)
    p.add_argument("--cold-start-frame", type=int, default=1000)
    p.add_argument("--eq-k", type=int, default=40)
    p.add_argument(
        "--eq-alt",
        type=str,
        default="M7",
        help="Equivalence method for block D (e.g. M7 or M5)",
    )
    p.add_argument("--skip-block-b", action="store_true", help="Skip SPIB1D block (e.g. missing training artifacts)")
    p.add_argument(
        "--require-spib",
        action="store_true",
        help="Fail if block B is requested but SPIB assignments file is missing",
    )
    p.add_argument("--skip-block-e", action="store_true", help="Omit long-budget block E")
    p.add_argument("--skip-cold-repeat", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    blocks_req = {x.strip().lower() for x in args.blocks.split(",") if x.strip()}
    if "all" in blocks_req:
        blocks_req = {"a", "b", "c", "d", "e", "cold"}

    eq_alts = ["M3", args.eq_alt.strip().upper()]
    manifests = build_manifest_entries(
        quick=args.quick,
        cold_start=args.cold_start_frame,
        eq_k=args.eq_k,
        eq_methods=eq_alts,
        random_repeats=args.random_repeats,
        random_seed=args.random_seed,
        include_e="e" in blocks_req and not args.skip_block_e,
    )

    _ensure_eq_index("M3", args.eq_k)
    alt = args.eq_alt.strip().upper()
    if alt != "M3":
        _ensure_eq_index(alt, args.eq_k)

    if "b" in blocks_req and not args.skip_block_b:
        if not spib_atlas_ready():
            msg = (
                f"SPIB map requires {SPIB_ASSIGNMENTS} — build SPIB analysis or pass --skip-block-b"
            )
            if args.require_spib:
                raise SystemExit(msg)
            print(f"WARNING: {msg}. Skipping block B.", flush=True)
            blocks_req.discard("b")

    runs: list[RunSpec] = []
    if "a" in blocks_req:
        runs.extend(manifests["block_a"])
    if "b" in blocks_req:
        runs.extend(manifests["block_b"])
    if "c" in blocks_req:
        runs.extend(manifests["block_c"])
    if "d" in blocks_req:
        runs.extend(manifests["block_d"])
    if "e" in blocks_req and not args.skip_block_e:
        runs.extend(manifests["block_e"])
    if "cold" in blocks_req and not args.skip_cold_repeat:
        runs.extend(manifests["cold_start_optional"])

    out_manifest: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "args": vars(args),
        "runs": [{"name": r.name, "description": r.description, "argv": r.argv} for r in runs],
    }
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    man_path = PLAN_ROOT / "manifest.json"

    if args.dry_run:
        print(json.dumps(out_manifest, indent=2))
        man_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
        print(f"Wrote {man_path}")
        return

    for spec in runs:
        cmd = [_py()] + spec.argv
        _run(cmd)

    man_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {man_path}")


if __name__ == "__main__":
    main()
