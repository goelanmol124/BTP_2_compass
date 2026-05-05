#!/usr/bin/env python3
"""
Post–Phase-8 automation for a single RunPod (or any Linux host).

Typical use after starting the first adaptive seeding run **without** TICA artifacts::

    nohup .venv-openmm/bin/python -u scripts/pod_post_seeding_chain.py \\
        --repo-root /workspace/btp_2_compass \\
        --wait-for-json outputs/seeding_a8q2/adaptive_seeding_results.json \\
        >> post_chain.log 2>&1 &

When ``adaptive_seeding_results.json`` appears (first run finished), this script:

1. Runs **Phase 7** — ``python -m src.tica_transfer`` (needs ``deeptime``).
2. Runs **Phase 8 again** with ``--tica_artifacts_dir`` → all three strategies
   into ``outputs/seeding_a8q2_3way`` (default).
3. Runs **Phase 9–style bundles** — ``metrics.run_phase9_bundle_for_seeding_run``
   for the first seeding tree and for the 3-way tree.

Idempotent: skips Phase 7 / second seeding / bundles if outputs already exist
unless ``--force``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _seeding_cmd(
    py: Path,
    repo_root: Path,
    sim_dir: Path,
    prefix: str,
    seeding_out: Path,
    tica_dir: Path | None,
) -> list[str]:
    gen_pdb = sim_dir / f"{prefix}_extended_npt_final.pdb"
    sys_xml = sim_dir / f"{prefix}_extended_npt_final.xml"
    prod_top = sim_dir / f"{prefix}_extended_prod_top.pdb"
    dcd_h = sim_dir / f"{prefix}_helix_prod.dcd"
    dcd_e = sim_dir / f"{prefix}_extended_prod.dcd"
    gt_h = sim_dir / f"{prefix}_helix_prod_top.pdb"
    gt_e = sim_dir / f"{prefix}_extended_prod_top.pdb"
    for p in (gen_pdb, sys_xml, prod_top, dcd_h, dcd_e, gt_h, gt_e):
        if not p.is_file():
            raise FileNotFoundError(f"Seeding prerequisite missing: {p}")

    cmd: list[str | Path] = [
        py,
        "-u",
        "-m",
        "src.adaptive_seeding",
        "--generator_pdb",
        str(gen_pdb),
        "--system_xml",
        str(sys_xml),
        "--prod_top_pdb",
        str(prod_top),
        "--gt_dcd_helix",
        str(dcd_h),
        "--gt_dcd_extended",
        str(dcd_e),
        "--gt_top_helix",
        str(gt_h),
        "--gt_top_extended",
        str(gt_e),
        "--output_dir",
        str(seeding_out),
        "--strategy",
        "all",
    ]
    if tica_dir is not None:
        ev = tica_dir / "a10_tica_singular_vectors_left.npy"
        mn = tica_dir / "a10_tica_mean0.npy"
        if ev.is_file() and mn.is_file():
            cmd.extend(["--tica_artifacts_dir", str(tica_dir)])
    return [str(x) for x in cmd]


def _run_checked(cmd: list[str], cwd: Path, label: str) -> None:
    print(f"[post-chain] {label}: {' '.join(cmd[:6])} …", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def wait_for_results_json(
    path: Path,
    poll_s: float,
    stable_s: float,
    also_require_no_process: bool,
) -> None:
    """Block until *path* exists and size stable, optionally no ``adaptive_seeding`` process."""
    path = Path(path)
    print(f"[post-chain] Waiting for {path} …", flush=True)
    last_size = -1
    stable_since: float | None = None
    check_proc = also_require_no_process
    while True:
        if path.is_file():
            sz = path.stat().st_size
            if sz > 32:
                if check_proc:
                    try:
                        out = subprocess.run(
                            ["pgrep", "-f", "src.adaptive_seeding"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if out.returncode == 0 and out.stdout.strip():
                            stable_since = None
                            last_size = sz
                            time.sleep(poll_s)
                            continue
                    except FileNotFoundError:
                        check_proc = False
                    except subprocess.TimeoutExpired:
                        pass
                if sz == last_size:
                    if stable_since is None:
                        stable_since = time.monotonic()
                    elif time.monotonic() - stable_since >= stable_s:
                        print(f"[post-chain] Ready: {path} ({sz} bytes).", flush=True)
                        return
                else:
                    stable_since = None
                    last_size = sz
        time.sleep(poll_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--wait-for-json",
        type=Path,
        default=None,
        help="Wait until this adaptive_seeding_results.json exists (first run).",
    )
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--stable-seconds", type=float, default=45.0)
    parser.add_argument(
        "--no-pgrep-wait",
        action="store_true",
        help="Do not require zero `pgrep -f src.adaptive_seeding` before continuing.",
    )
    parser.add_argument("--sim-dir", type=Path, default=Path("outputs/btp_equil/A8Q2"))
    parser.add_argument("--prefix", default="A8Q2")
    parser.add_argument(
        "--first-seeding-json",
        type=Path,
        default=Path("outputs/seeding_a8q2/adaptive_seeding_results.json"),
    )
    parser.add_argument(
        "--second-seeding-dir",
        type=Path,
        default=Path("outputs/seeding_a8q2_3way"),
    )
    parser.add_argument(
        "--tica-dir",
        type=Path,
        default=Path("outputs/tica_artifacts_a10"),
    )
    parser.add_argument(
        "--a10-features",
        type=Path,
        default=Path("outputs/artifacts/a10_features.npy"),
    )
    parser.add_argument(
        "--a8q2-features",
        type=Path,
        default=Path("outputs/artifacts/a8q2_features.npy"),
    )
    parser.add_argument(
        "--a8q2-nhb",
        type=Path,
        default=Path("outputs/artifacts/a8q2_nhb.npy"),
    )
    parser.add_argument(
        "--phase7-out",
        type=Path,
        default=Path("outputs/transfer_a8q2"),
    )
    parser.add_argument(
        "--phase9-first-bundle",
        type=Path,
        default=Path("outputs/phase9_bundle_round1"),
    )
    parser.add_argument(
        "--phase9-second-bundle",
        type=Path,
        default=Path("outputs/phase9_bundle_round2_3way"),
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Do not wait for --wait-for-json (first JSON must already exist).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run Phase 7 / second seeding even if outputs exist.",
    )
    parser.add_argument(
        "--run-second-seeding",
        action="store_true",
        help=(
            "Run the second full 3-way adaptive seeding tree and its Phase 9 round 2 bundle. "
            "Default is to skip this long second pass to reduce runtime."
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    py = repo_root / ".venv-openmm" / "bin" / "python"
    if not py.is_file():
        py = Path(sys.executable)

    wait_path = args.wait_for_json or args.first_seeding_json
    if not args.skip_wait:
        wait_for_results_json(
            repo_root / wait_path,
            poll_s=args.poll_interval,
            stable_s=args.stable_seconds,
            also_require_no_process=not args.no_pgrep_wait,
        )

    first_json = (repo_root / args.first_seeding_json).resolve()
    if not first_json.is_file():
        raise FileNotFoundError(f"First seeding results not found: {first_json}")

    sim_dir = (repo_root / args.sim_dir).resolve()
    tica_dir = (repo_root / args.tica_dir).resolve()
    phase7_out = (repo_root / args.phase7_out).resolve()
    transfer_json = phase7_out / "transfer_summary.json"

    ev = tica_dir / "a10_tica_singular_vectors_left.npy"
    mn = tica_dir / "a10_tica_mean0.npy"
    if not ev.is_file() or not mn.is_file():
        raise FileNotFoundError(
            f"TICA artifacts missing under {tica_dir} "
            "(need a10_tica_singular_vectors_left.npy + a10_tica_mean0.npy)."
        )

    lag_path = tica_dir / "a10_tica_chosen_lagtime.txt"
    if not lag_path.is_file():
        raise FileNotFoundError(f"Missing lag file: {lag_path}")
    lag = int(lag_path.read_text().strip())

    # --- Phase 7 ---
    if args.force or not transfer_json.is_file():
        phase7_out.mkdir(parents=True, exist_ok=True)
        cmd7 = [
            str(py),
            "-u",
            "-m",
            "src.tica_transfer",
            "--features_a10",
            str(repo_root / args.a10_features),
            "--features_a8q2",
            str(repo_root / args.a8q2_features),
            "--nhb_a8q2",
            str(repo_root / args.a8q2_nhb),
            "--tica_artifacts_dir",
            str(tica_dir),
            "--lagtime",
            str(lag),
            "--output_dir",
            str(phase7_out),
        ]
        for p, lab in [
            (repo_root / args.a10_features, "a10_features"),
            (repo_root / args.a8q2_features, "a8q2_features"),
            (repo_root / args.a8q2_nhb, "a8q2_nhb"),
        ]:
            if not p.is_file():
                raise FileNotFoundError(f"Missing {lab}: {p}")
        _run_checked(cmd7, repo_root, "Phase 7 tica_transfer")
    else:
        print(f"[post-chain] Skip Phase 7 (exists {transfer_json}).", flush=True)

    # --- Phase 9 bundle for first seeding (2-strategy) — before long 3-way MD ---
    from src.config import N_SEEDS, SEED_NS
    from src.metrics import run_phase9_bundle_for_seeding_run

    b1 = (repo_root / args.phase9_first_bundle).resolve()
    if args.force or not (b1 / "figures" / "fig_jsd_convergence.png").is_file():
        print(f"[post-chain] Phase 9 bundle (round 1, first seeding) → {b1}", flush=True)
        run_phase9_bundle_for_seeding_run(
            adaptive_results_json=first_json,
            sim_dir=sim_dir,
            prefix=args.prefix,
            transfer_summary_json=transfer_json if transfer_json.is_file() else None,
            bundle_dir=b1,
            seed_ns=float(SEED_NS),
            n_seeds=int(N_SEEDS),
        )
    else:
        print("[post-chain] Skip Phase 9 round 1 (figures exist).", flush=True)

    # --- Optional second full seeding + round-2 bundle ---
    if args.run_second_seeding:
        second_dir = (repo_root / args.second_seeding_dir).resolve()
        second_json = second_dir / "adaptive_seeding_results.json"
        if args.force or not second_json.is_file():
            cmd8 = _seeding_cmd(py, repo_root, sim_dir, args.prefix, second_dir, tica_dir)
            _run_checked(cmd8, repo_root, "Phase 8 adaptive_seeding (3-way)")
        else:
            print(f"[post-chain] Skip second seeding (exists {second_json}).", flush=True)

        # --- Phase 9 bundle for 3-way seeding ---
        b2 = (repo_root / args.phase9_second_bundle).resolve()
        sj = second_dir / "adaptive_seeding_results.json"
        if not sj.is_file():
            print(f"[post-chain] WARNING: no {sj} — skip Phase 9 round 2.", flush=True)
        elif args.force or not (b2 / "figures" / "fig_jsd_convergence.png").is_file():
            print(f"[post-chain] Phase 9 bundle (round 2, 3-way) → {b2}", flush=True)
            run_phase9_bundle_for_seeding_run(
                adaptive_results_json=sj,
                sim_dir=sim_dir,
                prefix=args.prefix,
                transfer_summary_json=transfer_json if transfer_json.is_file() else None,
                bundle_dir=b2,
                seed_ns=float(SEED_NS),
                n_seeds=int(N_SEEDS),
            )
        else:
            print("[post-chain] Skip Phase 9 round 2 (figures exist).", flush=True)
    else:
        print(
            "[post-chain] Skipping second seeding tree and Phase 9 round 2 "
            "(pass --run-second-seeding to enable).",
            flush=True,
        )

    print("[post-chain] All chain steps finished.", flush=True)


if __name__ == "__main__":
    main()
