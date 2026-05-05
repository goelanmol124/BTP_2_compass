#!/usr/bin/env python3
"""
Pod 2 post-production orchestrator (RunPod / Linux; see docs/EXPERIMENT_PROTOCOL.md §7–8, §11, §13.4).

Waits until A8Q2 Phase-4 production (both extended and helix replicas) has **finished
successfully**, then runs, in order:

  1. Optional ``git pull`` (only if ``REPO_ROOT/.git`` exists).
  2. Phase 5 ground-truth analysis for both replicas (``src.analysis.analyze_system``).
  3. Export pooled ``a8q2_features.npy`` and ``a8q2_nhb.npy`` (burn-in stripped
     per replica; N_HB for Phase 7 transfer / diagnostics).
  4. Adaptive seeding — Phase 8 (``python -m src.adaptive_seeding``).

Designed to be started **while** ``python -u -m src.simulate --system A8Q2 --phase all``
is still running: it polls until that process exits **and** both StateDataReporter logs
show completion (guards against a crashed process leaving partial logs).

Usage (on the pod, from repo root)::

    nohup python -u scripts/pod2_orchestrate.py \\
        --repo-root /workspace/btp_2_compass \\
        --sim-output-dir /workspace/btp_2_compass/outputs/btp_equil/A8Q2 \\
        > orchestrate.log 2>&1 &

Optional: ``--tica-artifacts-dir`` pointing to a folder with
``a10_tica_singular_vectors_left.npy`` and ``a10_tica_mean0.npy`` (from Phase 6 on A10)
to enable the **transferred** seeding strategy; otherwise that strategy is skipped by
``adaptive_seeding`` with a warning.
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path


def _reconfigure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


def _simulate_a8q2_running() -> bool:
    """True if a src.simulate process targeting A8Q2 is in the process table (Linux)."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "args="],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if out.returncode != 0 or not out.stdout:
        return False
    for line in out.stdout.splitlines():
        if "src.simulate" in line and "A8Q2" in line and "grep" not in line:
            return True
    return False


def _parse_last_state_data_row(log_path: Path) -> tuple[float, int, float] | None:
    """Return (progress_pct, step, time_ps) from the last non-comment row, or None."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        last = s
    if last is None:
        return None
    parts = [p.strip().strip('"') for p in last.split("\t")]
    if len(parts) < 3:
        return None
    try:
        prog_s = parts[0].rstrip("%")
        progress = float(prog_s)
        step = int(float(parts[1]))
        time_ps = float(parts[2])
    except ValueError:
        return None
    return progress, step, time_ps


def wait_for_production_complete(
    sim_dir: Path,
    prefix: str,
    *,
    prod_steps: int,
    target_time_ps: float,
    poll_s: float,
    require_process_exit: bool,
) -> None:
    """Block until both replicas are done and (optionally) simulate has exited."""
    labels = ("extended", "helix")
    log_files = {lab: sim_dir / f"{prefix}_{lab}_prod.log" for lab in labels}
    dcd_files = {lab: sim_dir / f"{prefix}_{lab}_prod.dcd" for lab in labels}

    step_tol = 10_000  # allow small reporter / rounding slack vs PROD_STEPS
    time_tol_ps = 500.0  # 0.5 ns slack on reported Time (ps)

    print(
        f"[orchestrate] Waiting for production: {prefix}_extended + {prefix}_helix\n"
        f"  expect step ≈ {prod_steps}, time ≈ {target_time_ps:.0f} ps\n"
        f"  poll every {poll_s:.0f} s, require_process_exit={require_process_exit}",
        flush=True,
    )

    while True:
        proc_running = _simulate_a8q2_running()
        logs_ok = True
        details: list[str] = []

        for lab in labels:
            lp = log_files[lab]
            dp = dcd_files[lab]
            if not lp.is_file():
                logs_ok = False
                details.append(f"{lab}: missing log")
                continue
            parsed = _parse_last_state_data_row(lp)
            if parsed is None:
                logs_ok = False
                details.append(f"{lab}: unreadable log tail")
                continue
            prog, step, t_ps = parsed
            if not dp.is_file() or dp.stat().st_size < 1024:
                logs_ok = False
                details.append(f"{lab}: dcd missing/empty")
                continue
            step_ok = step >= prod_steps - step_tol
            time_ok = t_ps >= target_time_ps - time_tol_ps
            prog_ok = prog >= 99.0
            if not (step_ok or time_ok or prog_ok):
                logs_ok = False
                details.append(
                    f"{lab}: prog={prog:.1f}% step={step} t={t_ps:.0f}ps (incomplete)"
                )

        process_ok = (not proc_running) if require_process_exit else True

        if logs_ok and process_ok:
            print("[orchestrate] Production complete for both replicas.", flush=True)
            return

        if not process_ok:
            details.append("simulate(A8Q2) still running")
        print(
            f"[orchestrate] {time.strftime('%Y-%m-%d %H:%M:%S')}  "
            + "; ".join(details),
            flush=True,
        )
        time.sleep(poll_s)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def try_git_pull(repo_root: Path) -> None:
    """Run ``git pull --ff-only`` only when explicitly enabled (avoids private-repo auth noise)."""
    git_dir = repo_root / ".git"
    if not git_dir.is_dir():
        print(
            "[orchestrate] No .git — skipping pull (sync src/ with scp/rsync from laptop).",
            flush=True,
        )
        return
    print("[orchestrate] git pull --ff-only …", flush=True)
    r = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        timeout=300,
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print(
            "[orchestrate] WARNING: git pull failed; continuing with tree on disk.",
            flush=True,
        )


def _reload_production_thresholds() -> tuple[int, float]:
    """Reload ``src.config`` so long waits pick up updated PROD_STEPS / PROD_US from disk."""
    import src.config as cfg

    importlib.reload(cfg)
    target_ps = float(cfg.PROD_US) * 1_000_000.0
    return int(cfg.PROD_STEPS), target_ps


class _OrchestratorLock:
    """Single-instance lock (Linux ``flock``). Prevents duplicate orchestrators on one pod."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._fh: object | None = None

    def acquire(self) -> None:
        try:
            import fcntl
        except ImportError:
            print(
                "[orchestrate] No fcntl — skipping single-instance lock (non-Linux?).",
                flush=True,
            )
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            try:
                stale = self.lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                stale = "?"
            raise SystemExit(
                f"[orchestrate] Another pod2_orchestrate instance is running "
                f"(lock {self.lock_path}, pid file hint: {stale!r}). Exit duplicate or use --no-lock."
            ) from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        atexit.register(self.release)

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None


def run_phase5_a8q2(
    sim_dir: Path,
    prefix: str,
    label: str,
    phase5_out: Path,
) -> None:
    from src.analysis import analyze_system

    helix_ref = sim_dir / f"{prefix}_helix.pdb"
    if not helix_ref.is_file():
        raise FileNotFoundError(f"Helix reference PDB missing: {helix_ref}")

    phase5_out.mkdir(parents=True, exist_ok=True)
    for lab in ("extended", "helix"):
        pprefix = f"{prefix}_{lab}"
        dcd = sim_dir / f"{pprefix}_prod.dcd"
        top = sim_dir / f"{pprefix}_prod_top.pdb"
        for p in (dcd, top):
            if not p.is_file():
                raise FileNotFoundError(f"Missing {p}")
        print(f"[orchestrate] Phase 5: {pprefix} …", flush=True)
        analyze_system(
            prefix=pprefix,
            label=label,
            prod_dcd=str(dcd),
            prod_top_pdb=str(top),
            helix_ref_pdb=str(helix_ref),
            output_dir=str(phase5_out),
        )


def export_a8q2_features(
    sim_dir: Path,
    prefix: str,
    out_npy: Path,
    burnin_frames: int,
) -> None:
    import numpy as np

    from src.analysis import BURNIN_FRAMES, compute_featurization, load_trajectory

    bi = burnin_frames if burnin_frames > 0 else BURNIN_FRAMES
    chunks: list = []
    for lab in ("extended", "helix"):
        pprefix = f"{prefix}_{lab}"
        dcd = sim_dir / f"{pprefix}_prod.dcd"
        top = sim_dir / f"{pprefix}_prod_top.pdb"
        print(f"[orchestrate] Featurize {pprefix} (strip {bi} frames) …", flush=True)
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        if traj.n_frames <= bi:
            raise RuntimeError(f"{dcd}: need > {bi} frames, got {traj.n_frames}")
        traj = traj[bi:]
        chunks.append(compute_featurization(traj))

    stacked = np.concatenate(chunks, axis=0)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, stacked)
    print(
        f"[orchestrate] Saved {out_npy}  shape={stacked.shape}",
        flush=True,
    )


def export_a8q2_nhb(
    sim_dir: Path,
    prefix: str,
    out_npy: Path,
    burnin_frames: int,
) -> None:
    import numpy as np

    from src.analysis import BURNIN_FRAMES, compute_nhb, load_trajectory

    bi = burnin_frames if burnin_frames > 0 else BURNIN_FRAMES
    chunks: list = []
    for lab in ("extended", "helix"):
        pprefix = f"{prefix}_{lab}"
        dcd = sim_dir / f"{pprefix}_prod.dcd"
        top = sim_dir / f"{pprefix}_prod_top.pdb"
        print(f"[orchestrate] N_HB featurize {pprefix} (strip {bi} frames) …", flush=True)
        traj = load_trajectory(str(dcd), str(top), protein_only=True)
        if traj.n_frames <= bi:
            raise RuntimeError(f"{dcd}: need > {bi} frames, got {traj.n_frames}")
        traj = traj[bi:]
        chunks.append(compute_nhb(traj).astype(np.float64))

    stacked = np.concatenate(chunks, axis=0)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, stacked)
    print(
        f"[orchestrate] Saved {out_npy}  shape={stacked.shape}",
        flush=True,
    )


def run_adaptive_seeding(
    repo_root: Path,
    sim_dir: Path,
    prefix: str,
    seeding_out: Path,
    tica_dir: Path | None,
) -> None:
    gen_pdb = sim_dir / f"{prefix}_extended_npt_final.pdb"
    sys_xml = sim_dir / f"{prefix}_extended_npt_final.xml"
    prod_top = sim_dir / f"{prefix}_extended_prod_top.pdb"
    dcd_h = sim_dir / f"{prefix}_helix_prod.dcd"
    dcd_e = sim_dir / f"{prefix}_extended_prod.dcd"
    gt_top_helix = sim_dir / f"{prefix}_helix_prod_top.pdb"
    gt_top_extended = sim_dir / f"{prefix}_extended_prod_top.pdb"

    for p in (gen_pdb, sys_xml, prod_top, dcd_h, dcd_e, gt_top_helix, gt_top_extended):
        if not p.is_file():
            raise FileNotFoundError(f"Seeding prerequisite missing: {p}")

    cmd: list[str] = [
        sys.executable,
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
        str(gt_top_helix),
        "--gt_top_extended",
        str(gt_top_extended),
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
        else:
            print(
                f"[orchestrate] WARNING: incomplete TICA dir {tica_dir} — "
                "transferred strategy will be disabled.",
                flush=True,
            )

    print("[orchestrate] Starting adaptive seeding …", flush=True)
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    print("[orchestrate] Adaptive seeding finished.", flush=True)


def main() -> None:
    _reconfigure_stdio_utf8()

    parser = argparse.ArgumentParser(
        description="Wait for A8Q2 production, then Phase 5 + features + seeding.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="btp_2_compass repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--sim-output-dir",
        type=Path,
        required=True,
        help="Directory with A8Q2_*_prod.dcd (e.g. outputs/btp_equil/A8Q2).",
    )
    parser.add_argument(
        "--prefix",
        default="A8Q2",
        help="System prefix for filenames (default: A8Q2).",
    )
    parser.add_argument(
        "--label",
        default="Ac-Ala₈Gln₂-NHMe",
        help="Plot title label for analysis.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=120.0,
        help="Seconds between completion checks while waiting (default: 120).",
    )
    parser.add_argument(
        "--no-wait-process",
        action="store_true",
        help="Do not require src.simulate to exit (only check log+DCD completeness).",
    )
    parser.add_argument(
        "--git-pull",
        action="store_true",
        help="Run git pull --ff-only before waiting (needs GitHub auth on the pod).",
    )
    parser.add_argument(
        "--skip-git-pull",
        action="store_true",
        help="Deprecated no-op (pull is off by default); use --git-pull to enable.",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Allow multiple orchestrators (not recommended on one pod).",
    )
    parser.add_argument(
        "--skip-phase5",
        action="store_true",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
    )
    parser.add_argument(
        "--skip-seeding",
        action="store_true",
    )
    parser.add_argument(
        "--phase5-dir",
        type=Path,
        default=None,
        help="Output dir for Phase 5 figures/CSV (default: outputs/phase5_a8q2/A8Q2).",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=None,
        help="Path for a8q2_features.npy (default: outputs/artifacts/a8q2_features.npy).",
    )
    parser.add_argument(
        "--nhb-out",
        type=Path,
        default=None,
        help="Path for a8q2_nhb.npy (default: same dir as --features-out).",
    )
    parser.add_argument(
        "--seeding-dir",
        type=Path,
        default=None,
        help="Adaptive seeding output (default: outputs/seeding_a8q2).",
    )
    parser.add_argument(
        "--tica-artifacts-dir",
        type=Path,
        default=None,
        help="Directory with a10_tica_*.npy from Phase 6 (optional).",
    )
    parser.add_argument(
        "--burnin-frames",
        type=int,
        default=-1,
        help="Burn-in frames stripped per replica (default: use analysis.BURNIN_FRAMES).",
    )
    parser.add_argument(
        "--auto-chain-after-seeding",
        action="store_true",
        help=(
            "After adaptive seeding completes, run scripts/pod_post_seeding_chain.py "
            "with --skip-wait: Phase 7 (tica_transfer), Phase 9 figures for this seeding, "
            "then a second 3-way seeding in outputs/seeding_a8q2_3way, then Phase 9 again. "
            "Requires outputs/tica_artifacts_a10 (or --chain-tica-dir) and a10/a8q2 .npy in outputs/artifacts/."
        ),
    )
    parser.add_argument(
        "--chain-tica-dir",
        type=Path,
        default=None,
        help="TICA artifact directory for --auto-chain-after-seeding (default: outputs/tica_artifacts_a10).",
    )
    parser.add_argument(
        "--chain-second-seeding-dir",
        type=Path,
        default=None,
        help="Second seeding output dir for --auto-chain-after-seeding (default: outputs/seeding_a8q2_3way).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sim_dir = args.sim_output_dir.resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if not args.no_lock:
        lock = _OrchestratorLock(repo_root / ".pod2_orchestrate.lock")
        lock.acquire()

    want_pull = (
        args.git_pull
        or _env_truthy("BTP_ENABLE_GIT_PULL")
    ) and not args.skip_git_pull
    if _env_truthy("BTP_SKIP_GIT_PULL"):
        want_pull = False
    if want_pull:
        try_git_pull(repo_root)
    else:
        print(
            "[orchestrate] Git pull skipped (default). "
            "Use --git-pull or BTP_ENABLE_GIT_PULL=1 after configuring GitHub auth; "
            "sync code with scp/rsync otherwise.",
            flush=True,
        )

    prod_steps, target_time_ps = _reload_production_thresholds()
    print(
        f"[orchestrate] Production thresholds from src.config: "
        f"PROD_STEPS={prod_steps}, target_time_ps={target_time_ps:.0f}",
        flush=True,
    )

    wait_for_production_complete(
        sim_dir,
        args.prefix,
        prod_steps=prod_steps,
        target_time_ps=target_time_ps,
        poll_s=args.poll_interval,
        require_process_exit=not args.no_wait_process,
    )

    phase5_out = (
        args.phase5_dir.resolve()
        if args.phase5_dir
        else (repo_root / "outputs" / "phase5_a8q2" / args.prefix)
    )
    features_out = (
        args.features_out.resolve()
        if args.features_out
        else (repo_root / "outputs" / "artifacts" / "a8q2_features.npy")
    )
    nhb_out = (
        args.nhb_out.resolve()
        if args.nhb_out
        else features_out.parent / "a8q2_nhb.npy"
    )
    seeding_out = (
        args.seeding_dir.resolve()
        if args.seeding_dir
        else (repo_root / "outputs" / "seeding_a8q2")
    )

    if not args.skip_phase5:
        try:
            import src.analysis as _analysis_mod

            importlib.reload(_analysis_mod)
            print("[orchestrate] Reloaded src.analysis for Phase 5.", flush=True)
        except Exception as exc:
            print(
                f"[orchestrate] WARNING: could not reload src.analysis ({exc!r}); "
                "using already-imported module.",
                flush=True,
            )
        run_phase5_a8q2(sim_dir, args.prefix, args.label, phase5_out)

    bi = args.burnin_frames if args.burnin_frames >= 0 else -1
    if not args.skip_features:
        export_a8q2_features(sim_dir, args.prefix, features_out, bi)
        export_a8q2_nhb(sim_dir, args.prefix, nhb_out, bi)

    if not args.skip_seeding:
        run_adaptive_seeding(
            repo_root,
            sim_dir,
            args.prefix,
            seeding_out,
            args.tica_artifacts_dir.resolve() if args.tica_artifacts_dir else None,
        )

    if (
        args.auto_chain_after_seeding
        and not args.skip_seeding
        and (seeding_out / "adaptive_seeding_results.json").is_file()
    ):
        tica_d = (
            args.chain_tica_dir.resolve()
            if args.chain_tica_dir
            else (repo_root / "outputs" / "tica_artifacts_a10")
        )
        ev = tica_d / "a10_tica_singular_vectors_left.npy"
        mn = tica_d / "a10_tica_mean0.npy"
        if not ev.is_file() or not mn.is_file():
            print(
                "[orchestrate] WARNING: --auto-chain-after-seeding skipped; "
                f"missing TICA npy under {tica_d}.",
                flush=True,
            )
        else:
            chain_script = repo_root / "scripts" / "pod_post_seeding_chain.py"
            if not chain_script.is_file():
                print(
                    f"[orchestrate] WARNING: chain script missing: {chain_script}",
                    flush=True,
                )
            else:
                chain_cmd: list[str] = [
                    sys.executable,
                    "-u",
                    str(chain_script),
                    "--repo-root",
                    str(repo_root),
                    "--skip-wait",
                    "--first-seeding-json",
                    str(seeding_out / "adaptive_seeding_results.json"),
                    "--tica-dir",
                    str(tica_d),
                    "--sim-dir",
                    str(sim_dir),
                    "--prefix",
                    args.prefix,
                ]
                if args.chain_second_seeding_dir is not None:
                    chain_cmd.extend(
                        [
                            "--second-seeding-dir",
                            str(args.chain_second_seeding_dir.resolve()),
                        ]
                    )
                print("[orchestrate] Starting post-seeding chain …", flush=True)
                subprocess.run(chain_cmd, cwd=str(repo_root), check=True)
                print("[orchestrate] Post-seeding chain finished.", flush=True)

    print("[orchestrate] All requested steps completed.", flush=True)


if __name__ == "__main__":
    main()
