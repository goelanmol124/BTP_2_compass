"""Orchestrate the full COMPASS experiment suite with caching.

Presets
-------
Base:  TICA-2D, 2 ns chunks, no noise → all 5 policies
Abl-A: CV type   (TICA-1D, TICA-2D, SPIB-1D, SPIB-2D), chunk=2 ns, noise=none
Abl-B: Chunk     (0.2, 0.5, 1, 2, 5, 10, 20 ns), TICA-2D, noise=none
Abl-C: Prior noise (none, low, med, high, uniform), TICA-2D, chunk=2 ns

Caching rules
-------------
P1 (single walk): preset-independent → run once, reuse everywhere
P2 (independent replicas): CV-agnostic, noise-agnostic → run once per chunk size
P3 (uniform random): same as P2
P4 (min-count): noise-agnostic → run once per (CV type × chunk size)
P5 (compass): full grid per (CV × chunk × noise)

Usage
-----
    python scripts/run_experiment_suite.py --phase precompute
    python scripts/run_experiment_suite.py --phase base --n-repeats 20
    python scripts/run_experiment_suite.py --phase ablation-a --n-repeats 20
    python scripts/run_experiment_suite.py --phase ablation-b --n-repeats 20
    python scripts/run_experiment_suite.py --phase ablation-c --n-repeats 20
    python scripts/run_experiment_suite.py --phase all --n-repeats 20 --n-workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


# ---------------------------------------------------------------------------
# Chunk size conversions
# ---------------------------------------------------------------------------
FRAME_SPACING_NS = 0.1  # 100 ps per saved frame

CHUNK_NS_TO_FRAMES: dict[float, int] = {
    0.2: 2,
    0.5: 5,
    1.0: 10,
    2.0: 20,
    5.0: 50,
    10.0: 100,
    20.0: 200,
}

# Total budget = 300 ns → n_steps = budget / chunk
TOTAL_BUDGET_NS = 300.0


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

CV_TYPES = ["tica2d", "tica1d", "spib1d", "spib2d"]
CHUNK_NS_LIST = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
NOISE_LEVELS = ["none", "low", "med", "high", "uniform"]

NOISE_TO_PRIOR_PARAMS: dict[str, dict[str, Any]] = {
    "none":    {"prior_quality": "none",   "prior_strength": 0.0},
    "low":     {"prior_quality": "noisy",  "prior_strength": 0.1},
    "med":     {"prior_quality": "noisy",  "prior_strength": 0.3},
    "high":    {"prior_quality": "noisy",  "prior_strength": 1.0},
    "uniform": {"prior_quality": "uniform","prior_strength": 1.0},
}

POLICIES_ALL = ["chrono", "independent_replicas", "random", "min_count", "compass"]
# Policies that are CV-agnostic and noise-agnostic (cache once per chunk)
POLICIES_CV_AGNOSTIC = ["chrono", "independent_replicas", "random"]
# Policies that are noise-agnostic but CV-dependent (cache once per CV×chunk)
POLICIES_NOISE_AGNOSTIC = ["min_count"]
# Policies requiring full grid
POLICIES_FULL = ["compass"]


@dataclass(frozen=True)
class RunSpec:
    """One benchmark run (1 repeat of 1 preset × 1 policy)."""

    policy: str
    cv_type: str
    chunk_ns: float
    noise: str
    repeat: int
    out_dir: Path

    def cache_key(self) -> str:
        return f"{self.policy}__{self.cv_type}__{self.chunk_ns}__{self.noise}__{self.repeat}"


def _out_root(repo: Path) -> Path:
    return repo / "outputs" / "v2" / "experiments"


def _run_dir(repo: Path, policy: str, cv_type: str, chunk_ns: float, noise: str, repeat: int) -> Path:
    chunk_str = f"{chunk_ns:.1f}ns".replace(".", "_")
    return (
        _out_root(repo)
        / f"p_{policy}"
        / f"cv_{cv_type}"
        / f"chunk_{chunk_str}"
        / f"noise_{noise}"
        / f"repeat_{repeat:02d}"
    )


def _is_done(out_dir: Path) -> bool:
    return (out_dir / "v2_metrics.csv").is_file()


# ---------------------------------------------------------------------------
# Build run list with caching
# ---------------------------------------------------------------------------

def build_run_list(
    repo: Path,
    phases: list[str],
    n_repeats: int,
) -> list[RunSpec]:
    specs: list[RunSpec] = []

    def _add(policy: str, cv_type: str, chunk_ns: float, noise: str, repeat: int) -> None:
        # Apply caching: P1/P2/P3 only differ by chunk_ns; reuse across CV and noise
        if policy in POLICIES_CV_AGNOSTIC:
            cv_key = "tica2d"   # canonical CV (irrelevant to these policies)
            noise_key = "none"  # canonical noise (irrelevant)
            canonical = _run_dir(repo, policy, cv_key, chunk_ns, noise_key, repeat)
            specs.append(RunSpec(policy, cv_key, chunk_ns, noise_key, repeat, canonical))
        elif policy in POLICIES_NOISE_AGNOSTIC:
            noise_key = "none"  # canonical noise
            canonical = _run_dir(repo, policy, cv_type, chunk_ns, noise_key, repeat)
            specs.append(RunSpec(policy, cv_type, chunk_ns, noise_key, repeat, canonical))
        else:
            specs.append(RunSpec(policy, cv_type, chunk_ns, noise, repeat,
                                  _run_dir(repo, policy, cv_type, chunk_ns, noise, repeat)))

    for phase in phases:
        if phase == "base":
            for policy in POLICIES_ALL:
                for rep in range(n_repeats):
                    _add(policy, "tica2d", 2.0, "none", rep)

        elif phase == "ablation-a":
            for cv in CV_TYPES:
                for policy in POLICIES_ALL:
                    for rep in range(n_repeats):
                        _add(policy, cv, 2.0, "none", rep)

        elif phase == "ablation-b":
            for chunk_ns in CHUNK_NS_LIST:
                for policy in POLICIES_ALL:
                    for rep in range(n_repeats):
                        _add(policy, "tica2d", chunk_ns, "none", rep)

        elif phase == "ablation-c":
            for noise in NOISE_LEVELS:
                for policy in POLICIES_ALL:
                    for rep in range(n_repeats):
                        _add(policy, "tica2d", 2.0, noise, rep)

    # Deduplicate by output directory path
    seen: set[str] = set()
    unique: list[RunSpec] = []
    for s in specs:
        key = str(s.out_dir)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

def _load_ref_data(repo: Path) -> dict[str, Any]:
    ref = repo / "outputs" / "v2" / "reference"
    result: dict[str, Any] = {}
    for name in ["pi_ref_bins", "f_helix_per_bin", "f_open_per_bin",
                 "bin_assignments", "source_bin_assignments_tica2d", "n_source_bins_tica2d"]:
        p = ref / f"{name}.npy"
        if p.is_file():
            result[name] = np.load(p)
    return result


def _get_n_steps(chunk_ns: float, budget_ns: float = TOTAL_BUDGET_NS) -> int:
    return max(1, int(round(budget_ns / chunk_ns)))


def execute_run(spec: RunSpec, repo: Path, oracle_cache: Path) -> str:
    """Execute one benchmark run and return status string."""
    if _is_done(spec.out_dir):
        return f"SKIP {spec.out_dir.name}"

    spec.out_dir.mkdir(parents=True, exist_ok=True)

    from dataclasses import replace as dc_replace
    from src.compass.replay_enhanced import ReplayConfig, build_atlas
    from src.compass.v2.bench import run_v2_benchmark
    from src.compass.v2.equivalence import EquivalenceMethod
    from src.compass.v2.evaluate import MilestoneTargets
    from src.compass.v2.policy import PolicyName

    chunk_frames = CHUNK_NS_TO_FRAMES[spec.chunk_ns]
    n_steps = _get_n_steps(spec.chunk_ns)
    noise_params = NOISE_TO_PRIOR_PARAMS[spec.noise]
    rng_seed = 42 + spec.repeat * 1000 + hash(spec.policy) % 997

    cfg = ReplayConfig(
        repo_root=repo,
        output_dir=spec.out_dir,
        chunk_frames=chunk_frames,
        n_steps=n_steps,
        random_repeats=1,
        n_bins_x=20,
        random_seed=int(rng_seed % (2**31)),
        helix_nhb_threshold=3,
        open_nhb_threshold=1,
        variant="balanced",
        schedule="two_phase",
        deficit_mode="information_gain",
        prior_quality=noise_params["prior_quality"],
        prior_strength=float(noise_params["prior_strength"]),
        map_kind=spec.cv_type,
        n_bins_y=None,
        softmax_temperature=0.0,
    )

    ref = _load_ref_data(repo)
    pi_ref = ref.get("pi_ref_bins")
    f_helix_bin = ref.get("f_helix_per_bin")
    f_open_bin = ref.get("f_open_per_bin")
    bin_assign = ref.get("bin_assignments")
    src_bin_assign = ref.get("source_bin_assignments_tica2d")
    n_src_bins_arr = ref.get("n_source_bins_tica2d")
    n_src_bins = int(n_src_bins_arr.item()) if n_src_bins_arr is not None else None

    policy = PolicyName(spec.policy)
    eq_method = EquivalenceMethod.M9_NHB_TORSION  # NHB+torsion hybrid as default

    t0 = time.time()
    metrics_df, cand_df, dec_df, summary = run_v2_benchmark(
        cfg,
        policy=policy,
        eq_method=eq_method,
        eq_k=40,
        oracle_cache=oracle_cache,
        segments_spec=None,
        cold_start_frame=None,
        initial_seeds_extra=[],
        max_time_delta_frames=500,
        same_segment_only=False,
        require_same_macro=False,
        macro_n_clusters=16,
        max_neighbors_per_anchor=40,
        milestones=MilestoneTargets(coverage=0.7, jsd=0.10, f_helix_err=0.02),
        allow_deterministic_reseed=True,
        ref_pi_bins=pi_ref,
        ref_f_helix_per_bin=f_helix_bin,
        ref_f_open_per_bin=f_open_bin,
        ref_bin_assignments=bin_assign,
        source_bin_assignments=src_bin_assign,
        n_source_bins=n_src_bins,
    )
    elapsed = time.time() - t0

    import pandas as pd
    metrics_df.to_csv(spec.out_dir / "v2_metrics.csv", index=False)
    cand_df.to_csv(spec.out_dir / "candidate_pool_sizes.csv", index=False)
    dec_df.to_csv(spec.out_dir / "seed_decisions.csv", index=False)
    meta = {
        "policy": spec.policy,
        "cv_type": spec.cv_type,
        "chunk_ns": spec.chunk_ns,
        "noise": spec.noise,
        "repeat": spec.repeat,
        "n_steps": n_steps,
        "chunk_frames": chunk_frames,
        "elapsed_s": round(elapsed, 2),
        **summary,
    }
    (spec.out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return f"OK  {spec.out_dir.relative_to(_out_root(repo))} ({elapsed:.1f}s)"


def _worker(args: tuple[RunSpec, Path, Path]) -> str:
    spec, repo, oracle_cache = args
    try:
        return execute_run(spec, repo, oracle_cache)
    except Exception as exc:
        return f"ERR {spec.out_dir.name}: {exc}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--phase",
        type=str,
        default="base",
        choices=["precompute", "base", "ablation-a", "ablation-b", "ablation-c", "all"],
    )
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--n-workers", type=int, default=1, help="Parallel workers (1=serial)")
    p.add_argument("--dry-run", action="store_true", help="Print run list but do not execute")
    args = p.parse_args(argv)

    repo = args.repo_root.resolve()
    oracle_cache = repo / "outputs" / "v2" / "oracle"
    oracle_cache.mkdir(parents=True, exist_ok=True)

    if args.phase == "precompute":
        from scripts.precompute_reference import run as precompute_run
        precompute_run(repo)
        return

    phases = (
        ["base", "ablation-a", "ablation-b", "ablation-c"]
        if args.phase == "all"
        else [args.phase]
    )

    runs = build_run_list(repo, phases, args.n_repeats)
    pending = [r for r in runs if not _is_done(r.out_dir)]

    print(f"Total runs: {len(runs)}  |  Pending: {len(pending)}  |  Workers: {args.n_workers}")
    if args.dry_run:
        for r in pending[:20]:
            print(f"  {r.policy:25s}  cv={r.cv_type:8s}  chunk={r.chunk_ns:4.1f}ns  "
                  f"noise={r.noise:8s}  rep={r.repeat:02d}")
        if len(pending) > 20:
            print(f"  ... and {len(pending)-20} more")
        return

    task_args = [(r, repo, oracle_cache) for r in pending]
    if args.n_workers <= 1:
        for ta in task_args:
            msg = _worker(ta)
            print(msg)
    else:
        with mp.Pool(args.n_workers) as pool:
            for msg in pool.imap_unordered(_worker, task_args, chunksize=1):
                print(msg)

    print(f"\nDone. Results in: {_out_root(repo)}")


if __name__ == "__main__":
    main()
