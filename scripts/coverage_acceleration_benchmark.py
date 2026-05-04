#!/usr/bin/env python3
"""Estimate 'acceleration': steps (and sampled ns) to reach coverage targets — COMPASS vs random MC.

Speedup = median_random_steps / compass_steps (values >1 mean COMPASS reaches the target in fewer steps).

Example:
  python scripts/coverage_acceleration_benchmark.py --repo-root . --n-steps 100 --random-repeats 48
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.compass.replay_enhanced import ReplayConfig
from src.compass.v2.bench import (
    PolicyName,
    _random_baseline_matching,
    load_equivalence_choice_method,
    run_v2_benchmark,
)
from src.compass.v2.equivalence import EquivalenceMethod
from src.compass.v2.evaluate import MilestoneTargets


def first_hit(
    g: pd.DataFrame,
    col: str,
    threshold: float,
) -> tuple[int | None, float | None]:
    g2 = g.sort_values("step")
    ok = g2[g2[col] >= threshold]
    if ok.empty:
        return None, None
    row = ok.iloc[0]
    return int(row["step"]), float(row["sampled_ns"]) if "sampled_ns" in row.index else None


def collect_hits_per_repeat(rand_df: pd.DataFrame, targets: list[float], col: str = "coverage") -> dict[str, list[int | None]]:
    out: dict[str, list[int | None]] = {f"{t:.3f}": [] for t in targets}
    ns_out: dict[str, list[float | None]] = {f"{t:.3f}": [] for t in targets}
    for rep, g in rand_df.groupby("repeat"):
        del rep
        for t in targets:
            k = f"{t:.3f}"
            st, ns = first_hit(g, col, t)
            out[k].append(st)
            ns_out[k].append(ns)
    return out, ns_out


def summarize_steps(steps: list[int | None]) -> dict[str, float | int | None]:
    arr = np.array([s for s in steps if s is not None], dtype=np.float64)
    n_miss = sum(1 for s in steps if s is None)
    if arr.size == 0:
        return {"median": None, "p25": None, "p75": None, "n_hit": 0, "n_miss": n_miss}
    return {
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n_hit": int(arr.size),
        "n_miss": n_miss,
    }


def summarize_ns(ns_list: list[float | None]) -> dict[str, float | None]:
    arr = np.array([x for x in ns_list if x is not None], dtype=np.float64)
    if arr.size == 0:
        return {"median_ns": None, "p25_ns": None, "p75_ns": None}
    return {
        "median_ns": float(np.median(arr)),
        "p25_ns": float(np.percentile(arr, 25)),
        "p75_ns": float(np.percentile(arr, 75)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--oracle-cache", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/v2/coverage_acceleration/summary.json"))
    p.add_argument("--random-seed", type=int, default=101)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--chunk-frames", type=int, default=10)
    p.add_argument("--eq-k", type=int, default=40)
    p.add_argument("--cold-start-frame", type=int, default=1000)
    p.add_argument("--random-repeats", type=int, default=48)
    p.add_argument(
        "--coverage-targets",
        type=str,
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.65",
        help="Comma-separated coverage thresholds in (0,1].",
    )
    p.add_argument("--include-chrono", action="store_true")
    args = p.parse_args()

    repo = args.repo_root.resolve()
    oracle = (args.oracle_cache or repo / "outputs" / "v2" / "oracle").resolve()
    out_path = args.out
    if out_path.suffix != ".json":
        out_path = out_path / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets = [float(x.strip()) for x in args.coverage_targets.split(",") if x.strip()]
    for t in targets:
        if not (0 < t <= 1):
            raise SystemExit(f"Invalid target {t}")

    eq_m, eq_note = load_equivalence_choice_method(oracle, fallback=EquivalenceMethod.M3_TORSION)

    bench_kw: dict[str, Any] = dict(
        segments_spec=None,
        cold_start_frame=args.cold_start_frame,
        initial_seeds_extra=[],
        max_time_delta_frames=500,
        same_segment_only=False,
        require_same_macro=False,
        macro_n_clusters=16,
        max_neighbors_per_anchor=40,
        milestones=MilestoneTargets(),
        allow_deterministic_reseed=False,
        eq_mask_m3_by_calibration=False,
        eq_m3_calibration_json=None,
    )

    cfg = ReplayConfig(
        repo_root=repo,
        output_dir=out_path.parent / "_tmp",
        chunk_frames=args.chunk_frames,
        n_steps=args.n_steps,
        random_repeats=20,
        n_bins_x=30,
        random_seed=args.random_seed,
        helix_nhb_threshold=3,
        open_nhb_threshold=1,
        variant="balanced",
        schedule="two_phase",
        deficit_mode="information_gain",
        prior_quality="none",
        prior_strength=0.5,
        map_kind="tica2d",
        n_bins_y=None,
        softmax_temperature=0.0,
    )

    mdf_comp, _, _, _ = run_v2_benchmark(
        replace(cfg, output_dir=out_path.parent / "compass_run"),
        policy=PolicyName.B5_COMPASS,
        eq_method=eq_m,
        eq_k=args.eq_k,
        oracle_cache=oracle,
        **bench_kw,
    )
    mdf_comp.to_csv(out_path.parent / "compass_metrics.csv", index=False)

    rk = {k: v for k, v in bench_kw.items() if k not in {"cold_start_frame", "segments_spec", "oracle_cache", "eq_k"}}
    rand_big, _ = _random_baseline_matching(
        cfg,
        eq_m,
        args.eq_k,
        oracle,
        None,
        args.cold_start_frame,
        args.random_repeats,
        benchmark_kwargs=rk,
    )
    rand_big.to_csv(out_path.parent / "random_metrics.csv", index=False)

    chrono_mdf = None
    if args.include_chrono:
        chrono_mdf, _, _, _ = run_v2_benchmark(
            replace(cfg, output_dir=out_path.parent / "chrono_run"),
            policy=PolicyName.B0_CHRONOLOGICAL,
            eq_method=eq_m,
            eq_k=args.eq_k,
            oracle_cache=oracle,
            **bench_kw,
        )
        chrono_mdf.to_csv(out_path.parent / "chrono_metrics.csv", index=False)

    payload: dict[str, Any] = {
        "settings": {
            "repo": str(repo),
            "cold_start_frame": args.cold_start_frame,
            "n_steps": args.n_steps,
            "chunk_frames": args.chunk_frames,
            "eq_method": eq_m.value,
            "eq_note": eq_note,
            "random_repeats": args.random_repeats,
            "random_seed": args.random_seed,
            "coverage_targets": targets,
        },
        "compass": {},
        "random_median_speedup_vs_compass": {},
        "notes": (
            "speedup_step = median_random_steps_to_target / compass_steps_to_target "
            "(>1 means COMPASS needs fewer steps). "
            "If compass never hits, speedup is null. "
            "Random uses median over repeats that hit; n_miss counts repeats that never reached target by horizon."
        ),
    }

    for t in targets:
        key = f"{t:.3f}"
        c_step, c_ns = first_hit(mdf_comp, "coverage", t)
        payload["compass"][key] = {
            "coverage_target": t,
            "first_step": c_step,
            "first_sampled_ns": c_ns,
        }

    step_lists, ns_lists = collect_hits_per_repeat(rand_big, targets, "coverage")
    payload["random"] = {}
    for t in targets:
        key = f"{t:.3f}"
        st = step_lists[key]
        ns = ns_lists[key]
        payload["random"][key] = {
            "coverage_target": t,
            "steps_to_hit": summarize_steps(st),
            "sampled_ns_to_hit": summarize_ns(ns),
        }
        c_step = payload["compass"][key]["first_step"]
        r_med = payload["random"][key]["steps_to_hit"]["median"]
        if c_step is None or r_med is None or c_step <= 0:
            spd = None
        else:
            spd = float(r_med / c_step)
        interp = (
            f"Random needed ~{spd:.2f}x as many steps (median) to reach coverage >= {t}"
            if spd is not None
            else "not_computable (COMPASS or random never hit, or insufficient data)"
        )
        n_hit_tgt = payload["random"][key]["steps_to_hit"]["n_hit"]
        payload["random_median_speedup_vs_compass"][key] = {
            "speedup_steps_median_random_over_compass": spd,
            "interpretation": interp,
            "random_n_hit_at_target": int(n_hit_tgt) if n_hit_tgt is not None else 0,
            "random_n_repeats": int(args.random_repeats),
        }

        c_ns = payload["compass"][key]["first_sampled_ns"]
        r_ns_m = payload["random"][key]["sampled_ns_to_hit"]["median_ns"]
        if c_ns is None or r_ns_m is None or c_ns <= 0:
            payload["random_median_speedup_vs_compass"][key]["speedup_sampled_ns_median"] = None
        else:
            payload["random_median_speedup_vs_compass"][key]["speedup_sampled_ns_median"] = float(r_ns_m / c_ns)

    if chrono_mdf is not None:
        payload["chrono"] = {}
        payload["chrono_median_speedup_vs_compass"] = {}
        for t_cut in targets:
            ck = f"{t_cut:.3f}"
            h_step, h_ns = first_hit(chrono_mdf, "coverage", t_cut)
            payload["chrono"][ck] = {"first_step": h_step, "first_sampled_ns": h_ns}
            c_step_ck = payload["compass"][ck]["first_step"]
            if h_step is None or c_step_ck is None or c_step_ck <= 0:
                payload["chrono_median_speedup_vs_compass"][ck] = None
            else:
                payload["chrono_median_speedup_vs_compass"][ck] = float(h_step / c_step_ck)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
