#!/usr/bin/env python3
"""Run v2 bench: COMPASS vs chronological (single chain) vs uniform-random (MC over seeds).

Writes per-policy folders under ``outputs/v2/exp_policy_ablation/`` and ``compare_summary.json``.
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


def _final_metrics(mdf: pd.DataFrame) -> dict[str, float]:
    last = mdf.iloc[-1]
    keys = [
        "f_helix_error",
        "f_open_error",
        "jsd",
        "coverage",
        "important_coverage",
        "source_mass_visited",
        "sampled_ns",
        "step",
    ]
    out: dict[str, float] = {}
    for k in keys:
        if k in last.index:
            out[k] = float(last[k])
    return out


def _pct_improve(compass_v: float, base_v: float, *, lower_is_better: bool) -> float:
    if abs(base_v) < 1e-15:
        return float("nan")
    if lower_is_better:
        return float((base_v - compass_v) / base_v * 100.0)
    return float((compass_v - base_v) / base_v * 100.0)


def _mc_final_per_repeat(rand_big: pd.DataFrame) -> pd.DataFrame:
    """One row per repeat: metrics at the last recorded step for that trace."""

    rows: list[pd.Series] = []
    for _, g in rand_big.groupby("repeat"):
        g2 = g.sort_values("step")
        rows.append(g2.iloc[-1])
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out-root", type=Path, default=Path("outputs/v2/exp_policy_ablation"))
    p.add_argument("--oracle-cache", type=Path, default=None)
    p.add_argument("--random-seed", type=int, default=101)
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--chunk-frames", type=int, default=10)
    p.add_argument("--eq-k", type=int, default=40)
    p.add_argument("--cold-start-frame", type=int, default=1000)
    p.add_argument("--random-mc-repeats", type=int, default=36)
    args = p.parse_args()

    repo = args.repo_root.resolve()
    oracle = (args.oracle_cache or repo / "outputs" / "v2" / "oracle").resolve()
    out_root = (repo / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

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
        output_dir=out_root / "_cfg_placeholder",
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

    summaries: dict[str, Any] = {
        "settings": {
            "repo": str(repo),
            "oracle_cache": str(oracle),
            "equivalence_method": eq_m.value,
            "equivalence_resolution": eq_note,
            "eq_k": args.eq_k,
            "random_seed": args.random_seed,
            "n_steps": args.n_steps,
            "chunk_frames": args.chunk_frames,
            "cold_start_frame": args.cold_start_frame,
            "random_mc_repeats": args.random_mc_repeats,
        },
        "per_policy_final": {},
        "improvement_vs_random_mean_pct": {},
        "compass_vs_random_mc_mean_pct": {},
        "chrono_vs_random_mc_mean_pct": {},
    }

    for name, policy in [
        ("compass", PolicyName.B5_COMPASS),
        ("chrono_single_chain", PolicyName.B0_CHRONOLOGICAL),
        ("random_single_trace", PolicyName.B1_RANDOM),
    ]:
        subdir = out_root / name
        subdir.mkdir(parents=True, exist_ok=True)
        sub_cfg = replace(cfg, output_dir=subdir)
        mdf, cand, dec, summ = run_v2_benchmark(
            sub_cfg,
            policy=policy,
            eq_method=eq_m,
            eq_k=args.eq_k,
            oracle_cache=oracle,
            **bench_kw,
        )
        mdf.to_csv(subdir / "v2_metrics.csv", index=False)
        cand.to_csv(subdir / "candidate_pool_sizes.csv", index=False)
        dec.to_csv(subdir / "seed_decisions.csv", index=False)
        row = _final_metrics(mdf)
        summaries["per_policy_final"][name] = row
        with open(subdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"final_row": row, **summ, "equivalence_resolution": eq_note}, f, indent=2)

    rk = {k: v for k, v in bench_kw.items() if k not in {"cold_start_frame", "segments_spec", "oracle_cache", "eq_k"}}
    rand_big, rand_summ = _random_baseline_matching(
        cfg,
        eq_m,
        args.eq_k,
        oracle,
        None,
        args.cold_start_frame,
        args.random_mc_repeats,
        benchmark_kwargs=rk,
    )
    mc_dir = out_root / "random_mc"
    mc_dir.mkdir(parents=True, exist_ok=True)
    rand_big.to_csv(mc_dir / "random_metrics.csv", index=False)
    rand_summ.to_csv(mc_dir / "random_summary.csv", index=False)

    mc_last = _mc_final_per_repeat(rand_big)
    base_mean = {
        "f_helix_error": float(mc_last["f_helix_error"].mean()),
        "f_open_error": float(mc_last["f_open_error"].mean()),
        "jsd": float(mc_last["jsd"].mean()),
        "coverage": float(mc_last["coverage"].mean()),
        "important_coverage": float(mc_last["important_coverage"].mean()),
        "source_mass_visited": float(mc_last["source_mass_visited"].mean()),
    }
    base_std = {
        "f_helix_error": float(mc_last["f_helix_error"].std(ddof=1)) if len(mc_last) > 1 else 0.0,
        "f_open_error": float(mc_last["f_open_error"].std(ddof=1)) if len(mc_last) > 1 else 0.0,
        "jsd": float(mc_last["jsd"].std(ddof=1)) if len(mc_last) > 1 else 0.0,
        "coverage": float(mc_last["coverage"].std(ddof=1)) if len(mc_last) > 1 else 0.0,
        "important_coverage": float(mc_last["important_coverage"].std(ddof=1)) if len(mc_last) > 1 else 0.0,
        "source_mass_visited": float(mc_last["source_mass_visited"].std(ddof=1)) if len(mc_last) > 1 else 0.0,
    }
    summaries["random_mc_final_mean_over_repeats"] = base_mean
    summaries["random_mc_final_std_over_repeats"] = base_std
    summaries["random_mc_median_sampled_ns_at_horizon"] = float(mc_last["sampled_ns"].median())

    last_ns = summaries["per_policy_final"]["compass"]["sampled_ns"]
    hit = rand_summ[(rand_summ["sampled_ns"] - last_ns).abs() < 1e-9]
    rs_last = hit.iloc[-1] if not hit.empty else rand_summ.sort_values("sampled_ns").iloc[-1]
    summaries["random_mc_summary_row_nearest_compass_ns"] = {}
    for c in rs_last.index:
        v = rs_last[c]
        if pd.isna(v):
            continue
        try:
            summaries["random_mc_summary_row_nearest_compass_ns"][str(c)] = float(v)
        except (TypeError, ValueError):
            summaries["random_mc_summary_row_nearest_compass_ns"][str(c)] = v

    comp = summaries["per_policy_final"]["compass"]

    lowers = {"f_helix_error", "f_open_error", "jsd"}
    highs = {"coverage", "important_coverage", "source_mass_visited"}
    for m, v_comp in comp.items():
        if m not in lowers | highs:
            continue
        if m not in base_mean:
            continue
        lo = m in lowers
        pct = _pct_improve(float(v_comp), base_mean[m], lower_is_better=lo)
        summaries["compass_vs_random_mc_mean_pct"][m] = pct
        summaries["improvement_vs_random_mean_pct"][m] = pct  # alias

    chrono = summaries["per_policy_final"]["chrono_single_chain"]
    for m in base_mean:
        lo = m in lowers
        if m not in chrono:
            continue
        summaries["chrono_vs_random_mc_mean_pct"][m] = _pct_improve(
            float(chrono[m]), base_mean[m], lower_is_better=lo
        )

    with open(out_root / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries["per_policy_final"], indent=2))
    print(json.dumps({"compass_vs_random_mc_mean_pct": summaries["compass_vs_random_mc_mean_pct"]}, indent=2))
    print(json.dumps({"chrono_vs_random_mc_mean_pct": summaries["chrono_vs_random_mc_mean_pct"]}, indent=2))


if __name__ == "__main__":
    main()
