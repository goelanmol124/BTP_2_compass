"""Rank equivalence methods using forward-window similarity on archived MD (oracle validation).

Composite score (lower is better): mixes torsion-feature JSD with optional NHB-path JSD,
both computed on *full* forward windows of ``chunk_frames`` (not 2-frame stubs).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.compass.online_state import TrajectorySegments, parse_msm_segments_arg
from src.compass.v2.equivalence import EquivalenceMethod, expand_candidates
from src.compass.v2.equivalence_index import build_and_save, load_index
from src.compass.v2.equivalence_index import load_macro_labels as load_macro_lab
from src.compass.v2.equivalence_index import _load_features
from src.compass.v2.forward_similarity import forward_feature_js, forward_nhb_js


def validate(
    repo_root: Path,
    oracle_cache: Path,
    *,
    chunk_frames: int,
    method_list: list[EquivalenceMethod],
    eq_k: int,
    anchors: int,
    segments_spec: str | None,
    rng: np.random.Generator,
    feat_weight: float = 0.5,
    nhb_weight: float = 0.5,
    same_segment_only_gate: bool = False,
    max_time_delta_frames: int = 800,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    feats = _load_features(repo_root)
    n_frames = feats.shape[0]
    segments = parse_msm_segments_arg(segments_spec, n_frames)
    seg_helper = TrajectorySegments(segments)
    macro = load_macro_lab(repo_root, oracle_cache)

    nhb_path = repo_root / "outputs" / "artifacts" / "a8q2_nhb.npy"
    nhb = np.load(nhb_path) if nhb_path.is_file() else None

    picks = rng.choice(n_frames, size=min(int(anchors), n_frames), replace=False)

    rankings: dict[str, Any] = {}
    winner = ""
    winner_score = float("inf")

    for m in method_list:
        path = oracle_cache / f"eq_{m.value}_k{eq_k}.npz"
        try:
            if not path.is_file():
                path = build_and_save(repo_root, oracle_cache, m, eq_k)
            eq = load_index(path)
        except Exception as exc:
            rankings[m.value] = {
                "error": repr(exc),
                "composite_score": float("inf"),
            }
            continue
        idx_tbl = eq["indices"]
        pooled_feat: list[float] = []
        pooled_nhb: list[float] = []
        pooled_composite: list[float] = []

        for ii in picks:
            visited = np.array([int(ii)])
            cand = expand_candidates(
                visited,
                idx_tbl,
                None,
                seg_helper,
                max_time_delta_frames=max_time_delta_frames if not same_segment_only_gate else None,
                same_segment_only=same_segment_only_gate,
                macro_labels=macro if m != EquivalenceMethod.M7_MACRO_KMEANS else None,
                require_same_macro=False,
                max_neighbors_per_anchor=eq_k,
            )
            jj_list = cand.tolist()
            jj_list = [int(x) for x in jj_list if int(x) != int(ii)][:eq_k]
            take = jj_list[:5]
            if not take:
                continue
            anc = np.array([ii] * len(take))
            nei = np.array(take, dtype=int)
            fjs = forward_feature_js(feats, anc, nei, chunk_frames)
            med_f = float(np.mean(fjs))
            pooled_feat.append(med_f)
            if nhb is not None and nhb.size == n_frames:
                nhjs = forward_nhb_js(nhb, anc, nei, chunk_frames)
                med_n = float(np.mean(nhjs))
                pooled_nhb.append(med_n)
                comp = feat_weight * med_f + nhb_weight * med_n
            else:
                med_n = float("nan")
                comp = med_f
            pooled_composite.append(comp)

        if not pooled_composite:
            rankings[m.value] = {"composite_score": float("inf"), "note": "no_valid_anchors"}
            continue

        med_composite = float(np.median(np.array(pooled_composite)))
        med_feat = float(np.median(np.array(pooled_feat)))
        med_nhb_out = float(np.median(np.array(pooled_nhb))) if pooled_nhb else float("nan")
        rankings[m.value] = {
            "median_composite_vs_neighbors": med_composite,
            "median_forward_feature_js_vs_neighbors": med_feat,
            "median_forward_nhb_js_vs_neighbors": med_nhb_out,
            "n_anchor_samples": len(pooled_composite),
            "weights_feat_nhb": [feat_weight, nhb_weight] if nhb is not None and pooled_nhb else None,
            "oracle_note": "lower_composite_prefers_near_identical_forward_windows_on_archive",
            "oracle_flavored": m in (EquivalenceMethod.M5_TICA_TARGET, EquivalenceMethod.M7_MACRO_KMEANS),
        }
        if med_composite < winner_score:
            winner_score = med_composite
            winner = m.value

    payload: dict[str, Any] = {
        "equivalence_choice": {"winner_method": winner, "winner_composite_score": winner_score},
        "validation_settings": {
            "chunk_frames": chunk_frames,
            "eq_k": eq_k,
            "anchors_requested": anchors,
            "same_segment_only_gate": same_segment_only_gate,
            "feat_weight": feat_weight,
            "nhb_weight": nhb_weight,
        },
        "per_method_rankings": rankings,
    }
    oracle_cache.mkdir(parents=True, exist_ok=True)
    outp = oracle_cache / "equivalence_choice.json"
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


NAME_MAP = {
    "M1": EquivalenceMethod.M1_RMSD_CA,
    "M3": EquivalenceMethod.M3_TORSION,
    "M5": EquivalenceMethod.M5_TICA_TARGET,
    "M7": EquivalenceMethod.M7_MACRO_KMEANS,
    "M8": EquivalenceMethod.M8_HYBRID_M1_M3,
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--oracle-cache", type=Path, default=None)
    p.add_argument("--chunk-frames", type=int, default=10)
    p.add_argument("--eq-k", type=int, default=40)
    p.add_argument("--anchors", type=int, default=200)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--aggregate-seeds", type=str, default="", help="e.g. '0,1,2' — mean composite per method.")
    p.add_argument("--msm-segments", type=str, default=None)
    p.add_argument("--methods", type=str, default="M3,M5,M7", help="M8 needs M1 DCDs present.")
    p.add_argument("--same-segment-only-gate", action="store_true")
    p.add_argument("--feat-weight", type=float, default=0.5)
    p.add_argument("--nhb-weight", type=float, default=0.5)
    args = p.parse_args(argv)
    repo = args.repo_root.resolve()
    cache = (args.oracle_cache or repo / "outputs" / "v2" / "oracle").resolve()

    methods = [NAME_MAP[s.strip()] for s in args.methods.split(",") if s.strip() in NAME_MAP]
    if not methods:
        raise SystemExit("No valid methods in --methods")

    if args.aggregate_seeds.strip():
        seeds = [int(x) for x in args.aggregate_seeds.split(",") if x.strip()]
        agg: dict[str, list[float]] = defaultdict(list)
        for sd in seeds:
            rng = np.random.default_rng(sd)
            pay = validate(
                repo,
                cache,
                chunk_frames=args.chunk_frames,
                method_list=methods,
                eq_k=args.eq_k,
                anchors=args.anchors,
                segments_spec=args.msm_segments,
                rng=rng,
                feat_weight=args.feat_weight,
                nhb_weight=args.nhb_weight,
                same_segment_only_gate=args.same_segment_only_gate,
            )
            for mid, row in pay["per_method_rankings"].items():
                if "median_composite_vs_neighbors" in row:
                    agg[mid].append(float(row["median_composite_vs_neighbors"]))
        means = {k: float(np.mean(v)) for k, v in agg.items() if v}
        win = min(means, key=means.get) if means else ""
        final = {
            "equivalence_choice": {
                "winner_method": win,
                "mean_composite_across_seeds": means.get(win),
                "per_method_mean_composite": means,
                "aggregate_seeds": seeds,
            },
            "note": "Written by multi-seed aggregation; per-seed details not kept.",
        }
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "equivalence_choice.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(json.dumps(final["equivalence_choice"], indent=2))
        return

    rng = np.random.default_rng(args.random_seed)
    out = validate(
        repo,
        cache,
        chunk_frames=args.chunk_frames,
        method_list=methods,
        eq_k=args.eq_k,
        anchors=args.anchors,
        segments_spec=args.msm_segments,
        rng=rng,
        feat_weight=args.feat_weight,
        nhb_weight=args.nhb_weight,
        same_segment_only_gate=args.same_segment_only_gate,
    )
    print(json.dumps(out["equivalence_choice"], indent=2))
    print(json.dumps(out["per_method_rankings"], indent=2))


if __name__ == "__main__":
    main()
