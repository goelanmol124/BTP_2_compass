"""Forward-window similarity scores (feature + optional NHB JSD) for equivalence validation/calibration."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def forward_feature_js(
    feats: np.ndarray,
    anchors: np.ndarray,
    neighbors: np.ndarray,
    chunk: int,
) -> np.ndarray:
    """Per-anchor–neighbor pair: median JSD over feature dims on aligned forward slices."""

    n_pairs = anchors.size
    scores = np.zeros(n_pairs)
    nf = feats.shape[1]
    for pi in range(n_pairs):
        i, j = int(anchors[pi]), int(neighbors[pi])
        hi = feats[i : i + chunk]
        hj = feats[j : j + chunk]
        m = min(len(hi), len(hj))
        if m < 4:
            scores[pi] = 1.0
            continue
        fi, fj = hi[:m], hj[:m]
        errs = []
        for d in range(min(nf, 40)):
            a = fi[:, d]
            b = fj[:, d]
            lo = min(float(np.min(a)), float(np.min(b)))
            hi_r = max(float(np.max(a)), float(np.max(b)))
            if hi_r <= lo:
                continue
            h1, _ = np.histogram(a, bins=8, range=(lo, hi_r), density=True)
            h2, _ = np.histogram(b, bins=8, range=(lo, hi_r), density=True)
            h1 = np.clip(h1 / max(h1.sum(), 1e-12), 1e-12, None)
            h2 = np.clip(h2 / max(h2.sum(), 1e-12), 1e-12, None)
            errs.append(float(jensenshannon(np.sqrt(h1), np.sqrt(h2), base=2) ** 2))
        scores[pi] = float(np.median(errs)) if errs else 1.0
    return scores


def forward_nhb_js(
    nhb: np.ndarray,
    anchors: np.ndarray,
    neighbors: np.ndarray,
    chunk: int,
) -> np.ndarray:
    """Per pair: JSD squared on aligned NHB count histograms."""

    n_pairs = anchors.size
    out = np.zeros(n_pairs)
    vmax = float(np.max(nhb))
    edges = np.arange(-0.5, max(vmax + 1.5, 8.0), 1.0)
    for pi in range(n_pairs):
        i, j = int(anchors[pi]), int(neighbors[pi])
        vi = np.asarray(nhb[i : i + chunk], dtype=np.float64)
        vj = np.asarray(nhb[j : j + chunk], dtype=np.float64)
        m = min(vi.size, vj.size)
        if m < 4:
            out[pi] = 1.0
            continue
        vi, vj = vi[:m], vj[:m]
        h1, _ = np.histogram(vi, bins=edges.size - 1, range=(edges[0], edges[-1]), density=True)
        h2, _ = np.histogram(vj, bins=edges.size - 1, range=(edges[0], edges[-1]), density=True)
        h1 = np.clip(h1 / max(h1.sum(), 1e-12), 1e-12, None)
        h2 = np.clip(h2 / max(h2.sum(), 1e-12), 1e-12, None)
        out[pi] = float(jensenshannon(np.sqrt(h1), np.sqrt(h2), base=2) ** 2)
    return out


def forward_composite_scores(
    feats: np.ndarray,
    nhb: np.ndarray | None,
    anchors: np.ndarray,
    neighbors: np.ndarray,
    chunk: int,
    *,
    feat_weight: float = 0.5,
    nhb_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(feat_js, nhb_js_or_nan, composite)`` arrays of length ``len(anchors)``."""

    fjs = forward_feature_js(feats, anchors, neighbors, chunk)
    n_frames = feats.shape[0]
    nh = np.full_like(fjs, np.nan, dtype=np.float64)
    comp = np.zeros_like(fjs, dtype=np.float64)
    if nhb is not None and np.asarray(nhb).size == n_frames:
        nh = forward_nhb_js(np.asarray(nhb), anchors, neighbors, chunk)
        comp = feat_weight * fjs + nhb_weight * nh
    else:
        comp = fjs.copy()
    return fjs, nh, comp


def percentile_summary(x: np.ndarray, pct: tuple[float, ...] = (10.0, 50.0, 90.0)) -> dict[str, float]:
    """Named percentiles as strings ``p10`` etc.; empty ``x`` returns NaNs."""

    if x.size == 0:
        return {f"p{int(q)}": float("nan") for q in pct}
    return {f"p{int(q)}": float(np.percentile(x, q)) for q in pct}


def youden_best_threshold(
    distances: np.ndarray,
    *,
    positive_low_composite: np.ndarray,
) -> tuple[float, float]:
    """Best ``d_star`` maximizing Youden's J for rule: predict *close* iff ``distance <= d_star``.

    ``positive_low_composite`` is True for pairs in the empirical *close* tier by composite.
    Returns ``(d_star, j_max)``. Cumulative over sorted distances.
    """

    d = np.asarray(distances, dtype=np.float64)
    y = np.asarray(positive_low_composite, dtype=bool)
    if d.size == 0 or y.size != d.size:
        return float("nan"), float("nan")
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(d, kind="mergesort")
    d_s = d[order]
    y_s = y[order]
    best_j = -2.0
    best_t = float("nan")
    tp = 0
    fp = 0
    i = 0
    while i < len(d_s):
        t_hi = d_s[i]
        j = i
        while j < len(d_s) and d_s[j] <= t_hi:
            if y_s[j]:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        j_val = tpr - fpr
        if j_val >= best_j:
            best_j = j_val
            best_t = t_hi
        i = j
    return float(best_t), float(best_j)
