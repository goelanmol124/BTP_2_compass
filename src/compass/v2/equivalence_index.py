"""Precomputed k-NN indices for equivalence methods (M1, M3, M5, M7, M8)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from src.compass.replay_enhanced import project_tica
from src.compass.v2.equivalence import EquivalenceMethod

try:
    from deeptime.decomposition import TICA as DeeptimeTICA
except ImportError:
    DeeptimeTICA = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class IndexMeta:
    method: str
    n_frames: int
    k: int
    lat: int
    notes: str


def _load_features(repo_root: Path) -> np.ndarray:
    p = repo_root / "outputs" / "artifacts" / "a8q2_features.npy"
    if not p.is_file():
        raise FileNotFoundError(p)
    return np.load(p).astype(np.float64)


def _load_concatenated_ca_coords(repo_root: Path) -> np.ndarray:
    """CA positions flattened (N_frames, 3*n_ca) superposed to first frame."""

    try:
        import mdtraj as md
    except ImportError as exc:
        raise RuntimeError("mdtraj required for M1 CA coordinates") from exc

    sim_dir = repo_root / "outputs" / "btp_equil" / "A8Q2"
    blocks: list[np.ndarray] = []
    for lab in ("extended", "helix"):
        dcd = sim_dir / f"A8Q2_{lab}_prod.dcd"
        top = sim_dir / f"A8Q2_{lab}_prod_top.pdb"
        if not dcd.is_file() or not top.is_file():
            raise FileNotFoundError(f"Need {dcd} and {top} for M1.")
        traj = md.load(str(dcd), top=str(top))
        if traj.n_frames <= BURNIN_FRAMES:
            raise RuntimeError(f"{dcd}: insufficient frames")
        traj = traj[BURNIN_FRAMES:]
        traj.superpose(traj, frame=0)
        ca = traj.xyz[:, traj.top.select("name CA"), :]
        blocks.append(ca.reshape(len(traj), -1))
    X = np.vstack(blocks).astype(np.float64)
    return X


def _neighbor_graph(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return indices (n, min(k+1,n)) and distances including self at column 0."""

    kk = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=kk, algorithm="auto", metric="euclidean")
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)
    return dist, idx


def _fit_target_tica_latent(features: np.ndarray, lag: int = 10) -> np.ndarray:
    if DeeptimeTICA is not None and len(features) > lag * 3:
        model = DeeptimeTICA(dim=2, lagtime=max(1, min(lag, len(features) // 20)))
        model.fit(features)
        return np.asarray(model.transform(features), dtype=np.float64)
    from sklearn.decomposition import PCA

    return PCA(n_components=2).fit_transform(features)


def _macro_kmeans_labels(target_xy: np.ndarray, n_clusters: int = 16) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1024)
    return km.fit_predict(np.asarray(target_xy, dtype=np.float64))


def _pca_features_2d(feats: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=0).fit_transform(np.asarray(feats, dtype=np.float64))


def build_method_matrix(
    repo_root: Path,
    method: EquivalenceMethod,
    *,
    n_clusters: int = 16,
) -> tuple[np.ndarray, str]:
    feats = _load_features(repo_root)
    if method == EquivalenceMethod.M1_RMSD_CA:
        X = _load_concatenated_ca_coords(repo_root)
        if X.shape[0] != feats.shape[0]:
            raise ValueError(f"M1 CA rows {X.shape[0]} != features rows {feats.shape[0]}")
        return X, "aligned_CA_flat_superpose_frame0"
    if method == EquivalenceMethod.M3_TORSION:
        return feats, "a8q2_features_sin_cos_torsions"
    if method == EquivalenceMethod.M5_TICA_TARGET:
        Z = _fit_target_tica_latent(feats)
        return Z, "deeptime_tica2_on_target_features_or_pca_fallback"
    if method == EquivalenceMethod.M7_MACRO_KMEANS:
        z = _pca_features_2d(feats)
        return z.astype(np.float64), f"kmeans_projection_nc{n_clusters}"
    raise ValueError(f"Unknown method matrix for {method}")


def intersect_neighbors(a_idx: np.ndarray, b_idx: np.ndarray, k_out: int) -> np.ndarray:
    """Intersect two neighbor lists (excluding self handled by caller). Row-wise capped."""

    n, ka = a_idx.shape
    _, kb = b_idx.shape
    out = np.full((n, k_out), -1, dtype=np.int64)
    for i in range(n):
        sa = []
        sb = []
        row_a = []
        row_b = []
        for jj in range(ka):
            j = int(a_idx[i, jj])
            if j != i:
                row_a.append(j)
        for jj in range(kb):
            j = int(b_idx[i, jj])
            if j != i:
                row_b.append(j)
        set_b = set(row_b)
        merged = []
        for j in row_a:
            if j in set_b:
                merged.append(j)
                if len(merged) >= k_out:
                    break
        merged.extend([x for x in row_a if x not in merged])
        merged = merged[:k_out]
        neg = merged + [-1] * (k_out - len(merged))
        out[i, :] = np.array(neg[:k_out], dtype=np.int64)
    return out


def build_and_save(
    repo_root: Path,
    cache_dir: Path,
    method: EquivalenceMethod,
    k: int,
    *,
    n_clusters: int = 16,
    m8_k1: int | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if method == EquivalenceMethod.M8_HYBRID_M1_M3:
        k1 = m8_k1 or max(k, 30)
        p1 = build_and_save(repo_root, cache_dir, EquivalenceMethod.M1_RMSD_CA, k1, n_clusters=n_clusters)
        p3 = build_and_save(repo_root, cache_dir, EquivalenceMethod.M3_TORSION, k1, n_clusters=n_clusters)
        d1 = np.load(p1)
        d3 = np.load(p3)
        idx = intersect_neighbors(d1["indices"], d3["indices"], k)
        out_path = cache_dir / f"eq_{EquivalenceMethod.M8_HYBRID_M1_M3.value}_k{k}.npz"
        np.savez_compressed(
            out_path,
            indices=idx,
            distances=np.zeros_like(idx, dtype=np.float64),
            method=EquivalenceMethod.M8_HYBRID_M1_M3.value,
            k=np.int32(k),
            sources=np.array([str(p1), str(p3)]),
        )
        meta = IndexMeta(
            method=EquivalenceMethod.M8_HYBRID_M1_M3.value,
            n_frames=int(idx.shape[0]),
            k=k,
            lat=0,
            notes="intersection_M1_M3",
        )
        (cache_dir / f"eq_{EquivalenceMethod.M8_HYBRID_M1_M3.value}_k{k}.meta.json").write_text(
            json.dumps(meta.__dict__, indent=2), encoding="utf-8"
        )
        return out_path

    X, note = build_method_matrix(repo_root, method, n_clusters=n_clusters)
    dist, idx = _neighbor_graph(X, k)
    out_path = cache_dir / f"eq_{method.value}_k{k}.npz"
    if method == EquivalenceMethod.M7_MACRO_KMEANS:
        macro_lab = _macro_kmeans_labels(X, n_clusters=n_clusters)
        np.savez_compressed(
            cache_dir / f"macro_kmeans_nc{n_clusters}.npz",
            labels=macro_lab,
            n_clusters=np.int32(n_clusters),
        )
        np.savez_compressed(
            out_path,
            indices=idx,
            distances=dist,
            method=method.value,
            k=np.int32(k),
            feature_note=note,
            macro_labels=macro_lab,
            n_clusters=np.int32(n_clusters),
        )
    else:
        np.savez_compressed(
            out_path, indices=idx, distances=dist, method=method.value, k=np.int32(k), feature_note=note
        )
    meta = IndexMeta(method=method.value, n_frames=int(X.shape[0]), k=k, lat=0, notes=note)
    (cache_dir / f"eq_{method.value}_k{k}.meta.json").write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")
    return out_path


def load_index(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_macro_labels(repo_root: Path, cache_dir: Path, n_clusters: int = 16) -> np.ndarray:
    p = cache_dir / f"macro_kmeans_nc{n_clusters}.npz"
    if not p.is_file():
        feats = _load_features(repo_root)
        z = _pca_features_2d(feats)
        labels = _macro_kmeans_labels(z, n_clusters=n_clusters)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, labels=labels, n_clusters=np.int32(n_clusters))
        return labels
    return np.load(p)["labels"]


def target_tica2_for_policy(repo_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, ev, a8_features) for A10-fitted TICA projection (policy latents)."""

    artifacts = repo_root / "outputs" / "artifacts"
    tica_dir = repo_root / "outputs" / "tica_artifacts_a10"
    a8 = np.load(artifacts / "a8q2_features.npy")
    mean = np.load(tica_dir / "a10_tica_mean0.npy")
    ev = np.load(tica_dir / "a10_tica_singular_vectors_left.npy")
    return mean, ev, a8
