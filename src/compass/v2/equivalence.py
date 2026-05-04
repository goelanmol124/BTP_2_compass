"""Equivalence methods and anti-cheat gates for candidate-cloud expansion."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.compass.online_state import TrajectorySegments


class EquivalenceMethod(str, Enum):
    M1_RMSD_CA = "M1"
    M3_TORSION = "M3"
    M5_TICA_TARGET = "M5"
    M7_MACRO_KMEANS = "M7"
    M8_HYBRID_M1_M3 = "M8"


def segment_id(segments: TrajectorySegments, frame: int) -> int | None:
    for si, (s, e) in enumerate(segments.ranges):
        if s <= frame < e:
            return si
    return None


def apply_gates(
    anchor: int,
    neighbor: int,
    segments: TrajectorySegments,
    *,
    max_time_delta_frames: int | None,
    same_segment_only: bool,
    macro_labels: np.ndarray | None,
    require_same_macro: bool,
) -> bool:
    """Return True if ``neighbor`` may be linked to ``anchor`` for the candidate pool."""

    if neighbor == anchor:
        return True
    if same_segment_only or max_time_delta_frames is not None:
        sa = segment_id(segments, anchor)
        sj = segment_id(segments, neighbor)
        if sa is None or sj is None:
            return False
        if same_segment_only and sa != sj:
            return False
        if max_time_delta_frames is not None and sa == sj:
            s, e = segments.ranges[sa]
            if abs(neighbor - anchor) > max_time_delta_frames:
                return False
    if require_same_macro and macro_labels is not None:
        if int(macro_labels[anchor]) != int(macro_labels[neighbor]):
            return False
    return True


def expand_candidates(
    visited_frames: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_mask: np.ndarray | None,
    segments: TrajectorySegments,
    *,
    max_time_delta_frames: int | None = 500,
    same_segment_only: bool = False,
    macro_labels: np.ndarray | None = None,
    require_same_macro: bool = False,
    max_neighbors_per_anchor: int = 50,
) -> np.ndarray:
    """Union of visited frames and gated neighbors."""

    visited_frames = np.asarray(visited_frames, dtype=int)
    out: set[int] = set(int(x) for x in visited_frames.tolist())
    n = neighbor_indices.shape[0]
    for v in visited_frames.tolist():
        if v < 0 or v >= n:
            continue
        row = neighbor_indices[v]
        for pos, j in enumerate(row[:max_neighbors_per_anchor]):
            j = int(j)
            if j < 0 or j >= n:
                continue
            if neighbor_mask is not None and not bool(neighbor_mask[v, pos]):
                continue
            if not apply_gates(
                v,
                j,
                segments,
                max_time_delta_frames=max_time_delta_frames,
                same_segment_only=same_segment_only,
                macro_labels=macro_labels,
                require_same_macro=require_same_macro,
            ):
                continue
            out.add(j)
    return np.array(sorted(out), dtype=int)
