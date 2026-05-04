"""Observed target trajectory state for online-faithful COMPASS replay."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectorySegments:
    """Half-open ranges [start, end) covering independent MD replicas."""

    ranges: list[tuple[int, int]]

    def containing(self, frame: int) -> tuple[int, int] | None:
        for s, e in self.ranges:
            if s <= frame < e:
                return (s, e)
        return None

    def clip_continuation(self, seed_frame: int, chunk_frames: int) -> tuple[int, int]:
        seg = self.containing(seed_frame)
        if seg is None:
            raise ValueError(f"Frame {seed_frame} not in any trajectory segment")
        s, e = seg
        end = min(seed_frame + chunk_frames, e)
        return seed_frame, end


def parse_msm_segments_arg(spec: str | None, n_frames: int) -> list[tuple[int, int]]:
    """Parse '0:4750,4750:9500' into half-open ranges."""

    if not spec:
        mid = n_frames // 2
        return [(0, mid), (mid, n_frames)]
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    if not out:
        return [(0, n_frames)]
    return out


@dataclass
class ObservedTargetState:
    """Frames seen so far on the mutant; seed reservoir for online decisions."""

    n_frames: int
    segments: TrajectorySegments
    seed_reservoir_stride: int = 1
    seed_reservoir_max: int | None = None
    allow_deterministic_reseed: bool = False
    observed_mask: np.ndarray = field(init=False)
    used_seed_frames: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.observed_mask = np.zeros(self.n_frames, dtype=bool)

    def mark_observed(self, start: int, stop: int) -> None:
        start = max(0, int(start))
        stop = min(self.n_frames, int(stop))
        if start < stop:
            self.observed_mask[start:stop] = True

    def seed_pool(self) -> np.ndarray:
        idx = np.flatnonzero(self.observed_mask)
        if idx.size == 0:
            return idx
        if self.seed_reservoir_stride > 1:
            idx = idx[:: self.seed_reservoir_stride]
        if self.seed_reservoir_max is not None and idx.size > self.seed_reservoir_max:
            idx = idx[: int(self.seed_reservoir_max)]
        if not self.allow_deterministic_reseed:
            idx = np.array([i for i in idx if i not in self.used_seed_frames], dtype=int)
        return idx

    def register_seed(self, seed_frame: int) -> None:
        self.used_seed_frames.add(int(seed_frame))

    @property
    def n_observed_frames(self) -> int:
        return int(np.sum(self.observed_mask))

    def observed_bins_count(self, bx: np.ndarray, by: np.ndarray) -> int:
        idx = np.flatnonzero(self.observed_mask)
        if idx.size == 0:
            return 0
        pairs = np.column_stack([bx[idx], by[idx]])
        return int(len({tuple(row) for row in pairs}))
