"""Reproducible point, range, uniform, and Zipf workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorkloadOperation:
    kind: str
    key: Any = None
    end_key: Any = None
    value: Any = None


def point_keys(keys: np.ndarray, count: int, seed: int, zipf: bool = False) -> list[int]:
    rng = np.random.default_rng(seed)
    if not zipf:
        return rng.choice(keys, size=count, replace=True).astype(int).tolist()
    ranks = np.minimum(rng.zipf(1.35, size=count) - 1, len(keys) - 1)
    return keys[np.asarray(ranks)].astype(int).tolist()


def mixed_workload(
    keys: np.ndarray,
    year_keys: list[tuple[int, int]],
    count: int,
    seed: int,
    distribution: str = "uniform",
) -> list[WorkloadOperation]:
    """Create 60% hits, 25% ranges, 10% inserts, and 5% misses."""
    if distribution not in {"uniform", "zipf"}:
        raise ValueError("distribution must be uniform or zipf")
    rng = np.random.default_rng(seed)
    existing = point_keys(keys, count, seed, zipf=distribution == "zipf")
    years = np.array([year for year, _ in year_keys], dtype=int)
    max_key = int(keys.max())
    operations: list[WorkloadOperation] = []
    kinds = rng.choice(["point", "range", "insert", "miss"], count, p=[0.60, 0.25, 0.10, 0.05])
    insert_number = 0
    for index, kind in enumerate(kinds):
        if kind == "point":
            operations.append(WorkloadOperation(kind, existing[index]))
        elif kind == "range":
            year = int(rng.choice(years))
            operations.append(WorkloadOperation(kind, (year, -1), (year + int(rng.integers(0, 5)), 10**19)))
        elif kind == "insert":
            insert_number += 1
            key = max_key + insert_number
            operations.append(WorkloadOperation(kind, key, value=key))
        else:
            operations.append(WorkloadOperation(kind, max_key + count + index + 1))
    return operations
