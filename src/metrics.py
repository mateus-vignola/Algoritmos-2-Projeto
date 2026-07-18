"""Shared metrics for the instrumented tree implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class OperationMetrics:
    """Counters accumulated by tree operations."""

    comparisons: int = 0
    nodes_visited: int = 0
    pages_read: int = 0
    pages_written: int = 0
    splits: int = 0

    def reset(self) -> None:
        """Set every counter back to zero."""
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, 0)

    def as_dict(self) -> dict[str, Any]:
        """Return counters in a serializable form."""
        return asdict(self)


def lower_bound(items: list[Any], key: Any, metrics: OperationMetrics) -> int:
    """Find the first item not less than key while counting comparisons."""
    low, high = 0, len(items)
    while low < high:
        middle = (low + high) // 2
        metrics.comparisons += 1
        if items[middle] < key:
            low = middle + 1
        else:
            high = middle
    return low


def upper_bound(items: list[Any], key: Any, metrics: OperationMetrics) -> int:
    """Find the first item greater than key while counting comparisons."""
    low, high = 0, len(items)
    while low < high:
        middle = (low + high) // 2
        metrics.comparisons += 1
        if key < items[middle]:
            high = middle
        else:
            low = middle + 1
    return low
