"""In-memory B+ tree with linked leaves and logical page metrics."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Optional, TypeVar

from .metrics import OperationMetrics, lower_bound, upper_bound

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class _BPlusNode(Generic[K, V]):
    leaf: bool = True
    keys: list[K] = field(default_factory=list)
    values: list[V] = field(default_factory=list)
    children: list["_BPlusNode[K, V]"] = field(default_factory=list)
    next_leaf: Optional["_BPlusNode[K, V]"] = None


class BPlusTree(Generic[K, V]):
    """B+ tree whose order is the maximum number of children per internal node.

    Separator keys in internal nodes have no associated record value. All values
    reside in linked leaves, which enables sequential range traversal.
    """

    name = "B+ tree"

    def __init__(self, order: int = 32) -> None:
        if order < 4 or order % 2:
            raise ValueError("order must be an even integer greater than or equal to 4")
        self.order = order
        self.max_keys = order - 1
        self.root: _BPlusNode[K, V] = _BPlusNode()
        self.metrics = OperationMetrics()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def reset_metrics(self) -> None:
        self.metrics.reset()

    def get_metrics(self) -> dict[str, Any]:
        return self.metrics.as_dict()

    def _visit(self) -> None:
        self.metrics.nodes_visited += 1
        self.metrics.pages_read += 1

    def _find_leaf(self, key: K) -> _BPlusNode[K, V]:
        node = self.root
        while not node.leaf:
            self._visit()
            node = node.children[upper_bound(node.keys, key, self.metrics)]
        self._visit()
        return node

    def search(self, key: K) -> Optional[V]:
        """Return the value for key, or None when absent."""
        leaf = self._find_leaf(key)
        index = lower_bound(leaf.keys, key, self.metrics)
        if index < len(leaf.keys):
            self.metrics.comparisons += 1
            if leaf.keys[index] == key:
                return leaf.values[index]
        return None

    def insert(self, key: K, value: V) -> None:
        """Insert a key-value pair, replacing the value for duplicate keys."""
        split = self._insert_recursive(self.root, key, value)
        if split is not None:
            separator, right = split
            self.root = _BPlusNode(
                leaf=False, keys=[separator], children=[self.root, right]
            )
            self.metrics.pages_written += 1

    def _insert_recursive(
        self, node: _BPlusNode[K, V], key: K, value: V
    ) -> Optional[tuple[K, _BPlusNode[K, V]]]:
        self._visit()
        if node.leaf:
            index = lower_bound(node.keys, key, self.metrics)
            if index < len(node.keys):
                self.metrics.comparisons += 1
                if node.keys[index] == key:
                    node.values[index] = value
                    self.metrics.pages_written += 1
                    return None
            node.keys.insert(index, key)
            node.values.insert(index, value)
            self._size += 1
            self.metrics.pages_written += 1
            if len(node.keys) <= self.max_keys:
                return None
            return self._split_leaf(node)

        child_index = upper_bound(node.keys, key, self.metrics)
        split = self._insert_recursive(node.children[child_index], key, value)
        if split is None:
            return None
        separator, right = split
        node.keys.insert(child_index, separator)
        node.children.insert(child_index + 1, right)
        self.metrics.pages_written += 1
        if len(node.keys) <= self.max_keys:
            return None
        return self._split_internal(node)

    def _split_leaf(self, leaf: _BPlusNode[K, V]) -> tuple[K, _BPlusNode[K, V]]:
        middle = len(leaf.keys) // 2
        right = _BPlusNode(
            leaf=True,
            keys=leaf.keys[middle:],
            values=leaf.values[middle:],
            next_leaf=leaf.next_leaf,
        )
        leaf.keys = leaf.keys[:middle]
        leaf.values = leaf.values[:middle]
        leaf.next_leaf = right
        self.metrics.splits += 1
        self.metrics.pages_written += 2
        return right.keys[0], right

    def _split_internal(
        self, node: _BPlusNode[K, V]
    ) -> tuple[K, _BPlusNode[K, V]]:
        middle = len(node.keys) // 2
        separator = node.keys[middle]
        right = _BPlusNode(
            leaf=False,
            keys=node.keys[middle + 1 :],
            children=node.children[middle + 1 :],
        )
        node.keys = node.keys[:middle]
        node.children = node.children[: middle + 1]
        self.metrics.splits += 1
        self.metrics.pages_written += 2
        return separator, right

    def range_search(self, start_key: K, end_key: K) -> list[tuple[K, V]]:
        """Return sorted key-value pairs in the inclusive interval."""
        if end_key < start_key:
            return []
        leaf = self._find_leaf(start_key)
        index = lower_bound(leaf.keys, start_key, self.metrics)
        output: list[tuple[K, V]] = []
        while leaf is not None:
            while index < len(leaf.keys):
                key = leaf.keys[index]
                self.metrics.comparisons += 1
                if key > end_key:
                    return output
                output.append((key, leaf.values[index]))
                index += 1
            leaf = leaf.next_leaf
            index = 0
            if leaf is not None:
                self._visit()
        return output

    def range_search_profile(
        self, start_key: K, end_key: K
    ) -> tuple[list[tuple[K, V]], int, int]:
        """Run a range query and separate leaf location from linked traversal."""
        if end_key < start_key:
            return [], 0, 0
        locate_start = time.perf_counter_ns()
        leaf = self._find_leaf(start_key)
        index = lower_bound(leaf.keys, start_key, self.metrics)
        locate_ns = time.perf_counter_ns() - locate_start
        traversal_start = time.perf_counter_ns()
        output: list[tuple[K, V]] = []
        while leaf is not None:
            while index < len(leaf.keys):
                key = leaf.keys[index]
                self.metrics.comparisons += 1
                if key > end_key:
                    return output, locate_ns, time.perf_counter_ns() - traversal_start
                output.append((key, leaf.values[index]))
                index += 1
            leaf = leaf.next_leaf
            index = 0
            if leaf is not None:
                self._visit()
        return output, locate_ns, time.perf_counter_ns() - traversal_start

    def height(self) -> int:
        height = 0
        node = self.root
        while True:
            height += 1
            if node.leaf:
                return height
            node = node.children[0]

    def _nodes(self) -> Iterator[_BPlusNode[K, V]]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children)

    def count_nodes(self) -> int:
        return sum(1 for _ in self._nodes())

    def structure_stats(self) -> dict[str, float | int]:
        nodes = list(self._nodes())
        leaves = sum(node.leaf for node in nodes)
        used = sum(len(node.keys) for node in nodes)
        return {
            "height": self.height(),
            "nodes": len(nodes),
            "internal_nodes": len(nodes) - leaves,
            "leaves": leaves,
            "occupancy": used / (len(nodes) * self.max_keys) if nodes else 0.0,
        }

    def estimated_size_bytes(self) -> int:
        total = sys.getsizeof(self) + sys.getsizeof(self.metrics)
        for node in self._nodes():
            total += sys.getsizeof(node)
            total += sys.getsizeof(node.keys) + sys.getsizeof(node.values)
            total += sys.getsizeof(node.children) + sys.getsizeof(node.next_leaf)
            total += sum(sys.getsizeof(key) for key in node.keys)
            total += sum(sys.getsizeof(value) for value in node.values)
        return total

    def delete(self, key: K) -> None:
        """Delete *key* and preserve B+ occupancy and separator invariants."""
        removed = self._delete_recursive(self.root, key)
        if not self.root.leaf and len(self.root.children) == 1:
            self.root = self.root.children[0]
        if removed:
            self._size -= 1

    def _delete_recursive(self, node: _BPlusNode[K, V], key: K) -> bool:
        self._visit()
        if node.leaf:
            index = lower_bound(node.keys, key, self.metrics)
            if index >= len(node.keys):
                return False
            self.metrics.comparisons += 1
            if node.keys[index] != key:
                return False
            node.keys.pop(index)
            node.values.pop(index)
            self.metrics.pages_written += 1
            return True

        child_index = upper_bound(node.keys, key, self.metrics)
        removed = self._delete_recursive(node.children[child_index], key)
        if not removed:
            return False
        self._rebalance_child(node, child_index)
        self._refresh_keys(node)
        return True

    def _rebalance_child(self, parent: _BPlusNode[K, V], index: int) -> None:
        child = parent.children[index]
        minimum = self.order // 2 if child.leaf else self.order // 2
        occupancy = len(child.keys) if child.leaf else len(child.children)
        if occupancy >= minimum:
            return

        left = parent.children[index - 1] if index > 0 else None
        right = parent.children[index + 1] if index + 1 < len(parent.children) else None
        left_occupancy = (
            len(left.keys) if left is not None and left.leaf
            else len(left.children) if left is not None else 0
        )
        right_occupancy = (
            len(right.keys) if right is not None and right.leaf
            else len(right.children) if right is not None else 0
        )

        if left is not None and left_occupancy > minimum:
            if child.leaf:
                child.keys.insert(0, left.keys.pop())
                child.values.insert(0, left.values.pop())
            else:
                child.children.insert(0, left.children.pop())
                self._refresh_keys(left)
                self._refresh_keys(child)
            self.metrics.pages_written += 3
            return
        if right is not None and right_occupancy > minimum:
            if child.leaf:
                child.keys.append(right.keys.pop(0))
                child.values.append(right.values.pop(0))
            else:
                child.children.append(right.children.pop(0))
                self._refresh_keys(right)
                self._refresh_keys(child)
            self.metrics.pages_written += 3
            return

        if left is not None:
            self._merge_nodes(left, child)
            parent.children.pop(index)
        elif right is not None:
            self._merge_nodes(child, right)
            parent.children.pop(index + 1)
        self.metrics.pages_written += 2

    def _merge_nodes(
        self, left: _BPlusNode[K, V], right: _BPlusNode[K, V]
    ) -> None:
        if left.leaf:
            left.keys.extend(right.keys)
            left.values.extend(right.values)
            left.next_leaf = right.next_leaf
        else:
            left.children.extend(right.children)
            self._refresh_keys(left)

    def _refresh_keys(self, node: _BPlusNode[K, V]) -> None:
        if not node.leaf:
            node.keys = [self._first_key(child) for child in node.children[1:]]

    @staticmethod
    def _first_key(node: _BPlusNode[K, V]) -> K:
        current = node
        while not current.leaf:
            current = current.children[0]
        return current.keys[0]
