"""In-memory B-tree with logical page-access instrumentation."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Optional, TypeVar

from .metrics import OperationMetrics, lower_bound

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class _BNode(Generic[K, V]):
    leaf: bool = True
    keys: list[K] = field(default_factory=list)
    values: list[V] = field(default_factory=list)
    children: list["_BNode[K, V]"] = field(default_factory=list)


class BTree(Generic[K, V]):
    """B-tree whose order is the maximum number of children per node.

    Values may be stored in leaves or internal nodes. Each visited node counts as
    one logical page read; every changed node counts as a logical page write.
    """

    name = "B-tree"

    def __init__(self, order: int = 32) -> None:
        if order < 4 or order % 2:
            raise ValueError("order must be an even integer greater than or equal to 4")
        self.order = order
        self.max_keys = order - 1
        self.root: _BNode[K, V] = _BNode()
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

    def search(self, key: K) -> Optional[V]:
        """Return the value for key, or None when the key is absent."""
        node = self.root
        while True:
            self._visit()
            index = lower_bound(node.keys, key, self.metrics)
            if index < len(node.keys):
                self.metrics.comparisons += 1
                if node.keys[index] == key:
                    return node.values[index]
            if node.leaf:
                return None
            node = node.children[index]

    def insert(self, key: K, value: V) -> None:
        """Insert a key-value pair, replacing the value for duplicate keys."""
        root = self.root
        if len(root.keys) == self.max_keys:
            new_root: _BNode[K, V] = _BNode(leaf=False, children=[root])
            self.root = new_root
            self._split_child(new_root, 0)
        if self._insert_non_full(self.root, key, value):
            self._size += 1

    def _insert_non_full(self, node: _BNode[K, V], key: K, value: V) -> bool:
        self._visit()
        index = lower_bound(node.keys, key, self.metrics)
        if index < len(node.keys):
            self.metrics.comparisons += 1
            if node.keys[index] == key:
                node.values[index] = value
                self.metrics.pages_written += 1
                return False
        if node.leaf:
            node.keys.insert(index, key)
            node.values.insert(index, value)
            self.metrics.pages_written += 1
            return True
        if len(node.children[index].keys) == self.max_keys:
            self._split_child(node, index)
            self.metrics.comparisons += 1
            if key == node.keys[index]:
                node.values[index] = value
                self.metrics.pages_written += 1
                return False
            if key > node.keys[index]:
                index += 1
        return self._insert_non_full(node.children[index], key, value)

    def _split_child(self, parent: _BNode[K, V], index: int) -> None:
        child = parent.children[index]
        middle = len(child.keys) // 2
        right = _BNode[K, V](leaf=child.leaf)
        promoted_key = child.keys[middle]
        promoted_value = child.values[middle]
        right.keys = child.keys[middle + 1 :]
        right.values = child.values[middle + 1 :]
        child.keys = child.keys[:middle]
        child.values = child.values[:middle]
        if not child.leaf:
            right.children = child.children[middle + 1 :]
            child.children = child.children[: middle + 1]
        parent.keys.insert(index, promoted_key)
        parent.values.insert(index, promoted_value)
        parent.children.insert(index + 1, right)
        self.metrics.splits += 1
        self.metrics.pages_written += 3

    def range_search(self, start_key: K, end_key: K) -> list[tuple[K, V]]:
        """Return sorted key-value pairs in the inclusive interval."""
        if end_key < start_key:
            return []
        return list(self._range_node(self.root, start_key, end_key))

    def _range_node(
        self, node: _BNode[K, V], start_key: K, end_key: K
    ) -> Iterator[tuple[K, V]]:
        self._visit()
        start_index = lower_bound(node.keys, start_key, self.metrics)
        index = start_index
        if not node.leaf:
            yield from self._range_node(node.children[index], start_key, end_key)
        while index < len(node.keys):
            key = node.keys[index]
            self.metrics.comparisons += 1
            if key > end_key:
                break
            if key >= start_key:
                yield key, node.values[index]
            index += 1
            if not node.leaf:
                yield from self._range_node(node.children[index], start_key, end_key)

    def height(self) -> int:
        height = 0
        node = self.root
        while True:
            height += 1
            if node.leaf:
                return height
            node = node.children[0]

    def _nodes(self) -> Iterator[_BNode[K, V]]:
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
        """Estimate Python object size without double-counting stored objects."""
        total = sys.getsizeof(self) + sys.getsizeof(self.metrics)
        for node in self._nodes():
            total += sys.getsizeof(node)
            total += sys.getsizeof(node.keys) + sys.getsizeof(node.values)
            total += sys.getsizeof(node.children)
            total += sum(sys.getsizeof(key) for key in node.keys)
            total += sum(sys.getsizeof(value) for value in node.values)
        return total

    def delete(self, key: K) -> None:
        """Delete *key* and rebalance nodes that fall below minimum occupancy."""
        removed = self._delete(self.root, key)
        if not self.root.leaf and not self.root.keys:
            self.root = self.root.children[0]
        if removed:
            self._size -= 1

    def _delete(self, node: _BNode[K, V], key: K) -> bool:
        self._visit()
        index = lower_bound(node.keys, key, self.metrics)
        found = index < len(node.keys) and node.keys[index] == key
        if index < len(node.keys):
            self.metrics.comparisons += 1

        if found:
            if node.leaf:
                node.keys.pop(index)
                node.values.pop(index)
                self.metrics.pages_written += 1
                return True
            return self._delete_internal_key(node, index)

        if node.leaf:
            return False

        child_index = index
        minimum_keys = self.order // 2 - 1
        if len(node.children[child_index].keys) == minimum_keys:
            child_index = self._ensure_child_capacity(node, child_index)
        return self._delete(node.children[child_index], key)

    def _delete_internal_key(self, node: _BNode[K, V], index: int) -> bool:
        minimum_degree = self.order // 2
        key = node.keys[index]
        left = node.children[index]
        right = node.children[index + 1]
        if len(left.keys) >= minimum_degree:
            predecessor_key, predecessor_value = self._maximum_item(left)
            node.keys[index] = predecessor_key
            node.values[index] = predecessor_value
            self.metrics.pages_written += 1
            return self._delete(left, predecessor_key)
        if len(right.keys) >= minimum_degree:
            successor_key, successor_value = self._minimum_item(right)
            node.keys[index] = successor_key
            node.values[index] = successor_value
            self.metrics.pages_written += 1
            return self._delete(right, successor_key)
        merged = self._merge_children(node, index)
        return self._delete(merged, key)

    def _maximum_item(self, node: _BNode[K, V]) -> tuple[K, V]:
        current = node
        while True:
            self._visit()
            if current.leaf:
                return current.keys[-1], current.values[-1]
            current = current.children[-1]

    def _minimum_item(self, node: _BNode[K, V]) -> tuple[K, V]:
        current = node
        while True:
            self._visit()
            if current.leaf:
                return current.keys[0], current.values[0]
            current = current.children[0]

    def _ensure_child_capacity(self, parent: _BNode[K, V], index: int) -> int:
        minimum_degree = self.order // 2
        child = parent.children[index]
        if index > 0 and len(parent.children[index - 1].keys) >= minimum_degree:
            left = parent.children[index - 1]
            child.keys.insert(0, parent.keys[index - 1])
            child.values.insert(0, parent.values[index - 1])
            parent.keys[index - 1] = left.keys.pop()
            parent.values[index - 1] = left.values.pop()
            if not left.leaf:
                child.children.insert(0, left.children.pop())
            self.metrics.pages_written += 3
            return index
        if (
            index < len(parent.children) - 1
            and len(parent.children[index + 1].keys) >= minimum_degree
        ):
            right = parent.children[index + 1]
            child.keys.append(parent.keys[index])
            child.values.append(parent.values[index])
            parent.keys[index] = right.keys.pop(0)
            parent.values[index] = right.values.pop(0)
            if not right.leaf:
                child.children.append(right.children.pop(0))
            self.metrics.pages_written += 3
            return index
        if index < len(parent.children) - 1:
            self._merge_children(parent, index)
            return index
        self._merge_children(parent, index - 1)
        return index - 1

    def _merge_children(self, parent: _BNode[K, V], index: int) -> _BNode[K, V]:
        left = parent.children[index]
        right = parent.children[index + 1]
        left.keys.append(parent.keys.pop(index))
        left.values.append(parent.values.pop(index))
        left.keys.extend(right.keys)
        left.values.extend(right.values)
        if not left.leaf:
            left.children.extend(right.children)
        parent.children.pop(index + 1)
        self.metrics.pages_written += 2
        return left
