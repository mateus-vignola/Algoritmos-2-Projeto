"""Correctness checks that must pass before performance measurements."""

from __future__ import annotations

from typing import Any

from .bplustree import BPlusTree
from .btree import BTree


def validate_tree_class(tree_class: type[Any], order: int = 8) -> None:
    empty = tree_class(order)
    assert empty.search(1) is None
    assert empty.range_search(1, 10) == []
    assert empty.height() == 1

    single = tree_class(order)
    single.insert(7, "seven")
    assert single.search(7) == "seven"
    single.insert(7, "SEVEN")
    assert len(single) == 1 and single.search(7) == "SEVEN"

    tree = tree_class(order)
    reference = {key: f"v{key}" for key in range(300)}
    insertion_order = list(range(0, 300, 2)) + list(range(299, 0, -2))
    for key in insertion_order:
        tree.insert(key, reference[key])
    assert len(tree) == len(reference)
    for key, value in reference.items():
        assert tree.search(key) == value
    assert tree.search(500) is None
    expected = [(key, reference[key]) for key in range(73, 188)]
    assert tree.range_search(73, 187) == expected
    assert tree.range_search(10, 9) == []
    assert tree.height() > 1 and tree.count_nodes() > 1

    for key in range(0, 300, 3):
        tree.delete(key)
        reference.pop(key)
    assert len(tree) == len(reference)
    assert tree.range_search(0, 299) == sorted(reference.items())
    for key in range(300):
        assert tree.search(key) == reference.get(key)
    tree.delete(999)
    assert len(tree) == len(reference)


def run_all_validations() -> None:
    for tree_class in (BTree, BPlusTree):
        for order in (4, 8, 32):
            validate_tree_class(tree_class, order)
