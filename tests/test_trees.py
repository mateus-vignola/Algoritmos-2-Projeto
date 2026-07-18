from __future__ import annotations

import random

import pytest

from src.bplustree import BPlusTree
from src.btree import BTree
from src.validation import validate_tree_class


@pytest.mark.parametrize("tree_class", [BTree, BPlusTree])
@pytest.mark.parametrize("order", [4, 8, 32])
def test_required_validation_cases(tree_class, order):
    validate_tree_class(tree_class, order)


@pytest.mark.parametrize("tree_class", [BTree, BPlusTree])
def test_randomized_operations_match_dict(tree_class):
    rng = random.Random(42)
    keys = list(range(1_000))
    rng.shuffle(keys)
    tree = tree_class(order=8)
    expected = {}
    for key in keys:
        expected[key] = key * 3
        tree.insert(key, key * 3)
    for key in range(1_000):
        assert tree.search(key) == expected[key]
    assert tree.range_search(123, 456) == [
        (key, expected[key]) for key in range(123, 457)
    ]
    assert tree.structure_stats()["occupancy"] <= 1.0


@pytest.mark.parametrize("tree_class", [BTree, BPlusTree])
def test_order_must_be_even(tree_class):
    with pytest.raises(ValueError):
        tree_class(3)
    with pytest.raises(ValueError):
        tree_class(5)


@pytest.mark.parametrize("tree_class", [BTree, BPlusTree])
@pytest.mark.parametrize("order", [4, 8, 32])
def test_deletion_rebalances_and_preserves_search_and_ranges(tree_class, order):
    rng = random.Random(123)
    insertion_order = list(range(500))
    rng.shuffle(insertion_order)
    tree = tree_class(order)
    expected = {key: key * 7 for key in insertion_order}
    for key, value in expected.items():
        tree.insert(key, value)

    deletion_order = insertion_order.copy()
    rng.shuffle(deletion_order)
    for index, key in enumerate(deletion_order):
        tree.delete(key)
        expected.pop(key)
        assert len(tree) == len(expected)
        if index % 25 == 0:
            assert tree.range_search(-1, 1_000) == sorted(expected.items())
            for probe in rng.sample(range(500), 20):
                assert tree.search(probe) == expected.get(probe)

    assert tree.height() == 1
    assert tree.range_search(-1, 1_000) == []
    tree.delete(999)
