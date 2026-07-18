"""End-to-end orchestration used by the notebook."""

from __future__ import annotations

import gc
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import (
    TREE_FACTORIES,
    append_csv,
    benchmark_build,
    benchmark_catalog_deletions,
    benchmark_catalog_insertions,
    benchmark_incremental,
    benchmark_mixed,
    benchmark_points,
    benchmark_ranges,
    make_range_intervals,
    summarize_results,
)
from .config import ExperimentConfig
from .workload import mixed_workload, point_keys


def _items(frame: pd.DataFrame, index_type: str, insertion_order: str) -> list[tuple[Any, int]]:
    if index_type == "primary_id":
        pairs = [(int(row.index_key), int(row.index_key)) for row in frame.itertuples()]
    elif index_type == "secondary_year_id":
        usable = frame.dropna(subset=["release_year"])
        pairs = [
            ((int(row.release_year), int(row.index_key)), int(row.index_key))
            for row in usable.itertuples()
        ]
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    return sorted(pairs) if insertion_order == "sorted" else pairs


def run_tree_experiments(
    samples: dict[int, pd.DataFrame], config: ExperimentConfig, output_path: Path
) -> pd.DataFrame:
    """Run construction, point, range, incremental, and mixed experiments."""
    if output_path.exists():
        output_path.unlink()
    rng = np.random.default_rng(config.random_seed)
    for sample_size, sample in samples.items():
        keys = sample["index_key"].astype("int64").to_numpy()
        hit_keys = point_keys(keys, config.query_count, config.random_seed)
        max_key = int(keys.max())
        miss_keys = list(range(max_key + config.query_count + 1, max_key + 2 * config.query_count + 1))
        incremental_items = [
            (max_key + index + 1, max_key + index + 1)
            for index in range(max(10, config.query_count // 10))
        ]
        secondary_random = _items(sample, "secondary_year_id", "random")
        secondary_sorted_keys = sorted(key for key, _ in secondary_random)
        intervals = make_range_intervals(secondary_sorted_keys)
        years = sorted({key[0] for key in secondary_sorted_keys})
        workload_ranges = secondary_sorted_keys or [(2000, 1)]

        for order in config.orders:
            for insertion_order in ("random", "sorted"):
                primary_items = _items(sample, "primary_id", insertion_order)
                secondary_items = _items(sample, "secondary_year_id", insertion_order)
                for repetition in range(config.repetitions):
                    structures = list(TREE_FACTORIES)
                    if (repetition + order) % 2:
                        structures.reverse()
                    for structure in structures:
                        context = {
                            "structure": structure,
                            "index_type": "primary_id",
                            "sample_size": sample_size,
                            "order": order,
                            "insertion_order": insertion_order,
                            "repetition": repetition,
                        }
                        primary, build_row = benchmark_build(
                            structure, order, primary_items, "primary_id", sample_size,
                            insertion_order, repetition,
                        )
                        append_csv([build_row], output_path)
                        append_csv(benchmark_points(primary, hit_keys, True, context), output_path)
                        append_csv(benchmark_points(primary, miss_keys, False, context), output_path)
                        append_csv(
                            benchmark_incremental(primary, incremental_items, context), output_path
                        )

                        secondary, secondary_build = benchmark_build(
                            structure, order, secondary_items, "secondary_year_id", sample_size,
                            insertion_order, repetition,
                        )
                        append_csv([secondary_build], output_path)
                        secondary_context = {**context, "index_type": "secondary_year_id"}
                        append_csv(benchmark_ranges(secondary, intervals, secondary_context), output_path)

                        for distribution in ("uniform", "zipf"):
                            operations = mixed_workload(
                                keys, workload_ranges, config.query_count,
                                config.random_seed + repetition,
                                distribution,
                            )
                            mixed_primary, _, _ = _rebuild(structure, order, primary_items)
                            mixed_secondary, _, _ = _rebuild(structure, order, secondary_items)
                            rows = benchmark_mixed(
                                mixed_primary, mixed_secondary, operations, distribution, context
                            )
                            append_csv(rows, output_path)
                        del primary, secondary
                        gc.collect()
    return pd.read_csv(output_path)


def _rebuild(
    structure: str, order: int, items: list[tuple[Any, Any]]
) -> tuple[Any, int, int]:
    from .benchmark import build_tree

    return build_tree(TREE_FACTORIES[structure], order, items)


def save_processed_results(raw: pd.DataFrame, destination: Path) -> pd.DataFrame:
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(raw)
    summary.to_csv(destination, index=False)
    return summary


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _category_items(frame: pd.DataFrame, insertion_order: str) -> list[tuple[tuple[str, int], int]]:
    pairs: list[tuple[tuple[str, int], int]] = []
    for row in frame.dropna(subset=["genre"]).itertuples():
        for category in str(row.genre).split(","):
            normalized = sys.intern(_normalize_text(category))
            if normalized:
                pairs.append(((normalized, int(row.index_key)), int(row.index_key)))
    return sorted(pairs) if insertion_order == "sorted" else pairs


def _title_items(frame: pd.DataFrame, insertion_order: str) -> list[tuple[tuple[str, int], int]]:
    pairs: list[tuple[tuple[str, int], int]] = []
    for row in frame.dropna(subset=["title"]).itertuples():
        normalized = _normalize_text(row.title)
        if normalized:
            key = int(row.index_key)
            pairs.append(((normalized, key), key))
    return sorted(pairs) if insertion_order == "sorted" else pairs


def run_catalog_operation_experiments(
    samples: dict[int, pd.DataFrame], config: ExperimentConfig, output_path: Path
) -> pd.DataFrame:
    """Benchmark the three searches plus catalog insertion and deletion."""
    if output_path.exists():
        output_path.unlink()
    for sample_size, sample in samples.items():
        rng = np.random.default_rng(config.random_seed + sample_size)
        keys = sample["index_key"].astype("int64").to_numpy()
        id_keys = point_keys(
            keys,
            config.query_count,
            config.random_seed + sample_size,
        )
        delete_count = min(config.query_count, len(keys))
        delete_keys = rng.choice(keys, delete_count, replace=False).astype(int).tolist()
        max_key = int(keys.max())
        insert_items = [
            (max_key + index + 1, max_key + index + 1)
            for index in range(config.query_count)
        ]
        available_categories = sorted({
            _normalize_text(category)
            for value in sample["genre"].dropna().astype(str)
            for category in value.split(",")
            if _normalize_text(category)
        })
        available_titles = []
        for value in sample["title"].dropna().astype(str):
            normalized_title = _normalize_text(value)
            if normalized_title:
                available_titles.append(normalized_title)
        category_queries = (
            rng.choice(available_categories, config.query_count, replace=True).tolist()
            if available_categories else []
        )
        title_queries = (
            rng.choice(available_titles, config.query_count, replace=True).tolist()
            if available_titles else []
        )
        category_intervals = [
            ((category, -1), (category, 10**19), category)
            for category in category_queries
        ]
        title_intervals = [
            ((title, -1), (title, 10**19), title)
            for title in title_queries
        ]

        insertion_order = "random"
        operation_specs = {
            "search_by_id": (
                "primary_id", lambda: _items(sample, "primary_id", insertion_order)
            ),
            "search_by_category": (
                "secondary_category_id", lambda: _category_items(sample, insertion_order)
            ),
            "search_by_title": (
                "secondary_title_id", lambda: _title_items(sample, insertion_order)
            ),
            "insert_catalog": (
                "primary_id", lambda: _items(sample, "primary_id", insertion_order)
            ),
            "delete_catalog": (
                "primary_id", lambda: _items(sample, "primary_id", insertion_order)
            ),
        }
        for operation, (index_type, build_items) in operation_specs.items():
            items = build_items()
            for order in config.orders:
                structures = list(TREE_FACTORIES)
                if (order // 32) % 2:
                    structures.reverse()
                for structure in structures:
                    search_tree = None
                    if operation.startswith("search_"):
                        search_tree, _, _ = _rebuild(structure, order, items)
                    for repetition in range(config.repetitions):
                        context = {
                            "structure": structure,
                            "index_type": index_type,
                            "sample_size": sample_size,
                            "order": order,
                            "insertion_order": insertion_order,
                            "repetition": repetition,
                        }
                        if operation == "search_by_id":
                            rows = benchmark_points(search_tree, id_keys, True, context)
                        elif operation in {"search_by_category", "search_by_title"}:
                            intervals = (
                                category_intervals
                                if operation == "search_by_category"
                                else title_intervals
                            )
                            rows = benchmark_ranges(search_tree, intervals, context)
                        elif operation == "insert_catalog":
                            mutation_tree, _, _ = _rebuild(structure, order, items)
                            rows = benchmark_catalog_insertions(
                                mutation_tree, insert_items, context
                            )
                            assert len(mutation_tree) == len(items) + len(insert_items)
                            del mutation_tree
                        else:
                            mutation_tree, _, _ = _rebuild(structure, order, items)
                            rows = benchmark_catalog_deletions(
                                mutation_tree, delete_keys, context
                            )
                            assert len(mutation_tree) == len(items) - len(delete_keys)
                            del mutation_tree
                        for row in rows:
                            row["operation"] = operation
                            row["distribution"] = "uniform"
                        append_csv(rows, output_path)
                    if search_tree is not None:
                        del search_tree
                    gc.collect()
            del items
            gc.collect()
    return pd.read_csv(output_path)
