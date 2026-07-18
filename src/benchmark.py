"""Benchmark runner and result consolidation for both tree indexes."""

from __future__ import annotations

import gc
import pickle
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import psutil

from .bplustree import BPlusTree
from .btree import BTree
from .workload import WorkloadOperation

TreeFactory = Callable[[int], Any]
TREE_FACTORIES: dict[str, TreeFactory] = {"B-tree": BTree, "B+ tree": BPlusTree}
RAW_COLUMNS = [
    "structure", "index_type", "sample_size", "order", "insertion_order",
    "operation", "operation_detail", "distribution", "interval_size",
    "cache_state", "repetition", "operation_index", "time_ms", "cpu_time_ms",
    "time_per_operation_ms",
    "locate_time_ms", "traversal_time_ms",
    "operations_per_second", "memory_peak_bytes", "estimated_index_bytes",
    "serialized_index_bytes", "index_file_bytes", "nodes_visited", "pages_read", "pages_written",
    "comparisons", "splits", "height", "nodes", "internal_nodes", "leaves",
    "occupancy", "count_results",
    "incremental_size_before_bytes", "incremental_size_after_bytes",
    "incremental_file_before_bytes", "incremental_file_after_bytes",
]

NS_PER_MS = 1_000_000


def environment_table() -> pd.DataFrame:
    """Capture hardware and software metadata for the experiment report."""
    uname = platform.uname()
    try:
        storage = "SSD/HDD não detectável de forma portátil; preencher manualmente"
    except OSError:
        storage = "não detectado"
    rows = {
        "execution_datetime": datetime.now().astimezone().isoformat(),
        "operating_system": f"{uname.system} {uname.release}",
        "python": sys.version.replace("\n", " "),
        "architecture": platform.machine(),
        "processor": uname.processor or platform.processor() or "não detectado",
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "storage_type": storage,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
    }
    return pd.DataFrame({"property": rows.keys(), "value": rows.values()})


def metric_delta(tree: Any, before: dict[str, int]) -> dict[str, int]:
    after = tree.get_metrics()
    return {key: int(after[key] - before.get(key, 0)) for key in after}


def _base_row(
    tree: Any,
    structure: str,
    index_type: str,
    sample_size: int,
    order: int,
    insertion_order: str,
    operation: str,
    repetition: int,
    structure_stats: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    stats = structure_stats if structure_stats is not None else tree.structure_stats()
    return {
        "structure": structure,
        "index_type": index_type,
        "sample_size": sample_size,
        "order": order,
        "insertion_order": insertion_order,
        "operation": operation,
        "distribution": "not_applicable",
        "cache_state": "warmed_or_normal",
        "interval_size": np.nan,
        "repetition": repetition,
        "height": stats["height"],
        "nodes": stats["nodes"],
        "internal_nodes": stats["internal_nodes"],
        "leaves": stats["leaves"],
        "occupancy": stats["occupancy"],
    }


def build_tree(
    factory: TreeFactory,
    order: int,
    items: Iterable[tuple[Any, Any]],
) -> tuple[Any, int, int]:
    """Build an index and return tree, elapsed nanoseconds, and peak traced bytes."""
    gc.collect()
    tracemalloc.start()
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    tree = factory(order)
    for key, value in items:
        tree.insert(key, value)
    elapsed = time.perf_counter_ns() - start
    cpu_elapsed = time.process_time_ns() - cpu_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tree._last_build_cpu_ns = cpu_elapsed
    return tree, elapsed, peak


def benchmark_build(
    structure: str,
    order: int,
    items: list[tuple[Any, Any]],
    index_type: str,
    sample_size: int,
    insertion_order: str,
    repetition: int,
) -> tuple[Any, dict[str, Any]]:
    tree, elapsed, peak = build_tree(TREE_FACTORIES[structure], order, items)
    row = _base_row(tree, structure, index_type, sample_size, order, insertion_order, "build", repetition)
    metrics = tree.get_metrics()
    file_size = pickle_file_size(tree)
    row.update(
        time_ms=elapsed / NS_PER_MS,
        cpu_time_ms=tree._last_build_cpu_ns / NS_PER_MS,
        time_per_operation_ms=elapsed / max(len(items), 1) / NS_PER_MS,
        operations_per_second=len(items) / (elapsed / 1e9) if elapsed else np.nan,
        memory_peak_bytes=peak,
        estimated_index_bytes=tree.estimated_size_bytes(),
        serialized_index_bytes=file_size,
        index_file_bytes=file_size,
        count_results=len(items),
        **metrics,
    )
    return tree, row


def benchmark_points(
    tree: Any,
    keys: list[Any],
    exists: bool,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    operation = "point_hit" if exists else "point_miss"
    if keys:
        tree.search(keys[0])
    rows = []
    structure_stats = tree.structure_stats()
    for operation_index, key in enumerate(keys):
        before = tree.get_metrics()
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        value = tree.search(key)
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        delta = metric_delta(tree, before)
        row = _base_row(
            tree, operation=operation, structure_stats=structure_stats, **context
        )
        row.update(
            operation_index=operation_index,
            time_ms=elapsed / NS_PER_MS,
            cpu_time_ms=cpu_elapsed / NS_PER_MS,
            time_per_operation_ms=elapsed / NS_PER_MS,
            count_results=int(value is not None),
            memory_peak_bytes=np.nan,
            estimated_index_bytes=np.nan,
            **delta,
        )
        rows.append(row)
    return rows


def benchmark_ranges(
    tree: Any,
    intervals: list[tuple[Any, Any, str]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    structure_stats = tree.structure_stats()
    if intervals:
        tree.range_search(intervals[0][0], intervals[0][1])
    for operation_index, (start_key, end_key, label) in enumerate(intervals):
        before = tree.get_metrics()
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        if hasattr(tree, "range_search_profile"):
            result, locate_time, traversal_time = tree.range_search_profile(start_key, end_key)
        else:
            result = tree.range_search(start_key, end_key)
            locate_time, traversal_time = np.nan, np.nan
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        delta = metric_delta(tree, before)
        row = _base_row(
            tree, operation="range", structure_stats=structure_stats, **context
        )
        row.update(
            operation_index=operation_index,
            interval_size=label,
            time_ms=elapsed / NS_PER_MS,
            cpu_time_ms=cpu_elapsed / NS_PER_MS,
            time_per_operation_ms=elapsed / NS_PER_MS,
            locate_time_ms=locate_time / NS_PER_MS,
            traversal_time_ms=traversal_time / NS_PER_MS,
            count_results=len(result),
            memory_peak_bytes=np.nan,
            estimated_index_bytes=np.nan,
            **delta,
        )
        rows.append(row)
    return rows


def benchmark_incremental(
    tree: Any,
    items: list[tuple[Any, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    structure_stats = tree.structure_stats()
    size_before = tree.estimated_size_bytes()
    file_before = pickle_file_size(tree)
    for operation_index, (key, value) in enumerate(items):
        before = tree.get_metrics()
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        tree.insert(key, value)
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        delta = metric_delta(tree, before)
        row = _base_row(
            tree, operation="incremental_insert", structure_stats=structure_stats,
            **context,
        )
        row.update(
            operation_index=operation_index,
            time_ms=elapsed / NS_PER_MS,
            cpu_time_ms=cpu_elapsed / NS_PER_MS,
            time_per_operation_ms=elapsed / NS_PER_MS,
            count_results=1,
            memory_peak_bytes=np.nan,
            estimated_index_bytes=np.nan,
            **delta,
        )
        rows.append(row)
    size_after = tree.estimated_size_bytes()
    file_after = pickle_file_size(tree)
    for row in rows:
        row["incremental_size_before_bytes"] = size_before
        row["incremental_size_after_bytes"] = size_after
        row["incremental_file_before_bytes"] = file_before
        row["incremental_file_after_bytes"] = file_after
    return rows


def benchmark_catalog_insertions(
    tree: Any,
    items: list[tuple[Any, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Measure individual insertions of new catalog records in milliseconds."""
    rows: list[dict[str, Any]] = []
    structure_stats = tree.structure_stats()
    for operation_index, (key, value) in enumerate(items):
        before = tree.get_metrics()
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        tree.insert(key, value)
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        row = _base_row(
            tree, operation="insert_catalog", structure_stats=structure_stats,
            **context,
        )
        row.update(
            operation_index=operation_index,
            time_ms=elapsed / NS_PER_MS,
            cpu_time_ms=cpu_elapsed / NS_PER_MS,
            time_per_operation_ms=elapsed / NS_PER_MS,
            count_results=1,
            memory_peak_bytes=np.nan,
            estimated_index_bytes=np.nan,
            **metric_delta(tree, before),
        )
        rows.append(row)
    return rows


def benchmark_catalog_deletions(
    tree: Any,
    keys: list[Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Measure individual removals of existing catalog records in milliseconds."""
    rows: list[dict[str, Any]] = []
    structure_stats = tree.structure_stats()
    for operation_index, key in enumerate(keys):
        before = tree.get_metrics()
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        tree.delete(key)
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        row = _base_row(
            tree, operation="delete_catalog", structure_stats=structure_stats,
            **context,
        )
        row.update(
            operation_index=operation_index,
            time_ms=elapsed / NS_PER_MS,
            cpu_time_ms=cpu_elapsed / NS_PER_MS,
            time_per_operation_ms=elapsed / NS_PER_MS,
            count_results=1,
            memory_peak_bytes=np.nan,
            estimated_index_bytes=np.nan,
            **metric_delta(tree, before),
        )
        rows.append(row)
    return rows


def pickle_file_size(tree: Any) -> int:
    """Serialize an index to a real temporary file and return its byte size."""
    with NamedTemporaryFile(prefix="tree_index_", suffix=".pkl") as stream:
        pickle.dump(tree, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        return stream.tell()


def benchmark_mixed(
    primary_tree: Any,
    range_tree: Any,
    operations: list[WorkloadOperation],
    distribution: str,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute a realistic workload against primary and secondary indexes."""
    rows: list[dict[str, Any]] = []
    structure_stats = primary_tree.structure_stats()
    for operation_index, operation in enumerate(operations):
        primary_before = primary_tree.get_metrics()
        range_before = range_tree.get_metrics()
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        if operation.kind in {"point", "miss"}:
            result = primary_tree.search(operation.key)
            count = int(result is not None)
        elif operation.kind == "range":
            result = range_tree.range_search(operation.key, operation.end_key)
            count = len(result)
        else:
            primary_tree.insert(operation.key, operation.value)
            range_tree.insert((2100, operation.key), operation.value)
            count = 1
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.process_time_ns() - cpu_start
        primary_delta = metric_delta(primary_tree, primary_before)
        range_delta = metric_delta(range_tree, range_before)
        combined = {key: primary_delta[key] + range_delta[key] for key in primary_delta}
        row = _base_row(
            primary_tree, operation="mixed_workload",
            structure_stats=structure_stats, **context,
        )
        row.update(
            operation_index=operation_index,
            operation_detail=operation.kind,
            distribution=distribution,
            time_ms=elapsed / NS_PER_MS,
            cpu_time_ms=cpu_elapsed / NS_PER_MS,
            time_per_operation_ms=elapsed / NS_PER_MS,
            count_results=count,
            memory_peak_bytes=np.nan,
            estimated_index_bytes=np.nan,
            **combined,
        )
        rows.append(row)
    return rows


def make_range_intervals(sorted_keys: list[Any], targets: tuple[int, ...] = (10, 100, 1_000)) -> list[tuple[Any, Any, str]]:
    """Create key-space intervals with known approximate result counts."""
    if not sorted_keys:
        return []
    intervals: list[tuple[Any, Any, str]] = []
    center = len(sorted_keys) // 2
    requested = list(targets) + [max(1, len(sorted_keys) // 10), len(sorted_keys)]
    for count in requested:
        actual = min(count, len(sorted_keys))
        left = max(0, min(center - actual // 2, len(sorted_keys) - actual))
        right = left + actual - 1
        intervals.append((sorted_keys[left], sorted_keys[right], str(count)))
    return intervals


def append_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Append partial results so a long run survives later failures."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=RAW_COLUMNS)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Consolidate latency, variability, structural, and speedup metrics."""
    if raw.empty:
        return raw.copy()
    raw = raw.copy()
    for column in (
        "structure", "index_type", "insertion_order", "operation",
        "distribution", "interval_size",
    ):
        if column in raw:
            raw[column] = raw[column].astype("string").fillna("not_applicable")
    dimensions = [
        "structure", "index_type", "sample_size", "order", "insertion_order",
        "operation", "distribution", "interval_size",
    ]
    dimensions = [column for column in dimensions if column in raw.columns]
    grouped = raw.groupby(dimensions, dropna=False)
    summary = grouped.agg(
        mean_time_ms=("time_ms", "mean"),
        mean_cpu_time_ms=("cpu_time_ms", "mean"),
        median_time_ms=("time_ms", "median"),
        std_time_ms=("time_ms", "std"),
        min_time_ms=("time_ms", "min"),
        max_time_ms=("time_ms", "max"),
        p95_time_ms=("time_ms", lambda values: np.percentile(values, 95)),
        p99_time_ms=("time_ms", lambda values: np.percentile(values, 99)),
        mean_memory_bytes=("memory_peak_bytes", "mean"),
        std_memory_bytes=("memory_peak_bytes", "std"),
        mean_nodes_visited=("nodes_visited", "mean"),
        mean_pages_read=("pages_read", "mean"),
        std_pages_read=("pages_read", "std"),
        mean_pages_written=("pages_written", "mean"),
        mean_splits=("splits", "mean"),
        std_splits=("splits", "std"),
        mean_height=("height", "mean"),
        std_height=("height", "std"),
        mean_results=("count_results", "mean"),
    ).reset_index()
    comparison_keys = [column for column in dimensions if column != "structure"]
    pivot = summary.pivot_table(
        index=comparison_keys, columns="structure", values="median_time_ms"
    ).reset_index()
    if {"B-tree", "B+ tree"}.issubset(pivot.columns):
        pivot["speedup_b_over_bplus"] = pivot["B-tree"] / pivot["B+ tree"]
        pivot["difference_percent_bplus_vs_b"] = (pivot["B+ tree"] - pivot["B-tree"]) / pivot["B-tree"] * 100
        summary = summary.merge(
            pivot[comparison_keys + ["speedup_b_over_bplus", "difference_percent_bplus_vs_b"]],
            on=comparison_keys,
            how="left",
        )
    return summary


def numeric_interpretation(summary: pd.DataFrame, operation: str) -> str:
    """Generate claims only from measurements that actually exist."""
    subset = summary.loc[summary["operation"] == operation].dropna(
        subset=["median_time_ms"]
    )
    if subset.empty:
        return f"Não há medições disponíveis para {operation}."
    by_structure = subset.groupby("structure")["median_time_ms"].median().sort_values()
    winner = str(by_structure.index[0])
    if len(by_structure) < 2:
        return f"A única estrutura medida em {operation} foi {winner}."
    fastest, second = float(by_structure.iloc[0]), float(by_structure.iloc[1])
    difference = (second - fastest) / second * 100 if second else 0.0
    variability = subset.groupby("structure")["std_time_ms"].median().fillna(0)
    return (
        f"Em {operation}, {winner} apresentou a menor mediana agregada "
        f"({fastest:.6f} ms), uma diferença de {difference:.1f}% frente à outra estrutura. "
        f"A mediana dos desvios-padrão por estrutura variou de "
        f"{variability.min():.6f} a {variability.max():.6f} ms. "
        "A diferença deve ser interpretada junto das dispersões e configurações individuais."
    )
