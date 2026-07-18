"""SQLite baseline measurements; these are a reference, not an identical tree."""

from __future__ import annotations

import gc
import sqlite3
import time
from typing import Any

import numpy as np

from .database import configure_reference_indexes


def benchmark_sqlite(
    connection: sqlite3.Connection,
    point_movie_ids: list[str],
    year_intervals: list[tuple[int, int]],
    repetitions: int,
    sample_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = {
        "none": "sqlite_no_additional_index",
        "movie_id": "sqlite_movie_id_index",
        "movie_id_and_year": "sqlite_movie_id_and_year_index",
    }
    for mode, index_label in modes.items():
        configure_reference_indexes(connection, mode)
        for repetition in range(repetitions):
            gc.collect()
            if point_movie_ids:
                connection.execute(
                    "SELECT title FROM movies WHERE movie_id = ?", (point_movie_ids[0],)
                ).fetchone()
            for operation_index, movie_id in enumerate(point_movie_ids):
                start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                result = connection.execute(
                    "SELECT title FROM movies WHERE movie_id = ?", (str(movie_id),)
                ).fetchone()
                elapsed = time.perf_counter_ns() - start
                cpu_elapsed = time.process_time_ns() - cpu_start
                rows.append({
                    "structure": "SQLite", "index_type": index_label,
                    "sample_size": sample_size, "order": np.nan,
                    "insertion_order": "database", "operation": "point_hit",
                    "distribution": "uniform", "interval_size": np.nan,
                    "cache_state": "warmed_or_normal",
                    "repetition": repetition, "operation_index": operation_index,
                    "time_ms": elapsed / 1_000_000,
                    "cpu_time_ms": cpu_elapsed / 1_000_000,
                    "time_per_operation_ms": elapsed / 1_000_000,
                    "count_results": int(result is not None),
                    "memory_peak_bytes": np.nan, "nodes_visited": np.nan,
                    "pages_read": np.nan, "pages_written": np.nan,
                    "splits": np.nan, "height": np.nan,
                })
            for operation_index, (year_start, year_end) in enumerate(year_intervals):
                start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                result = connection.execute(
                    "SELECT index_key FROM movies WHERE release_year BETWEEN ? AND ? ORDER BY release_year, index_key",
                    (int(year_start), int(year_end)),
                ).fetchall()
                elapsed = time.perf_counter_ns() - start
                cpu_elapsed = time.process_time_ns() - cpu_start
                rows.append({
                    "structure": "SQLite", "index_type": index_label,
                    "sample_size": sample_size, "order": np.nan,
                    "insertion_order": "database", "operation": "range",
                    "distribution": "uniform", "interval_size": f"{year_end - year_start + 1}_years",
                    "cache_state": "warmed_or_normal",
                    "repetition": repetition, "operation_index": operation_index,
                    "time_ms": elapsed / 1_000_000,
                    "cpu_time_ms": cpu_elapsed / 1_000_000,
                    "time_per_operation_ms": elapsed / 1_000_000,
                    "count_results": len(result), "memory_peak_bytes": np.nan,
                    "nodes_visited": np.nan, "pages_read": np.nan,
                    "pages_written": np.nan, "splits": np.nan, "height": np.nan,
                })
    return rows
