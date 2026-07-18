from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmark import benchmark_build, summarize_results
from src.config import ExperimentConfig
from src.experiment import run_catalog_operation_experiments
from src.plots import create_figures, create_results_overview_figures


def test_build_records_real_file_size():
    _, row = benchmark_build(
        "B-tree", 8, [(key, key) for key in range(50)],
        "primary_id", 50, "sorted", 0,
    )
    assert row["index_file_bytes"] > 0
    assert row["serialized_index_bytes"] == row["index_file_bytes"]
    assert row["cpu_time_ms"] > 0
    assert "time_ns" not in row


def test_summary_accepts_sqlite_text_interval_and_tree_missing_interval():
    raw = pd.DataFrame([
        {"structure": "B-tree", "index_type": "primary_id", "sample_size": 10,
         "order": 8, "insertion_order": "random", "operation": "point_hit",
         "distribution": "uniform", "interval_size": np.nan, "time_ms": 0.100,
         "cpu_time_ms": 0.090, "memory_peak_bytes": np.nan, "nodes_visited": 1,
         "pages_read": 1, "pages_written": 0, "splits": 0, "height": 1,
         "count_results": 1},
        {"structure": "B+ tree", "index_type": "primary_id", "sample_size": 10,
         "order": 8, "insertion_order": "random", "operation": "point_hit",
         "distribution": "uniform", "interval_size": np.nan, "time_ms": 0.080,
         "cpu_time_ms": 0.070, "memory_peak_bytes": np.nan, "nodes_visited": 1,
         "pages_read": 1, "pages_written": 0, "splits": 0, "height": 1,
         "count_results": 1},
        {"structure": "SQLite", "index_type": "sqlite_movie_id_index", "sample_size": 10,
         "order": np.nan, "insertion_order": "database", "operation": "range",
         "distribution": "uniform", "interval_size": "10_years", "time_ms": 0.120,
         "cpu_time_ms": 0.100, "memory_peak_bytes": np.nan, "nodes_visited": np.nan,
         "pages_read": np.nan, "pages_written": np.nan, "splits": np.nan,
         "height": np.nan, "count_results": 3},
    ])
    summary = summarize_results(raw)
    assert len(summary) == 3
    speedup = summary.loc[summary.structure == "B-tree", "speedup_b_over_bplus"].iloc[0]
    assert speedup == 1.25


def test_catalog_operations_generate_five_clean_figures(tmp_path):
    movies = pd.DataFrame({
        "index_key": range(1, 21),
        "movie_id": [f"tt{key:04d}" for key in range(1, 21)],
        "title": ["Repeated" if key % 4 == 0 else f"Movie {key}" for key in range(1, 21)],
        "genre": ["Drama, Comedy" if key % 2 else "Action" for key in range(1, 21)],
    })
    config = ExperimentConfig(run_mode="quick", project_root=tmp_path)
    config.orders = (8,)
    config.quick_repetitions = 1
    config.quick_queries = 4
    config.create_directories()
    raw = run_catalog_operation_experiments(
        {len(movies): movies}, config, config.raw_results_dir / "focused.csv"
    )
    summary = summarize_results(raw)
    figures = create_figures(raw, summary)
    assert set(raw.operation) == {
        "search_by_id", "search_by_category", "search_by_title",
        "insert_catalog", "delete_catalog",
    }
    assert set(figures) == {
        "01_search_by_id", "02_search_by_category", "03_search_by_title",
        "04_insert_catalog", "05_delete_catalog",
    }
    for figure in figures.values():
        assert len(figure.data) == 2
        assert all(trace.type == "heatmap" for trace in figure.data)
        assert all(list(trace.x) == ["8"] for trace in figure.data)
        assert all(list(trace.y) == ["20"] for trace in figure.data)
        assert all(len(trace.z) == 1 and len(trace.z[0]) == 1 for trace in figure.data)
        assert figure.layout.coloraxis.colorbar.title.text == "Tempo médio<br>(ms)"
    assert raw.loc[raw.operation == "search_by_id", "count_results"].eq(1).all()
    assert raw.groupby(["operation", "structure"]).size().eq(4).all()
    assert "time_ms" in raw and "time_ns" not in raw


def test_new_run_removes_old_outputs_but_preserves_raw_dataset(tmp_path):
    config = ExperimentConfig(run_mode="quick", project_root=tmp_path)
    config.create_directories(clear_previous_outputs=False)
    raw_dataset = config.raw_data_dir / "kagglehub_location.txt"
    raw_dataset.write_text("cache", encoding="utf-8")
    stale_files = [
        config.processed_data_dir / "movies_clean.csv",
        config.raw_results_dir / "old.csv",
        config.processed_results_dir / "old.csv",
        tmp_path / "figures/html/old.html",
        tmp_path / "figures/static/old.png",
        tmp_path / "streaming_catalog.db",
    ]
    for stale in stale_files:
        stale.write_text("old", encoding="utf-8")
    config.create_directories(clear_previous_outputs=True)
    assert raw_dataset.exists()
    assert not any(stale.exists() for stale in stale_files)


def test_old_results_cannot_generate_legacy_figures():
    old = pd.DataFrame({
        "operation": ["build"], "structure": ["B-tree"], "sample_size": [1000],
        "order": [32], "insertion_order": ["random"], "time_ms": [0.001],
    })
    with pytest.raises(ValueError, match="dados antigos não serão reutilizados"):
        create_figures(old, old)


def test_requested_orders_sizes_form_three_by_three_heatmaps():
    rows = []
    for operation in (
        "search_by_id", "search_by_category", "search_by_title",
        "insert_catalog", "delete_catalog",
    ):
        for structure in ("B-tree", "B+ tree"):
            for order in (32, 128, 256):
                for size in (10_000, 100_000, 700_000):
                    rows.append({
                        "operation": operation,
                        "structure": structure,
                        "order": order,
                        "sample_size": size,
                        "insertion_order": "random",
                        "time_ms": float(order + size) / 1_000_000,
                    })
    raw = pd.DataFrame(rows)
    figures = create_figures(raw, raw)
    assert len(figures) == 5
    for figure in figures.values():
        assert len(figure.data) == 2
        for heatmap in figure.data:
            assert list(heatmap.x) == ["32", "128", "256"]
            assert list(heatmap.y) == ["10 mil", "100 mil", "700 mil"]
            assert np.asarray(heatmap.z).shape == (3, 3)


def test_results_notebook_figures_are_compact_aggregates():
    rows = []
    for operation_index, operation in enumerate(
        (
            "search_by_id", "search_by_category", "search_by_title",
            "insert_catalog", "delete_catalog",
        ), start=1
    ):
        for structure_index, structure in enumerate(("B-tree", "B+ tree"), start=1):
            for order in (32, 128, 256):
                for size in (10_000, 100_000, 700_000):
                    for repetition in range(3):
                        rows.append({
                            "operation": operation,
                            "structure": structure,
                            "order": order,
                            "sample_size": size,
                            "insertion_order": "random",
                            "time_ms": float(
                                operation_index * structure_index * size / order
                                + repetition
                            ) / 1_000_000,
                        })
    figures = create_results_overview_figures(pd.DataFrame(rows))
    assert set(figures) == {
        "06_best_largest_size", "07_structure_comparison", "08_scalability"
    }
    assert len(figures["06_best_largest_size"].data) == 10
    assert all(len(trace.y) == 1 for trace in figures["06_best_largest_size"].data)
    assert len(figures["07_structure_comparison"].data) == 1
    assert np.asarray(figures["07_structure_comparison"].data[0].z).shape == (5, 3)
    assert len(figures["08_scalability"].data) == 10
    assert all(len(trace.y) == 3 for trace in figures["08_scalability"].data)


def test_default_experimental_grid_matches_current_request():
    config = ExperimentConfig()
    assert config.orders == (32, 128, 256)
    assert config.sample_sizes == (10_000, 100_000, 700_000)
    assert config.query_count == 150
    assert config.repetitions == 2
