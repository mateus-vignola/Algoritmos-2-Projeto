from __future__ import annotations

import pandas as pd

from src.data import (
    clean_movies, discover_source_files, infer_column_mapping, read_dataset,
    read_initial_sample, reproducible_samples,
)
from src.database import connect_database, create_catalog, query_by_id, query_year_range
from src.sqlite_benchmark import benchmark_sqlite


def sample_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "imdbId": ["tt0003", "tt0001", "tt0002", "tt0002", None],
            "Name": ["C", "A", "B", "duplicate", "missing"],
            "startYear": [2003, 2001, "2002", 2002, 2000],
            "averageRating": [7.1, "8.0", 6.5, 2.0, 1.0],
            "numVotes": [10, 20, "30", 0, 1],
        }
    )


def test_flexible_mapping_and_cleaning():
    mapping = infer_column_mapping(list(sample_raw().columns))
    assert mapping["movie_id"] == "imdbId"
    movies, quality, _ = clean_movies(sample_raw())
    assert len(movies) == 3
    assert movies["index_key"].tolist() == [3, 1, 2]
    assert quality["removed"].sum() == 2
    samples = reproducible_samples(movies, (2, 10), 42, include_full=True)
    assert set(samples) == {2, 3}


def test_sqlite_round_trip(tmp_path):
    movies, _, _ = clean_movies(sample_raw())
    connection = connect_database(tmp_path / "catalog.db")
    try:
        create_catalog(connection, movies)
        assert query_by_id(connection, 1)[1] == "tt0001"
        assert query_year_range(connection, 2001, 2002) == [(2001, 1), (2002, 2)]
        rows = benchmark_sqlite(connection, ["tt0001"], [(2001, 2002)], 1, len(movies))
        assert len(rows) == 6
        assert {row["index_type"] for row in rows} == {
            "sqlite_no_additional_index",
            "sqlite_movie_id_index",
            "sqlite_movie_id_and_year_index",
        }
    finally:
        connection.close()


def test_json_batch_discovery_and_flattening(tmp_path):
    batches = [
        [{
            "id": f"tt{number:04d}", "titleText": {"text": f"Movie {number}"},
            "releaseYear": {"year": 2000 + number},
            "runtime": {"seconds": 6_000},
            "ratingsSummary": {"aggregateRating": 7.5, "voteCount": 10},
            "genres": {"genres": [{"text": "Drama"}]},
            "countriesOfOrigin": {"countries": [{"text": "Brazil"}]},
            "spokenLanguages": {"spokenLanguages": [{"text": "Portuguese"}]},
        }]
        for number in (2, 1)
    ]
    import json
    for number, records in zip((2, 1), batches):
        (tmp_path / f"movies_batch_{number}.json").write_text(json.dumps(records), encoding="utf-8")
    sources = discover_source_files(tmp_path)
    assert [path.name for path in sources] == ["movies_batch_1.json", "movies_batch_2.json"]
    preview, metadata = read_initial_sample(sources)
    frame, _ = read_dataset(sources)
    assert metadata["format"] == "json_batches"
    assert preview.iloc[0].movie_id == "tt0001"
    assert frame.runtime_minutes.tolist() == [100.0, 100.0]
    assert frame.genre.tolist() == ["Drama", "Drama"]
