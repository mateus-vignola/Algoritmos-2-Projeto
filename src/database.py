"""SQLite catalog creation and reference queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import pandas as pd

MOVIE_COLUMNS = (
    "index_key", "movie_id", "title", "release_year", "rating", "votes",
    "genre", "runtime_minutes", "country", "language",
)


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_catalog(connection: sqlite3.Connection, movies: pd.DataFrame) -> None:
    """Replace and populate the catalog in one explicit transaction."""
    connection.execute("DROP TABLE IF EXISTS movies")
    connection.execute(
        """CREATE TABLE movies (
        index_key INTEGER PRIMARY KEY,
        movie_id TEXT NOT NULL,
        title TEXT,
        release_year INTEGER,
        rating REAL,
        votes INTEGER,
        genre TEXT,
        runtime_minutes INTEGER,
        country TEXT,
        language TEXT
        )"""
    )
    rows = movies.loc[:, MOVIE_COLUMNS].astype(object).where(pd.notna(movies.loc[:, MOVIE_COLUMNS]), None)
    placeholders = ",".join("?" for _ in MOVIE_COLUMNS)
    with connection:
        connection.executemany(
            f"INSERT INTO movies ({','.join(MOVIE_COLUMNS)}) VALUES ({placeholders})",
            rows.itertuples(index=False, name=None),
        )


def configure_secondary_indexes(connection: sqlite3.Connection, enabled: bool) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_movies_year_key")
    if enabled:
        connection.execute("CREATE INDEX idx_movies_year_key ON movies(release_year, index_key)")
    connection.commit()


def configure_reference_indexes(connection: sqlite3.Connection, mode: str) -> None:
    """Set SQLite to no extra index, movie-ID index, or both requested indexes."""
    if mode not in {"none", "movie_id", "movie_id_and_year"}:
        raise ValueError("unknown SQLite index mode")
    connection.execute("DROP INDEX IF EXISTS idx_movies_movie_id")
    connection.execute("DROP INDEX IF EXISTS idx_movies_year_key")
    if mode in {"movie_id", "movie_id_and_year"}:
        connection.execute("CREATE INDEX idx_movies_movie_id ON movies(movie_id)")
    if mode == "movie_id_and_year":
        connection.execute(
            "CREATE INDEX idx_movies_year_key ON movies(release_year, index_key)"
        )
    connection.commit()


def query_by_id(connection: sqlite3.Connection, key: int) -> tuple | None:
    return connection.execute("SELECT * FROM movies WHERE index_key = ?", (int(key),)).fetchone()


def query_year_range(connection: sqlite3.Connection, start: int, end: int) -> list[tuple[int, int]]:
    cursor = connection.execute(
        """SELECT release_year, index_key FROM movies
        WHERE release_year BETWEEN ? AND ? ORDER BY release_year, index_key""",
        (int(start), int(end)),
    )
    return [(int(year), int(key)) for year, key in cursor if year is not None]


def database_pages(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "page_size": int(connection.execute("PRAGMA page_size").fetchone()[0]),
        "page_count": int(connection.execute("PRAGMA page_count").fetchone()[0]),
    }
