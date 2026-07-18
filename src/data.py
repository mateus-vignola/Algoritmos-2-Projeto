"""Dataset download, discovery, schema inference, and cleaning."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

DATASET_HANDLE = "pavan4kalyan/imdb-dataset-of-600k-international-movies"

ALIASES: dict[str, tuple[str, ...]] = {
    "movie_id": ("id", "imdb_id", "imdbid", "title_id", "tconst", "movie_id"),
    "title": ("title", "name", "movie_title", "primary_title", "original_title"),
    "release_year": ("year", "release_year", "start_year", "date_published", "release_date"),
    "rating": ("rating", "avg_rating", "average_rating", "imdb_rating", "score"),
    "votes": ("votes", "num_votes", "vote_count", "imdb_votes"),
    "genre": ("genre", "genres"),
    "runtime_minutes": ("runtime", "duration", "runtime_minutes", "minutes"),
    "country": ("country", "countries", "country_of_origin"),
    "language": ("language", "languages", "original_language"),
}


def download_dataset(destination: Path, handle: str = DATASET_HANDLE) -> Path:
    """Download through kagglehub and record its cache path without duplicating files."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub before downloading the dataset") from exc
    destination.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download(handle))
    (destination / "kagglehub_location.txt").write_text(str(cache_path), encoding="utf-8")
    return cache_path


def inventory_files(folder: Path) -> pd.DataFrame:
    """List dataset files with extensions and sizes."""
    rows = [
        {"path": str(path), "extension": path.suffix.lower(), "size_bytes": path.stat().st_size}
        for path in folder.rglob("*")
        if path.is_file()
    ]
    return pd.DataFrame(rows).sort_values("size_bytes", ascending=False).reset_index(drop=True)


def choose_main_file(folder: Path) -> Path:
    """Choose the largest plausible delimited data file without assuming its name."""
    candidates = [
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No CSV, TSV, or TXT file found below {folder}")
    return max(candidates, key=lambda path: path.stat().st_size)


def discover_source_files(folder: Path) -> list[Path]:
    """Find one delimited main file or a naturally ordered set of JSON batches."""
    delimited = [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    ]
    if delimited:
        return [max(delimited, key=lambda path: path.stat().st_size)]
    json_files = [path for path in folder.rglob("*.json") if path.is_file()]
    if not json_files:
        raise FileNotFoundError(f"No supported delimited or JSON data files below {folder}")
    by_name: dict[str, Path] = {}
    for path in json_files:
        current = by_name.get(path.name)
        if current is None or len(path.parts) > len(current.parts):
            by_name[path.name] = path

    def batch_number(path: Path) -> tuple[int, str]:
        match = re.search(r"batch[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
        return (int(match.group(1)) if match else 10**12, path.name)

    return sorted(by_name.values(), key=batch_number)


def detect_text_format(path: Path) -> tuple[str, str]:
    """Test common encodings and use csv.Sniffer to infer a delimiter."""
    raw = path.read_bytes()[:131_072]
    decoded = None
    selected_encoding = "utf-8"
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            decoded = raw.decode(encoding)
            selected_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise UnicodeError(f"Could not decode a sample from {path}")
    try:
        delimiter = csv.Sniffer().sniff(decoded, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return selected_encoding, delimiter


def _nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _nested_texts(record: dict[str, Any], container: tuple[str, ...]) -> str | None:
    values = _nested(record, *container)
    if not isinstance(values, list):
        return None
    texts = [item.get("text") for item in values if isinstance(item, dict) and item.get("text")]
    return ", ".join(texts) if texts else None


def _extract_json_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten only experiment fields, avoiding multi-gigabyte normalization."""
    rows = []
    for record in records:
        runtime_seconds = _nested(record, "runtime", "seconds")
        rows.append({
            "movie_id": record.get("id"),
            "title": _nested(record, "titleText", "text") or _nested(record, "originalTitleText", "text"),
            "release_year": _nested(record, "releaseYear", "year"),
            "rating": _nested(record, "ratingsSummary", "aggregateRating"),
            "votes": _nested(record, "ratingsSummary", "voteCount"),
            "genre": _nested_texts(record, ("genres", "genres")),
            "runtime_minutes": runtime_seconds / 60 if isinstance(runtime_seconds, (int, float)) else None,
            "country": _nested_texts(record, ("countriesOfOrigin", "countries")),
            "language": _nested_texts(record, ("spokenLanguages", "spokenLanguages")),
        })
    return pd.DataFrame(rows)


def read_initial_sample(sources: Sequence[Path], rows: int = 5) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a tiny sample from either a delimited file or the first JSON batch."""
    first = sources[0]
    if first.suffix.lower() == ".json":
        with first.open(encoding="utf-8") as stream:
            records = json.load(stream)
        if not isinstance(records, list):
            records = [records]
        return _extract_json_records(records[:rows]), {
            "format": "json_batches", "encoding": "utf-8", "files": str(len(sources))
        }
    encoding, delimiter = detect_text_format(first)
    return pd.read_csv(first, encoding=encoding, sep=delimiter, nrows=rows), {
        "format": "delimited", "encoding": encoding, "delimiter": repr(delimiter),
        "files": "1",
    }


def read_dataset(source: Path | Sequence[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load one delimited file or stream and flatten a collection of JSON batches."""
    sources = [source] if isinstance(source, Path) else list(source)
    if not sources:
        raise ValueError("At least one source file is required")
    first = sources[0]
    if first.suffix.lower() != ".json":
        encoding, delimiter = detect_text_format(first)
        frame = pd.read_csv(first, encoding=encoding, sep=delimiter, low_memory=False)
        return frame, {
            "format": "delimited", "encoding": encoding,
            "delimiter": repr(delimiter), "path": str(first),
        }
    frames: list[pd.DataFrame] = []
    for path in sources:
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)
        if not isinstance(records, list):
            records = [records]
        frames.append(_extract_json_records(records))
    return pd.concat(frames, ignore_index=True), {
        "format": "json_batches", "encoding": "utf-8",
        "files": str(len(sources)), "path": str(first.parent),
    }


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def infer_column_mapping(columns: list[str]) -> dict[str, str]:
    """Map heterogeneous source names onto the experimental schema."""
    normalized = {normalize_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[target] = normalized[alias]
                break
        if target not in mapping:
            for normalized_name, original in normalized.items():
                if any(alias in normalized_name for alias in aliases if len(alias) > 3):
                    mapping[target] = original
                    break
    return mapping


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_movies(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build the canonical movie table and a transparent quality report."""
    mapping = infer_column_mapping([str(column) for column in raw.columns])
    if "movie_id" not in mapping:
        raise ValueError("No plausible IMDb identifier column was found")
    output = pd.DataFrame(index=raw.index)
    for target in ALIASES:
        output[target] = raw[mapping[target]] if target in mapping else pd.NA
    report: list[dict[str, Any]] = []
    initial = len(output)
    ids = output["movie_id"].astype("string").str.strip()
    valid = ids.notna() & ids.ne("") & ids.ne("<NA>")
    output = output.loc[valid].copy()
    output["movie_id"] = ids.loc[valid]
    report.append({"step": "remove_missing_key", "before": initial, "after": len(output), "removed": initial - len(output)})
    before = len(output)
    output = output.drop_duplicates("movie_id", keep="first")
    report.append({"step": "remove_duplicate_key", "before": before, "after": len(output), "removed": before - len(output)})

    extracted = pd.to_numeric(output["movie_id"].str.extract(r"(\d+)", expand=False), errors="coerce")
    if extracted.notna().all() and extracted.is_unique:
        output["index_key"] = extracted.astype("int64")
    else:
        ordered_ids = sorted(output["movie_id"].tolist())
        deterministic = {movie_id: index + 1 for index, movie_id in enumerate(ordered_ids)}
        output["index_key"] = output["movie_id"].map(deterministic).astype("int64")

    for column in ("release_year", "rating", "votes", "runtime_minutes"):
        output[column] = _numeric(output[column])
    output["release_year"] = output["release_year"].where(output["release_year"].between(1870, 2100)).astype("Int64")
    output["votes"] = output["votes"].round().astype("Int64")
    output["runtime_minutes"] = output["runtime_minutes"].round().astype("Int64")
    for column in ("title", "genre", "country", "language"):
        output[column] = output[column].astype("string").replace({"": pd.NA})
    output = output[["index_key", *ALIASES.keys()]].reset_index(drop=True)
    report.append({"step": "final", "before": len(output), "after": len(output), "removed": 0})
    quality = pd.DataFrame(report)
    quality["missing_values_final"] = [np.nan] * (len(quality) - 1) + [int(output.isna().sum().sum())]
    return output, quality, mapping


def reproducible_samples(
    movies: pd.DataFrame, sizes: tuple[int, ...], seed: int, include_full: bool = False
) -> dict[int, pd.DataFrame]:
    """Create nested reproducible samples in one fixed random order."""
    shuffled = movies.sample(frac=1, random_state=seed).reset_index(drop=True)
    valid_sizes = sorted({size for size in sizes if size <= len(shuffled)})
    if include_full and len(shuffled) not in valid_sizes:
        valid_sizes.append(len(shuffled))
    return {size: shuffled.iloc[:size].copy() for size in valid_sizes}
