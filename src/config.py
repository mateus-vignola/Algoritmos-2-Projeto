"""Central experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """All knobs and paths needed to reproduce an experiment."""

    run_mode: str = "quick"
    random_seed: int = 42
    orders: tuple[int, ...] = (32, 128, 256)
    quick_sizes: tuple[int, ...] = (10_000, 100_000, 700_000)
    full_sizes: tuple[int, ...] = (10_000, 100_000, 700_000)
    quick_repetitions: int = 2
    full_repetitions: int = 2
    quick_queries: int = 150
    full_queries: int = 150
    include_full_dataset: bool = False
    export_static_images: bool = False
    project_root: Path = field(default_factory=lambda: Path.cwd().resolve())

    def __post_init__(self) -> None:
        if self.run_mode not in {"quick", "full"}:
            raise ValueError("run_mode must be 'quick' or 'full'")

    @property
    def sample_sizes(self) -> tuple[int, ...]:
        return self.quick_sizes if self.run_mode == "quick" else self.full_sizes

    @property
    def repetitions(self) -> int:
        return self.quick_repetitions if self.run_mode == "quick" else self.full_repetitions

    @property
    def query_count(self) -> int:
        return self.quick_queries if self.run_mode == "quick" else self.full_queries

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def raw_results_dir(self) -> Path:
        return self.project_root / "results" / "raw"

    @property
    def processed_results_dir(self) -> Path:
        return self.project_root / "results" / "processed"

    def create_directories(self, clear_previous_outputs: bool = True) -> None:
        """Create the project layout and, by default, start with clean outputs."""
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.raw_results_dir,
            self.processed_results_dir,
            self.project_root / "figures" / "html",
            self.project_root / "figures" / "static",
        ):
            path.mkdir(parents=True, exist_ok=True)
        if clear_previous_outputs:
            self.clear_generated_outputs()

    def clear_generated_outputs(self) -> None:
        """Remove artifacts from older runs while preserving data/raw and .gitkeep."""
        targets = {
            self.processed_data_dir: ("*.csv", "*.json", "*.parquet"),
            self.raw_results_dir: ("*.csv", "*.json"),
            self.processed_results_dir: ("*.csv", "*.json"),
            self.project_root / "figures" / "html": ("*.html",),
            self.project_root / "figures" / "static": ("*.png", "*.svg"),
        }
        for folder, patterns in targets.items():
            for pattern in patterns:
                for artifact in folder.glob(pattern):
                    if artifact.is_file():
                        artifact.unlink()
        database = self.project_root / "streaming_catalog.db"
        if database.exists():
            database.unlink()
