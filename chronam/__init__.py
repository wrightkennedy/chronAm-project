"""
ChronAM package

Provides tools for local parquet data or [optionally] streaming data from Hugging Face, merging geojson,
filling missing newspaper metadata, running collocation analysis, and
visualizing results.

Local-first (DuckDB) backend by default.
Legacy HuggingFace downloader is exposed as download_data_hf for optional use.
"""

__version__ = "0.2.0"

from .config import init_project
from .merge import merge_geojson
from .fetch_metadata import fetch_missing_metadata
from .collocate import run_collocation, build_collocation_output_paths
from .visualize import plot_bar, plot_rank_changes

# Local (DuckDB) is the default
from .duckdb_io import download_data  # noqa: F401

# Legacy remote downloader (lazy import so 'datasets' isn't required unless used)
def download_data_hf(*args, **kwargs):
    from .download import download_data as _hf_download
    return _hf_download(*args, **kwargs)

__all__ = [
    "init_project",
    "download_data",       # local DuckDB
    "download_data_hf",    # legacy HF
    "merge_geojson",
    "fetch_missing_metadata",
    "run_collocation",
    "build_collocation_output_paths",
    "plot_bar",
    "plot_rank_changes",
]
