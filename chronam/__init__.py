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
from .duckdb_io import download_data  # noqa: F401


def merge_geojson(*args, **kwargs):
    from .merge import merge_geojson as _merge_geojson
    return _merge_geojson(*args, **kwargs)


def fetch_missing_metadata(*args, **kwargs):
    from .fetch_metadata import fetch_missing_metadata as _fetch
    return _fetch(*args, **kwargs)


def run_collocation(*args, **kwargs):
    from .collocate import run_collocation as _run
    return _run(*args, **kwargs)


def build_collocation_output_paths(*args, **kwargs):
    from .collocate import build_collocation_output_paths as _build_paths
    return _build_paths(*args, **kwargs)


def run_topic_model(*args, **kwargs):
    from .topics import run_topic_model as _run_topics
    return _run_topics(*args, **kwargs)


def build_topic_model_output_paths(*args, **kwargs):
    from .topics import build_topic_model_output_paths as _build_topic_paths
    return _build_topic_paths(*args, **kwargs)


def plot_bar(*args, **kwargs):
    from .visualize import plot_bar as _plot_bar
    return _plot_bar(*args, **kwargs)


def plot_rank_changes(*args, **kwargs):
    from .visualize import plot_rank_changes as _plot_rank_changes
    return _plot_rank_changes(*args, **kwargs)


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
    "run_topic_model",
    "build_topic_model_output_paths",
    "plot_bar",
    "plot_rank_changes",
]
