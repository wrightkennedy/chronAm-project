"""Utilities for packaged ChronAM dataset metrics."""

from __future__ import annotations

import csv
import os
from datetime import date
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Optional

from .config import default_yearly_summary_path

_METRIC_COLUMN_MAP: Mapping[str, str] = {
    "keyword_frequency": "keyword_frequency",
    "article_count": "total_articles",
    "page_count": "total_pages",
    "issue_count": "total_issues",
    "newspaper_count": "total_newspapers",
    "word_count": "total_words",
}


@lru_cache(maxsize=1)
def _load_yearly_summary() -> Dict[str, Dict[str, int]]:
    """Load the packaged yearly summary CSV into a dictionary keyed by year."""
    path = default_yearly_summary_path()
    if not path or not os.path.exists(path):
        return {}

    summary: Dict[str, Dict[str, int]] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                year_raw = (row.get("year") or "").strip()
                if not year_raw:
                    continue
                metrics: Dict[str, int] = {}
                for metric_key, column in _METRIC_COLUMN_MAP.items():
                    try:
                        value = int(float(row.get(column, "") or 0))
                    except (TypeError, ValueError):
                        continue
                    metrics[metric_key] = value
                if metrics:
                    summary[year_raw] = metrics
    except Exception:
        return {}
    return summary


def available_metrics() -> Iterable[str]:
    """Return the metric identifiers exposed by the yearly summary."""
    return _METRIC_COLUMN_MAP.keys()


def metrics_for_year(year: int) -> Dict[str, int]:
    """Return the metrics dictionary for a given year (empty dict if missing)."""
    summary = _load_yearly_summary()
    return dict(summary.get(str(year), {}))


def metric_total_for_year(year: int, metric: str) -> Optional[int]:
    """Return the yearly total for ``metric`` if available."""
    summary = _load_yearly_summary()
    metrics = summary.get(str(year))
    if not metrics:
        return None
    return metrics.get(metric)


def metric_total_for_years(years: Iterable[int], metric: str) -> int:
    """Sum ``metric`` across the provided years."""
    total = 0
    for year in years:
        value = metric_total_for_year(year, metric)
        if isinstance(value, int):
            total += value
    return total


def metric_total_for_range(start_year: int, end_year: int, metric: str) -> int:
    """Convenience wrapper to sum ``metric`` between two year bounds (inclusive)."""
    if end_year < start_year:
        start_year, end_year = end_year, start_year
    return metric_total_for_years(range(start_year, end_year + 1), metric)


def metric_total_for_dates(start_date: date, end_date: date, metric: str) -> int:
    """
    Approximate the metric total for a date span by scaling the yearly totals.

    The packaged summary is yearly, so this computes a proportional total based on the
    fraction of each year covered by the date range. Returns a rounded integer.
    """
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    summary = _load_yearly_summary()
    if not summary:
        return 0

    start_year = start_date.year
    end_year = end_date.year
    total_value = 0.0

    for year in range(start_year, end_year + 1):
        metrics = summary.get(str(year))
        if not metrics:
            continue
        year_total = metrics.get(metric)
        if not year_total:
            continue

        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        overlap_start = max(year_start, start_date)
        overlap_end = min(year_end, end_date)
        if overlap_start > overlap_end:
            continue

        overlap_days = (overlap_end - overlap_start).days + 1
        year_days = (year_end - year_start).days + 1
        fraction = overlap_days / year_days if year_days else 0.0
        total_value += year_total * fraction

    return int(round(total_value))


def metric_total_for_year_within_dates(year: int, start_date: date, end_date: date, metric: str) -> int:
    """Return the yearly total clipped to the overlap with the provided dates."""
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    overlap_start = max(year_start, start_date)
    overlap_end = min(year_end, end_date)
    if overlap_start > overlap_end:
        return 0
    return metric_total_for_dates(overlap_start, overlap_end, metric)
