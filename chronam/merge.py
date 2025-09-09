
"""
merge.py
Merge raw search JSON with newspaper metadata CSV (XY coordinates + City/State)
into a GeoJSON FeatureCollection.

Requirements satisfied:
- Adds City and State from CSV into GeoJSON properties.
- No circular imports. Self-contained _load_xy and helpers.
"""

import os
import json
from typing import List, Dict, Any, Optional

import pandas as pd

try:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import Point  # type: ignore
except Exception:  # pragma: no cover
    gpd = None  # type: ignore
    Point = None  # type: ignore

from .config import init_project  # type: ignore


def _load_xy(csv_path: str) -> pd.DataFrame:
    """Load the ChronAm newspapers XY CSV and standardize column names."""
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    required = {"SN", "Long", "Lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Optional City/State/Title
    for col in ["City", "State", "Title"]:
        if col not in df.columns:
            df[col] = None
    return df


def _normalize_articles(payload: Dict[str, Any]) -> pd.DataFrame:
    """Normalize the Search Dataset JSON payload into a DataFrame of articles."""
    # Two possible shapes seen in practice:
    #   {start_date, end_date, search_term, match_count, articles: [..dicts..]}
    # or a dict with 'records' key; we support 'articles' primary, fallback to 'records'
    articles = payload.get("articles")
    if articles is None:
        articles = payload.get("records", [])
    if not isinstance(articles, list):
        raise ValueError("JSON does not contain a list of 'articles' or 'records'.")
    # Build DataFrame
    df = pd.DataFrame(articles)
    # Ensure required columns exist
    for col in ["lccn", "date", "article", "page", "url", "newspaper_name"]:
        if col not in df.columns:
            df[col] = None
    # Coerce date to datetime (naive)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def _to_geodataframe(merged: pd.DataFrame) -> "gpd.GeoDataFrame":  # type: ignore
    """Convert merged DataFrame with Long/Lat columns to a GeoDataFrame."""
    if gpd is None or Point is None:
        raise RuntimeError("geopandas/shapely is required to write GeoJSON.")
    geometry = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lon, lat in zip(merged["Long"], merged["Lat"])
    ]
    gdf = gpd.GeoDataFrame(merged, geometry=geometry, crs="EPSG:4326")
    return gdf


def _infer_names(jpath: str, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Infer search_term, start_date, end_date for output naming.
    """
    term = payload.get("search_term")
    start = payload.get("start_date")
    end = payload.get("end_date")

    # Try to infer from filename if missing
    if not term or not start or not end:
        base = os.path.splitext(os.path.basename(jpath))[0]
        # Expect pattern like: {term}_{start}_{end}.json
        parts = base.split("_")
        if len(parts) >= 3:
            # last two tokens are likely dates (YYYY-MM-DD)
            maybe_end = parts[-1]
            maybe_start = parts[-2]
            if not start:
                start = maybe_start
            if not end:
                end = maybe_end
            if not term:
                term = "_".join(parts[:-2])
        elif not term:
            term = base
    return {"term": term or "term", "start": start or "all", "end": end or "all"}


def merge_geojson(
    project_dir: str,
    csv_path: Optional[str] = None,
    json_path: Optional[str] = None,
    search_term: Optional[str] = None,
    year: Optional[str] = None,
) -> List[str]:
    """
    Merge raw JSON search results with newspaper coordinates (XY) and output GeoJSON.
    If json_path is provided, merges exactly that JSON file.
    Otherwise merges all JSON files matching the search_term (and year, if specified).

    Returns a list of output GeoJSON file paths created.
    """
    paths = init_project(project_dir)
    raw_dir = paths["raw"]
    proc_dir = paths["processed"]
    csv_path = csv_path or paths["csv"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    df_xy = _load_xy(csv_path)

    # Determine input JSON files
    json_files: List[str] = []
    if json_path:
        if not os.path.exists(json_path):
            raise FileNotFoundError(json_path)
        json_files = [json_path]
    else:
        # Gather all JSONs in raw_dir; filter by search_term/year if provided
        for name in os.listdir(raw_dir):
            if not name.lower().endswith(".json"):
                continue
            if search_term and not name.startswith(search_term + "_"):
                continue
            if year and not name.endswith(f"_{year}.json"):
                continue
            json_files.append(os.path.join(raw_dir, name))
        json_files.sort()

    if not json_files:
        raise FileNotFoundError("No JSON files found to merge.")

    outputs: List[str] = []
    for jfile in json_files:
        with open(jfile, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Normalize articles
        df = _normalize_articles(payload)
        if df.empty:
            continue

        # Merge with XY metadata
        # Prefer join on lccn <-> SN; fallback to Title if lccn missing.
        if "lccn" in df.columns:
            merged = df.merge(df_xy, left_on="lccn", right_on="SN", how="left")
        else:
            merged = df.merge(df_xy, left_on="newspaper_name", right_on="Title", how="left")

        # Ensure City/State are present in properties
        for col in ["City", "State", "Title", "SN", "Long", "Lat"]:
            if col not in merged.columns:
                merged[col] = None

        # Convert to GeoDataFrame
        gdf = _to_geodataframe(merged)

        # Create output path
        name_parts = _infer_names(jfile, payload)
        out_dir = os.path.join(proc_dir, name_parts["term"])
        os.makedirs(out_dir, exist_ok=True)
        out_name = f'merged_{name_parts["term"]}_{name_parts["start"]}_{name_parts["end"]}.geojson'
        out_path = os.path.join(out_dir, out_name)


        # Ensure JSON-serializable types
        if "date" in gdf.columns:
            gdf["date"] = gdf["date"].astype(str)
        for col in ["page"]:
            if col in gdf.columns:
                gdf[col] = gdf[col].astype(str)
        # Select properties to keep (article-centric + XY + location)
        keep_cols = [
            "article_id", "lccn", "date", "page", "article", "url", "filename", "newspaper_name",
            "SN", "Title", "City", "State", "Long", "Lat"
        ]
        existing_cols = [c for c in keep_cols if c in gdf.columns]
        gdf = gdf[existing_cols + ["geometry"]]

        # Save GeoJSON
        gdf.to_file(out_path, driver="GeoJSON")
        outputs.append(out_path)

    return outputs
