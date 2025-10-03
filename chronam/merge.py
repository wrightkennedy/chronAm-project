
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


class MergeResult(list):
    """List of output paths annotated with merge statistics."""

    def __init__(self, paths: List[str], stats: Dict[str, Any]):
        super().__init__(paths)
        self.stats = stats


def _load_xy(csv_path: str) -> pd.DataFrame:
    """Load the ChronAm newspapers XY CSV and standardize column names."""
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    # Keep a single row per SN to avoid join blow-up
    if "SN" in df.columns:
        df = df.drop_duplicates(subset=["SN"], keep="first")
    for col in ["Date_start", "Date_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    start_cols = [c for c in ["Date_start"] if c in df.columns]
    end_cols = [c for c in ["Date_end"] if c in df.columns]
    if start_cols:
        start_frame = df[start_cols].copy()
        df["_start"] = start_frame.bfill(axis=1).iloc[:, 0]
    else:
        df["_start"] = pd.NaT
    if end_cols:
        end_frame = df[end_cols].copy()
        df["_end"] = end_frame.bfill(axis=1).iloc[:, 0]
    else:
        df["_end"] = pd.NaT
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
    for col in ["lccn", "date", "article", "page", "url"]:
        if col not in df.columns:
            df[col] = None
    # Coerce date to datetime (naive)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


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
    unmatched_csv_path: Optional[str] = None,
) -> List[str]:
    """
    Merge raw JSON search results with newspaper coordinates (XY) and output GeoJSON.
    If json_path is provided, merges exactly that JSON file.
    Otherwise merges all JSON files matching the search_term (and year, if specified).

    Returns a list of output GeoJSON file paths created.
    If unmatched_csv_path is provided, writes a CSV of unmatched articles there.
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
    unmatched_frames: List[pd.DataFrame] = []
    overall_stats: Dict[str, Any] = {
        "matched_lccn": 0,
        "matched_title": 0,
        "total_articles": 0,
        "unmatched": 0,
        "files": [],
    }
    for jfile in json_files:
        with open(jfile, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Normalize articles
        df = _normalize_articles(payload)
        if df.empty:
            continue

        metadata_cols = [
            col for col in ["SN", "Title", "City", "State", "Long", "Lat", "_start", "_end"]
            if col in df_xy.columns
        ]

        merged = df.copy()
        merged["_match_source"] = None
        merged["_title_key"] = None
        if "date" in merged.columns:
            merged["_article_ts"] = pd.to_datetime(merged["date"], errors="coerce")
        else:
            merged["_article_ts"] = pd.NaT

        search_start = _parse_date(payload.get("start_date"))
        search_end = _parse_date(payload.get("end_date"))

        if merged["_article_ts"].notna().any():
            if search_start is None:
                search_start = merged["_article_ts"].min()
            if search_end is None:
                search_end = merged["_article_ts"].max()

        if "lccn" in merged.columns and merged["lccn"].notna().any():
            merged = merged.merge(df_xy[metadata_cols], left_on="lccn", right_on="SN", how="left")
            merged.loc[merged["SN"].notna(), "_match_source"] = "lccn"
        else:
            for col in metadata_cols:
                if col not in merged.columns:
                    merged[col] = None

        title_source_col = next((c for c in ("newspaper_name", "Title", "title") if c in merged.columns), None)
        if title_source_col:
            merged["_title_key"] = (
                merged[title_source_col]
                .astype(str)
                .str.strip()
                .str.casefold()
            )
            valid_titles = merged["_title_key"].astype(bool) & (merged["_title_key"] != "nan")
            fallback_mask = merged["SN"].isna() & valid_titles
            if fallback_mask.any():
                df_xy_title = df_xy[df_xy["Title"].notna()].copy()
                df_xy_title["_title_key"] = (
                    df_xy_title["Title"]
                    .astype(str)
                    .str.strip()
                    .str.casefold()
                )
                df_xy_title = df_xy_title[df_xy_title["_title_key"].astype(bool)]

                if search_start is not None or search_end is not None:
                    mask = pd.Series(True, index=df_xy_title.index)
                    if search_start is not None:
                        mask &= df_xy_title["_end"].isna() | (df_xy_title["_end"] >= search_start)
                    if search_end is not None:
                        mask &= df_xy_title["_start"].isna() | (df_xy_title["_start"] <= search_end)
                    df_xy_title = df_xy_title[mask]

                if not df_xy_title.empty:
                    df_xy_title = df_xy_title.drop_duplicates(subset=["_title_key"], keep="first")
                    metadata_lookup = df_xy_title.set_index("_title_key")[metadata_cols]
                    title_keys = merged.loc[fallback_mask, "_title_key"]
                    mapped = metadata_lookup.reindex(title_keys)
                    mapped.index = title_keys.index

                    if not mapped.empty:
                        article_dates = merged.loc[fallback_mask, "_article_ts"]
                        valid = pd.Series(True, index=mapped.index)
                        if "_start" in mapped.columns:
                            valid &= mapped["_start"].isna() | (article_dates >= mapped["_start"])
                        if "_end" in mapped.columns:
                            valid &= mapped["_end"].isna() | (article_dates <= mapped["_end"])
                        if not valid.all():
                            mapped.loc[~valid, metadata_cols] = None

                        for col in metadata_cols:
                            if col in mapped.columns:
                                existing = merged.loc[fallback_mask, col]
                                merged.loc[fallback_mask, col] = existing.where(existing.notna(), mapped[col])

                        fallback_matched = fallback_mask & merged["SN"].notna()
                        if fallback_matched.any():
                            current_sources = merged.loc[fallback_matched, "_match_source"]
                            merged.loc[fallback_matched, "_match_source"] = current_sources.where(
                                current_sources.notna(), "title"
                            )

        if "article_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["article_id"], keep="first")

        if "_match_source" in merged.columns:
            merged.loc[merged["SN"].notna() & merged["_match_source"].isna(), "_match_source"] = "lccn"

        match_counts = {"lccn": 0, "title": 0}
        if "_match_source" in merged.columns:
            for key in match_counts.keys():
                mask = merged["_match_source"] == key
                if mask.any():
                    if "article_id" in merged.columns:
                        match_counts[key] = int(merged.loc[mask & merged["article_id"].notna(), "article_id"].nunique())
                    else:
                        match_counts[key] = int(mask.sum())

        if "Title" in merged.columns and "newspaper_name" in merged.columns:
            merged["Title"] = merged["Title"].fillna(merged["newspaper_name"])
        if "newspaper_name" in merged.columns:
            merged = merged.drop(columns=["newspaper_name"])

        if "_match_source" in merged.columns:
            merged = merged.rename(columns={"_match_source": "match_source"})
        if "match_source" in merged.columns:
            unmatched_mask = merged["SN"].isna()
            merged.loc[unmatched_mask & merged["match_source"].isna(), "match_source"] = "unmatched"
        helper_cols = ["_article_ts", "_start", "_end", "_title_key"]
        unmatched_rows = merged[merged["SN"].isna()].drop(columns=[c for c in helper_cols if c in merged.columns], errors="ignore")
        if not unmatched_rows.empty:
            unmatched_frames.append(unmatched_rows.copy())

        merged = merged.drop(columns=[c for c in helper_cols if c in merged.columns], errors="ignore")

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
            "article_id", "lccn", "date", "page", "article", "url", "filename",
            "Title", "City", "State"
        ]
        existing_cols = [c for c in keep_cols if c in gdf.columns]
        gdf = gdf[existing_cols + ["geometry"]]

        # Save GeoJSON
        gdf.to_file(out_path, driver="GeoJSON")
        outputs.append(out_path)

        overall_stats["matched_lccn"] += match_counts.get("lccn", 0)
        overall_stats["matched_title"] += match_counts.get("title", 0)
        total_articles = int(len(gdf))
        overall_stats["total_articles"] += total_articles
        unmatched_count = int(len(unmatched_rows))
        overall_stats["unmatched"] += unmatched_count
        overall_stats["files"].append({
            "path": out_path,
            "matched_lccn": match_counts.get("lccn", 0),
            "matched_title": match_counts.get("title", 0),
            "total_articles": total_articles,
            "unmatched": unmatched_count,
        })

    unmatched_df = pd.concat(unmatched_frames, ignore_index=True) if unmatched_frames else pd.DataFrame()
    overall_stats["unmatched"] = int(len(unmatched_df))
    overall_stats["unmatched_path"] = None
    if unmatched_csv_path and not unmatched_df.empty:
        os.makedirs(os.path.dirname(unmatched_csv_path), exist_ok=True)
        unmatched_df.to_csv(unmatched_csv_path, index=False)
        overall_stats["unmatched_path"] = unmatched_csv_path

    return MergeResult(outputs, overall_stats)
