# chronam/duckdb_io.py
"""
chronam/duckdb_io.py — clean DuckDB backend for the fixed Parquet schema

Per-year files:  AmericanStories_<YYYY>.parquet
Schema:
    full_article_id : INT64
    article_id      : STRING
    lccn            : STRING
    date            : STRING (YYYY-MM-DD)
    page            : STRING (e.g., "p1")
    article         : STRING
    url             : STRING
    filename        : STRING

Output JSON (per year) matches the app's existing downstream expectations:
{
  "year": "1885",
  "start_date": "1885-01-01",
  "end_date": "1885-12-31",
  "search_term": "railroad",
  "match_count": 123,
  "articles": [
    {
      "article_id": "...",
      "lccn": "sn82014381",
      "newspaper_name": "The Washington herald.",  # filled via CSV when available
      "date": "1885-01-01",
      "page": "p1",
      "headline": null,
      "byline": null,
      "article": "...text...",
      "url": "https://...",
      "filename": "1885-01-01_p1_sn82014381_....json"
    }
  ]
}
"""


from typing import Optional, List, Dict
import os, json, re, threading
import pandas as pd
import duckdb
from .config import init_project

DEFAULT_PARQUET_PREFIX = "AmericanStories"
SEARCH_LOCATIONS_REL = ["data/parquet", "parquet"]
_like_escape_re = re.compile(r"([_%])")

def _like_pattern(term: str) -> str:
    return "%" + _like_escape_re.sub(r"\\\1", term) + "%"

def download_data(
    project_dir: str,
    search_term: str,
    start_date_str: str,
    end_date_str: str,
    *,
    parquet_dir: Optional[str] = None,
    parquet_prefix: str = DEFAULT_PARQUET_PREFIX,
    max_saved_articles_per_year: Optional[int] = None,  # kept for API compat; still acts per-file scan
    progress_callback=None,
    cancel_event: Optional[threading.Event] = None,
    cleaning_options: Optional[Dict[str, bool]] = None,
) -> List[str]:
    """
    Query local Parquet with DuckDB and write *one* JSON payload for the full date range
    into data/raw/<term>_<start>_<end>.json.

    Returns: [path_to_single_json]
    """
    # Validate dates
    from datetime import datetime
    try:
        _ = datetime.strptime(start_date_str, "%Y-%m-%d")
        _ = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date_str and end_date_str must be 'YYYY-MM-DD'.")

    # Resolve paths
    paths = init_project(project_dir)
    raw_dir = paths["raw"]
    os.makedirs(raw_dir, exist_ok=True)

    # Optional enrichment: lccn -> Title / newspaper_name map if available
    def _load_lccn_title_map(csv_path: str) -> Dict[str, str]:
        if not (csv_path and os.path.exists(csv_path)):
            return {}
        try:
            df = pd.read_csv(csv_path)
            # Most common headings in the repo CSV(s)
            lccn_col = next((c for c in df.columns if c.strip().lower() in ("sn", "lccn")), None)
            title_col = next((c for c in df.columns if c.strip().lower() in ("title", "newspaper", "name")), None)
            if not (lccn_col and title_col):
                return {}
            return dict(zip(df[lccn_col].astype(str), df[title_col].astype(str)))
        except Exception:
            return {}

    lccn_to_title = _load_lccn_title_map(paths.get("csv"))

    # Where the Parquet corpus lives
    def _resolve_parquet_dir(project_dir: str, parquet_dir: Optional[str]) -> str:
        if parquet_dir and os.path.isdir(parquet_dir):
            return parquet_dir
        for rel in SEARCH_LOCATIONS_REL:
            cand = os.path.join(project_dir, rel)
            if os.path.isdir(cand):
                return cand
        raise FileNotFoundError("Parquet directory not found (checked: data/parquet, parquet).")

    parquet_root = _resolve_parquet_dir(project_dir, parquet_dir)

    def _file_for_year(parquet_dir: str, year: int, prefix: str) -> str:
        return os.path.join(parquet_dir, f"{prefix}_{year}.parquet")

    def _years_between(start: str, end: str):
        ys = int(start[:4]); ye = int(end[:4])
        if ys > ye:
            ys, ye = ye, ys
        return range(ys, ye + 1)

    like_pat = _like_pattern(search_term)

    # DuckDB
    con = duckdb.connect()
    try:
        threads = max(os.cpu_count() or 4, 4)
        con.execute(f"PRAGMA threads={threads}")
    except Exception:
        pass

    # Accumulate across all per-year Parquet files
    all_records: List[Dict] = []

    for y in _years_between(start_date_str, end_date_str):
        if cancel_event and cancel_event.is_set():
            return []
        fpath = _file_for_year(parquet_root, y, parquet_prefix)
        if not os.path.exists(fpath):
            continue

        sql = """
            SELECT
                CAST(article_id AS VARCHAR) AS article_id,
                CAST(lccn AS VARCHAR)       AS lccn,
                CAST(date AS VARCHAR)       AS date,
                CAST(page AS VARCHAR)       AS page,
                CAST(article AS VARCHAR)    AS article,
                CAST(url AS VARCHAR)        AS url,
                CAST(filename AS VARCHAR)   AS filename
            FROM read_parquet(?)
            WHERE date >= ? AND date <= ?
              AND lower(COALESCE(article, '')) LIKE lower(?)
            ORDER BY date
        """
        params = [fpath, start_date_str, end_date_str, like_pat]
        if max_saved_articles_per_year:
            sql += " LIMIT ?"
            params.append(int(max_saved_articles_per_year))

        df = con.execute(sql, params).fetchdf()
        if cancel_event and cancel_event.is_set():
            return []
        if df.empty:
            if progress_callback:
                progress_callback(0)
            continue

        # Enrich with newspaper_name where possible
        if lccn_to_title:
            df["newspaper_name"] = df["lccn"].map(lccn_to_title)

        all_records.extend(df.to_dict("records"))
        if cancel_event and cancel_event.is_set():
            return []
        if progress_callback:
            progress_callback(int(len(df)))

    if cancel_event and cancel_event.is_set():
        return []

    if not all_records:
        return []

    cleaning_options = cleaning_options or {}
    lowercase_articles = bool(cleaning_options.get('lowercase_articles'))
    urls_to_pdf = bool(cleaning_options.get('urls_to_pdf'))
    collapse_hyphenated = bool(cleaning_options.get('collapse_hyphenated_breaks'))

    if lowercase_articles or urls_to_pdf or collapse_hyphenated:
        hyphen_pattern = re.compile(r'-\s+') if collapse_hyphenated else None
        jp2_pattern = re.compile(r'\.jp2(?=$|[?#])', re.IGNORECASE) if urls_to_pdf else None
        for record in all_records:
            article_text = record.get('article')
            if isinstance(article_text, str):
                if hyphen_pattern:
                    article_text = hyphen_pattern.sub('', article_text)
                if lowercase_articles:
                    article_text = article_text.lower()
                record['article'] = article_text
            if jp2_pattern:
                url_val = record.get('url')
                if isinstance(url_val, str):
                    record['url'] = jp2_pattern.sub('.pdf', url_val)

    # Write a single payload (empty-safe)
    out_file = os.path.join(raw_dir, f"{search_term}_{start_date_str}_{end_date_str}.json")
    payload = {
        "start_date": start_date_str,
        "end_date": end_date_str,
        "search_term": search_term,
        "match_count": int(len(all_records)),
        "articles": all_records,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return [out_file]
