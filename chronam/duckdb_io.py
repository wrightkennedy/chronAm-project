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


from typing import Optional, List, Dict, Any, Union
from collections import defaultdict
import os, json, re, threading
import pandas as pd
import duckdb
from .config import init_project
from .utils import term_directory_name, write_metadata_file

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
    summary_only: bool = False,
    cleaning_options: Optional[Dict[str, bool]] = None,
    metadata_enabled: bool = True,
) -> Union[List[str], Dict[str, Any]]:
    """
    Query local Parquet with DuckDB.

    When ``summary_only`` is False (default), write a single JSON payload for the full date
    range into ``data/processed/<term>/<term>_<start>_<end>.json`` and return the path.

    When ``summary_only`` is True, return aggregated per-year metrics instead of writing
    the JSON payload. The caller is responsible for persisting any downstream artifacts.
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
    processed_dir = paths["processed"]
    os.makedirs(processed_dir, exist_ok=True)
    term_dir = os.path.join(processed_dir, term_directory_name(search_term))
    os.makedirs(term_dir, exist_ok=True)

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

    year_sequence = list(_years_between(start_date_str, end_date_str))
    keyword_pattern = re.compile(re.escape(search_term), re.IGNORECASE) if search_term.strip() else None
    word_pattern = re.compile(r'\b\w+\b')

    # Accumulate across all per-year Parquet files
    all_records: List[Dict] = []
    summary_stats: Dict[str, Dict[str, int]] = {}
    pages_seen = defaultdict(set)
    issues_seen = defaultdict(set)
    newspapers_seen = defaultdict(set)

    for y in year_sequence:
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

        if summary_only:
            year_key = str(y)
            stats = summary_stats.setdefault(
                year_key,
                {
                    "keyword_frequency": 0,
                    "article_count": 0,
                    "word_count": 0,
                }
            )
            for row in df.itertuples(index=False):
                stats["article_count"] += 1
                article_text = getattr(row, "article", "") or ""
                if article_text:
                    stats["word_count"] += len(word_pattern.findall(article_text))
                    if keyword_pattern:
                        stats["keyword_frequency"] += len(keyword_pattern.findall(article_text))
                elif keyword_pattern:
                    # No article text — still ensure key exists
                    stats["keyword_frequency"] += 0
                date_val = getattr(row, "date", "") or ""
                page_val = getattr(row, "page", "") or ""
                lccn_val = getattr(row, "lccn", "") or ""
                pages_seen[year_key].add((date_val, page_val, lccn_val))
                issues_seen[year_key].add((date_val, lccn_val))
                if lccn_val:
                    newspapers_seen[year_key].add(lccn_val)
        else:
            all_records.extend(df.to_dict("records"))

        if cancel_event and cancel_event.is_set():
            return []
        if progress_callback:
            progress_callback(int(len(df)))

    if cancel_event and cancel_event.is_set():
        return []

    if summary_only:
        per_year: List[Dict[str, Any]] = []
        totals = {
            "keyword_frequency": 0,
            "article_count": 0,
            "page_count": 0,
            "issue_count": 0,
            "newspaper_count": 0,
            "word_count": 0,
        }
        for year in year_sequence:
            year_key = str(year)
            stats = summary_stats.get(year_key, {})
            page_count = len(pages_seen.get(year_key, set()))
            issue_count = len(issues_seen.get(year_key, set()))
            newspaper_count = len(newspapers_seen.get(year_key, set()))
            row = {
                "year": year_key,
                "keyword_frequency": int(stats.get("keyword_frequency", 0)),
                "article_count": int(stats.get("article_count", 0)),
                "page_count": page_count,
                "issue_count": issue_count,
                "newspaper_count": newspaper_count,
                "word_count": int(stats.get("word_count", 0)),
            }
            per_year.append(row)
            totals["keyword_frequency"] += row["keyword_frequency"]
            totals["article_count"] += row["article_count"]
            totals["page_count"] += row["page_count"]
            totals["issue_count"] += row["issue_count"]
            totals["newspaper_count"] += row["newspaper_count"]
            totals["word_count"] += row["word_count"]

        return {
            "summary_only": True,
            "per_year": per_year,
            "totals": totals,
            "search_term": search_term,
            "start_date": start_date_str,
            "end_date": end_date_str,
        }

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
    out_file = os.path.join(term_dir, f"{search_term}_{start_date_str}_{end_date_str}.json")
    payload = {
        "start_date": start_date_str,
        "end_date": end_date_str,
        "search_term": search_term,
        "match_count": int(len(all_records)),
        "articles": all_records,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    metadata_payload = {
        'tool': 'search_dataset',
        'parameters': {
            'search_term': search_term,
            'start_date': start_date_str,
            'end_date': end_date_str,
            'parquet_prefix': parquet_prefix,
            'parquet_dir': parquet_root,
        },
        'cleaning_options': cleaning_options or {},
        'records': len(all_records),
    }
    write_metadata_file(project_dir, out_file, metadata_payload, enabled=metadata_enabled)

    return [out_file]
