
"""
collocate.py
Core collocation analysis pipeline.

Key features implemented here:
- Toggles for include_page_count, include_first_last_date, include_cooccurrence_rate,
  include_relative_position, drop_stopwords.
- Supports "Use JSON results" (plain JSON) and "Use GeoJSON" inputs.
- Time binning aligned to the provided start_date with custom bin size and units.
- Writes two CSVs: collocates_metrics_{term}_{start}_{end}.csv and
  collocates_by_time_{term}_{start}_{end}.csv.
- Optionally writes a filtered occurrences GeoJSON (no dependency on geopandas).
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

from .config import init_project  # type: ignore


# A robust built-in English stopword list (no external deps)
STOPWORDS = {
    # Articles & pronouns
    "a","an","the","this","that","these","those","i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them",
    "their","theirs","themselves","what","which","who","whom","whose","someone","something","one",
    # Aux verbs & common verbs
    "am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing",
    "would","should","could","ought","may","might","must","can","shall","will",
    # Prepositions & conjunctions
    "and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between",
    "into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over",
    "under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few",
    "more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can",
    "will","just","don","should","now",
    # Historical/scan artifacts
    "—","–","-","—","•","…","“","”","‘","’","``","''","'",'"',"&",";",":","(",")","[","]","{","}","/","\\",".",",","?","!",
    # numerals as strings
    "one","two","three","four","five","six","seven","eight","nine","ten"
}

WORD_RE = re.compile(r"[A-Za-z0-9']+")  # keep letters, digits, apostrophes


@dataclass
class CollocationOptions:
    include_page_count: bool = False
    include_first_last_date: bool = False
    include_cooccurrence_rate: bool = False
    include_relative_position: bool = False
    drop_stopwords: bool = False
    window: int = 5  # window on each side


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [w.lower() for w in WORD_RE.findall(text)]


def _find_phrase_positions(tokens: List[str], phrase_tokens: List[str]) -> List[int]:
    """Return start indexes where phrase_tokens occur in tokens."""
    L = len(phrase_tokens)
    if L == 0 or not tokens:
        return []
    starts = []
    for i in range(0, len(tokens) - L + 1):
        if tokens[i:i+L] == phrase_tokens:
            starts.append(i)
    return starts


def _get_bin_edges(start: pd.Timestamp, end: pd.Timestamp, unit: str, size: int) -> List[pd.Timestamp]:
    """Generate bin edges aligned to the start with variable-length offsets (months/years supported)."""
    edges = [start]
    if unit == "days":
        delta = pd.Timedelta(days=size)
        current = start
        while current < end:
            current = current + delta
            edges.append(current)
    elif unit == "weeks":
        delta = pd.Timedelta(weeks=size)
        current = start
        while current < end:
            current = current + delta
            edges.append(current)
    elif unit == "months":
        offset = pd.DateOffset(months=size)
        current = start
        while current < end:
            current = current + offset
            edges.append(current)
    elif unit == "years":
        offset = pd.DateOffset(years=size)
        current = start
        while current < end:
            current = current + offset
            edges.append(current)
    else:
        # default to months if unknown
        offset = pd.DateOffset(months=size)
        current = start
        while current < end:
            current = current + offset
            edges.append(current)
    # Ensure last edge > end
    if edges[-1] <= end:
        if unit == "days":
            edges.append(edges[-1] + pd.Timedelta(days=size))
        elif unit == "weeks":
            edges.append(edges[-1] + pd.Timedelta(weeks=size))
        elif unit == "months":
            edges.append(edges[-1] + pd.DateOffset(months=size))
        else:
            edges.append(edges[-1] + pd.DateOffset(years=size))
    return edges


def _assign_time_bin(dates: pd.Series, edges: List[pd.Timestamp]) -> pd.Series:
    labels = [edges[i].date().isoformat() for i in range(len(edges)-1)]
    return pd.cut(pd.to_datetime(dates), bins=edges, right=False, labels=labels, include_lowest=True)


def _load_json(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    articles = payload.get("articles") or payload.get("records", [])
    df = pd.DataFrame(articles)
    # Ensure required cols
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for col in ["article_id","lccn","page","article","url","filename","newspaper_name"]:
        if col not in df.columns:
            df[col] = None
    return df


def _load_geojson(geojson_path: str) -> pd.DataFrame:
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    props = [feat.get("properties", {}) for feat in feats]
    df = pd.DataFrame(props)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # Ensure fields exist
    for col in ["article_id","lccn","page","article","url","filename","newspaper_name", "City", "State"]:
        if col not in df.columns:
            df[col] = None
    return df


def _filter_df(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str],
               city: Optional[str], state: Optional[str], is_geo: bool) -> pd.DataFrame:
    out = df.copy()
    if start_date:
        sd = pd.to_datetime(start_date, errors="coerce")
        out = out[pd.to_datetime(out["date"], errors="coerce") >= sd]
    if end_date:
        ed = pd.to_datetime(end_date, errors="coerce")
        out = out[pd.to_datetime(out["date"], errors="coerce") <= ed]
    if is_geo:
        if city:
            out = out[(out["City"].astype(str).str.lower() == city.lower())]
        if state:
            out = out[(out["State"].astype(str).str.lower() == state.lower())]
    out = out.dropna(subset=["article", "date"])
    return out


def _collocate_from_df(df: pd.DataFrame, term: str, opts: CollocationOptions
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process DataFrame of articles and compute collocation metrics and by-time counts.
    """
    term_tokens = _tokenize(term)
    if opts.drop_stopwords:
        term_tokens = [t for t in term_tokens if t not in STOPWORDS]
    if not term_tokens:
        raise ValueError("Search term is empty after token processing.")

    total_articles = 0
    per_collocate_count = Counter()  # total token co-occurrence count (within windows)
    per_collocate_article_ids = defaultdict(set)  # set of article_ids where collocate appears
    per_collocate_pages = defaultdict(set)
    per_collocate_dates = defaultdict(list)
    per_collocate_rel_positions = defaultdict(list)

    # Prepare by-time nested counters
    by_time_counts: Dict[str, Counter] = defaultdict(Counter)

    for _, row in df.iterrows():
        text = row.get("article", "")
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = _tokenize(text)
        if opts.drop_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]

        starts = _find_phrase_positions(tokens, term_tokens)
        if not starts:
            # It's possible that API search results always contain the term, but robustly skip otherwise
            continue
        total_articles += 1
        aid = row.get("article_id") or row.get("filename") or f"row{_}"
        page = row.get("page")
        dt = row.get("date")
        pd_dt = pd.to_datetime(dt, errors="coerce")

        for st in starts:
            left = max(0, st - opts.window)
            right = min(len(tokens), st + len(term_tokens) + opts.window)
            # collocates exclude the term tokens themselves
            neighbors = tokens[left:st] + tokens[st+len(term_tokens):right]
            for j, tok in enumerate(neighbors):
                if not tok or tok == "" or tok.isdigit():
                    continue
                per_collocate_count[tok] += 1
                per_collocate_article_ids[tok].add(aid)
                if page:
                    per_collocate_pages[tok].add(str(page))
                if pd.notna(pd_dt):
                    per_collocate_dates[tok].append(pd_dt)
                if opts.include_relative_position:
                    # relative position from the first token of the phrase; negative => before
                    rel = j - len(tokens[left:st])
                    per_collocate_rel_positions[tok].append(rel)

            # by-time counting will be assigned after this loop when we know bins

    # Build metrics DataFrame
    if not per_collocate_count:
        return pd.DataFrame(columns=["collocate_term","frequency"]), pd.DataFrame()

    metrics = pd.DataFrame({
        "collocate_term": list(per_collocate_count.keys()),
        "frequency": list(per_collocate_count.values())
    })

    # Additional metrics
    if opts.include_page_count:
        metrics["page_count"] = metrics["collocate_term"].map(lambda t: len(per_collocate_pages.get(t, set())))
    if opts.include_first_last_date:
        metrics["first_date"] = metrics["collocate_term"].map(
            lambda t: (min(per_collocate_dates[t]).date().isoformat() if per_collocate_dates.get(t) else None)
        )
        metrics["last_date"] = metrics["collocate_term"].map(
            lambda t: (max(per_collocate_dates[t]).date().isoformat() if per_collocate_dates.get(t) else None)
        )
    if opts.include_cooccurrence_rate:
        metrics["article_count"] = metrics["collocate_term"].map(lambda t: len(per_collocate_article_ids.get(t, set())))
        metrics["cooccurrence_rate"] = metrics["article_count"] / max(1, total_articles)
    if opts.include_relative_position:
        metrics["mean_relative_position"] = metrics["collocate_term"].map(
            lambda t: (float(np.mean(per_collocate_rel_positions[t])) if per_collocate_rel_positions.get(t) else np.nan)
        )

    # Sort by frequency descending
    metrics = metrics.sort_values(["frequency","collocate_term"], ascending=[False, True]).reset_index(drop=True)

    # We'll compute by-time counts below (requires bin assignment)
    return metrics, pd.DataFrame()


def _build_by_time(df: pd.DataFrame, term: str, opts: CollocationOptions,
                   start_date: Optional[str], end_date: Optional[str],
                   size: int, unit: str) -> pd.DataFrame:
    """Return DataFrame of counts per time bin and term with ordinal ranks."""
    if df.empty:
        return pd.DataFrame()

    # Prepare tokens for lookup
    term_tokens = _tokenize(term)
    if opts.drop_stopwords:
        term_tokens = [t for t in term_tokens if t not in STOPWORDS]

    # Build bin edges aligned to start_date (or min date)
    dates_series = pd.to_datetime(df["date"], errors="coerce")
    sdt = pd.to_datetime(start_date, errors="coerce") if start_date else dates_series.min()
    edt = pd.to_datetime(end_date, errors="coerce") if end_date else dates_series.max()
    sdt = pd.to_datetime(sdt).normalize()
    edt = pd.to_datetime(edt).normalize()
    edges = _get_bin_edges(sdt, edt, unit, size)
    labels = [edges[i].date().isoformat() for i in range(len(edges)-1)]

    # Assign bins to rows for grouping
    df = df.copy()
    df["time_bin"] = _assign_time_bin(df["date"], edges)

    # For efficiency we pre-tokenize all articles
    df["tokens"] = df["article"].astype(str).map(_tokenize)
    if opts.drop_stopwords:
        df["tokens"] = df["tokens"].map(lambda toks: [t for t in toks if t not in STOPWORDS])

    # Create per-bin counters
    records = []
    for label in labels:
        # subset rows within this time bin
        sub = df[df["time_bin"] == label]
        if sub.empty:
            continue
        counter = Counter()
        for toks in sub["tokens"]:
            starts = _find_phrase_positions(toks, term_tokens)
            if not starts:
                continue
            for st in starts:
                L = len(term_tokens)
                left = max(0, st - opts.window)
                right = min(len(toks), st + L + opts.window)
                neighbors = toks[left:st] + toks[st+L:right]
                for tok in neighbors:
                    if not tok or tok == "" or tok.isdigit():
                        continue
                    counter[tok] += 1
        if not counter:
            continue
        # Convert to rows with rank
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for rank, (tok, freq) in enumerate(items, start=1):
            records.append({"time_bin": label, "collocate_term": tok, "frequency": freq, "ordinal_rank": rank})

    if not records:
        return pd.DataFrame()

    by_time = pd.DataFrame.from_records(records)
    return by_time


def _safe_term(term: Optional[str]) -> str:
    if not term:
        return "term"
    return re.sub(r"[^A-Za-z0-9._-]", "", re.sub(r"\s+", "_", term)) or "term"


def run_collocation(
    project_dir: str,
    city: Optional[str] = None,
    state: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    term: Optional[str] = None,
    time_bin_unit: Optional[str] = None,  # e.g., "1 months", "2 weeks"
    json_path: Optional[str] = None,
    geojson_path: Optional[str] = None,
    include_page_count: bool = False,
    include_first_last_date: bool = False,
    include_cooccurrence_rate: bool = False,
    include_relative_position: bool = False,
    drop_stopwords: bool = False,
    write_occurrences_geojson: bool = False,
) -> Optional[str]:
    """
    Execute collocation analysis. Writes outputs into data/processed.

    Returns path to occurrences GeoJSON if write_occurrences_geojson is True, else None.
    """
    if not json_path and not geojson_path:
        raise ValueError("Provide either json_path or geojson_path")

    opts = CollocationOptions(
        include_page_count=include_page_count,
        include_first_last_date=include_first_last_date,
        include_cooccurrence_rate=include_cooccurrence_rate,
        include_relative_position=include_relative_position,
        drop_stopwords=drop_stopwords,
    )

    paths = init_project(project_dir)
    proc = paths["processed"]

    # Load and filter dataframe
    is_geo = bool(geojson_path)
    if is_geo:
        df = _load_geojson(geojson_path)  # includes City/State if present
    else:
        df = _load_json(json_path)  # type: ignore

    df = _filter_df(df, start_date, end_date, city, state, is_geo=is_geo)
    if df.empty:
        # Still write empty CSVs to keep UI predictable
        safe = _safe_term(term)
        start_lbl = start_date or "all"
        end_lbl = end_date or "all"
        metrics_csv = os.path.join(proc, f"collocates_metrics_{safe}_{start_lbl}_{end_lbl}.csv")
        bytime_csv = os.path.join(proc, f"collocates_by_time_{safe}_{start_lbl}_{end_lbl}.csv")
        pd.DataFrame(columns=["collocate_term","frequency"]).to_csv(metrics_csv, index=False)
        pd.DataFrame(columns=["time_bin","collocate_term","frequency","ordinal_rank"]).to_csv(bytime_csv, index=False)
        return None

    # Ensure term is present
    if not term:
        # try to infer from filename or payload structure
        if json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            term = payload.get("search_term")
        if not term and geojson_path:
            base = os.path.basename(geojson_path)
            if base.startswith("merged_"):
                term = base[len("merged_"):].split("_")[0]
    if not term:
        raise ValueError("Search term is required to run collocation.")

    # Build metrics (without time dimension)
    metrics, _ = _collocate_from_df(df, term, opts)

    # Write metrics CSV
    safe = _safe_term(term)
    start_lbl = start_date or "all"
    end_lbl = end_date or "all"
    metrics_csv = os.path.join(proc, f"collocates_metrics_{safe}_{start_lbl}_{end_lbl}.csv")
    metrics.to_csv(metrics_csv, index=False)

    # Build by-time CSV if time_bin_unit provided
    bytime_csv = os.path.join(proc, f"collocates_by_time_{safe}_{start_lbl}_{end_lbl}.csv")
    if time_bin_unit and isinstance(time_bin_unit, str):
        parts = time_bin_unit.strip().split()
        if len(parts) == 2 and parts[0].isdigit():
            size = int(parts[0])
            unit = parts[1].lower()
        else:
            # default 1 month
            size, unit = 1, "months"
        by_time = _build_by_time(df, term, opts, start_date, end_date, size, unit)
        by_time.to_csv(bytime_csv, index=False)
    else:
        # if no time bin requested, write an empty structure (prevents FileNotFoundError downstream)
        pd.DataFrame(columns=["time_bin","collocate_term","frequency","ordinal_rank"]).to_csv(bytime_csv, index=False)

    # Optionally write occurrences geojson (filtered subset)
    if write_occurrences_geojson and geojson_path:
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            feats = gj.get("features", [])
            # Filter by city/state/dates
            sel = []
            for ft in feats:
                pr = ft.get("properties", {})
                dt = pd.to_datetime(pr.get("date"), errors="coerce")
                if start_date and pd.notna(dt) and dt < pd.to_datetime(start_date):
                    continue
                if end_date and pd.notna(dt) and dt > pd.to_datetime(end_date):
                    continue
                if city and str(pr.get("City","")).lower() != city.lower():
                    continue
                if state and str(pr.get("State","")).lower() != state.lower():
                    continue
                # Keep if the article text contains the term (fallback check)
                txt = pr.get("article", "")
                toks = _tokenize(txt)
                tks = _tokenize(term)
                if _find_phrase_positions(toks, tks):
                    sel.append(ft)
            out = {
                "type": "FeatureCollection",
                "name": f"occurrences_{safe}_{start_lbl}_{end_lbl}",
                "crs": gj.get("crs", {"type":"name", "properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}}),
                "features": sel,
            }
            out_path = os.path.join(proc, f"occurrences_{safe}_{start_lbl}_{end_lbl}.geojson")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f)
            return out_path
        except Exception:
            # don't fail the whole run if we can't write occurrences
            return None

    return None
