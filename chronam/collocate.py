
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
import hashlib
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

from .config import init_project  # type: ignore
from .exceptions import OperationCancelledError
from .utils import term_directory_name, write_metadata_file


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


def _check_cancel(cancel_event: Optional[threading.Event]):
    if cancel_event and cancel_event.is_set():
        raise OperationCancelledError()


@dataclass
class CollocationOptions:
    include_page_count: bool = False
    include_first_last_date: bool = False
    include_cooccurrence_rate: bool = False
    include_relative_position: bool = False
    drop_stopwords: bool = False
    window_left: int = 5
    window_right: int = 5

    @property
    def window(self) -> int:
        return max(self.window_left, self.window_right)


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


def _collocate_from_df(
    df: pd.DataFrame,
    term: str,
    opts: CollocationOptions,
    cancel_event: Optional[threading.Event] = None,
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

    track_pages = opts.include_page_count

    for _, row in df.iterrows():
        _check_cancel(cancel_event)
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
        page = row.get("page") if track_pages else None
        lccn_val = (row.get("lccn") or row.get("SN")) if track_pages else None
        dt = row.get("date")
        pd_dt = pd.to_datetime(dt, errors="coerce")
        date_key = (
            pd_dt.date().isoformat() if (track_pages and pd.notna(pd_dt)) else (str(dt).strip() if (track_pages and dt) else "")
        )

        for st in starts:
            _check_cancel(cancel_event)
            left_idx = max(0, st - max(0, int(opts.window_left)))
            right_idx = min(len(tokens), st + len(term_tokens) + max(0, int(opts.window_right)))
            left_segment = tokens[left_idx:st]
            right_segment = tokens[st+len(term_tokens):right_idx]
            neighbors = left_segment + right_segment
            left_len = len(left_segment)
            for j, tok in enumerate(neighbors):
                if not tok or tok == "" or tok.isdigit():
                    continue
                per_collocate_count[tok] += 1
                per_collocate_article_ids[tok].add(aid)
                if track_pages:
                    page_key: Optional[Tuple[str, str, str]] = None
                    if page or lccn_val or date_key:
                        page_key = (
                            str(lccn_val or ""),
                            date_key,
                            str(page or ""),
                        )
                    elif aid:
                        page_key = ("", "", f"article:{aid}")
                    if page_key is not None:
                        per_collocate_pages[tok].add(page_key)
                if pd.notna(pd_dt):
                    per_collocate_dates[tok].append(pd_dt)
                if opts.include_relative_position:
                    # relative position from the first token of the phrase; negative => before
                    rel = j - left_len
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


def _build_by_time(
    df: pd.DataFrame,
    term: str,
    opts: CollocationOptions,
    start_date: Optional[str],
    end_date: Optional[str],
    size: int,
    unit: str,
    cancel_event: Optional[threading.Event] = None,
) -> pd.DataFrame:
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
        _check_cancel(cancel_event)
        # subset rows within this time bin
        sub = df[df["time_bin"] == label]
        if sub.empty:
            continue
        counter = Counter()
        for toks in sub["tokens"]:
            _check_cancel(cancel_event)
            starts = _find_phrase_positions(toks, term_tokens)
            if not starts:
                continue
            for st in starts:
                L = len(term_tokens)
                left_idx = max(0, st - max(0, int(opts.window_left)))
                right_idx = min(len(toks), st + L + max(0, int(opts.window_right)))
                left_segment = toks[left_idx:st]
                right_segment = toks[st+L:right_idx]
                neighbors = left_segment + right_segment
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


def _safe_component(value: Optional[str], default: str) -> str:
    if not value:
        return default
    return _safe_term(str(value)) or default


def _build_output_stem(
    term: str,
    start_date: Optional[str],
    end_date: Optional[str],
    city: Optional[str],
    state: Optional[str],
    time_bin_unit: Optional[str],
    ignore_bin: bool,
    options: Dict[str, bool],
) -> str:
    safe_term = _safe_term(term)
    start_lbl = _safe_component(start_date, "all")
    end_lbl = _safe_component(end_date, "all")
    city_lbl = _safe_component(city, "allcities")
    state_lbl = _safe_component(state, "allstates")
    if ignore_bin or not time_bin_unit:
        bin_lbl = "nobin"
    else:
        bin_lbl = _safe_component(time_bin_unit.replace(" ", ""), "bin")
    payload = {
        "city": city or "all",
        "state": state or "all",
        "start": start_date or "all",
        "end": end_date or "all",
        "time_bin_unit": None if ignore_bin else time_bin_unit,
        "options": options,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return "_".join([safe_term, start_lbl, end_lbl, city_lbl, state_lbl, bin_lbl, digest])


def _drop_suffix(drop_terms: Optional[List[str]]) -> str:
    if not drop_terms:
        return "_nodrop"
    cleaned = [str(term).strip() for term in drop_terms if str(term).strip()]
    return "_drop" if cleaned else "_nodrop"


def _normalize_term_groups(term_groups: Optional[List[dict]]) -> List[dict]:
    if not term_groups:
        return []
    normalized: List[dict] = []
    for entry in term_groups:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name', '')).strip()
        raw_terms = entry.get('terms') or []
        terms: List[str] = []
        seen_terms: Set[str] = set()
        for term in raw_terms:
            term_str = str(term).strip()
            if not term_str:
                continue
            lowered = term_str.lower()
            if lowered in seen_terms:
                continue
            seen_terms.add(lowered)
            terms.append(term_str)
        if not name or not terms:
            continue
        normalized.append({
            'name': name,
            'terms': terms,
            'terms_lower': [t.lower() for t in terms],
        })
    return normalized


def _group_signature(groups: List[dict]) -> str:
    if not groups:
        return ''
    payload = [
        {
            'name': group['name'],
            'terms': sorted(group.get('terms_lower', [])),
        }
        for group in groups
        if group.get('terms_lower')
    ]
    if not payload:
        return ''
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:6]
    return f"_group_{digest}"


def _suffix_with_groups(drop_terms: Optional[List[str]], groups: List[dict]) -> str:
    suffix = _drop_suffix(drop_terms)
    signature = _group_signature(groups)
    return f"{suffix}{signature}" if signature else suffix


def _as_int_if_close(value: Optional[float]) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (int,)):
        return value
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(float_val) and abs(float_val - round(float_val)) < 1e-9:
        return int(round(float_val))
    return float_val


def _aggregate_metric_subset(subset: pd.DataFrame, group_name: str, columns: List[str]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    freq_total = None
    if 'frequency' in subset.columns:
        freq_total = pd.to_numeric(subset['frequency'], errors='coerce').fillna(0).sum()
    article_total = None
    if 'article_count' in subset.columns:
        article_total = pd.to_numeric(subset['article_count'], errors='coerce').fillna(0).sum()
    co_rate = None
    if 'cooccurrence_rate' in subset.columns and article_total not in (None, 0):
        denom_candidates = []
        for _, row in subset[['article_count', 'cooccurrence_rate']].dropna().iterrows():
            rate = row.get('cooccurrence_rate')
            count_val = row.get('article_count')
            if not rate:
                continue
            try:
                denom = float(count_val) / float(rate)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if math.isfinite(denom):
                denom_candidates.append(denom)
        total_articles = float(sum(denom_candidates) / len(denom_candidates)) if denom_candidates else None
        if total_articles:
            co_rate = article_total / total_articles if total_articles else None
    weighted_position = None
    if 'mean_relative_position' in subset.columns and 'frequency' in subset.columns:
        valid = subset[['mean_relative_position', 'frequency']].dropna()
        if not valid.empty:
            weights = pd.to_numeric(valid['frequency'], errors='coerce').fillna(0)
            weight_sum = float(weights.sum())
            if weight_sum > 0:
                weighted_vals = pd.to_numeric(valid['mean_relative_position'], errors='coerce').fillna(0)
                weighted_position = float((weighted_vals * weights).sum() / weight_sum)
    for col in columns:
        if col == 'collocate_term':
            aggregated[col] = group_name
        elif col == 'frequency' and freq_total is not None:
            aggregated[col] = _as_int_if_close(freq_total)
        elif col == 'article_count' and article_total is not None:
            aggregated[col] = _as_int_if_close(article_total)
        elif col == 'page_count' and col in subset.columns:
            total_pages = pd.to_numeric(subset['page_count'], errors='coerce').fillna(0).sum()
            aggregated[col] = _as_int_if_close(total_pages)
        elif col == 'cooccurrence_rate' and co_rate is not None:
            aggregated[col] = float(co_rate)
        elif col == 'mean_relative_position' and weighted_position is not None:
            aggregated[col] = weighted_position
        elif col == 'first_date' and col in subset.columns:
            aggregated[col] = subset['first_date'].dropna().min()
        elif col == 'last_date' and col in subset.columns:
            aggregated[col] = subset['last_date'].dropna().max()
        elif col in subset.columns and pd.api.types.is_numeric_dtype(subset[col]):
            total_val = pd.to_numeric(subset[col], errors='coerce').fillna(0).sum()
            aggregated[col] = _as_int_if_close(total_val)
        elif col in subset.columns:
            non_null = subset[col].dropna()
            aggregated[col] = non_null.iloc[0] if not non_null.empty else None
        else:
            aggregated[col] = None
    return aggregated


def _apply_term_groups_to_metrics(metrics: pd.DataFrame, groups: List[dict]) -> pd.DataFrame:
    if metrics.empty or not groups:
        return metrics
    working = metrics.copy()
    norm_col = '__group_norm__'
    working[norm_col] = working['collocate_term'].astype(str).str.lower()
    columns = [col for col in working.columns if col != norm_col]
    rows_to_drop: Set[int] = set()
    new_rows: List[Dict[str, Any]] = []
    assigned_terms: Set[str] = set()
    for group in groups:
        terms_lower = [t for t in group.get('terms_lower', []) if t not in assigned_terms]
        if not terms_lower:
            continue
        subset_mask = working[norm_col].isin(terms_lower)
        subset = working[subset_mask]
        if subset.empty:
            continue
        assigned_terms.update(terms_lower)
        rows_to_drop.update(subset.index.tolist())
        subset_no_norm = subset.drop(columns=[norm_col], errors='ignore')
        new_rows.append(_aggregate_metric_subset(subset_no_norm, group['name'], columns))
    if rows_to_drop:
        working = working.drop(index=list(rows_to_drop))
    working = working.drop(columns=[norm_col], errors='ignore')
    if new_rows:
        working = pd.concat([working, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    if 'frequency' in working.columns:
        working = working.sort_values(['frequency', 'collocate_term'], ascending=[False, True]).reset_index(drop=True)
    else:
        working = working.sort_values(['collocate_term']).reset_index(drop=True)
    return working


def _recalculate_ordinal_ranks(by_time: pd.DataFrame) -> pd.DataFrame:
    if by_time.empty:
        if 'ordinal_rank' not in by_time.columns:
            by_time = by_time.copy()
            by_time['ordinal_rank'] = pd.Series(dtype=int)
        return by_time
    working = by_time.copy()
    working = working.drop(columns=['ordinal_rank'], errors='ignore')
    frames: List[pd.DataFrame] = []
    for time_bin, segment in working.groupby('time_bin', sort=False):
        ordered = segment.sort_values(['frequency', 'collocate_term'], ascending=[False, True]).reset_index(drop=True)
        ordered['ordinal_rank'] = range(1, len(ordered) + 1)
        frames.append(ordered)
    return pd.concat(frames, ignore_index=True) if frames else working


def _apply_term_groups_to_by_time(by_time: pd.DataFrame, groups: List[dict]) -> pd.DataFrame:
    if by_time.empty or not groups:
        return by_time
    working = by_time.copy()
    norm_col = '__group_norm__'
    working[norm_col] = working['collocate_term'].astype(str).str.lower()
    rows_to_drop: Set[int] = set()
    additions: List[Dict[str, Any]] = []
    assigned_terms: Set[str] = set()
    for group in groups:
        terms_lower = [t for t in group.get('terms_lower', []) if t not in assigned_terms]
        if not terms_lower:
            continue
        subset = working[working[norm_col].isin(terms_lower)]
        if subset.empty:
            continue
        assigned_terms.update(terms_lower)
        rows_to_drop.update(subset.index.tolist())
        grouped = subset.groupby('time_bin', as_index=False)['frequency'].sum()
        for _, row in grouped.iterrows():
            additions.append({
                'time_bin': row['time_bin'],
                'collocate_term': group['name'],
                'frequency': row['frequency'],
            })
    if rows_to_drop:
        working = working.drop(index=list(rows_to_drop))
    working = working.drop(columns=[norm_col], errors='ignore')
    if additions:
        working = pd.concat([working, pd.DataFrame(additions)], ignore_index=True, sort=False)
    if 'ordinal_rank' in working.columns or additions:
        working = _recalculate_ordinal_ranks(working)
    return working


def _build_output_paths(
    processed_dir: str,
    term: str,
    start_date: Optional[str],
    end_date: Optional[str],
    city: Optional[str],
    state: Optional[str],
    time_bin_unit: Optional[str],
    ignore_bin: bool,
    options: Dict[str, Any],
    filename_suffix: str,
) -> Dict[str, Optional[str]]:
    stem = _build_output_stem(term, start_date, end_date, city, state, time_bin_unit, ignore_bin, options)
    stem_with_suffix = f"{stem}{filename_suffix}" if filename_suffix else stem
    term_dir = os.path.join(processed_dir, term_directory_name(term))
    metrics = os.path.join(term_dir, f"collocates_metrics_{stem_with_suffix}.csv")
    by_time = None if ignore_bin or not time_bin_unit else os.path.join(term_dir, f"collocates_by_time_{stem_with_suffix}.csv")
    occurrences = os.path.join(term_dir, f"occurrences_{stem_with_suffix}.geojson")
    return {
        "stem": stem_with_suffix,
        "metrics": metrics,
        "by_time": by_time,
        "occurrences": occurrences,
    }


def build_collocation_output_paths(
    project_dir: str,
    *,
    term: str,
    start_date: Optional[str],
    end_date: Optional[str],
    city: Optional[str],
    state: Optional[str],
    time_bin_unit: Optional[str],
    ignore_bin: bool,
    options: Dict[str, Any],
    drop_terms: Optional[List[str]] = None,
    term_groups: Optional[List[dict]] = None,
    metadata_enabled: bool = True,
) -> Dict[str, Optional[str]]:
    processed = init_project(project_dir)["processed"]
    normalized_groups = _normalize_term_groups(term_groups)
    suffix = _suffix_with_groups(drop_terms, normalized_groups)
    return _build_output_paths(processed, term, start_date, end_date, city, state, time_bin_unit, ignore_bin, options, suffix)


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
    window_left: int = 5,
    window_right: int = 5,
    write_occurrences_geojson: bool = False,
    ignore_bin: bool = False,
    write_by_time: bool = True,
    drop_terms: Optional[List[str]] = None,
    term_groups: Optional[List[dict]] = None,
    metadata_enabled: bool = True,
    write_metrics: bool = True,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Optional[str]]:
    """
    Execute collocation analysis. Writes outputs into data/processed/<term>/.

    Returns path to occurrences GeoJSON if write_occurrences_geojson is True, else None.
    """
    if not json_path and not geojson_path:
        raise ValueError("Provide either json_path or geojson_path")

    window_left = int(max(0, min(99, window_left)))
    window_right = int(max(0, min(99, window_right)))

    opts = CollocationOptions(
        include_page_count=include_page_count,
        include_first_last_date=include_first_last_date,
        include_cooccurrence_rate=include_cooccurrence_rate,
        include_relative_position=include_relative_position,
        drop_stopwords=drop_stopwords,
        window_left=window_left,
        window_right=window_right,
    )
    opt_dict = {
        "include_page_count": include_page_count,
        "include_first_last_date": include_first_last_date,
        "include_cooccurrence_rate": include_cooccurrence_rate,
        "include_relative_position": include_relative_position,
        "drop_stopwords": drop_stopwords,
        "window_left": window_left,
        "window_right": window_right,
    }

    paths = init_project(project_dir)
    proc = paths["processed"]

    # Load and filter dataframe
    is_geo = bool(geojson_path)
    if is_geo:
        df = _load_geojson(geojson_path)  # includes City/State if present
    else:
        df = _load_json(json_path)  # type: ignore

    df = _filter_df(df, start_date, end_date, city, state, is_geo=is_geo)
    _check_cancel(cancel_event)

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

    drop_terms = drop_terms or []
    drop_set = {str(t).strip() for t in drop_terms if str(t).strip()}
    normalized_groups = _normalize_term_groups(term_groups)
    suffix = _suffix_with_groups(drop_terms, normalized_groups)

    metadata_paths: Dict[str, str] = {}
    metadata_common = {
        'tool': 'collocation_analysis',
        'parameters': {
            'city': city,
            'state': state,
            'start_date': start_date,
            'end_date': end_date,
            'term': term,
            'time_bin_unit': None if ignore_bin else time_bin_unit,
            'ignore_bin': bool(ignore_bin),
            'options': opt_dict,
            'drop_terms': sorted(drop_set),
            'term_groups': [
                {'name': group['name'], 'terms': list(group['terms'])}
                for group in normalized_groups
            ],
            'write_occurrences_geojson': bool(write_occurrences_geojson),
        },
        'inputs': {
            'json_path': json_path,
            'geojson_path': geojson_path,
        },
    }

    metrics_path: Optional[str] = None
    by_time_path: Optional[str] = None

    if df.empty:
        # Still write empty CSVs to keep UI predictable
        empty_paths = _build_output_paths(
            proc,
            term,
            start_date,
            end_date,
            city,
            state,
            time_bin_unit,
            ignore_bin,
            opt_dict,
            suffix,
        )
        metrics_dir = os.path.dirname(empty_paths.get("metrics") or proc)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        _check_cancel(cancel_event)
        if write_metrics and empty_paths.get("metrics"):
            pd.DataFrame(columns=["collocate_term", "frequency"]).to_csv(empty_paths["metrics"], index=False)
            metrics_meta = dict(metadata_common)
            metrics_meta.update({
                'output_type': 'metrics_csv',
                'row_count': 0,
            })
            meta_path = write_metadata_file(project_dir, empty_paths["metrics"], metrics_meta, enabled=metadata_enabled)
            if meta_path:
                metadata_paths['metrics'] = meta_path
            metrics_path = empty_paths["metrics"]
        if write_by_time and empty_paths.get("by_time"):
            _check_cancel(cancel_event)
            pd.DataFrame(columns=["time_bin", "collocate_term", "frequency", "ordinal_rank"]).to_csv(empty_paths["by_time"], index=False)
            by_time_meta = dict(metadata_common)
            by_time_meta.update({
                'output_type': 'by_time_csv',
                'row_count': 0,
            })
            meta_path = write_metadata_file(project_dir, empty_paths["by_time"], by_time_meta, enabled=metadata_enabled)
            if meta_path:
                metadata_paths['by_time'] = meta_path
            by_time_path = empty_paths["by_time"]
        return {
            "metrics": metrics_path,
            "by_time": by_time_path,
            "occurrences": None,
            "metadata": metadata_paths,
        }

    # Build metrics (without time dimension)
    metrics, _ = _collocate_from_df(df, term, opts, cancel_event=cancel_event)

    if drop_set:
        metrics = metrics[~metrics['collocate_term'].isin(drop_set)].reset_index(drop=True)

    metrics = _apply_term_groups_to_metrics(metrics, normalized_groups)

    # Write metrics CSV
    output_paths = _build_output_paths(proc, term, start_date, end_date, city, state, time_bin_unit, ignore_bin, opt_dict, suffix)
    metrics_dir = os.path.dirname(output_paths.get("metrics") or proc)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    if write_metrics and output_paths.get("metrics"):
        metrics.to_csv(output_paths["metrics"], index=False)
        metrics_meta = dict(metadata_common)
        metrics_meta.update({
            'output_type': 'metrics_csv',
            'row_count': int(len(metrics)),
        })
        meta_path = write_metadata_file(project_dir, output_paths["metrics"], metrics_meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths['metrics'] = meta_path
        metrics_path = output_paths["metrics"]

    # Build by-time CSV if requested
    if write_by_time and output_paths["by_time"] and time_bin_unit and isinstance(time_bin_unit, str):
        parts = time_bin_unit.strip().split()
        if len(parts) == 2 and parts[0].isdigit():
            size = int(parts[0])
            unit = parts[1].lower()
        else:
            # default 1 month
            size, unit = 1, "months"
        by_time = _build_by_time(df, term, opts, start_date, end_date, size, unit, cancel_event=cancel_event)
        if drop_set:
            by_time = by_time[~by_time['collocate_term'].isin(drop_set)].reset_index(drop=True)
        by_time = _apply_term_groups_to_by_time(by_time, normalized_groups)
        by_time.to_csv(output_paths["by_time"], index=False)
        by_time_meta = dict(metadata_common)
        by_time_meta.update({
            'output_type': 'by_time_csv',
            'row_count': int(len(by_time)),
        })
        meta_path = write_metadata_file(project_dir, output_paths["by_time"], by_time_meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths['by_time'] = meta_path
        by_time_path = output_paths["by_time"]

    # Optionally write occurrences geojson (filtered subset)
    occurrence_path = None
    if write_occurrences_geojson and geojson_path:
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            feats = gj.get("features", [])
            # Filter by city/state/dates
            sel = []
            for ft in feats:
                _check_cancel(cancel_event)
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
                "name": os.path.splitext(os.path.basename(output_paths["occurrences"]))[0],
                "crs": gj.get("crs", {"type":"name", "properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}}),
                "features": sel,
            }
            out_path = output_paths["occurrences"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f)
            occurrence_path = out_path
            occur_meta = dict(metadata_common)
            occur_meta.update({
                'output_type': 'occurrences_geojson',
                'feature_count': len(sel),
            })
            meta_path = write_metadata_file(project_dir, occurrence_path, occur_meta, enabled=metadata_enabled)
            if meta_path:
                metadata_paths['occurrences'] = meta_path
        except Exception:
            occurrence_path = None

    result = {
        "metrics": metrics_path,
        "by_time": by_time_path,
        "occurrences": occurrence_path,
        "metadata": metadata_paths,
    }
    return result
