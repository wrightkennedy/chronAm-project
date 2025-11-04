import html
import json
import os
import re
import uuid
import csv
import threading
from collections import Counter, defaultdict
from string import Template as StrTemplate

from jinja2 import Template as JinjaTemplate
from branca.element import MacroElement
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple, Optional, Callable, Set

import folium
from folium import Html, Popup
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster

from .collocate import STOPWORDS as _STOPWORDS, WORD_RE as _WORD_RE
from .exceptions import OperationCancelledError
from .utils import write_metadata_file
from .metrics import metric_total_for_year_within_dates


def _tok(text: Any, drop_stop: bool = False) -> List[str]:
    s = str(text or '')
    toks = [w.lower() for w in _WORD_RE.findall(s)]
    if drop_stop:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def _check_cancel(cancel_event: Optional[threading.Event]):
    if cancel_event and cancel_event.is_set():
        raise OperationCancelledError()

def _find_positions(tokens: List[str], phrase_tokens: List[str]) -> List[int]:
    if not tokens or not phrase_tokens:
        return []
    L = len(phrase_tokens)
    pos = []
    for i in range(0, len(tokens) - L + 1):
        if tokens[i:i+L] == phrase_tokens:
            pos.append(i)
    return pos


COLLOCATE_RANK_LIMIT = 150
COLLOCATE_SELECTOR_LIMIT = 300


def _city_state_key(city: Any, state: Any) -> str:
    city_norm = str(city or '').strip().lower()
    state_norm = str(state or '').strip().lower()
    return f"{city_norm}||{state_norm}"


def _sorted_counter_terms(counter: Counter) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _prepare_term_group_lookup(term_groups: Optional[List[dict]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not term_groups:
        return lookup
    for entry in term_groups:
        if not isinstance(entry, dict):
            continue
        display = str(entry.get('name', '')).strip()
        if not display:
            continue
        display_lower = display.lower()
        lookup.setdefault(display_lower, display)
        terms = entry.get('terms') or []
        seen: Set[str] = set()
        for term in terms:
            term_str = str(term).strip()
            if not term_str:
                continue
            term_lower = term_str.lower()
            if term_lower in seen:
                continue
            seen.add(term_lower)
            lookup[term_lower] = display
    return lookup


def _build_collocate_rank_index(
    groups: List[Dict[str, Any]],
    popup_dataset: Dict[str, Any],
    search_term: Optional[str],
    *,
    drop_stopwords: bool,
    window: int,
    drop_terms: Optional[List[str]],
    term_groups: Optional[List[dict]] = None,
    top_n: int = COLLOCATE_RANK_LIMIT,
    term_scope: str = 'global',
    time_key: Optional[str] = None,
    focus_mode: str = 'all',
    focus_city: Optional[str] = None,
    focus_state: Optional[str] = None,
    manual_terms: Optional[List[str]] = None,
    rank_limit: int = COLLOCATE_RANK_LIMIT,
    selector_limit: int = COLLOCATE_SELECTOR_LIMIT,
    cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[
        List[str],
        Dict[str, Dict[str, Dict[str, int]]],
        int,
        Dict[str, Dict[str, Set[int]]],
        Dict[str, Counter],
        Dict[str, str],
    ]:
    term_tokens = [tok for tok in _tok(search_term or '', drop_stop=False) if tok]
    if drop_stopwords:
        term_tokens = [tok for tok in term_tokens if tok not in _STOPWORDS]
    if not term_tokens:
        return [], {}, 0, {}

    try:
        window_size = int(window)
    except (TypeError, ValueError):
        window_size = 5
    window_size = max(1, window_size)

    drop_set = {
        str(term).strip().lower()
        for term in (drop_terms or [])
        if isinstance(term, str) and str(term).strip()
    }

    group_lookup = _prepare_term_group_lookup(term_groups)

    def _resolve_term(value: Any) -> str:
        raw = str(value or '').strip()
        if not raw:
            return ''
        mapped = group_lookup.get(raw.lower())
        return mapped if mapped else raw

    manual_terms_norm: List[str] = []
    manual_terms_seen: Set[str] = set()
    if manual_terms:
        for term in manual_terms:
            if not isinstance(term, str):
                continue
            raw = term.strip()
            if not raw:
                continue
            canonical = _resolve_term(raw)
            canonical_lower = canonical.lower()
            if canonical_lower and canonical_lower not in manual_terms_seen:
                manual_terms_seen.add(canonical_lower)
                manual_terms_norm.append(canonical)

    try:
        requested_top = int(top_n)
    except (TypeError, ValueError):
        requested_top = rank_limit
    top_limit = max(1, min(requested_top, rank_limit))
    if manual_terms_norm:
        top_limit = max(top_limit, len(manual_terms_norm))

    term_scope_norm = str(term_scope or 'global').strip().lower()
    focus_mode_norm = str(focus_mode or 'all').strip().lower()
    focus_city_norm = str(focus_city or '').strip().lower()
    focus_state_norm = str(focus_state or '').strip().lower()

    rank_index: Dict[str, Dict[str, Dict[str, int]]] = {}
    global_counts: Counter = Counter()
    aggregate_time_global: Dict[str, Counter] = defaultdict(Counter)
    time_key_labels: Dict[str, str] = {}
    focus_city_counter: Counter = Counter()
    focus_city_time: Dict[str, Counter] = defaultdict(Counter)
    focus_state_counter: Counter = Counter()
    focus_state_time: Dict[str, Counter] = defaultdict(Counter)
    group_rank_data: Dict[str, Dict[str, Any]] = {}
    city_term_hits: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
    rank_max = 0

    for group in groups:
        _check_cancel(cancel_event)
        entries = group.get('entries') or []
        if not entries:
            continue

        group_id = group.get('id')
        dataset_entry = popup_dataset.get(group_id) if isinstance(popup_dataset, dict) else {}
        time_bins = dataset_entry.get('time_bins') if isinstance(dataset_entry, dict) else {}

        index_time_keys: Dict[int, Set[str]] = defaultdict(set)
        if isinstance(time_bins, dict):
            for bin_key, info in time_bins.items():
                _check_cancel(cancel_event)
                if not isinstance(info, dict):
                    continue
                indexes = info.get('indexes') or []
                label = info.get('time_label') or ''
                alt_keys: List[str] = []
                if bin_key is not None:
                    alt_keys.append(str(bin_key))
                if label:
                    alt_keys.append(str(label))
                    if bin_key is not None:
                        time_key_labels.setdefault(str(bin_key), str(label))
                for idx in indexes:
                    try:
                        idx_int = int(idx)
                    except (TypeError, ValueError):
                        continue
                    bucket = index_time_keys[idx_int]
                    for key_variant in alt_keys:
                        if key_variant:
                            bucket.add(key_variant)

        first_props = entries[0].get('props', {}) if entries else {}
        city_raw = first_props.get('City')
        state_raw = first_props.get('State')
        city_key = _city_state_key(city_raw, state_raw)
        city_norm = str(city_raw or '').strip().lower()
        state_norm = str(state_raw or '').strip().lower()
        group_counter: Counter = Counter()
        time_counters: Dict[str, Counter] = defaultdict(Counter)
        term_length = len(term_tokens)
        term_hits = city_term_hits.setdefault(city_key, defaultdict(set))

        for idx, entry in enumerate(entries):
            _check_cancel(cancel_event)
            article_text = entry.get('_article_full') or entry.get('props', {}).get('article')
            if not article_text or not isinstance(article_text, str):
                continue
            tokens = _tok(article_text, drop_stop=drop_stopwords)
            if not tokens:
                continue
            starts = _find_positions(tokens, term_tokens)
            if not starts:
                continue
            for start in starts:
                _check_cancel(cancel_event)
                left = max(0, start - window_size)
                right = min(len(tokens), start + term_length + window_size)
                neighbors = tokens[left:start] + tokens[start + term_length:right]
                for tok in neighbors:
                    _check_cancel(cancel_event)
                    if not tok or tok.isdigit():
                        continue
                    tok_norm = tok.lower()
                    if tok_norm in drop_set:
                        continue
                    canonical = _resolve_term(tok)
                    if not canonical:
                        continue
                    group_counter[canonical] += 1
                    global_counts[canonical] += 1
                    term_hits[canonical].add(idx)
                    for time_key in index_time_keys.get(idx, ()):  # time-specific accumulation
                        time_counters[time_key][canonical] += 1

        if not group_counter:
            continue

        for key, counter in time_counters.items():
            _check_cancel(cancel_event)
            if counter:
                aggregate_time_global[key].update(counter)

        if focus_mode_norm == 'city' and focus_city_norm:
            city_match = city_norm == focus_city_norm and (
                not focus_state_norm or state_norm == focus_state_norm
            )
            if city_match:
                focus_city_counter.update(group_counter)
                for key, counter in time_counters.items():
                    _check_cancel(cancel_event)
                    if counter:
                        focus_city_time[key].update(counter)
        elif focus_mode_norm == 'state' and focus_state_norm:
            if state_norm == focus_state_norm and state_norm:
                focus_state_counter.update(group_counter)
                for key, counter in time_counters.items():
                    _check_cancel(cancel_event)
                    if counter:
                        focus_state_time[key].update(counter)

        group_rank_data[city_key] = {
            'ordered': _sorted_counter_terms(group_counter),
            'time': time_counters,
        }

    if not global_counts:
        return [], {}, 0, {}

    base_counter = global_counts
    base_time = aggregate_time_global
    if focus_mode_norm == 'city' and focus_city_counter:
        base_counter = focus_city_counter
        base_time = focus_city_time
    elif focus_mode_norm == 'state' and focus_state_counter:
        base_counter = focus_state_counter
        base_time = focus_state_time

    selected_terms_ordered = [term for term, _freq in _sorted_counter_terms(base_counter) if term][:top_limit]
    if manual_terms_norm:
        manual_filtered = [term for term in manual_terms_norm if term in base_counter or term in global_counts]
        if manual_filtered:
            selected_terms_ordered = manual_filtered
        else:
            manual_terms_norm = []
    if term_scope_norm.startswith('time') and time_key:
        raw_key = str(time_key).strip()
        key_variants: List[str] = [raw_key] if raw_key else []
        label_variant = time_key_labels.get(raw_key)
        if label_variant:
            key_variants.append(label_variant)
        combined_counter: Counter = Counter()
        for data in group_rank_data.values():
            time_counters = data.get('time') or {}
            for candidate in key_variants:
                if not candidate:
                    continue
                counter = time_counters.get(candidate)
                if counter:
                    combined_counter.update(counter)
        if not combined_counter:
            for candidate in key_variants:
                if not candidate:
                    continue
                counter = aggregate_time_global.get(candidate)
                if counter:
                    combined_counter.update(counter)
        if combined_counter:
            scoped_terms = [term for term, _freq in _sorted_counter_terms(combined_counter) if term][:top_limit]
            if scoped_terms:
                selected_terms_ordered = scoped_terms
    if not selected_terms_ordered:
        selected_terms_ordered = [term for term, _freq in _sorted_counter_terms(global_counts) if term][:top_limit]

    selected_terms_set = set(selected_terms_ordered)
    selector_cap = min(selector_limit, max(top_limit, len(selected_terms_ordered)))

    hits_result: Dict[str, Dict[str, Set[int]]] = {}
    for city_key, term_map in city_term_hits.items():
        filtered_terms: Dict[str, Set[int]] = {}
        for term, indexes in term_map.items():
            if not indexes:
                continue
            if selected_terms_set and term not in selected_terms_set:
                continue
            filtered_terms[term] = set(indexes)
        if filtered_terms:
            hits_result[city_key] = filtered_terms

    for city_key, data in group_rank_data.items():
        ordered_terms = data.get('ordered') or []
        limited_terms = ordered_terms[:rank_limit]
        if selected_terms_set:
            limited_terms = [item for item in limited_terms if item[0] in selected_terms_set]
        limited_terms = limited_terms[:top_limit]
        if limited_terms:
            rank_map = {term: pos + 1 for pos, (term, _freq) in enumerate(limited_terms)}
            if rank_map:
                city_entry = rank_index.setdefault(city_key, {})
                city_entry[''] = rank_map
                rank_max = max(rank_max, len(rank_map))
        time_counters = data.get('time') or {}
        for key, counter in time_counters.items():
            if not counter:
                continue
            ordered_time = _sorted_counter_terms(counter)
            ordered_time = ordered_time[:rank_limit]
            if selected_terms_set:
                ordered_time = [item for item in ordered_time if item[0] in selected_terms_set]
            ordered_time = ordered_time[:top_limit]
            if not ordered_time:
                continue
            rank_map_time = {term: pos + 1 for pos, (term, _freq) in enumerate(ordered_time)}
            if rank_map_time:
                city_entry = rank_index.setdefault(city_key, {})
                city_entry[key] = rank_map_time
                rank_max = max(rank_max, len(rank_map_time))

    if not selected_terms_ordered:
        return [], rank_index, rank_max, hits_result, aggregate_time_global, time_key_labels

    collocate_terms_list = selected_terms_ordered[:selector_cap]
    return collocate_terms_list, rank_index, rank_max, hits_result, aggregate_time_global, time_key_labels


# ----------------------------
# Helpers: dates and formatting
# ----------------------------


def _format_date(dt: Optional[datetime]) -> str:
    if not dt:
        return ''
    return dt.strftime('%Y-%m-%d')


def _export_collocate_csv(
    out_path: str,
    *,
    groups: List[Dict[str, Any]],
    popup_dataset: Dict[str, Any],
    summary: Dict[str, Any],
    collocate_map_variant: str,
    rank_index: Dict[str, Dict[str, Dict[str, int]]],
    collocate_hits_by_city: Dict[str, Dict[str, Set[int]]],
    collocate_terms_list: List[str],
    time_index: List[datetime],
    range_start: Optional[datetime],
    range_end: Optional[datetime],
    collocate_rank_term_scope: str,
    collocate_rank_time_key: Optional[str],
) -> Optional[str]:
    rows: List[List[str]] = []

    def city_key(city: Any, state: Any) -> str:
        return f"{str(city or '').strip().lower()}||{str(state or '').strip().lower()}"

    bin_ranges: Dict[str, Tuple[Optional[datetime], Optional[datetime]]] = {}
    if time_index:
        for idx, start_dt in enumerate(time_index):
            end_dt = None
            if idx + 1 < len(time_index):
                end_dt = time_index[idx + 1] - timedelta(days=1)
            else:
                end_dt = range_end
            if end_dt and start_dt and end_dt < start_dt:
                end_dt = start_dt
            bin_ranges[str(idx + 1)] = (start_dt, end_dt)

    overall_start = summary.get('date_range', ('', ''))[0]
    overall_end = summary.get('date_range', ('', ''))[1]

    for group in groups:
        group_id = group.get('id')
        dataset_entry = popup_dataset.get(group_id) if isinstance(popup_dataset, dict) else None
        if not isinstance(dataset_entry, dict):
            continue
        lat = dataset_entry.get('lat')
        lon = dataset_entry.get('lon')
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            lat_f = None
            lon_f = None
        city_raw = dataset_entry.get('city') or ''
        state_raw = dataset_entry.get('state') or ''
        place_label = dataset_entry.get('place_label') or ''
        city_display = str(city_raw).strip() or str(place_label).strip() or 'Unknown'
        c_key = city_key(city_raw, state_raw)
        hits_map = collocate_hits_by_city.get(c_key, {})

        if collocate_map_variant == 'top_term':
            time_bins = dataset_entry.get('time_bins') if isinstance(dataset_entry.get('time_bins'), dict) else {}
            if time_bins:
                def _bin_sort(item):
                    key = item[0]
                    try:
                        return float(key)
                    except (TypeError, ValueError):
                        return key
                for bin_key, info in sorted(time_bins.items(), key=_bin_sort):
                    rank_map = (rank_index.get(c_key) or {}).get(bin_key) or {}
                    if not rank_map:
                        continue
                    top_term = None
                    top_rank = None
                    for term, rank_val in rank_map.items():
                        try:
                            rank_num = int(rank_val)
                        except (TypeError, ValueError):
                            continue
                        if top_rank is None or rank_num < top_rank or (rank_num == top_rank and term < top_term):
                            top_rank = rank_num
                            top_term = term
                    if not top_term:
                        continue
                    indexes = info.get('indexes') or []
                    allowed = set()
                    for idx in indexes:
                        try:
                            allowed.add(int(idx))
                        except (TypeError, ValueError):
                            continue
                    term_hits = set(hits_map.get(top_term, []))
                    count = len([idx for idx in term_hits if idx in allowed])
                    start_dt, end_dt = bin_ranges.get(bin_key, (range_start, range_end))
                    rows.append([
                        _format_date(start_dt),
                        _format_date(end_dt),
                        str(top_term),
                        str(count),
                        city_display,
                        str(lat_f) if lat_f is not None else '',
                        str(lon_f) if lon_f is not None else '',
                    ])
            else:
                rank_map = (rank_index.get(c_key) or {}).get('') or {}
                if not rank_map:
                    continue
                top_term = None
                top_rank = None
                for term, rank_val in rank_map.items():
                    try:
                        rank_num = int(rank_val)
                    except (TypeError, ValueError):
                        continue
                    if top_rank is None or rank_num < top_rank or (rank_num == top_rank and term < top_term):
                        top_rank = rank_num
                        top_term = term
                if not top_term:
                    continue
                count = len(set(hits_map.get(top_term, [])))
                rows.append([
                    overall_start,
                    overall_end,
                    str(top_term),
                    str(count),
                    city_display,
                    str(lat_f) if lat_f is not None else '',
                    str(lon_f) if lon_f is not None else '',
                ])
        else:
            rank_scope = ''
            time_scope = str(collocate_rank_time_key or '').strip()
            if collocate_rank_term_scope and str(collocate_rank_term_scope).strip().lower().startswith('time') and time_scope:
                rank_scope = time_scope
            rank_map = (rank_index.get(c_key) or {}).get(rank_scope) or {}
            if not rank_map:
                continue
            if not collocate_terms_list:
                term_iter = sorted(rank_map.items(), key=lambda item: item[1])
            else:
                term_iter = [(term, rank_map.get(term)) for term in collocate_terms_list if term in rank_map]
            allowed = None
            if rank_scope:
                dataset_entry_tb = dataset_entry.get('time_bins') if isinstance(dataset_entry.get('time_bins'), dict) else {}
                info = dataset_entry_tb.get(rank_scope)
                if info:
                    allowed = set()
                    for idx in info.get('indexes', []):
                        try:
                            allowed.add(int(idx))
                        except (TypeError, ValueError):
                            continue
                start_dt, end_dt = bin_ranges.get(rank_scope, (range_start, range_end))
            else:
                start_dt = range_start
                end_dt = range_end

            for term, rank_val in term_iter:
                if rank_val is None:
                    continue
                try:
                    rank_num = int(rank_val)
                except (TypeError, ValueError):
                    continue
                hits = set(hits_map.get(term, []))
                if allowed is not None:
                    count = len([idx for idx in hits if idx in allowed])
                else:
                    count = len(hits)
                rows.append([
                    overall_start if start_dt is None else _format_date(start_dt),
                    overall_end if end_dt is None else _format_date(end_dt),
                    str(term),
                    str(rank_num),
                    str(count),
                    city_display,
                    str(lat_f) if lat_f is not None else '',
                    str(lon_f) if lon_f is not None else '',
                ])

    if not rows:
        return None

    headers_top = ['Time Start', 'Time End', 'Top Collocate', 'Article Count', 'City', 'Latitude', 'Longitude']
    headers_rank = ['Start Date', 'End Date', 'Collocate', 'Collocate Rank', 'Article Count', 'City', 'Latitude', 'Longitude']
    headers = headers_top if collocate_map_variant == 'top_term' else headers_rank

    with open(out_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    return out_path

def _parse_date(raw: Any) -> Optional[datetime]:
    """Parse a date string from properties['date'] into a datetime (UTC naive)."""
    if not raw:
        return None
    s = str(raw).strip()
    # Strip time if ISO
    if "T" in s:
        s = s.split("T", 1)[0]
    # Try multiple formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)  # last resort
    except Exception:
        return None


def _format_date_str(raw: Any) -> str:
    """Return a YYYY-MM-DD string if possible, else original string."""
    dt = _parse_date(raw)
    if dt:
        return dt.strftime("%Y-%m-%d")
    s = (str(raw).strip()) if raw else ""
    if "T" in s:
        s = s.split("T", 1)[0]
    return s


def _first_line_excerpt(article: Any, max_chars: int = 75) -> str:
    """Return the first line trimmed to max_chars with ellipsis if needed."""
    if not article:
        return ""
    first = str(article).splitlines()[0].strip()
    if len(first) <= max_chars:
        return first
    return first[: max_chars - 1].rstrip() + "…"


def _esc(value: Any) -> str:
    """HTML-escape arbitrary user data for popup/table output."""
    if value is None:
        return ""
    escaped = html.escape(str(value))
    # Prevent Jinja from treating brace sequences like {{ }} or {% %} as template tags
    return escaped.replace('{', '&#123;').replace('}', '&#125;')


def _sanitize_element_id(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw or "")
    safe = safe.strip("_")
    if not safe:
        safe = f"id_{uuid.uuid4().hex}"
    return safe


def _slug(value: Any) -> str:
    text = str(value or '').strip().lower()
    if not text:
        return 'value'
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text or 'value'


def _highlight_term(text: str, term: Optional[str]) -> str:
    if not text:
        return ""
    term_str = (term or "").strip()
    if not term_str:
        return _esc(text)
    pattern = re.compile(re.escape(term_str), re.IGNORECASE)
    parts = []
    last = 0
    for match in pattern.finditer(text):
        parts.append(_esc(text[last:match.start()]))
        parts.append(f"<mark>{_esc(match.group(0))}</mark>")
        last = match.end()
    parts.append(_esc(text[last:]))
    return ''.join(parts)


def _keyword_snippet(text: Any, term: Any, window_chars: int = 60) -> str:
    """Return +-window_chars characters surrounding the keyword if present."""
    if not text or not term:
        return ""
    term_str = str(term).strip()
    if not term_str:
        return ""
    text_str = str(text)
    match = re.search(re.escape(term_str), text_str, re.IGNORECASE)
    if not match:
        return ""
    start_idx = max(0, match.start() - window_chars)
    end_idx = min(len(text_str), match.end() + window_chars)
    snippet_core = text_str[start_idx:end_idx].strip()
    snippet_core = re.sub(r"\s+", " ", snippet_core)
    if not snippet_core:
        return ""
    snippet = snippet_core
    if start_idx > 0:
        snippet = '…' + snippet
    if end_idx < len(text_str):
        snippet = snippet + '…'
    return _highlight_term(snippet, term_str)


def _count_term_occurrences(text: Any, term: Optional[str]) -> int:
    if not text or not term:
        return 0
    term_str = term.strip()
    if not term_str:
        return 0
    return len(re.findall(re.escape(term_str), str(text), re.IGNORECASE))


def _word_count(text: Any) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))


def _truncate_plain_text(text: Any, max_chars: int = 420) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars].rstrip() + "…"


def _article_excerpt(text: Any, term: Any, max_chars: int = 1200) -> str:
    if not text:
        return ""
    term_str = str(term).strip() if term else ""
    text_str = re.sub(r"\s+", " ", str(text)).strip()
    if max_chars and len(text_str) > max_chars:
        text_str = text_str[:max_chars].rstrip() + '…'
    return _highlight_term(text_str, term_str)


def _compute_group_stats(entries: List[Dict[str, Any]], search_term: Optional[str]) -> Dict[str, float]:
    stats = {
        'article_count': len(entries),
        'page_count': 0,
        'key_term_frequency': 0,
        'word_count': 0,
    }
    pages = set()
    term = (search_term or '').strip() or None
    for entry in entries:
        props = entry.get('props') or {}
        page = props.get('page')
        if page:
            pages.add(str(page))
        article_text = props.get('article') or ''
        stats['word_count'] += _word_count(article_text)
        if term:
            stats['key_term_frequency'] += _count_term_occurrences(article_text, term)
    stats['page_count'] = len(pages) if pages else len(entries)
    return stats


def _compute_group_value(stats: Dict[str, float], metric: str, normalize: bool, denominator: Optional[str]) -> float:
    value = float(stats.get(metric, 0))
    if normalize and denominator:
        denom_value = float(stats.get(denominator, 0))
        if denom_value <= 0:
            return 0.0
        value = value / denom_value
    return max(value, 0.0)


def _format_metric_value(metric: str, value: Optional[float], normalized: bool) -> str:
    if value is None:
        return 'n/a'
    if metric in ('article_count', 'page_count') and not normalized:
        return f"{int(round(value)):,}"
    return f"{value:.4f}"


def _detect_search_term(geojson_path: str, data: Dict[str, Any]) -> str:
    meta = data.get('metadata') or {}
    for key in ('search_term', 'SearchTerm', 'term', 'Term'):
        val = meta.get(key)
        if val:
            return str(val)

    for feat in data.get('features', []) or []:
        props = feat.get('properties') or {}
        for key in ('search_term', 'SearchTerm', 'term', 'Term'):
            val = props.get(key)
            if val:
                return str(val)

    base = os.path.basename(geojson_path or '')
    name, _ = os.path.splitext(base)
    ignore = {'occurrences', 'merged', 'heatmap', 'points', 'graduated', 'attributes', 'create', 'map'}
    for token in name.split('_'):
        if not token:
            continue
        lowered = token.lower()
        if lowered in ignore:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            continue
        return token
    return ''


def _unit_to_keyword(unit: str) -> str:
    """Normalize unit to one of: day/week/month/year."""
    u = (unit or "").strip().lower()
    if u.startswith("day"):
        return "day"
    if u.startswith("week"):
        return "week"
    if u.startswith("month"):
        return "month"
    if u.startswith("year"):
        return "year"
    return "month"


def _add_step(dt: datetime, unit: str, step: int) -> datetime:
    """Increment datetime dt by 'step' units of 'day, week, month, year'."""
    unit = _unit_to_keyword(unit)
    step = max(1, int(step or 1))
    if unit == "day":
        return dt + timedelta(days=step)
    if unit == "week":
        return dt + timedelta(weeks=step)
    if unit == "month":
        # naive month add
        y = dt.year
        m = dt.month + step
        y += (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = min(dt.day, [31,
                         29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                         31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
        return dt.replace(year=y, month=m, day=d)
    if unit == "year":
        try:
            return dt.replace(year=dt.year + step)
        except ValueError:
            # Feb 29 to Feb 28 fallback
            if dt.month == 2 and dt.day == 29:
                return dt.replace(year=dt.year + step, day=28)
            raise
    return dt


def _build_time_index(min_dt: datetime, max_dt: datetime, unit: str, step: int) -> List[datetime]:
    """Build a list of datetimes from min_dt to >= max_dt stepping unit/step."""
    idx: List[datetime] = []
    cur = min_dt
    while cur <= max_dt:
        idx.append(cur)
        cur = _add_step(cur, unit, step)
    if len(idx) < 2:
        idx.append(_add_step(min_dt, unit, step))
    return idx


# ----------------------------
# Core
# ----------------------------

def _extract_points(features: List[Dict[str, Any]], cancel_event: Optional[threading.Event] = None) -> List[Dict[str, Any]]:
    """
    Extract and validate points/features from GeoJSON features.
    Returns list of dicts with: lat, lon, props(dict), date(dt)
    """
    out: List[Dict[str, Any]] = []
    for feat in features:
        _check_cancel(cancel_event)
        if not feat or not isinstance(feat, dict):
            continue
        geometry = feat.get("geometry")
        if not geometry or not isinstance(geometry, dict):
            continue
        coords = geometry.get("coordinates")
        if not coords or not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        if lon is None or lat is None:
            continue
        try:
            latf = float(lat)
            lonf = float(lon)
        except (TypeError, ValueError):
            continue

        props = feat.get("properties") or {}
        dtx = _parse_date(props.get("date"))
        out.append({"lat": latf, "lon": lonf, "props": props, "date": dtx})
    return out


def _popup_html(
    props: Dict[str, Any],
    search_term: Optional[str],
    popup_id: str,
    *,
    lightweight: bool = False,
) -> str:
    summary = _first_line_excerpt(props.get("article") or "", 160)
    snippet_html = _keyword_snippet(props.get("article"), search_term)
    url = (props.get("url") or "").strip()
    pdf_url = url.replace(".jp2", ".pdf") if url else ""

    date_str = _format_date_str(props.get("date") or "")
    newspaper = (props.get("Title") or props.get("newspaper_name") or "").strip()
    city = (props.get("City") or "").strip()
    state = (props.get("State") or "").strip()
    location = ", ".join([p for p in (city, state) if p])
    page = (props.get("page") or "").strip()

    show_more_id = _sanitize_element_id(f"{popup_id}-more")

    meta_rows = []
    if date_str:
        meta_rows.append(f'<div><span style="font-weight:600;">Date:</span> {_esc(date_str)}</div>')
    if newspaper:
        meta_rows.append(f'<div><span style="font-weight:600;">Newspaper:</span> {_esc(newspaper)}</div>')
    if location:
        meta_rows.append(f'<div><span style="font-weight:600;">Place:</span> {_esc(location)}</div>')
    if page:
        meta_rows.append(f'<div><span style="font-weight:600;">Page:</span> {_esc(page)}</div>')

    lines: List[str] = []
    if summary:
        lines.append(f'<div style="margin-bottom:4px;">{_esc(summary)}</div>')
    if snippet_html:
        lines.append(
            '<div style="margin-bottom:4px;"><span style="font-weight:600;">Context:</span> '
            f'{snippet_html}</div>'
        )
    if pdf_url:
        lines.append(
            f'<div><a href="{_esc(pdf_url)}" target="_blank" rel="noopener">Source Image</a></div>'
        )

    article_html = ""
    if not lightweight:
        article_html = _article_excerpt(props.get("article"), search_term, max_chars=1200)

    if meta_rows or article_html:
        button = (
            f"<button type=\"button\" style=\"margin-top:6px;\" "
            f"onclick=\"var el=document.getElementById('{show_more_id}');"
            "if(!el){return;}var hidden=el.style.display==='none';"
            "el.style.display=hidden?'block':'none';"
            "this.textContent=hidden?'Show less':'Show more';\">Show more</button>"
        )
        lines.append(button)
        hidden_html = ''.join(meta_rows)
        if article_html:
            hidden_html += f'<div style="margin-top:6px;">{article_html}</div>'
        lines.append(
            f'<div id="{show_more_id}" style="display:none; margin-top:6px; max-height:200px; overflow:auto;">'
            f'{hidden_html}</div>'
        )

    if not lines:
        lines.append('<div>No additional information available.</div>')

    return '<div style="font-size:14px; line-height:1.25;">' + "\n".join(lines) + "</div>"


def _feature_label(props: Dict[str, Any]) -> str:
    date_str = _format_date_str(props.get("date") or "")
    newspaper = (props.get("Title") or props.get("newspaper_name") or "").strip()
    city = (props.get("City") or "").strip()
    state = (props.get("State") or "").strip()
    location = ", ".join([p for p in (city, state) if p])
    pieces = [part for part in (date_str, newspaper) if part]
    if location:
        pieces.append(location)
    if not pieces:
        headline = (props.get("headline") or props.get("Headline") or "").strip()
        if headline:
            pieces.append(headline)
    label = " — ".join(pieces)
    return label or "Feature"


def _entry_payload(
    entry: Dict[str, Any],
    search_term: Optional[str],
    *,
    embed_article: bool = True,
    lightweight: bool = False,
) -> Dict[str, Any]:
    props = entry.get('props') or {}
    article_text = props.get('article')
    if not article_text:
        article_text = entry.get('_article_full', '') or ''
    first_line = _first_line_excerpt(article_text, 160)
    snippet_html = _keyword_snippet(article_text, search_term)
    url_val = (props.get('url') or '').strip()
    pdf_url = url_val.replace('.jp2', '.pdf') if url_val else ''
    date_val = _format_date_str(props.get('date') or '')
    newspaper_val = (props.get('Title') or props.get('newspaper_name') or '').strip()
    city_val = (props.get('City') or '').strip()
    state_val = (props.get('State') or '').strip()
    place_val = ', '.join([p for p in (city_val, state_val) if p])

    payload = {
        'first_line': _esc(first_line),
        'context': '' if lightweight else (snippet_html or ''),
        'pdf_url': _esc(pdf_url) if pdf_url else '',
        'date': _esc(date_val),
        'newspaper': _esc(newspaper_val),
        'place': _esc(place_val),
        'page': _esc((props.get('page') or '').strip()),
        'article_html': _article_excerpt(article_text, search_term, max_chars=3000)
        if (embed_article and article_text)
        else '',
        'article_preview': (
            '' if lightweight else (_article_excerpt(article_text, search_term, max_chars=600) if article_text else '')
        ),
    }
    label_value = _feature_label(props)
    if label_value:
        payload['label'] = label_value
    row_id = entry.get('_popup_row_id')
    if row_id:
        payload['attr_row_id'] = row_id
    return payload


def _group_header(entries: List[Dict[str, Any]], stats: Dict[str, Any], search_term: Optional[str]) -> Tuple[str, int, str]:
    first_props = entries[0].get('props', {}) if entries else {}
    city_name = (first_props.get('City') or '').strip() or 'this location'
    term_text = (search_term or '').strip()
    article_count = int(stats.get('article_count', len(entries))) if stats else len(entries)
    if term_text:
        title_text = f'Articles mentioning "{term_text}" in {city_name}'
    else:
        title_text = f'Articles in {city_name}'
    return title_text, article_count, city_name


def _group_popup_html(
    group: Dict[str, Any],
    search_term: Optional[str],
    group_id: str,
    *,
    lightweight: bool = False,
    title_text: Optional[str] = None,
    article_count: Optional[int] = None,
) -> str:
    entries: List[Dict[str, Any]] = group.get('entries') or []
    if not entries:
        return '<div style="font-size:14px;">No data available.</div>'

    stats = group.get('stats') or {}
    metric_key = group.get('metric_key')
    metric_display = group.get('metric_display')
    metric_norm_display = group.get('metric_normalized_display')
    normalized = bool(group.get('normalized'))
    denominator_label = group.get('denominator_label')
    metric_value = group.get('value')
    raw_metric_value = stats.get(metric_key) if stats else None

    header_title, header_articles, _ = _group_header(entries, stats, search_term)
    if title_text is None:
        title_text = header_title
    if article_count is None:
        article_count = header_articles

    metric_lines: List[str] = []
    if metric_key:
        if normalized:
            formatted = _format_metric_value(metric_key, metric_value, True)
            label = metric_norm_display or metric_display or 'Metric'
            metric_lines.append(f"{label}: {formatted}")
            if raw_metric_value is not None:
                metric_lines.append(
                    f"Raw {metric_display or 'value'}: {_format_metric_value(metric_key, raw_metric_value, False)}"
                )
            if denominator_label:
                metric_lines.append(f"Normalized by {denominator_label} per city")
        else:
            if metric_key == 'article_count':
                metric_lines = []
            else:
                formatted = _format_metric_value(metric_key, metric_value, False)
                label = metric_display or 'Metric'
                metric_lines.append(f"{label}: {formatted}")

    nav_buttons_html = (
        '<div data-nav-controls style="display:flex; align-items:center; gap:6px;">'
        '<button type="button" data-step="-1" class="collocate-time-button" '
        'title="Previous article" aria-label="Previous article">‹</button>'
        '<button type="button" data-step="1" class="collocate-time-button" '
        'title="Next article" aria-label="Next article">›</button>'
        '</div>'
    )

    select_attrs = ['data-map-select', 'style="width:100%;"']
    options_html = ''
    if lightweight:
        select_attrs.append('data-options-source="json"')
    else:
        option_parts = []
        for idx, entry in enumerate(entries):
            props = entry.get('props', {})
            label = _feature_label(props)
            option_parts.append(f'<option value="{idx}">{_esc(label)}</option>')
        options_html = "".join(option_parts)

    select_html = '<select ' + ' '.join(select_attrs) + '>' + options_html + '</select>'

    select_block = f'<div style="margin-top:6px;">{select_html}</div>'
    footer_html = (
        '<div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center; gap:6px;">'
        '<span data-location-progress style="font-size:12px; color:#555;"></span>'
        '<div style="display:flex; align-items:center; gap:6px;">'
        '<span data-article-progress style="font-size:12px; color:#555;"></span>'
        f'{nav_buttons_html}'
        '</div>'
        '</div>'
    )

    header_html = (
        '<div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">'
        f'<div data-popup-header style="margin-bottom:4px; font-weight:600;">{_esc(title_text)}</div>'
        '<div data-popup-actions style="display:flex; align-items:center; gap:4px;">'
        '<button type="button" data-pin-toggle="1" '
        'style="border:none; background:none; cursor:pointer; padding:0; font-size:16px; line-height:1; color:#2b6cb0;" '
        'title="Pin popup">📌</button>'
        '<button type="button" data-dock-toggle="1" '
        'style="border:none; background:none; cursor:pointer; padding:0; font-size:16px; line-height:1; color:#2b6cb0;" '
        'title="Dock popup">⧉</button>'
        '</div>'
        '</div>'
    )

    container_html = '<div data-detail-container style="margin-top:6px; font-size:14px; line-height:1.3; min-height:120px;">Loading…</div>'

    return (
        f'<div data-popup-root="1" data-group-id="{_esc(group_id)}" '
        f'data-total-entries="{len(entries)}" '
        'data-docked="0" '
        'style="font-size:14px; line-height:1.25; min-width:260px; position:relative; padding-right:48px; box-sizing:border-box; '
        'max-height: calc(100vh - 200px); overflow-y:auto;">'
        f'{header_html}'
        f'{select_block}'
        f'{container_html}'
        f'{footer_html}'
        '</div>'
    )


def _group_points(points: List[Dict[str, Any]], precision: int = 6, cancel_event: Optional[threading.Event] = None) -> List[Dict[str, Any]]:
    """Group point dictionaries by rounded lat/lon for shared popups."""
    grouped_map: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for pt in points:
        _check_cancel(cancel_event)
        key = (round(pt["lat"], precision), round(pt["lon"], precision))
        grouped_map.setdefault(key, []).append(pt)

    groups: List[Dict[str, Any]] = []
    for idx, (key, entries) in enumerate(grouped_map.items()):
        _check_cancel(cancel_event)
        groups.append({
            'location': key,
            'entries': entries,
            'id': _sanitize_element_id(f"group-{idx}"),
        })
    return groups


def _add_point_markers(
    map_obj: folium.Map,
    groups: List[Dict[str, Any]],
    search_term: Optional[str],
    radius_func: Callable[[Dict[str, Any]], float],
    popup_width: int = 360,
    lightweight: bool = False,
    popup_dataset: Optional[Dict[str, Any]] = None,
    *,
    ghost_markers: bool = False,
) -> None:
    """Add grouped point markers with selection popups to the map."""
    for idx, group in enumerate(groups):
        entries = group.get('entries') or []
        if not entries:
            continue
        group_id = group.get('id') or f'group-{idx}'
        dataset_entry = popup_dataset.get(group_id) if isinstance(popup_dataset, dict) else None
        override_title = dataset_entry.get('title') if isinstance(dataset_entry, dict) else None
        override_articles = dataset_entry.get('article_count') if isinstance(dataset_entry, dict) else None
        popup_html_str = _group_popup_html(
            group,
            search_term,
            group_id,
            lightweight=lightweight,
            title_text=override_title,
            article_count=override_articles,
        )
        if isinstance(dataset_entry, dict):
            dataset_entry.setdefault('template', popup_html_str)
        popup_obj = Popup(Html(popup_html_str, script=True), max_width=popup_width)
        lat, lon = group.get('location', (entries[0]['lat'], entries[0]['lon']))
        radius = radius_func(group) if callable(radius_func) else 4.0
        try:
            radius_value = max(1.0, float(radius))
        except (TypeError, ValueError):
            radius_value = 4.0
        if ghost_markers:
            radius_value = max(radius_value, 8.0)
        marker_kwargs = {
            'location': [lat, lon],
            'radius': radius_value,
            'weight': 0 if ghost_markers else 1,
            'opacity': 0.0 if ghost_markers else 1.0,
            'fill': True,
            'fill_opacity': 0.0 if ghost_markers else 0.85,
            'color': '#2b6cb0',
            'fill_color': '#2b6cb0',
            'interactive': True,
        }
        if popup_obj is not None:
            marker_kwargs['popup'] = popup_obj
        marker = folium.CircleMarker(
            **marker_kwargs,
        )
        try:
            metric_val = float(group.get('value', 0.0))
        except (TypeError, ValueError):
            metric_val = 0.0
        marker.options.update({
            'groupId': group_id,
            'metricValue': metric_val,
            'baseOpacity': marker.options.get('opacity', 0.0 if ghost_markers else 1.0),
            'baseFillOpacity': marker.options.get('fillOpacity', 0.0 if ghost_markers else 0.85),
            'ghostMarker': bool(ghost_markers),
        })
        marker.add_to(map_obj)


def _heat_slices(
    points: List[Dict[str, Any]],
    time_index: List[datetime],
    linger_unit: str,
    linger_step: int
) -> List[List[List[float]]]:
    """
    Build HeatMapWithTime slices: list where each entry is a list of [lat, lon] for that time slice.
    We include a point in slices from its date slice up to linger length.
    """
    slices: List[List[List[float]]] = [[] for _ in time_index]

    for p in points:
        dt = p["date"]
        if dt is None:
            continue

        # map dt to a slice index (first index with time >= dt)
        insert_i = 0
        for i, t in enumerate(time_index):
            if dt >= t:
                insert_i = i
            else:
                break

        # compute linger end as dt + linger duration (approx)
        linger_end = dt
        if int(linger_step or 0) > 0:
            linger_end = _add_step(dt, linger_unit, int(linger_step))

        for j, t in enumerate(time_index):
            if j < insert_i:
                continue
            if int(linger_step or 0) > 0:
                if t >= linger_end:
                    break
            else:
                if j != insert_i:
                    break
            point_entry: List[float] = [p["lat"], p["lon"]]
            weight = p.get('value')
            try:
                weight_val = float(weight) if weight is not None else None
            except (TypeError, ValueError):
                weight_val = None
            if weight_val is not None and weight_val > 0:
                point_entry.append(weight_val)
            slices[j].append(point_entry)

    return slices


def _assign_time_bins(
    dt: Optional[datetime],
    time_index: List[datetime],
    linger_unit: str,
    linger_step: int
) -> List[int]:
    if dt is None or not time_index:
        return []

    insert_i = 0
    for i, t in enumerate(time_index):
        if dt >= t:
            insert_i = i
        else:
            break

    indices: List[int] = []
    linger_duration = max(0, int(linger_step or 0))
    linger_end = dt
    if linger_duration > 0:
        linger_end = _add_step(dt, linger_unit, linger_duration)

    for j, t in enumerate(time_index):
        if j < insert_i:
            continue
        if linger_duration > 0:
            if t >= linger_end:
                break
            indices.append(j)
        else:
            if j == insert_i:
                indices.append(j)
                break

    return indices


def _graduated_radius_resolver(groups: List[Dict[str, Any]], min_radius: float, max_radius: float):
    values = [float(g.get('value', 0.0)) for g in groups if g.get('value') is not None]
    if not values:
        return lambda _group: min_radius
    min_val = min(values)
    max_val = max(values)
    if max_val <= min_val:
        return lambda _group: max_radius

    span = max_val - min_val

    def _resolver(group: Dict[str, Any]) -> float:
        val = float(group.get('value', 0.0))
        scale = (val - min_val) / span
        radius = min_radius + scale * (max_radius - min_radius)
        return max(min_radius, min(max_radius, radius))

    return _resolver


def _write_attribute_table(
    points: List[Dict[str, Any]],
    out_path: str,
    max_rows: Optional[int] = None,
    omit_article: bool = False,
    include_columns: Optional[List[str]] = None,
    hyperlink_columns: Optional[List[str]] = None,
) -> Optional[str]:
    """Render a simple HTML attribute table for the supplied points."""
    columns: List[str] = ["Latitude", "Longitude"]
    seen = {"Latitude", "Longitude"}
    drop_props = set()
    if include_columns:
        ordered_cols: List[str] = []
        for col in include_columns:
            col_str = str(col)
            if omit_article and col_str == 'article':
                continue
            if col_str not in seen:
                seen.add(col_str)
                ordered_cols.append(col_str)
        columns.extend(ordered_cols)
    else:
        if omit_article:
            drop_props.add('article')
        for entry in points:
            props = entry.get("props") if isinstance(entry, dict) else None
            if not isinstance(props, dict):
                continue
            for key in props.keys():
                key_str = str(key)
                if key_str in drop_props:
                    continue
                if key_str not in seen:
                    seen.add(key_str)
                    columns.append(key_str)

    link_set = {str(col) for col in (hyperlink_columns or [])}

    for entry in points:
        if isinstance(entry, dict):
            entry['_attr_in_table'] = False

    def _attr_string(pairs: List[Tuple[str, Any]]) -> str:
        parts = []
        for name, value in pairs:
            if value is None:
                continue
            parts.append(f'{name}="{html.escape(str(value), quote=True)}"')
        return (' ' + ' '.join(parts)) if parts else ''

    def _td(content: str, attrs: Optional[List[Tuple[str, Any]]] = None) -> str:
        return f'<td{_attr_string(attrs or [])}>{content}</td>'

    rows: List[str] = []
    truncated = False
    for idx, entry in enumerate(points):
        if max_rows is not None and idx >= max_rows:
            truncated = True
            break
        props = entry.get("props") if isinstance(entry, dict) else {}
        if not isinstance(props, dict):
            props = {}
        lat_cell = _esc(_format_coord(entry.get("lat")))
        lon_cell = _esc(_format_coord(entry.get("lon")))
        row_attrs: List[Tuple[str, Any]] = []
        row_id = entry.get('_popup_row_id')
        if row_id:
            row_attrs.append(('data-entry-key', row_id))
            row_attrs.append(('id', row_id))

        cells = [_td(lat_cell), _td(lon_cell)]
        full_article_text = entry.get('_article_full', '') if isinstance(entry, dict) else ''
        for key in columns[2:]:
            value = props.get(key, '')
            cell_html: str
            if key == 'article':
                base_text = str(value or '').strip()
                if not base_text and full_article_text:
                    base_text = str(full_article_text)
                display_text = _truncate_plain_text(base_text, max_chars=420)
                attrs_list: List[Tuple[str, Any]] = [('data-column', 'article')]
                if base_text:
                    attrs_list.append(('data-article-text', base_text))
                cell_html = _esc(display_text)
                cells.append(_td(cell_html, attrs_list))
                continue
            elif key in link_set and value:
                href = html.escape(str(value), quote=True)
                text = html.escape(str(value))
                cell_html = f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'
            else:
                cell_html = _esc(value)
            cells.append(_td(cell_html))
        rows.append(f'<tr{_attr_string(row_attrs)}>' + ''.join(cells) + '</tr>')
        if isinstance(entry, dict):
            entry['_attr_in_table'] = True

    if not rows:
        rows.append(
            '<tr><td colspan="{}">{}</td></tr>'.format(
                len(columns),
                _esc("No point data available."),
            )
        )
    elif truncated:
        rows.append(
            '<tr><td colspan="{}">{}</td></tr>'.format(
                len(columns),
                _esc(f"Output truncated to the first {max_rows} rows in lightweight mode."),
            )
        )

    head_cells = ''.join(f'<th>{_esc(col)}</th>' for col in columns)
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="utf-8" />',
        '<title>Attribute Table</title>',
        '<style>',
        'body { font-family: Arial, Helvetica, sans-serif; margin: 16px; background: #fafafa; }',
        'h2 { margin-top: 0; }',
        '.table-wrapper { max-height: 75vh; overflow: auto; border: 1px solid #d9d9d9; background: #fff; }',
        'table { border-collapse: collapse; width: 100%; font-size: 14px; }',
        'th, td { border: 1px solid #d0d0d0; padding: 4px 6px; text-align: left; }',
        'thead th { position: sticky; top: 0; background: #f3f6fa; z-index: 1; }',
        'tbody tr:nth-child(odd) { background: #f9fbfd; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h2>Attribute Table ({len(points)} features)</h2>',
        '<div class="table-wrapper">',
        '<table>',
        f'<thead><tr>{head_cells}</tr></thead>',
        '<tbody>',
        ''.join(rows),
        '</tbody>',
        '</table>',
        '</div>',
        '</body>',
        '</html>',
    ]

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
    except OSError:
        return None
    return out_path


def _format_coord(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


class _ZoomTopRight(MacroElement):
    _template = JinjaTemplate(
        """
        {% macro script(this, kwargs) %}
        {{this._parent.get_name()}}.zoomControl.setPosition('topright');
        {% endmacro %}
        """
    )


# ----------------------------
# Public API
# ----------------------------

def create_map(
    geojson_path: str,
    mode: str = "points",
    time_unit: str = "month",
    time_step: int = 1,
    linger_unit: str = "week",
    linger_step: int = 0,
    disable_time: bool = False,
    heat_radius: Optional[int] = None,
    heat_value: Optional[float] = None,
    grad_min_radius: Optional[int] = None,
    grad_max_radius: Optional[int] = None,
    metric: Optional[str] = None,
    normalize: bool = False,
    normalize_denominator: Optional[str] = None,
    lightweight: bool = False,
    table_mode: str = "full",
    table_row_limit: Optional[int] = None,
    collocate_rank_mode: bool = False,
    collocate_drop_stopwords: bool = False,
    collocate_window: int = 5,
    collocate_drop_terms: Optional[List[str]] = None,
    collocate_term_groups: Optional[List[dict]] = None,
    collocate_rank_top_n: int = COLLOCATE_RANK_LIMIT,
    collocate_rank_term_scope: str = 'global',
    collocate_rank_time_key: Optional[str] = None,
    collocate_rank_focus: str = 'all',
    collocate_rank_focus_city: Optional[str] = None,
    collocate_rank_focus_state: Optional[str] = None,
    collocate_rank_time_label: Optional[str] = None,
    collocate_rank_focus_label: Optional[str] = None,
    collocate_rank_colorize: bool = False,
    collocate_time_slider: bool = False,
    collocate_rank_terms: Optional[List[str]] = None,
    collocate_map_variant: str = 'rank',
    metadata_enabled: bool = False,
    project_dir: Optional[str] = None,
    time_start_override: Optional[str] = None,
    time_end_override: Optional[str] = None,
    collocate_export_csv: bool = False,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Optional[str]]:
    """
    Create a leaflet map next to the GeoJSON.

    Modes:
      - "points": static points (CircleMarker dots) with popups.
      - "cluster": clustered point markers that expand on zoom/click.
      - "heatmap": heat density view. Uses a time slider if dates are present and time is enabled.
      - "graduated": scaled circle markers sized by the chosen metric.

    Time slider parameters (heatmap mode only):
      - time_unit: 'day' | 'week' | 'month' | 'year'   (default 'month')
      - time_step: integer step for the slider increments (default 1)
      - linger_unit: same options as time_unit (default 'week')
      - linger_step: how long (in linger_unit) a point remains visible after its date (default 0)

    Additional options:
      - disable_time: force a static heat layer even when dates are available.
      - heat_radius: override the radius for the heatmap kernel (default 15).
      - heat_value: multiplier applied to heatmap weights (default 1.0).
      - grad_min_radius / grad_max_radius: radius range for graduated markers.
      - metric: 'article_count' | 'page_count' | 'key_term_frequency'.
      - normalize: divide the metric by a denominator when True.
      - normalize_denominator: 'word_count' | 'article_count' | 'page_count'.
      - lightweight: reduce popup detail and attribute table size for very large outputs.
      - table_mode: 'full' | 'article' | 'minimal' – controls attribute table columns.
      - table_row_limit: optional max rows in attribute table (None/<=0 for all rows).
      - collocate_rank_mode / collocate_drop_stopwords / collocate_window / collocate_drop_terms: configure
        lightweight collocate rank visualisation on point maps.
      - collocate_term_groups: optional grouping definitions applied to collocate terms (list of
        dictionaries with "name" and "terms" entries).
      - collocate_rank_top_n: limit collocate list and rank output to the top-N terms (default 150).
      - collocate_rank_term_scope: 'global' to rank across entire period, 'time' to use a specific time bin.
      - collocate_rank_time_key: time-bin key (1-based index) when collocate_rank_term_scope='time'.
      - collocate_rank_focus: 'all' (default), 'city', or 'state' to control which locations determine the top terms.
      - collocate_rank_focus_city / collocate_rank_focus_state: labels used when collocate_rank_focus filters by city or state.
      - collocate_rank_colorize: when True, apply a graduated color ramp based on collocate article counts.
      - collocate_rank_terms: optional explicit list of collocate terms to include (overrides top-N when provided).
      - collocate_time_slider: when True, expose an interactive time slider for collocate views (points mode only).
      - collocate_map_variant: 'rank' (default) for the ranked-term explorer, or 'top_term' to show each location's top term.
      - time_start_override / time_end_override: optional ISO dates to force the slider/time-bin extent
        to respect user-selected parameters when metadata is missing.
      - collocate_export_csv: when True, write a CSV summarizing collocate data per location.

    Returns:
        dict with 'map_path' and optional 'attribute_table'.
    """

    permitted_modes = {"points", "heatmap", "graduated", "cluster"}
    mode_normalized = (mode or "points").strip().lower()
    if mode_normalized not in permitted_modes:
        mode_normalized = "points"
    mode = mode_normalized

    variant_norm = (collocate_map_variant or 'rank').strip().lower()
    if variant_norm not in {'rank', 'top_term'}:
        variant_norm = 'rank'
    collocate_map_variant = variant_norm
    top_term_variant = collocate_map_variant == 'top_term'

    _check_cancel(cancel_event)

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features") or []
    if not isinstance(features, list):
        raise ValueError("GeoJSON does not contain a valid 'features' list.")

    pts = _extract_points(features, cancel_event=cancel_event)
    _check_cancel(cancel_event)
    for entry in pts:
        _check_cancel(cancel_event)
        props = entry.get('props') or {}
        entry['_article_full'] = props.get('article') or ''
    for idx, entry in enumerate(pts):
        _check_cancel(cancel_event)
        entry['_popup_row_id'] = _sanitize_element_id(f'feature-row-{idx}')
    groups = _group_points(pts, cancel_event=cancel_event)
    _check_cancel(cancel_event)
    search_term = _detect_search_term(geojson_path, data)

    start_override = (time_start_override or '').strip()
    end_override = (time_end_override or '').strip()

    city_entries_map: Dict[str, List[Dict[str, Any]]] = {}

    allowed_metrics = {"article_count", "page_count", "key_term_frequency"}
    metric_key = (metric or "article_count").strip().lower()
    if metric_key not in allowed_metrics:
        metric_key = "article_count"

    metric_definitions = {
        'article_count': {
            'metric_display': 'Articles',
            'normalized_display': 'Articles / Total Articles',
            'denominator': 'article_count',
            'denom_label': 'total articles',
        },
        'page_count': {
            'metric_display': 'Pages',
            'normalized_display': 'Pages / Total Pages',
            'denominator': 'page_count',
            'denom_label': 'total pages',
        },
        'key_term_frequency': {
            'metric_display': 'Term Frequency',
            'normalized_display': 'Term Frequency / Total Words',
            'denominator': 'word_count',
            'denom_label': 'total words',
        },
    }

    metric_info = metric_definitions.get(metric_key, metric_definitions['article_count'])
    metric_display = metric_info['metric_display']
    metric_normalized_display = metric_info['normalized_display']
    denom_label = metric_info['denom_label']

    normalize_flag = bool(normalize)
    denominator_key = metric_info['denominator'] if normalize_flag else None

    time_enabled = mode == 'heatmap' and not disable_time

    articles_count = len(pts)
    city_set: set = set()
    newspaper_ids: set = set()
    dates_dt: List[datetime] = []
    for p in pts:
        props = p.get('props', {})
        city = props.get('City')
        if city not in (None, ""):
            city_set.add(str(city))
        sn = props.get('SN') or props.get('lccn')
        if sn not in (None, ""):
            newspaper_ids.add(str(sn))
        else:
            title = props.get('Title') or props.get('newspaper_name')
            if title not in (None, ""):
                newspaper_ids.add(str(title))
        dt = p.get('date')
        if isinstance(dt, datetime):
            dates_dt.append(dt)

    metadata = data.get('metadata') or data.get('properties') or {}
    start_meta = start_override or metadata.get('start_date') or metadata.get('StartDate')
    end_meta = end_override or metadata.get('end_date') or metadata.get('EndDate')
    start_dt_meta = _parse_date(start_meta) if start_meta else None
    end_dt_meta = _parse_date(end_meta) if end_meta else None

    if dates_dt:
        min_dt = min(dates_dt)
        max_dt = max(dates_dt)
    else:
        min_dt = max_dt = None

    dated_pts = [p for p in pts if p.get("date") is not None]
    rank_time_bins = collocate_rank_mode and bool(dated_pts)
    use_time_slider = bool(dated_pts) and time_enabled
    time_index: List[datetime] = []
    time_labels: List[str] = []
    range_start = start_dt_meta or min_dt
    range_end = end_dt_meta or max_dt
    if range_start and range_end and range_start > range_end:
        range_start, range_end = range_end, range_start
    if (use_time_slider or rank_time_bins) and min_dt and max_dt:
        time_index = _build_time_index(range_start, range_end, time_unit, max(1, int(time_step or 1)))
        time_labels = [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in time_index]

    collocate_time_slider_enabled = bool(
        collocate_time_slider
        and collocate_rank_mode
        and time_index
        and len(time_index) > 1
    )
    collocate_time_bins_payload: List[Dict[str, str]] = []
    collocate_time_default_key: Optional[str] = None
    if collocate_time_slider_enabled:
        for idx, dt in enumerate(time_index):
            label_iso = time_labels[idx] if idx < len(time_labels) else dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            try:
                display_label = dt.strftime('%Y-%m-%d')
            except Exception:
                display_label = label_iso
            collocate_time_bins_payload.append({'key': str(idx + 1), 'label': display_label, 'iso': label_iso})
        if len(collocate_time_bins_payload) <= 1:
            collocate_time_slider_enabled = False
            collocate_time_bins_payload = []
        else:
            if collocate_rank_term_scope and collocate_rank_term_scope.strip().lower().startswith('time'):
                if collocate_rank_time_key:
                    candidate_key = str(collocate_rank_time_key).strip()
                    if any(bin_info['key'] == candidate_key for bin_info in collocate_time_bins_payload):
                        collocate_time_default_key = candidate_key
            if collocate_time_default_key is None and collocate_time_bins_payload:
                collocate_time_default_key = collocate_time_bins_payload[0]['key']

    start_str = start_meta or (min_dt.strftime('%Y-%m-%d') if min_dt else '')
    end_str = end_meta or (max_dt.strftime('%Y-%m-%d') if max_dt else '')
    if not start_str and end_str:
        start_str = end_str
    if not end_str and start_str:
        end_str = start_str
    date_range = (start_str, end_str) if start_str or end_str else ()

    popup_width = 320 if lightweight else 360

    start_date_for_metrics: Optional[date] = None
    end_date_for_metrics: Optional[date] = None
    if isinstance(range_start, datetime):
        start_date_for_metrics = range_start.date()
    elif isinstance(range_start, date):
        start_date_for_metrics = range_start
    if isinstance(range_end, datetime):
        end_date_for_metrics = range_end.date()
    elif isinstance(range_end, date):
        end_date_for_metrics = range_end
    if start_date_for_metrics is None and min_dt:
        start_date_for_metrics = min_dt.date()
    if end_date_for_metrics is None and max_dt:
        end_date_for_metrics = max_dt.date()

    embed_articles = not lightweight
    values: List[float] = []
    popup_dataset: Dict[str, Any] = {}
    for group in groups:
        _check_cancel(cancel_event)
        entries = group.get("entries", [])
        stats = _compute_group_stats(entries, search_term)
        group["stats"] = stats
        value = _compute_group_value(stats, metric_key, normalize_flag, denominator_key)
        group["value"] = value
        group["metric_key"] = metric_key
        group["metric_display"] = metric_display
        group["metric_normalized_display"] = metric_normalized_display
        group["normalized"] = normalize_flag
        group["denominator_label"] = denom_label if normalize_flag else ''

        location = group.get('location') or (
            (entries[0]['lat'], entries[0]['lon']) if entries else (0.0, 0.0)
        )
        try:
            loc_lat = float(location[0])
        except (TypeError, ValueError, IndexError):
            loc_lat = 0.0
        try:
            loc_lon = float(location[1])
        except (TypeError, ValueError, IndexError):
            loc_lon = 0.0

        entry_payloads = []
        year_match_counts: Dict[str, int] = {}
        for entry_idx, entry in enumerate(entries):
            _check_cancel(cancel_event)
            payload = _entry_payload(
                entry,
                search_term,
                embed_article=embed_articles,
                lightweight=lightweight,
            )
            payload['full_index'] = entry_idx
            entry_date = str(entry.get('date') or '').strip()
            if entry_date:
                year_token = entry_date[:4]
                if year_token.isdigit():
                    payload['dataset_year'] = year_token
                    year_match_counts[year_token] = year_match_counts.get(year_token, 0) + 1
                    try:
                        year_int = int(year_token)
                    except (TypeError, ValueError):
                        year_int = None
                    if year_int is not None:
                        start_for_year = start_date_for_metrics or date(year_int, 1, 1)
                        end_for_year = end_date_for_metrics or date(year_int, 12, 31)
                        try:
                            year_total_val = metric_total_for_year_within_dates(
                                year_int,
                                start_for_year,
                                end_for_year,
                                "article_count",
                            )
                        except Exception:
                            year_total_val = None
                        if year_total_val:
                            payload['dataset_year_total'] = int(year_total_val)
                            payload['dataset_metric_label'] = 'articles'
            entry_payloads.append(payload)

        for entry in entries:
            _check_cancel(cancel_event)
            entry["value"] = value

        first_props = entries[0].get('props', {}) if entries else {}
        city_raw = str(first_props.get('City') or '').strip()
        state_raw = str(first_props.get('State') or '').strip()
        city_key = _city_state_key(city_raw, state_raw)
        place_label = ', '.join([p for p in (city_raw, state_raw) if p])

        city_entries_map.setdefault(city_key, entries)

        title_text, article_count, _ = _group_header(entries, stats, search_term)
        dataset_entry: Dict[str, Any] = {
            'entries': entry_payloads,
            'value': value,
            'full_value': value,
            'article_count': article_count,
            'full_article_count': article_count,
            'title': title_text,
            'full_title': title_text,
            'metric_display': metric_display,
            'metric_normalized_display': metric_normalized_display,
            'normalized': normalize_flag,
            'denominator_label': denom_label if normalize_flag else '',
            'lat': loc_lat,
            'lon': loc_lon,
            'coords': [{'lat': loc_lat, 'lon': loc_lon}],
            'location_index': 1,
            'location_total': 1,
            'location_label': '',
            'search_term': search_term or '',
            'city': city_raw,
            'state': state_raw,
            'place_label': place_label,
        }
        if year_match_counts:
            dataset_entry['year_match_counts'] = year_match_counts

        if (use_time_slider or rank_time_bins) and time_index:
            time_bins: Dict[str, Dict[str, Any]] = {}
            for idx_entry, entry in enumerate(entries):
                dt = entry.get('date')
                bin_indices = _assign_time_bins(dt, time_index, linger_unit, int(linger_step or 0))
                for bin_idx in bin_indices:
                    _check_cancel(cancel_event)
                    if bin_idx < len(time_labels):
                        label = time_labels[bin_idx]
                    else:
                        label = time_index[bin_idx].strftime('%Y-%m-%dT%H:%M:%SZ')
                    key = str(bin_idx + 1)
                    bucket = time_bins.setdefault(key, {'indexes': [], 'label': label})
                    bucket['indexes'].append(idx_entry)

            if time_bins:
                time_payload: Dict[str, Any] = {}
                for bin_key, info in time_bins.items():
                    _check_cancel(cancel_event)
                    indexes = info.get('indexes') or []
                    if not indexes:
                        continue
                    bin_entries = [entries[i] for i in indexes if 0 <= i < len(entries)]
                    if not bin_entries:
                        continue
                    bin_stats = _compute_group_stats(bin_entries, search_term)
                    bin_value = _compute_group_value(bin_stats, metric_key, normalize_flag, denominator_key)
                    bin_title, bin_count, _ = _group_header(bin_entries, bin_stats, search_term)
                    label_value = info.get('label') or ''
                    time_payload[bin_key] = {
                        'indexes': indexes,
                        'value': bin_value,
                        'article_count': bin_count,
                        'title': bin_title,
                        'time_label': label_value,
                    }
                if time_payload:
                    dataset_entry['time_bins'] = time_payload

        popup_dataset[group['id']] = dataset_entry
        values.append(value)

    collocate_terms_list: List[str] = []
    rank_index: Dict[str, Dict[str, Dict[str, int]]] = {}
    rank_max_value = 0
    collocate_term_stats: Dict[str, Dict[str, int]] = {}
    collocate_hits_by_city: Dict[str, Dict[str, Set[int]]] = {}
    initial_collocate_term: str = ''
    initial_collocate_summary_text: str = ''
    if collocate_rank_mode:
        (
            collocate_terms_list,
            rank_index,
            rank_max_value,
            collocate_hits_by_city,
            collocate_time_totals,
            collocate_time_labels,
        ) = _build_collocate_rank_index(
            groups,
            popup_dataset,
            search_term,
            drop_stopwords=collocate_drop_stopwords,
            window=collocate_window,
            drop_terms=collocate_drop_terms,
            term_groups=collocate_term_groups,
            top_n=collocate_rank_top_n,
            term_scope=collocate_rank_term_scope,
            time_key=collocate_rank_time_key,
            focus_mode=collocate_rank_focus,
            focus_city=collocate_rank_focus_city,
            focus_state=collocate_rank_focus_state,
            manual_terms=collocate_rank_terms,
            cancel_event=cancel_event,
        )
        if collocate_hits_by_city:
            for group in groups:
                group_id = group.get('id')
                if not group_id:
                    continue
                dataset_entry = popup_dataset.get(group_id)
                if not isinstance(dataset_entry, dict):
                    continue
                city_val = dataset_entry.get('city')
                state_val = dataset_entry.get('state')
                city_key = _city_state_key(city_val, state_val)
                term_map = collocate_hits_by_city.get(city_key)
                if not term_map:
                    continue
                hits_for_dataset: Dict[str, List[int]] = {}
                for term, indexes in term_map.items():
                    if not indexes:
                        continue
                    normalized_indexes: Set[int] = set()
                    for idx in indexes:
                        try:
                            normalized_indexes.add(int(idx))
                        except (TypeError, ValueError):
                            continue
                    if normalized_indexes:
                        hits_for_dataset[term] = sorted(normalized_indexes)
                if hits_for_dataset:
                    dataset_entry['collocate_hits'] = hits_for_dataset

        if collocate_hits_by_city:
            term_article_counts: Dict[str, int] = defaultdict(int)
            term_cities: Dict[str, Set[str]] = defaultdict(set)
            term_newspapers: Dict[str, Set[str]] = defaultdict(set)

            for city_key, term_map in collocate_hits_by_city.items():
                entries = city_entries_map.get(city_key) or []
                for term, indexes in term_map.items():
                    if not indexes:
                        continue
                    term_article_counts[term] += len(indexes)
                    term_cities[term].add(city_key)
                    for idx in indexes:
                        try:
                            idx_int = int(idx)
                        except (TypeError, ValueError):
                            continue
                        if idx_int < 0 or idx_int >= len(entries):
                            continue
                        entry_obj = entries[idx_int]
                        props = entry_obj.get('props', {}) if isinstance(entry_obj, dict) else {}
                        paper_id = (
                            props.get('SN')
                            or props.get('lccn')
                            or props.get('Title')
                            or props.get('newspaper_name')
                            or ''
                        )
                        paper_str = str(paper_id).strip()
                        if paper_str:
                            term_newspapers[term].add(paper_str)

            for term, total_articles in term_article_counts.items():
                collocate_term_stats[term] = {
                    'articles': int(total_articles),
                    'newspapers': len(term_newspapers.get(term, set())),
                    'cities': len(term_cities.get(term, set())),
                }

        initial_collocate_term = collocate_terms_list[0] if collocate_terms_list else ''

        def _format_collocate_summary(term: str) -> str:
            if not term:
                return 'Collocate term: none selected'
            stats = collocate_term_stats.get(term) or {}
            articles_val = stats.get('articles', 0)
            newspapers_val = stats.get('newspapers', 0)
            cities_val = stats.get('cities', 0)
            return (
                f'Collocate term "{term}": '
                f'{articles_val:,} articles | {newspapers_val:,} newspapers | {cities_val:,} cities'
            )

        initial_collocate_summary_text = _format_collocate_summary(initial_collocate_term)

        scope_norm = str(collocate_rank_term_scope or '').strip().lower()
        if scope_norm.startswith('time') and collocate_rank_time_key:
            raw_key = str(collocate_rank_time_key).strip()
            key_variants = [raw_key] if raw_key else []
            label_variant = collocate_time_labels.get(raw_key)
            if label_variant:
                key_variants.append(label_variant)
            combined_counter = Counter()
            for candidate in key_variants:
                if not candidate:
                    continue
                counter = collocate_time_totals.get(candidate)
                if counter:
                    combined_counter.update(counter)
            if combined_counter:
                scoped_terms = [
                    term for term, _freq in _sorted_counter_terms(combined_counter) if term
                ][:collocate_rank_top_n]
                if scoped_terms:
                    collocate_terms_list = scoped_terms
                    initial_collocate_term = collocate_terms_list[0]
                    initial_collocate_summary_text = _format_collocate_summary(initial_collocate_term)

    if groups and not any(v > 0 for v in values):
        for group in groups:
            group["value"] = 1.0
            for entry in group.get("entries", []):
                entry["value"] = 1.0

    base_name = os.path.splitext(os.path.basename(geojson_path))[0]
    out_dir = os.path.dirname(geojson_path)

    articles_count = len(pts)
    city_set: set = set()
    newspaper_ids: set = set()
    dates_dt: List[datetime] = []
    for p in pts:
        props = p.get('props', {})
        city = props.get('City')
        if city not in (None, ""):
            city_set.add(str(city))
        sn = props.get('SN') or props.get('lccn')
        if sn not in (None, ""):
            newspaper_ids.add(str(sn))
        else:
            title = props.get('Title') or props.get('newspaper_name')
            if title not in (None, ""):
                newspaper_ids.add(str(title))
        dt = p.get('date')
        if isinstance(dt, datetime):
            dates_dt.append(dt)

    metadata = data.get('metadata') or data.get('properties') or {}
    start_meta = start_override or metadata.get('start_date') or metadata.get('StartDate')
    end_meta = end_override or metadata.get('end_date') or metadata.get('EndDate')

    if dates_dt:
        min_dt = min(dates_dt)
        max_dt = max(dates_dt)
    else:
        min_dt = max_dt = None

    dated_pts = [p for p in pts if p.get("date") is not None]
    rank_time_bins = collocate_rank_mode and bool(dated_pts)
    use_time_slider = bool(dated_pts) and time_enabled
    time_index: List[datetime] = []
    time_labels: List[str] = []
    range_start = start_dt_meta or min_dt
    range_end = end_dt_meta or max_dt
    if range_start and range_end and range_start > range_end:
        range_start, range_end = range_end, range_start
    if (use_time_slider or rank_time_bins) and min_dt and max_dt:
        time_index = _build_time_index(range_start, range_end, time_unit, max(1, int(time_step or 1)))
        time_labels = [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in time_index]

    start_str = start_meta or (min_dt.strftime('%Y-%m-%d') if min_dt else '')
    end_str = end_meta or (max_dt.strftime('%Y-%m-%d') if max_dt else '')
    if not start_str and end_str:
        start_str = end_str
    if not end_str and start_str:
        end_str = start_str
    date_range = (start_str, end_str) if start_str or end_str else ()

    metric_display_summary = metric_normalized_display if normalize_flag else metric_display

    allowed_table_modes = {'full', 'article', 'minimal'}
    table_mode_norm = (table_mode or 'full').strip().lower()
    if table_mode_norm not in allowed_table_modes:
        table_mode_norm = 'full'

    row_limit_val: Optional[int] = None
    if table_row_limit:
        try:
            parsed_limit = int(table_row_limit)
            if parsed_limit > 0:
                row_limit_val = parsed_limit
        except (TypeError, ValueError):
            row_limit_val = None

    summary = {
        'geojson_name': os.path.basename(geojson_path),
        'term': search_term or '',
        'date_range': date_range,
        'articles': articles_count,
        'newspapers': len(newspaper_ids),
        'cities': len(city_set),
        'metric_display': metric_display_summary,
        'metric_key': metric_key,
        'normalized': normalize_flag,
        'denominator_label': denom_label if normalize_flag else '',
        'mode': mode,
        'time_enabled': time_enabled,
        'time_unit': time_unit,
        'time_step': time_step,
        'linger_unit': linger_unit,
        'linger_step': linger_step,
        'lightweight': bool(lightweight),
        'table_mode': table_mode_norm,
        'table_row_limit': row_limit_val or 0,
        'collocate_drop_terms': list(collocate_drop_terms or []),
        'collocate_term_groups': list(collocate_term_groups or []),
    }

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix_parts = [mode, metric_key]
    if normalize_flag:
        suffix_parts.append('norm')
    if lightweight:
        suffix_parts.append('lite')
    if collocate_rank_mode:
        scope_tag = 'global'
        if collocate_rank_term_scope and collocate_rank_term_scope.strip().lower().startswith('time'):
            key_val = (collocate_rank_time_key or '').strip()
            scope_tag = f'time-{key_val}' if key_val else 'time'
        suffix_parts.append(f'scope-{_slug(scope_tag)}')

        focus_mode_norm = (collocate_rank_focus or 'all').strip().lower()
        focus_tag = focus_mode_norm or 'all'
        if focus_mode_norm == 'city':
            components = [collocate_rank_focus_city or '']
            if collocate_rank_focus_state:
                components.append(collocate_rank_focus_state)
            focus_tag = 'city-' + '-'.join(filter(None, components))
        elif focus_mode_norm == 'state':
            focus_tag = f"state-{collocate_rank_focus_state or ''}"
        suffix_parts.append(f'focus-{_slug(focus_tag)}')
    suffix_parts.append(timestamp)
    suffix = '_'.join(suffix_parts)

    map_filename = f"{base_name}_{suffix}.html"
    table_filename = f"{base_name}_{suffix}_attributes.html"
    out_html = os.path.join(out_dir, map_filename)
    attr_path = os.path.join(out_dir, table_filename)

    table_kwargs: Dict[str, Any] = {'hyperlink_columns': ['url']}
    if table_mode_norm == 'article':
        table_kwargs['include_columns'] = ['date', 'Title', 'article', 'url']
    elif table_mode_norm == 'minimal':
        table_kwargs['include_columns'] = ['date', 'Title', 'City', 'State', 'lccn', 'page', 'url', 'article']
        table_kwargs['omit_article'] = False

    max_rows = row_limit_val
    if lightweight:
        light_limit = min(len(pts), 1000) if pts else None
        if light_limit:
            max_rows = light_limit if max_rows is None else min(max_rows, light_limit)
        include_cols = table_kwargs.get('include_columns')
        if include_cols:
            if 'article' not in include_cols:
                include_cols.append('article')
        else:
            table_kwargs['include_columns'] = [
                'date',
                'Title',
                'headline',
                'Headline',
                'City',
                'State',
                'page',
                'url',
                'article',
            ]
        table_kwargs['omit_article'] = False

    if max_rows:
        table_kwargs['max_rows'] = max_rows

    heat_radius_val = 15
    if heat_radius is not None:
        try:
            candidate_radius = int(heat_radius)
            if candidate_radius > 0:
                heat_radius_val = candidate_radius
        except (TypeError, ValueError):
            pass

    heat_multiplier = 1.0
    if heat_value is not None:
        try:
            candidate_multiplier = float(heat_value)
            if candidate_multiplier > 0:
                heat_multiplier = candidate_multiplier
        except (TypeError, ValueError):
            pass

    grad_min_val = max(1, int(grad_min_radius) if grad_min_radius else 6)
    grad_max_val = max(grad_min_val + 1, int(grad_max_radius) if grad_max_radius else 28)

    m = folium.Map(location=[37.8, -96.0], zoom_start=4)
    m.add_child(_ZoomTopRight())

    def point_radius(group: Dict[str, Any]) -> float:
        count = len(group.get("entries") or [])
        return 5 if count > 1 else 3

    if mode == "heatmap":
        if use_time_slider and dated_pts:
            idx = time_index
            slices = _heat_slices(dated_pts, idx, linger_unit, int(linger_step or 0))
            heat_data: List[List[List[float]]] = []
            for frame in slices:
                frame_pts: List[List[float]] = []
                for item in frame:
                    lat, lon = item[0], item[1]
                    base_weight = item[2] if len(item) > 2 else None
                    if base_weight is not None and base_weight > 0:
                        weight_val = base_weight * heat_multiplier
                        frame_pts.append([lat, lon, weight_val])
                    elif heat_multiplier > 0:
                        frame_pts.append([lat, lon, heat_multiplier])
                    else:
                        frame_pts.append([lat, lon])
                heat_data.append(frame_pts)
            index_labels = time_labels if time_labels else [t.strftime("%Y-%m-%dT00:00:00Z") for t in idx]
            HeatMapWithTime(
                heat_data,
                index=index_labels,
                auto_play=False,
                max_opacity=0.7,
                radius=heat_radius_val,
            ).add_to(m)
        else:
            coords: List[List[float]] = []
            for p in pts:
                base_weight = p.get("value")
                if base_weight is not None and base_weight > 0:
                    weight_val = base_weight * heat_multiplier
                    coords.append([p["lat"], p["lon"], weight_val])
                elif heat_multiplier > 0:
                    coords.append([p["lat"], p["lon"], heat_multiplier])
                else:
                    coords.append([p["lat"], p["lon"]])
            if coords:
                HeatMap(coords, max_opacity=0.7, radius=heat_radius_val).add_to(m)

        _add_point_markers(
            m,
            groups,
            search_term,
            point_radius,
            popup_width=popup_width,
            lightweight=lightweight,
            popup_dataset=popup_dataset,
            ghost_markers=True,
        )

    elif mode == "graduated":
        resolver = _graduated_radius_resolver(groups, float(grad_min_val), float(grad_max_val))
        _add_point_markers(
            m,
            groups,
            search_term,
            resolver,
            popup_width=popup_width,
            lightweight=lightweight,
            popup_dataset=popup_dataset,
        )
    elif mode == "cluster":
        icon_create_function = (
            "function(cluster) {"
            " var sum = 0;"
            " cluster.getAllChildMarkers().forEach(function(marker) {"
            "   var v = marker.options && marker.options.metricValue;"
            "   if (typeof v === 'number' && !isNaN(v)) { sum += v; }"
            "   else if (v) { var num = parseFloat(v); if (!isNaN(num)) sum += num; }"
            " });"
            " var formatted;"
            " if (%s) { formatted = sum.toFixed(4); }"
            " else { formatted = Math.round(sum).toLocaleString(); }"
            " var absSum = Math.abs(sum);"
            " var c = 'marker-cluster marker-cluster-small';"
            " if (absSum >= 100) { c = 'marker-cluster marker-cluster-large'; }"
            " else if (absSum >= 10) { c = 'marker-cluster marker-cluster-medium'; }"
            " return L.divIcon({ html: '<div><span>' + formatted + '</span></div>', className: c, iconSize: new L.Point(40, 40) });"
            "}"
        ) % ('true' if normalize_flag else 'false')

        cluster = MarkerCluster(
            name='Markers',
            options={
                'showCoverageOnHover': False,
                'zoomToBoundsOnClick': False,
                'spiderfyOnMaxZoom': False,
            },
            icon_create_function=icon_create_function,
        )
        cluster.add_to(m)
        _add_point_markers(
            cluster,
            groups,
            search_term,
            point_radius,
            popup_width=popup_width,
            lightweight=lightweight,
            popup_dataset=popup_dataset,
            ghost_markers=False,
        )
    else:
        _add_point_markers(
            m,
            groups,
            search_term,
            point_radius,
            popup_width=popup_width,
            lightweight=lightweight,
            popup_dataset=popup_dataset,
        )


    if lightweight:
        for entry in pts:
            props = entry.get('props') if isinstance(entry, dict) else None
            if isinstance(props, dict) and 'article' in props:
                props['article'] = ''

    attr_file = _write_attribute_table(pts, attr_path, **table_kwargs)
    if attr_file:
        summary['attribute_table'] = attr_file

    valid_row_ids = {
        entry.get('_popup_row_id')
        for entry in pts
        if isinstance(entry, dict) and entry.get('_attr_in_table')
    }
    if not attr_file:
        valid_row_ids = set()

    if popup_dataset:
        def _prune_attr_ids(payloads: Optional[List[Dict[str, Any]]]) -> None:
            if not payloads:
                return
            for payload in payloads:
                if not isinstance(payload, dict):
                    continue
                row_id = payload.get('attr_row_id')
                if row_id and row_id not in valid_row_ids:
                    payload['attr_row_id'] = ''

        for dataset_entry in popup_dataset.values():
            if not isinstance(dataset_entry, dict):
                continue
            _prune_attr_ids(dataset_entry.get('entries'))
            _prune_attr_ids(dataset_entry.get('full_entries'))

    lazy_popup_mode = bool(attr_file and mode == 'heatmap' and lightweight)
    if lazy_popup_mode and popup_dataset:
        def _strip_inline_articles(payloads: Optional[List[Dict[str, Any]]]) -> None:
            if not payloads:
                return
            for payload in payloads:
                if not isinstance(payload, dict):
                    continue
                if payload.get('attr_row_id'):
                    payload['article_html'] = ''

        for dataset_entry in popup_dataset.values():
            if not isinstance(dataset_entry, dict):
                continue
            _strip_inline_articles(dataset_entry.get('entries'))
            _strip_inline_articles(dataset_entry.get('full_entries'))

    if collocate_rank_mode:
        metric_display_summary = 'Ranked Collocates'
        if top_term_variant:
            metric_display_summary = 'Top Collocate Term'

    search_results_text = f"{articles_count:,} articles | {len(newspaper_ids):,} newspapers | {len(city_set):,} cities"
    search_results_line = f'<div id="collocateSearchResults" style="margin-top:4px;"><strong>Search Results:</strong> {_esc(search_results_text)}</div>'

    header_lines: List[str] = []

    title_html = ''
    # Collocate maps get a custom title; otherwise include basic info lines
    if collocate_rank_mode:
        term_text = (summary.get('term') or '').strip()
        term_segment = f"of '{_esc(term_text)}'" if term_text else ''
        date_segment = ''
        if summary.get('date_range'):
            start, end = summary['date_range']
            date_text = start if start == end else ' – '.join([s for s in (start, end) if s])
            if date_text:
                date_segment = f"between {_esc(date_text)}"
        title_bits = ['Top Ranked Collocates']
        if term_segment:
            title_bits.append(term_segment)
        if date_segment:
            title_bits.append(date_segment)
        title_html = '<div style="font-weight:700; font-size:16px; margin-bottom:4px;">' + ' '.join(title_bits) + '</div>'
    else:
        if summary.get('date_range'):
            start, end = summary['date_range']
            date_text = start if start == end else ' – '.join([s for s in (start, end) if s])
            if date_text:
                header_lines.append(f'<div><strong>Date range:</strong> {_esc(date_text)}</div>')
        if summary.get('term'):
            header_lines.append(f'<div><strong>Term:</strong> {_esc(summary["term"])}</div>')

    if title_html:
        header_lines.insert(0, title_html)

    if lightweight:
        header_lines.append('<div style="font-size:12px; font-weight:400;"><em>Lightweight mode: popups and table trimmed for size.</em></div>')

    header_lines.append(f'<div><strong>Data Source:</strong> {_esc(summary["geojson_name"])}</div>')

    header_lines.append(search_results_line)

    if attr_file:
        link_name = os.path.basename(attr_file)
        header_lines.append(
            f'<div><a href="{html.escape(link_name)}" target="_blank" rel="noopener">Open attribute table</a></div>'
        )

    header_lines.append('<div style="height:6px;"></div>')

    header_lines.append(f'<div><strong>Mapped metric:</strong> {_esc(metric_display_summary)}</div>')
    if summary.get('normalized') and denom_label:
        header_lines.append(
            f'<div style="font-size:12px; color:#555;">Normalized per city by {_esc(denom_label)}</div>'
        )
    if time_enabled:
        time_text = f"{time_step} {time_unit}"
        linger_text = f"{linger_step} {linger_unit}"
        header_lines.append(
            f'<div><strong>Time bin:</strong> {_esc(time_text)} | <strong>Linger:</strong> {_esc(linger_text)}</div>'
        )

    if collocate_rank_mode:
        if top_term_variant:
            summary_line_text = 'Top collocate term per location.'
        else:
            summary_line_text = initial_collocate_summary_text or 'Collocate term: none selected'
        ranking_scope_text = 'Entire period'
        if collocate_rank_term_scope and collocate_rank_term_scope.strip().lower().startswith('time'):
            key_text = collocate_rank_time_key or ''
            if collocate_rank_time_label:
                ranking_scope_text = collocate_rank_time_label
            elif key_text:
                ranking_scope_text = f'Time bin {key_text}'
            else:
                ranking_scope_text = 'First time bin'
        header_lines.append(
            f'<div><strong>Home time Bin:</strong> {_esc(ranking_scope_text)}</div>'
        )

        focus_mode_norm = (collocate_rank_focus or '').strip().lower()
        if collocate_rank_focus_label:
            focus_desc = collocate_rank_focus_label
        elif focus_mode_norm == 'city' and collocate_rank_focus_city:
            if collocate_rank_focus_state:
                focus_desc = f'City — {collocate_rank_focus_city}, {collocate_rank_focus_state}'
            else:
                focus_desc = f'City — {collocate_rank_focus_city}'
        elif focus_mode_norm == 'state' and collocate_rank_focus_state:
            focus_desc = f'State — {collocate_rank_focus_state}'
        else:
            focus_desc = 'All cities'
        header_lines.append(
            f'<div><strong>Ranking Location:</strong> {_esc(focus_desc)}</div>'
        )

        slider_placeholder = '<div id="collocateTimeSliderContainer" style="margin-top:6px;"></div>' if collocate_time_slider_enabled and collocate_time_bins_payload else ''
        if top_term_variant:
            header_lines.append(
                '<div style="margin-top:6px;">Markers display the top-ranked collocate term for each location. Circle size reflects the number of articles containing that term.</div>'
            )
            if slider_placeholder:
                header_lines.append(slider_placeholder)
            header_lines.append(
                '<div id="collocateSummaryLine" '
                'style="color:#2d3748; margin-top:6px; position:relative; display:inline-block; min-width:320px;">'
                f'<span data-summary-content="1">{_esc(summary_line_text)}</span>'
                '<span aria-hidden="true" data-summary-buffer="1" '
                'style="visibility:hidden; pointer-events:none; white-space:nowrap; display:inline-block;">'
                'Top collocate term sample: 999,999 articles | Unique terms: 9,999 | Time: 1901-01-01'
                '</span>'
                '</div>'
            )
        else:
            if collocate_terms_list:
                select_opts_parts = []
                for idx, term in enumerate(collocate_terms_list[:200], start=1):
                    select_opts_parts.append(
                        f'<option value="{_esc(term)}">({_esc(str(idx))}) {_esc(term)}</option>'
                    )
                select_opts = ''.join(select_opts_parts)
                scope_desc = ranking_scope_text
                if scope_desc.lower() == 'entire period':
                    scope_desc = 'entire period'
                focus_desc = ''
                focus_mode_norm = (collocate_rank_focus or '').strip().lower()
                if focus_mode_norm == 'city' and collocate_rank_focus_city:
                    focus_desc = f' for {collocate_rank_focus_city}'
                    if collocate_rank_focus_state:
                        focus_desc += f', {collocate_rank_focus_state}'
                elif focus_mode_norm == 'state' and collocate_rank_focus_state:
                    focus_desc = f' for {collocate_rank_focus_state}'
                scope_text = _esc(scope_desc)
                focus_text = _esc(focus_desc)
                header_lines.append(
                    f'<div>Top {len(collocate_terms_list)} collocates based on {scope_text}{focus_text}</div>'
                )
                header_lines.append(
                    '<div style="margin-top:6px;">'
                    '<label style="font-weight:600; margin-right:6px;">Collocate term:</label>'
                    f'<select id="collocateTermSelect" style="min-width:220px;">{select_opts}</select>'
                    '</div>'
                )
                if slider_placeholder:
                    header_lines.append(slider_placeholder)
            else:
                if slider_placeholder:
                    header_lines.append(slider_placeholder)

            header_lines.append(
                '<div id="collocateSummaryLine" '
                'style="color:#c53030; margin-top:6px; position:relative; display:inline-block; min-width:320px;">'
                f'<span data-summary-content="1">{_esc(summary_line_text)}</span>'
                '<span aria-hidden="true" data-summary-buffer="1" '
                'style="visibility:hidden; pointer-events:none; white-space:nowrap; display:inline-block;">'
                'Collocate term sample: 999,999 articles | 9,999 newspapers | 9,999 cities | Time: 1901-01-01'
                '</span>'
                '</div>'
            )

    header_html = (
        '<div style="position: fixed; top: 5px; left: 5px; z-index:9999;">'
        '<div style="max-width: 560px; background: rgba(255,255,255,0.92); '
        'padding: 8px 12px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.2); '
        'font-size: 13px; line-height: 1.4;">'
        + ''.join(header_lines)
        + '</div></div>'
    )
    m.get_root().html.add_child(folium.Element(header_html))

    if rank_index and not top_term_variant:
        label_style = (
            '<style>'
            '.collocate-rank-label { '
            'background: transparent; '
            'border: none; '
            'box-shadow: none; '
            'color: #ffffff; '
            'font-weight: 600; '
            'pointer-events: none; '
            'padding: 0; '
            'margin: 0; '
            'text-shadow: 0 1px 2px rgba(0,0,0,0.55); '
            '}'
            '.collocate-rank-label:before { display: none; }'
            '.collocate-rank-label .leaflet-tooltip-content { '
            'margin: 0 !important; '
            'padding: 0 !important; '
            'line-height: 1; '
            '}'
            '.collocate-time-button { '
            'border: 1px solid #cbd5e0; '
            'background: #f7fafc; '
            'color: #2d3748; '
            'border-radius: 999px; '
            'padding: 2px 10px; '
            'font-size: 14px; '
            'line-height: 1; '
            'cursor: pointer; '
            'transition: background 0.2s ease, color 0.2s ease; '
            '}'
            '.collocate-time-button[data-clock-role="toggle"] { '
            'position: relative; '
            'font-size: 18px; '
            'padding: 2px 8px; '
            '}'
            '.collocate-time-button[data-clock-role="toggle"][data-clock-disabled="1"]::after { '
            'content: ""; '
            'position: absolute; '
            'left: 50%; '
            'top: 50%; '
            'width: 70%; '
            'height: 2px; '
            'background: #c53030; '
            'border-radius: 999px; '
            'pointer-events: none; '
            'transform: translate(-50%, -50%) rotate(45deg); '
            '}'
            '.collocate-time-button:hover { background: #edf2f7; color: #1a202c; }'
            '.collocate-time-button:disabled { opacity: 0.4; cursor: default; }'
            '#collocateTimeRange { accent-color: #2b6cb0; }'
            '</style>'
        )
        m.get_root().html.add_child(folium.Element(label_style))

    config_payload = {
        'attribute_table': os.path.basename(attr_file) if attr_file else '',
        'search_term': search_term or '',
        'inline_articles': bool(embed_articles),
        'time_labels': time_labels if (use_time_slider or collocate_time_slider_enabled) and time_labels else [],
        'map_mode': mode,
        'click_radius_px': heat_radius_val,
        'collocate_summary': collocate_term_stats if collocate_term_stats else {},
        'collocate_colorize': bool(collocate_rank_colorize),
        'initial_collocate_term': initial_collocate_term,
        'collocate_map_variant': collocate_map_variant,
        'collocate_export_csv': bool(collocate_export_csv),
    }

    # Embed collocate rank index when available
    if rank_index:
        config_payload['collocate_ranks'] = rank_index
        config_payload['collocate_terms'] = collocate_terms_list
        config_payload['rank_max'] = rank_max_value or COLLOCATE_RANK_LIMIT
        config_payload['collocate_settings'] = {
            'requested_top_n': collocate_rank_top_n,
            'terms_returned': len(collocate_terms_list),
            'term_scope': collocate_rank_term_scope,
            'time_key': collocate_rank_time_key or '',
            'focus': collocate_rank_focus,
            'focus_city': collocate_rank_focus_city or '',
            'focus_state': collocate_rank_focus_state or '',
            'time_label': collocate_rank_time_label or '',
            'focus_label': collocate_rank_focus_label or '',
            'colorize': bool(collocate_rank_colorize),
        }
        if collocate_time_slider_enabled and collocate_time_bins_payload:
            config_payload['collocate_time_slider'] = True
            config_payload['collocate_time_bins'] = collocate_time_bins_payload
            if collocate_time_default_key is not None:
                config_payload['collocate_time_default'] = collocate_time_default_key
    config_json = json.dumps(config_payload, ensure_ascii=False).replace('</', '<\\/')
    config_script = (
        '<script id="map-config" type="application/json">'
        f"{config_json}"
        '</script>'
    )
    m.get_root().html.add_child(folium.Element(config_script))

    popup_json_js = json.dumps(popup_dataset, ensure_ascii=False).replace('</', '<\\/')
    data_script = (
        '<script id="popup-data" type="application/json">'
        f"{popup_json_js}"
        '</script>'
    )
    m.get_root().html.add_child(folium.Element(data_script))

    map_var = m.get_name()
    cluster_block = ''
    if mode == 'cluster':
        cluster_var = cluster.get_name()
        metric_label_js = json.dumps(metric_display_summary)
        normalized_js = 'true' if normalize_flag else 'false'
        cluster_block = f"""
  whenLayerReady('{cluster_var}', function(clusterLayer) {{
    if (!clusterLayer) {{
      return;
    }}
    clusterLayer.options.iconCreateFunction = function(cluster) {{
      var sum = 0;
      cluster.getAllChildMarkers().forEach(function(marker) {{
        var v = marker.options && marker.options.metricValue;
        if (typeof v === 'number' && !isNaN(v)) {{
          sum += v;
        }} else if (v) {{
          var num = parseFloat(v);
          if (!isNaN(num)) sum += num;
        }}
      }});
      var formatted;
      if ({normalized_js}) {{
        formatted = sum.toFixed(4);
      }} else {{
        formatted = Math.round(sum).toLocaleString();
      }}
      var absSum = Math.abs(sum);
      var c = 'marker-cluster marker-cluster-small';
      if (absSum >= 100) {{
        c = 'marker-cluster marker-cluster-large';
      }} else if (absSum >= 10) {{
        c = 'marker-cluster marker-cluster-medium';
      }}
      return L.divIcon({{
        html: '<div><span>' + formatted + '</span></div>',
        className: c,
        iconSize: new L.Point(40, 40)
      }});
    }};
    if (clusterLayer.refreshClusters) {{
      clusterLayer.refreshClusters();
    }}
    clusterLayer.on('clusterclick', function(e) {{
      if (e && e.originalEvent) {{
        if (typeof e.originalEvent.preventDefault === 'function') {{
          e.originalEvent.preventDefault();
        }}
        if (typeof e.originalEvent.stopPropagation === 'function') {{
          e.originalEvent.stopPropagation();
        }}
        if (typeof L !== 'undefined' && L.DomEvent && typeof L.DomEvent.stop === 'function') {{
          L.DomEvent.stop(e.originalEvent);
        }}
      }}
      loadData(function(dataset) {{
        dataset = dataset || {{}};
        var markers = e.layer.getAllChildMarkers();
        var ids = [];
        var metricSum = 0;
        markers.forEach(function(marker) {{
          var gid = marker.options && marker.options.groupId;
          if (gid && ids.indexOf(gid) === -1) ids.push(gid);
          var mv = marker.options && marker.options.metricValue;
          if (typeof mv === 'number' && !isNaN(mv)) {{
            metricSum += mv;
          }} else if (mv) {{
            var num = parseFloat(mv);
            if (!isNaN(num)) metricSum += num;
          }}
        }});
        if (!ids.length) return;
        var metricLabel = {metric_label_js};
        var isNormalized = {normalized_js};
        var metricText = isNormalized ? (metricSum).toFixed(4) : Math.round(metricSum).toLocaleString();
        var container = document.createElement('div');
        container.style.minWidth = '280px';
        container.style.fontSize = '14px';
        container.style.lineHeight = '1.3';
        var summary = document.createElement('div');
        summary.style.fontSize = '12px';
        summary.style.color = '#555';
        var cityCount = ids.length;
        var cityLabel = cityCount === 1 ? 'City' : 'Cities';
        var totalArticles = 0;
        var hasArticleCounts = true;
        ids.forEach(function(gid) {{
          var data = dataset[gid];
          var articleCount = data && data.article_count;
          var parsed = Number(articleCount);
          if (Number.isFinite(parsed)) {{
            totalArticles += parsed;
          }} else {{
            hasArticleCounts = false;
          }}
        }});
        var articleDisplay;
        if (hasArticleCounts) {{
          articleDisplay = Math.round(totalArticles).toLocaleString();
        }} else if (!isNormalized) {{
          articleDisplay = Math.round(metricSum).toLocaleString();
        }} else {{
          articleDisplay = metricText;
        }}
        summary.textContent = 'Articles: ' + articleDisplay + ' from ' + cityCount + ' ' + cityLabel;
        container.appendChild(summary);
        if (isNormalized || metricLabel !== 'Articles') {{
          var metricDetail = document.createElement('div');
          metricDetail.style.marginTop = '2px';
          metricDetail.style.fontSize = '12px';
          metricDetail.style.color = '#555';
          metricDetail.textContent = metricLabel + ': ' + metricText;
          container.appendChild(metricDetail);
        }}
        var select = null;
        if (ids.length > 1) {{
          var selectLabel = document.createElement('div');
          selectLabel.style.marginTop = '6px';
          selectLabel.style.fontWeight = '600';
          selectLabel.textContent = 'Select a location';
          container.appendChild(selectLabel);
          select = document.createElement('select');
          select.style.width = '100%';
          select.style.marginTop = '4px';
          ids.forEach(function(gid) {{
            var data = dataset[gid];
            var opt = document.createElement('option');
            opt.value = gid;
            opt.textContent = (data && data.title) || gid;
            select.appendChild(opt);
          }});
          container.appendChild(select);
        }}
        var detailHost = document.createElement('div');
        detailHost.style.marginTop = '8px';
        container.appendChild(detailHost);
        function renderGroup(gid) {{
          detailHost.innerHTML = '';
          var data = dataset[gid];
          if (!data) {{
            detailHost.textContent = 'No data.';
            return;
          }}
          if (data.template) {{
            var wrapper = document.createElement('div');
            wrapper.innerHTML = data.template;
            var root = wrapper.firstElementChild;
            if (root) {{
              detailHost.appendChild(root);
              attach(root);
              attachDockToggle(root, clusterLayer._map, null);
            }}
          }} else {{
            detailHost.textContent = 'No data.';
          }}
        }}
        renderGroup(ids[0]);
        if (select) {{
          select.addEventListener('change', function() {{
            renderGroup(this.value);
          }});
        }}
        var mapRef = clusterLayer._map;
        if (!mapRef || typeof mapRef.openPopup !== 'function') {{
          return;
        }}
        L.popup({{maxWidth: 360}}).setLatLng(e.latlng).setContent(container).openOn(mapRef);
      }});
      return false;
    }});
  }});
"""
    script_template = StrTemplate(r"""
(function() {
  var popupCache = null;
  var config = {};
  var attrTableUrl = '';
  var searchTerm = '';
  var timeLabels = [];
  var mapMode = '';
  var clickRadiusPx = 24;
  var collocateRanks = null;
  var collocateTerms = [];
  var rankMax = 0;
  var collocateSummary = {};
  var collocateMapVariant = 'rank';
  var collocateColorize = false;
  var initialCollocateTerm = '';
  var selectedCollocate = '';
  var collocateTimeEnabled = false;
  var collocateTimeBins = [];
  var selectedTimeBinKey = null;
  var collocateTimeManuallyDisabled = false;
  var collocateTimeLastActiveIndex = 1;
  var activeMap = null;
  var highlightLayer = null;
  var currentHighlightId = null;
  var dockState = {
    root: null,
    map: null,
    popup: null,
    lat: null,
    lon: null,
    preferred: false,
  };
  var topTermColors = {};
  var topTermPalette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666', '#8c564b', '#bcbd22', '#17becf', '#ff9896', '#9467bd', '#fdae61', '#3288bd', '#f46d43', '#74add1', '#d53e4f'];
  var topTermPaletteIndex = 0;
  var topTermLegend = null;
  var topTermLegendList = null;
  var topTermLegendContainer = null;
  var topTermLabelToggle = null;
  var topTermLabelsEnabled = true;
  var topTermStyleInjected = false;
  var topTermLegendItems = {};
  var topTermActiveFilterTerm = '';
  var topTermCollapseButton = null;
  var topTermLegendCollapsed = false;

  (function parseConfig() {
    var tag = document.getElementById('map-config');
    if (!tag) {
      return;
    }
    try {
      config = JSON.parse(tag.textContent || tag.innerText || '{}') || {};
    } catch (err) {
      console.error('Failed to parse map config', err);
      config = {};
    }
    if (config.attribute_table) {
      attrTableUrl = String(config.attribute_table).trim();
    }
    if (attrTableUrl) {
      try {
        var resolvedAttrUrl = new URL(attrTableUrl, window.location && window.location.href ? window.location.href : undefined);
        var pageProtocol = (window.location && window.location.protocol) || '';
        if (pageProtocol === 'file:' && resolvedAttrUrl && resolvedAttrUrl.protocol === 'file:') {
          console.warn('Attribute table loading disabled when viewing the map over file://. Use a local web server to enable full text.');
          attrTableUrl = '';
        }
      } catch (cfgUrlErr) {
        if (window.location && window.location.protocol === 'file:') {
          console.warn('Attribute table loading disabled when viewing the map over file://. Use a local web server to enable full text.');
          attrTableUrl = '';
        }
      }
    }
    if (config.search_term) {
      searchTerm = String(config.search_term).trim();
    }
    if (Array.isArray(config.time_labels)) {
      timeLabels = config.time_labels.filter(function(item) {
        return item !== null && typeof item !== 'undefined';
      });
    }
    if (config.map_mode) {
      mapMode = String(config.map_mode).trim().toLowerCase();
    }
    if (config.collocate_map_variant) {
      collocateMapVariant = String(config.collocate_map_variant).trim().toLowerCase();
    }
    if (typeof config.click_radius_px === 'number' && Number.isFinite(config.click_radius_px)) {
      clickRadiusPx = Math.max(4, Number(config.click_radius_px));
    }
    if (config.collocate_ranks) { collocateRanks = config.collocate_ranks; }
    if (Array.isArray(config.collocate_terms)) { collocateTerms = config.collocate_terms.slice(); }
    if (typeof config.rank_max === 'number' && Number.isFinite(config.rank_max)) { rankMax = config.rank_max|0; }
    if (config.collocate_summary && typeof config.collocate_summary === 'object') {
      collocateSummary = config.collocate_summary;
    }
    if (typeof config.collocate_colorize === 'boolean') {
      collocateColorize = config.collocate_colorize;
    }
    if (typeof config.initial_collocate_term === 'string') {
      initialCollocateTerm = String(config.initial_collocate_term).trim();
    }
    if (config.collocate_time_slider) {
      collocateTimeEnabled = true;
    }
    if (Array.isArray(config.collocate_time_bins)) {
      collocateTimeBins = config.collocate_time_bins
        .map(function(item) {
          if (!item) {
            return null;
          }
          var key = '';
          if (Object.prototype.hasOwnProperty.call(item, 'key')) {
            key = String(item.key || '').trim();
          }
          if (!key) {
            return null;
          }
          var label = '';
          if (Object.prototype.hasOwnProperty.call(item, 'label')) {
            label = String(item.label || '').trim();
          }
          var iso = '';
          if (Object.prototype.hasOwnProperty.call(item, 'iso')) {
            iso = String(item.iso || '').trim();
          }
          if (!label) {
            label = iso || key;
          }
          return { key: key, label: label, iso: iso };
        })
        .filter(function(entry) { return entry !== null; });
    }
    if (collocateTimeEnabled && !collocateTimeBins.length) {
      collocateTimeEnabled = false;
    }
    if (collocateTimeEnabled) {
      var defaultTimeKey = '';
      if (typeof config.collocate_time_default === 'string') {
        defaultTimeKey = config.collocate_time_default.trim();
      }
      if (defaultTimeKey && collocateTimeBins.some(function(entry) { return entry.key === defaultTimeKey; })) {
        selectedTimeBinKey = defaultTimeKey;
      } else {
        selectedTimeBinKey = null;
      }
    }
  })();

  if (!selectedCollocate && !isTopTermMode()) {
    if (initialCollocateTerm) {
      selectedCollocate = initialCollocateTerm;
    } else if (collocateTerms.length) {
      selectedCollocate = String(collocateTerms[0] || '').trim();
    }
  }

  function isTopTermMode() {
    return collocateMapVariant === 'top_term';
  }

  function ensureTopTermStyle() {
    if (topTermStyleInjected) {
      return;
    }
    topTermStyleInjected = true;
    var style = document.createElement('style');
    style.textContent = `
.top-term-legend { position: fixed; top: 60px; right: 12px; z-index: 9999; background: rgba(255,255,255,0.94); box-shadow: 0 1px 4px rgba(0,0,0,0.25); border-radius: 6px; padding: 8px 10px; max-width: 240px; font-size: 13px; line-height: 1.4; }
.top-term-legend h3 { margin: 0; font-size: 14px; font-weight: 600; color: #2d3748; }
.top-term-legend .legend-body { max-height: 200px; overflow-y: auto; margin-top: 6px; transition: max-height 0.25s ease, margin-top 0.25s ease; }
.top-term-legend.collapsed .legend-body { max-height: 0; margin-top: 0; overflow: hidden; }
.top-term-legend .legend-header-controls { display: flex; align-items: center; gap: 8px; }
.top-term-legend .legend-header-spacer { flex: 1 1 auto; }
.top-term-legend button.legend-toggle { border: none; background: none; cursor: pointer; font-size: 14px; line-height: 1; padding: 0 4px; color: #2d3748; }
.top-term-legend button.legend-toggle:focus { outline: 2px solid #3182ce; outline-offset: 2px; }
.top-term-legend .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; font-size: 12px; color: #2d3748; }
.top-term-legend .legend-item:last-child { margin-bottom: 0; }
.top-term-legend .swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.2); flex: 0 0 auto; }
.top-term-legend .term-label { flex: 1 1 auto; }
.top-term-legend .count-label { color: #4a5568; }
.top-term-legend .legend-controls { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 12px; color: #2d3748; }
.top-term-legend input[type="checkbox"] { vertical-align: middle; }
.top-term-label { background: rgba(255,255,255,0.85); border-radius: 4px; padding: 2px 6px; border: 1px solid rgba(0,0,0,0.15); box-shadow: 0 1px 3px rgba(0,0,0,0.3); color: #1a202c; font-weight: 600; }
`.trim();
    document.head.appendChild(style);
  }

  function setTopTermLegendCollapsed(collapsed) {
    topTermLegendCollapsed = !!collapsed;
    if (topTermLegendContainer) {
      if (topTermLegendCollapsed) {
        topTermLegendContainer.classList.add('collapsed');
      } else {
        topTermLegendContainer.classList.remove('collapsed');
      }
    }
    if (topTermLegendList) {
      if (topTermLegendCollapsed) {
        topTermLegendList.style.maxHeight = '0';
        topTermLegendList.style.marginTop = '0';
      } else {
        topTermLegendList.style.maxHeight = '200px';
        topTermLegendList.style.marginTop = '6px';
      }
    }
    if (topTermCollapseButton) {
      topTermCollapseButton.textContent = topTermLegendCollapsed ? '▾' : '▴';
      var label = topTermLegendCollapsed ? 'Expand term list' : 'Collapse term list';
      topTermCollapseButton.setAttribute('aria-label', label);
      topTermCollapseButton.title = label;
    }
  }

  function colorForTerm(term) {
    var key = String(term || '').trim();
    if (!key) {
      return '#cbd5e0';
    }
    if (!Object.prototype.hasOwnProperty.call(topTermColors, key)) {
      var color = topTermPalette[topTermPaletteIndex % topTermPalette.length];
      topTermPaletteIndex += 1;
      topTermColors[key] = color;
    }
    return topTermColors[key];
  }

  function lightenColor(color, ratio) {
    var base = String(color || '').trim();
    if (!base) { base = '#cbd5e0'; }
    var target = '#f7fafc';
    var parse = function(h) {
      h = String(h || '').replace('#','');
      if (h.length === 3) {
        h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
      }
      while (h.length < 6) {
        h += '0';
      }
      return {
        r: parseInt(h.slice(0,2), 16),
        g: parseInt(h.slice(2,4), 16),
        b: parseInt(h.slice(4,6), 16)
      };
    };
    var mix = function(a,b,t) { return Math.round(a + (b - a) * t); };
    var src = parse(base);
    var dst = parse(target);
    var t = Math.min(1, Math.max(0, ratio || 0.5));
    var r = mix(src.r, dst.r, t).toString(16).padStart(2, '0');
    var g = mix(src.g, dst.g, t).toString(16).padStart(2, '0');
    var b = mix(src.b, dst.b, t).toString(16).padStart(2, '0');
    return '#' + r + g + b;
  }

  function ensureTopTermLegend() {
    ensureTopTermStyle();
    if (topTermLegendContainer && document.body.contains(topTermLegendContainer)) {
      topTermLegendContainer.style.display = '';
      topTermLegendList = topTermLegendContainer.querySelector('.legend-body');
      topTermCollapseButton = topTermLegendContainer.querySelector('[data-top-term-collapse]');
      setTopTermLegendCollapsed(topTermLegendCollapsed);
      return;
    }
    var container = document.createElement('div');
    container.className = 'top-term-legend';
    container.setAttribute('data-top-term-legend', '1');

    var headerRow = document.createElement('div');
    headerRow.className = 'legend-controls';
    headerRow.style.display = 'flex';
    headerRow.style.alignItems = 'center';
    headerRow.style.gap = '8px';
    var title = document.createElement('h3');
    title.textContent = 'Top Terms';
    headerRow.appendChild(title);

    var headerSpacer = document.createElement('span');
    headerSpacer.className = 'legend-header-spacer';
    headerRow.appendChild(headerSpacer);

    var collapseBtn = document.createElement('button');
    collapseBtn.type = 'button';
    collapseBtn.className = 'legend-toggle';
    collapseBtn.setAttribute('data-top-term-collapse', '1');
    collapseBtn.textContent = topTermLegendCollapsed ? '▾' : '▴';
    collapseBtn.title = topTermLegendCollapsed ? 'Expand term list' : 'Collapse term list';
    collapseBtn.setAttribute('aria-label', collapseBtn.title);
    collapseBtn.addEventListener('click', function(ev) {
      ev.preventDefault();
      setTopTermLegendCollapsed(!topTermLegendCollapsed);
    });
    headerRow.appendChild(collapseBtn);

    var toggleLabel = document.createElement('label');
    toggleLabel.style.display = 'flex';
    toggleLabel.style.alignItems = 'center';
    toggleLabel.style.gap = '4px';
    toggleLabel.style.cursor = 'pointer';
    toggleLabel.style.cursor = 'pointer';
    var toggle = document.createElement('input');
    toggle.type = 'checkbox';
    toggle.checked = topTermLabelsEnabled;
    toggle.addEventListener('change', function() {
      topTermLabelsEnabled = !!this.checked;
      if (activeMap) {
        refreshCollocateSizes(activeMap);
        updateCollocateSummaryLineFromDataset(popupCache);
      }
    });
    toggleLabel.appendChild(toggle);
    toggleLabel.appendChild(document.createTextNode('Show labels'));
    headerRow.appendChild(toggleLabel);

    var list = document.createElement('div');
    list.className = 'legend-body';

    container.appendChild(headerRow);
    container.appendChild(list);

    var parent = document.querySelector('.leaflet-container') || document.body;
    parent.appendChild(container);

    // Prevent map interactions when using the legend
    try {
      addScrollGuards(container);
      if (typeof L !== 'undefined' && L && L.DomEvent) {
        if (typeof L.DomEvent.disableClickPropagation === 'function') {
          L.DomEvent.disableClickPropagation(container);
        }
        if (typeof L.DomEvent.disableScrollPropagation === 'function') {
          L.DomEvent.disableScrollPropagation(container);
        }
      } else {
        container.addEventListener('click', function(ev) {
          if (typeof ev.stopPropagation === 'function') {
            ev.stopPropagation();
          }
        });
        container.addEventListener('dblclick', function(ev) {
          if (typeof ev.stopPropagation === 'function') {
            ev.stopPropagation();
          }
        });
        container.addEventListener('wheel', function(ev) {
          if (typeof ev.stopPropagation === 'function') {
            ev.stopPropagation();
          }
        }, { passive: true });
      }
    } catch (legendGuardErr) {
      // best-effort
    }

    topTermLegendContainer = container;
    topTermLegend = container;
    topTermLegendList = list;
    topTermLabelToggle = toggle;
    topTermCollapseButton = collapseBtn;
    setTopTermLegendCollapsed(topTermLegendCollapsed);
  }

  function removeTopTermLegend() {
    if (topTermLegendContainer) {
      topTermLegendContainer.style.display = 'none';
    }
  }

  function updateTopTermLegendActiveState() {
    if (!topTermLegendItems) {
      return;
    }
    var active = (topTermActiveFilterTerm || '').trim();
    Object.keys(topTermLegendItems).forEach(function(term) {
      var el = topTermLegendItems[term];
      if (!el) { return; }
      var isActive = active && term === active;
      el.dataset.active = isActive ? '1' : '0';
      el.style.opacity = (active && !isActive) ? '0.5' : '1';
    });
  }

  function setTopTermFilter(term) {
    var value = String(term || '').trim();
    if (topTermActiveFilterTerm === value) {
      topTermActiveFilterTerm = '';
    } else {
      topTermActiveFilterTerm = value;
    }
    updateTopTermLegendActiveState();
    if (activeMap) {
      refreshCollocateSizes(activeMap);
      updateCollocateSummaryLineFromDataset(popupCache);
    }
  }

  function updateTopTermLegend(termCounts) {
    if (!isTopTermMode()) {
      removeTopTermLegend();
      return;
    }
    ensureTopTermLegend();
    if (!topTermLegendList) {
      return;
    }
    while (topTermLegendList.firstChild) {
      topTermLegendList.removeChild(topTermLegendList.firstChild);
    }
    topTermLegendItems = {};
    var entries = Object.keys(termCounts || {}).map(function(term) {
      return { term: term, count: termCounts[term] || 0 };
    });
    entries.sort(function(a, b) {
      if (b.count !== a.count) {
        return b.count - a.count;
      }
      return a.term.localeCompare(b.term);
    });
    if (!entries.length) {
      var empty = document.createElement('div');
      empty.style.color = '#4a5568';
      empty.style.fontStyle = 'italic';
      empty.textContent = 'No collocate terms available.';
      topTermLegendList.appendChild(empty);
      return;
    }
    entries.forEach(function(entry) {
      var item = document.createElement('div');
      item.className = 'legend-item';
      item.setAttribute('data-term', entry.term);
      var swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.background = colorForTerm(entry.term);
      var termLabel = document.createElement('span');
      termLabel.className = 'term-label';
      termLabel.textContent = entry.term;
      var countLabel = document.createElement('span');
      countLabel.className = 'count-label';
      var countText = entry.count === 1 ? '1 city' : entry.count + ' cities';
      countLabel.textContent = countText;
      item.appendChild(swatch);
      item.appendChild(termLabel);
      item.appendChild(countLabel);
      item.addEventListener('click', function() {
        setTopTermFilter(entry.term);
      });
      topTermLegendItems[entry.term] = item;
      topTermLegendList.appendChild(item);
    });
    if (topTermLabelToggle) {
      topTermLabelToggle.checked = topTermLabelsEnabled;
    }
    updateTopTermLegendActiveState();
    setTopTermLegendCollapsed(topTermLegendCollapsed);
  }

  function applyTopTermLabelState(layer, term) {
    if (!layer || typeof layer.setTooltipContent === 'function') {
      // handled later via bind/unbind
    }
    var labelText = term ? String(term) : '';
    if (!labelText) {
      if (typeof layer.unbindTooltip === 'function') {
        try { layer.unbindTooltip(); } catch (err) {}
      }
      return;
    }
    var highlighted = !(layer && layer.options && layer.options.topTermHighlighted === false);
    var baseColor = colorForTerm(term);
    var bgColor = highlighted ? baseColor : lightenColor(baseColor, 0.7);
    var textColor = highlighted ? '#1a202c' : '#4a5568';
    if (topTermLabelsEnabled) {
      if (typeof layer.unbindTooltip === 'function') {
        try { layer.unbindTooltip(); } catch (err) {}
      }
      try {
        layer.bindTooltip(labelText, {
          permanent: true,
          direction: 'center',
          className: 'top-term-label',
          opacity: highlighted ? 1 : 0.6,
        });
        var tooltip = layer.getTooltip && layer.getTooltip();
        if (tooltip && typeof tooltip.getElement === 'function') {
          var el = tooltip.getElement();
          if (el) {
            el.style.background = bgColor;
            el.style.borderColor = 'rgba(0,0,0,0.3)';
            el.style.color = textColor;
          }
        }
      } catch (bindErr) {}
    } else {
      if (typeof layer.unbindTooltip === 'function') {
        try { layer.unbindTooltip(); } catch (err) {}
      }
      try {
        layer.bindTooltip(labelText, {
          permanent: false,
          direction: 'top',
          opacity: highlighted ? 1 : 0.6,
          sticky: true,
        });
        var tooltip = layer.getTooltip && layer.getTooltip();
        if (tooltip && typeof tooltip.getElement === 'function') {
          var el = tooltip.getElement();
          if (el) {
            el.style.background = bgColor;
            el.style.borderColor = 'rgba(0,0,0,0.2)';
            el.style.color = textColor;
          }
        }
      } catch (hoverErr) {}
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function highlightAndEscape(raw) {
    if (!raw) {
      return '';
    }
    if (!searchTerm) {
      return escapeHtml(raw);
    }
    var text = String(raw);
    var lowerText = text.toLowerCase();
    var termLower = searchTerm.toLowerCase();
    if (!termLower) {
      return escapeHtml(text);
    }
    var termLength = termLower.length;
    var idx = 0;
    var next = lowerText.indexOf(termLower, idx);
    var pieces = '';
    while (next !== -1) {
      pieces += escapeHtml(text.slice(idx, next));
      pieces += '<mark>' + escapeHtml(text.slice(next, next + termLength)) + '</mark>';
      idx = next + termLength;
      if (termLength === 0) {
        idx += 1;
      }
      next = lowerText.indexOf(termLower, idx);
    }
    pieces += escapeHtml(text.slice(idx));
    return pieces;
  }

  function cssEscapeValue(value) {
    var str = String(value);
    if (typeof CSS !== 'undefined' && CSS && typeof CSS.escape === 'function') {
      return CSS.escape(str);
    }
    return str.replace(/[^a-zA-Z0-9_-]/g, function(ch) {
      var hex = ch.charCodeAt(0).toString(16).toUpperCase();
      return '\\\\' + hex + ' ';
    });
  }

  var articleLoader = (function() {
    var cache = {};
    var doc = null;
    var pending = null;

    function loadHtml() {
      if (!attrTableUrl) {
        return Promise.reject(new Error('No attribute table.'));
      }

      function xhrPromise() {
        return new Promise(function(resolve, reject) {
          try {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', attrTableUrl, true);
            xhr.onreadystatechange = function() {
              if (xhr.readyState === 4) {
                if (xhr.status === 0 || (xhr.status >= 200 && xhr.status < 300)) {
                  resolve(xhr.responseText);
                } else {
                  reject(new Error('status ' + xhr.status));
                }
              }
            };
            xhr.onerror = function() {
              reject(new Error('network error'));
            };
            xhr.send();
          } catch (xhrErr) {
            reject(xhrErr);
          }
        });
      }

      if (typeof fetch === 'function') {
        return fetch(attrTableUrl)
          .then(function(resp) {
            if (!resp.ok && resp.status !== 0) {
              throw new Error('status ' + resp.status);
            }
            return resp.text();
          })
          .catch(function() {
            return xhrPromise();
          });
      }

      return xhrPromise();
    }

    function requestDocument(callback) {
      if (!attrTableUrl) {
        callback(null);
        return;
      }
      if (doc) {
        callback(doc);
        return;
      }
      if (pending) {
        pending.push(callback);
        return;
      }
      pending = [callback];
      loadHtml()
        .then(function(html) {
          if (!html) {
            throw new Error('Empty attribute table response');
          }
          var parser = new DOMParser();
          doc = parser.parseFromString(html, 'text/html');
          var callbacks = pending || [];
          pending = null;
          callbacks.forEach(function(cb) {
            cb(doc);
          });
        })
        .catch(function(err) {
          console.error('Failed to load attribute table', err);
          var callbacks = pending || [];
          pending = null;
          callbacks.forEach(function(cb) {
            cb(null);
          });
        });
    }

    function get(rowId, callback) {
      if (!rowId) {
        callback(null);
        return;
      }
      if (Object.prototype.hasOwnProperty.call(cache, rowId)) {
        callback(cache[rowId]);
        return;
      }
      requestDocument(function(docNode) {
        if (!docNode) {
          callback(null);
          return;
        }
        var selector = '[data-entry-key="' + cssEscapeValue(rowId) + '"] td[data-column="article"]';
        var cell = docNode.querySelector(selector);
        if (!cell) {
          cache[rowId] = null;
          callback(null);
          return;
        }
        var text = cell.textContent || '';
        cache[rowId] = text;
        callback(text);
      });
    }

    return {
      get: get,
    };
  })();

  function currentTimeKey(mapObj) {
    if (collocateTimeEnabled) {
      if (!selectedTimeBinKey) {
        return '';
      }
      return String(selectedTimeBinKey);
    }
    if (!mapObj || !mapObj.timeDimension || typeof mapObj.timeDimension.getCurrentTime !== 'function') {
      return '';
    }
    var t = mapObj.timeDimension.getCurrentTime();
    try { return new Date(t).toISOString().replace('.000Z','Z'); } catch(e) { return ''; }
  }
  function cityKey(city, state) {
    var c = String(city||'').trim().toLowerCase();
    var s = String(state||'').trim().toLowerCase();
    return c + '||' + s;
  }
  function lookupRankForTerm(data, timeKey, term) {
    if (!collocateRanks || !term) {
      return null;
    }
    var ck = cityKey(data && data.city, data && data.state);
    if (!ck || !Object.prototype.hasOwnProperty.call(collocateRanks, ck)) {
      return null;
    }
    var byCity = collocateRanks[ck];
    if (timeKey && byCity) {
      var candidates = resolveTimeKeys(timeKey);
      if (!Array.isArray(candidates) || !candidates.length) {
        candidates = [String(timeKey)];
      }
      for (var i = 0; i < candidates.length; i++) {
        var key = candidates[i];
        if (!key || !Object.prototype.hasOwnProperty.call(byCity, key)) {
          continue;
        }
        var bucket = byCity[key];
        if (bucket && Object.prototype.hasOwnProperty.call(bucket, term)) {
          var val = Number(bucket[term]);
          return Number.isFinite(val) ? val : null;
        }
      }
    }
    if ((!collocateTimeEnabled || !timeKey) && byCity && Object.prototype.hasOwnProperty.call(byCity, '') && byCity[''] && Object.prototype.hasOwnProperty.call(byCity[''], term)) {
      var baseVal = Number(byCity[''][term]);
      return Number.isFinite(baseVal) ? baseVal : null;
    }
    return null;
  }
  function countArticlesForTerm(data, term, timeKey) {
    if (!data || !term || !data.collocate_hits || !Object.prototype.hasOwnProperty.call(data.collocate_hits, term)) {
      return 0;
    }
    var hits = data.collocate_hits[term];
    if (!Array.isArray(hits) || !hits.length) {
      return 0;
    }
    if (timeKey) {
      var allowed = collectTimeIndexes(data, timeKey);
      if (!allowed || !allowed.size) {
        return 0;
      }
      var count = 0;
      hits.forEach(function(idx) {
        var num = Number(idx);
        if (Number.isFinite(num) && allowed.has(num)) {
          count += 1;
        }
      });
      return count;
    }
    return hits.length;
  }
  function computeSelectedTermInfo(data, timeKey) {
    var term = String(selectedCollocate || '').trim();
    if (!term) {
      var emptyInfo = { term: '', rank: null, count: 0 };
      data._currentTopTermInfo = emptyInfo;
      return emptyInfo;
    }
    var rank = lookupRankForTerm(data, timeKey, term);
    var count = Array.isArray(data.entries) ? data.entries.length : 0;
    var info = {
      term: term,
      rank: Number.isFinite(rank) ? Number(rank) : null,
      count: count,
    };
    data._currentTopTermInfo = info;
    return info;
  }
  function computeTopTermInfo(data, timeKey) {
    var bestTerm = '';
    var bestCount = 0;
    var bestRankScore = Number.POSITIVE_INFINITY;
    var evaluateTerm = function(term) {
      if (!term) { return; }
      var count = countArticlesForTerm(data, term, timeKey);
      if (!Number.isFinite(count) || count <= 0) {
        return;
      }
      var rankVal = lookupRankForTerm(data, timeKey, term);
      var rankScore = Number.isFinite(rankVal) ? Number(rankVal) : Number.POSITIVE_INFINITY;
      if (count > bestCount || (count === bestCount && (rankScore < bestRankScore || (rankScore === bestRankScore && term < bestTerm)))) {
        bestTerm = term;
        bestCount = count;
        bestRankScore = rankScore;
      }
    };

    if (data && data.collocate_hits) {
      Object.keys(data.collocate_hits).forEach(evaluateTerm);
    }

    if (!bestTerm && collocateRanks) {
      var ck = cityKey(data && data.city, data && data.state);
      if (ck && Object.prototype.hasOwnProperty.call(collocateRanks, ck)) {
        var byCity = collocateRanks[ck];
        if (byCity) {
          var rankMap = null;
          var variants = resolveTimeKeys(timeKey);
          if (Array.isArray(variants) && variants.length) {
            for (var i = 0; i < variants.length; i++) {
              var key = variants[i];
              if (key && Object.prototype.hasOwnProperty.call(byCity, key)) {
                rankMap = byCity[key];
                if (rankMap) { break; }
              }
            }
          }
          if (!rankMap && Object.prototype.hasOwnProperty.call(byCity, '')) {
            rankMap = byCity[''];
          }
          if (rankMap) {
            Object.keys(rankMap).forEach(function(term) {
              evaluateTerm(term);
            });
          }
        }
      }
    }

    if (!bestTerm) {
      var emptyInfo = { term: '', rank: null, count: 0 };
      data._currentTopTermInfo = emptyInfo;
      return emptyInfo;
    }
    var finalRank = Number.isFinite(bestRankScore) ? Number(bestRankScore) : lookupRankForTerm(data, timeKey, bestTerm);
    if (!Number.isFinite(finalRank)) {
      finalRank = null;
    }
    var info = {
      term: bestTerm,
      rank: finalRank,
      count: bestCount,
    };
    data._currentTopTermInfo = info;
    return info;
  }
  function getTermInfoForGroup(data, timeKey) {
    if (isTopTermMode()) {
      return computeTopTermInfo(data, timeKey);
    }
    return computeSelectedTermInfo(data, timeKey);
  }
  function rankToRadius(rank) {
    if (!Number.isFinite(rank) || rank <= 0) return 3;
    var rMin = 3, rMax = 18;
    if (!rankMax || rankMax <= 1) return rMax;
    var t = 1 - ((rank - 1) / (rankMax - 1));
    return Math.max(rMin, Math.min(rMax, rMin + t * (rMax - rMin)));
  }
  function countToRadius(count, maxCount) {
    var minR = 4;
    var maxR = 22;
    var safeCount = Number(count);
    if (!Number.isFinite(safeCount) || safeCount <= 0) {
      return minR;
    }
    var safeMax = Number(maxCount);
    if (!Number.isFinite(safeMax) || safeMax <= 0) {
      safeMax = safeCount;
    }
    var ratio = safeCount / safeMax;
    ratio = Math.min(1, Math.max(0, ratio));
    var eased = Math.sqrt(ratio);
    return Math.max(minR, Math.min(maxR, minR + eased * (maxR - minR)));
  }
  function refreshCollocateSizes(mapObj) {
    if (!mapObj || typeof mapObj.eachLayer !== 'function') {
      return;
    }
    loadData(function(dataset) {
      if (!dataset) {
        return;
      }
      // Determine current time key first, then filter dataset accordingly
      var timeKey = currentTimeKey(mapObj);
      applyCollocateFilterToDataset(dataset, timeKey || null);
      var baseColor = '#2b6cb0';
      var colorScale = null;
      if (collocateColorize) {
        colorScale = createColorScale(dataset);
      }
      // timeKey already resolved above
      var updates = [];
      mapObj.eachLayer(function(layer) {
        if (!layer || !layer.options || !layer.options.groupId) {
          return;
        }
        var gid = layer.options.groupId;
        var data = dataset[gid];
        if (!data) {
          return;
        }
        var info = getTermInfoForGroup(data, timeKey);
        updates.push({ layer: layer, data: data, info: info || { term: '', rank: null, count: 0 } });
      });
      var maxCount = 0;
      if (isTopTermMode()) {
        updates.forEach(function(entry) {
          var count = Number(entry.info.count) || 0;
          if (count > maxCount) {
            maxCount = count;
          }
        });
        if (!Number.isFinite(maxCount) || maxCount <= 0) {
          maxCount = 1;
        }
      }
      var highlightTerm = (topTermActiveFilterTerm || '').trim();
      var termCounts = {};
      updates.forEach(function(entry) {
        var layer = entry.layer;
        var data = entry.data;
        var info = entry.info || { term: '', rank: null, count: 0 };
        var rank = Number.isFinite(info.rank) ? Number(info.rank) : null;
        var term = info.term || '';
        var count = Number(info.count) || 0;
        var hasArticles = count > 0;
        var radius = isTopTermMode() ? countToRadius(count, maxCount) : rankToRadius(rank);
        if (!isTopTermMode()) {
          count = Array.isArray(data.entries) ? data.entries.length : 0;
          hasArticles = count > 0;
        }
        if (typeof layer.setRadius === 'function') {
          layer.setRadius(radius);
        } else if (layer.options) {
          layer.options.radius = radius;
          if (layer._radius && typeof layer.redraw === 'function') {
            try { layer.redraw(); } catch (e) {}
          }
        }
        layer.options = layer.options || {};
        layer.options.collocateRank = rank;
        layer.options.currentCollocateTerm = term;
        layer.options.currentCollocateCount = count;
        var collocateCount = count;
        var fillColor = baseColor;
        var strokeColor = baseColor;
        var smallMarker = !hasArticles;
        if (!smallMarker) {
          if (isTopTermMode()) {
            fillColor = colorForTerm(term);
            strokeColor = '#2d3748';
          } else if (collocateColorize && typeof colorScale === 'function') {
            fillColor = colorScale(collocateCount);
            strokeColor = '#4a5568';
          }
        } else {
          radius = Math.max(2.5, Math.min(radius, 3));
          if (typeof layer.setRadius === 'function') {
            layer.setRadius(radius);
          } else {
            layer.options.radius = radius;
          }
          fillColor = '#4a5568';
          strokeColor = '#2d3748';
        }
        var visible = !smallMarker;
        var isHighlighted = !highlightTerm || (term && term === highlightTerm);
        layer.options.topTermHighlighted = isHighlighted;
        if (!smallMarker && isTopTermMode() && term && visible) {
          termCounts[term] = (termCounts[term] || 0) + 1;
        }
        if (!smallMarker && isTopTermMode() && highlightTerm && !isHighlighted) {
          fillColor = lightenColor(fillColor, 0.6);
          strokeColor = lightenColor(strokeColor, 0.6);
        }
        var baseOpacity = (typeof layer.options.baseOpacity === 'number') ? layer.options.baseOpacity : (typeof layer.options.opacity === 'number' ? layer.options.opacity : 0.5);
        var baseFill = (typeof layer.options.baseFillOpacity === 'number') ? layer.options.baseFillOpacity : (typeof layer.options.fillOpacity === 'number' ? layer.options.fillOpacity : 0.85);
        var opacityMultiplier = (!smallMarker && isTopTermMode() && highlightTerm && !isHighlighted) ? 0.25 : 1;
        var layerOpacity;
        var layerFillOpacity;
        if (smallMarker) {
          layerOpacity = 0.8;
          layerFillOpacity = 0.6;
        } else {
          layerOpacity = baseOpacity * opacityMultiplier;
          layerFillOpacity = baseFill * opacityMultiplier;
        }
        if (typeof layer.setStyle === 'function') {
          layer.setStyle({
            color: strokeColor,
            fillColor: fillColor,
            weight: collocateColorize ? 1.0 : (typeof layer.options.weight === 'number' ? layer.options.weight : 1),
            opacity: layerOpacity,
            fillOpacity: layerFillOpacity,
          });
        }
        layer.options.color = strokeColor;
        layer.options.fillColor = fillColor;
        layer.options.weight = collocateColorize ? 1.0 : (typeof layer.options.weight === 'number' ? layer.options.weight : 1);
        layer.options.opacity = layerOpacity;
        layer.options.fillOpacity = layerFillOpacity;
        var isGhost = !!layer.options.ghostMarker;
        var interactive = !isGhost && !smallMarker && (!highlightTerm || isHighlighted);
        layer.options.interactive = interactive;
        if (layer._path && layer._path.style) {
          layer._path.style.pointerEvents = interactive ? 'auto' : 'none';
          layer._path.style.opacity = layerOpacity;
          layer._path.style.fillOpacity = layerFillOpacity;
        }
        if (!interactive && typeof layer.closePopup === 'function') {
          layer.closePopup();
        }
        if (!interactive && currentHighlightId === layer.options.groupId) {
          clearHighlight();
        }
        if (!smallMarker && isTopTermMode()) {
          applyTopTermLabelState(layer, term);
        } else {
          var labelText = '';
          // Only show a label when the location has visible entries in this time bin
          if (!smallMarker && Number.isFinite(rank) && rank > 0) {
            labelText = String(Math.round(rank));
          }
          if (typeof layer.bindTooltip === 'function') {
            var tooltip = (typeof layer.getTooltip === 'function') ? layer.getTooltip() : null;
            if (!tooltip) {
              try {
                layer.bindTooltip(labelText, {
                  permanent: true,
                  direction: 'center',
                  className: 'collocate-rank-label',
                  opacity: 1,
                });
                tooltip = (typeof layer.getTooltip === 'function') ? layer.getTooltip() : null;
              } catch (tooltipErr) {
                tooltip = null;
              }
            }
            if (typeof layer.setTooltipContent === 'function') {
              try { layer.setTooltipContent(labelText); } catch (setErr) {}
            } else if (tooltip && typeof tooltip.setContent === 'function') {
              tooltip.setContent(labelText);
            }
            var tooltipEl = null;
            if (tooltip && typeof tooltip.getElement === 'function') {
              tooltipEl = tooltip.getElement();
            } else {
              var tmpTooltip = (typeof layer.getTooltip === 'function') ? layer.getTooltip() : null;
              if (tmpTooltip && typeof tmpTooltip.getElement === 'function') {
                tooltipEl = tmpTooltip.getElement();
              }
            }
            if (tooltipEl) {
              tooltipEl.style.display = labelText ? '' : 'none';
              var fontSize = Math.max(11, Math.round(radius * 1.05));
              tooltipEl.style.fontSize = fontSize + 'px';
            }
          }
        }
      });
      refreshPopupAfterFilter(mapObj);
      updateCollocateSummaryLineFromDataset(dataset);
      if (isTopTermMode()) {
        updateTopTermLegend(termCounts);
      } else {
        removeTopTermLegend();
      }
      updateHighlightFromDataset(dataset);
    });
  }

  function loadData(callback) {
    if (popupCache) { callback(popupCache); return; }
    var tag = document.getElementById('popup-data');
    if (!tag) { callback({}); return; }
    try {
      popupCache = JSON.parse(tag.textContent || tag.innerText || '{}');
    } catch (err) {
      console.error('Failed to parse map popup data', err);
      popupCache = {};
    }
    if (popupCache && typeof popupCache === 'object') {
      Object.keys(popupCache).forEach(function(key) {
        var data = popupCache[key];
        if (!data) {
          return;
        }
        if (!Array.isArray(data.full_entries) && Array.isArray(data.entries)) {
          data.full_entries = data.entries.slice();
        }
        var baseSource = Array.isArray(data.full_entries) ? data.full_entries : (Array.isArray(data.entries) ? data.entries : []);
        var baseCount = Number.isFinite(data.full_article_count) ? Number(data.full_article_count) : (Array.isArray(baseSource) ? baseSource.length : 0);
        setBaseEntries(data, baseSource, baseCount);
      });
        applyCollocateFilterToDataset(popupCache, null);
      updateCollocateSummaryLineFromDataset(popupCache);
    }
    callback(popupCache);
  }
  function formatTimeKey(timeValue) {
    if (timeValue === null || typeof timeValue === 'undefined') {
      return null;
    }
    if (typeof timeValue === 'string') {
      var trimmed = timeValue.replace(/\s+$$/, '');
      if (/T/.test(trimmed)) {
        return trimmed.replace('.000Z', 'Z');
      }
      var numeric = Number(trimmed);
      if (Number.isFinite(numeric)) {
        return formatTimeKey(numeric);
      }
    }
    try {
      var date = new Date(timeValue);
      if (Number.isNaN(date.getTime())) {
        return null;
      }
      var iso = date.toISOString();
      return iso.replace('.000Z', 'Z');
    } catch (err) {
      return null;
    }
  }

  function resolveTimeKeys(timeValue) {
    var keys = [];
    if (timeValue === null || typeof timeValue === 'undefined') {
      return keys;
    }
    var primary = String(timeValue);
    keys.push(primary);
    var formatted = formatTimeKey(timeValue);
    if (formatted && keys.indexOf(formatted) === -1) {
      keys.push(formatted);
    }
    if (Array.isArray(timeLabels) && timeLabels.length) {
      var isoCandidates = [];
      if (formatted) {
        isoCandidates.push(formatted);
      }
      if (typeof primary === 'string' && /T/.test(primary)) {
        isoCandidates.push(primary.replace('.000Z', 'Z'));
      }
      isoCandidates.forEach(function(iso) {
        if (!iso) {
          return;
        }
        timeLabels.forEach(function(label, idx) {
          if (label === iso) {
            var keyStr = String(idx + 1);
            if (keys.indexOf(keyStr) === -1) {
              keys.push(keyStr);
            }
          }
        });
      });
    }
    if (typeof timeValue === 'number' && Array.isArray(timeLabels) && timeLabels.length) {
      var idx = Math.round(timeValue);
      if (Number.isFinite(idx) && idx >= 1 && idx <= timeLabels.length) {
        var label = timeLabels[idx - 1];
        if (label && keys.indexOf(label) === -1) {
          keys.push(label);
        }
      }
    }
    if (collocateTimeEnabled && Array.isArray(collocateTimeBins) && collocateTimeBins.length) {
      collocateTimeBins.forEach(function(entry) {
        if (!entry) {
          return;
        }
        if (entry.key === primary || entry.label === primary || entry.iso === primary) {
          if (entry.key && keys.indexOf(entry.key) === -1) {
            keys.push(entry.key);
          }
          if (entry.iso && keys.indexOf(entry.iso) === -1) {
            keys.push(entry.iso);
          }
        }
      });
    }
    if (typeof timeValue === 'string' && Array.isArray(timeLabels) && timeLabels.length) {
      var parsed = Number(timeValue);
      if (Number.isFinite(parsed) && parsed >= 1 && parsed <= timeLabels.length) {
        var altLabel = timeLabels[parsed - 1];
        if (altLabel && keys.indexOf(altLabel) === -1) {
          keys.push(altLabel);
        }
      }
    }
    return keys;
  }

  function collectTimeIndexes(data, timeKey) {
    if (!timeKey || !data || !data.time_bins) {
      return null;
    }
    var variants = resolveTimeKeys(timeKey);
    if (!Array.isArray(variants) || !variants.length) {
      variants = [String(timeKey)];
    }
    var allowed = new Set();
    var matched = false;
    variants.forEach(function(key) {
      if (!key || !Object.prototype.hasOwnProperty.call(data.time_bins, key)) {
        return;
      }
      var bin = data.time_bins[key];
      if (!bin || !Array.isArray(bin.indexes) || !bin.indexes.length) {
        return;
      }
      matched = true;
      bin.indexes.forEach(function(idx) {
        var num = Number(idx);
        if (Number.isFinite(num)) {
          allowed.add(num);
        }
      });
    });
    if (!matched || !allowed.size) {
      return null;
    }
    return allowed;
  }

  function labelForTimeKey(key) {
    if (!collocateTimeEnabled || !collocateTimeBins.length) {
      return '';
    }
    if (!key) {
      return 'All bins';
    }
    for (var i = 0; i < collocateTimeBins.length; i++) {
      var entry = collocateTimeBins[i];
      if (entry && entry.key === key) {
        return entry.label || entry.key;
      }
    }
    return key;
  }

  function initCollocateTimeControls() {
    if (!collocateTimeEnabled || !collocateTimeBins.length) {
      collocateTimeManuallyDisabled = false;
      collocateTimeLastActiveIndex = 1;
      return;
    }
    var host = document.getElementById('collocateTimeSliderContainer');
    if (!host) {
      return;
    }
    host.innerHTML = '';

    var title = document.createElement('div');
    title.style.display = 'flex';
    title.style.alignItems = 'baseline';
    title.style.gap = '6px';
    title.style.fontWeight = '600';
    title.style.flexWrap = 'wrap';

    var titlePrefix = document.createElement('span');
    titlePrefix.textContent = 'Time bin:';

    var sliderLabel = document.createElement('span');
    sliderLabel.id = 'collocateTimeRangeLabel';
    sliderLabel.style.fontWeight = '600';
    sliderLabel.style.whiteSpace = 'nowrap';
    sliderLabel.style.flex = '0 0 auto';
    sliderLabel.style.color = '#2b6cb0';

    title.appendChild(titlePrefix);
    title.appendChild(sliderLabel);
    host.appendChild(title);

    var row = document.createElement('div');
    row.style.marginTop = '4px';
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '6px';
    row.style.flexWrap = 'nowrap';

    var prevBtn = document.createElement('button');
    prevBtn.type = 'button';
    prevBtn.textContent = '‹';
    prevBtn.title = 'Previous time bin';
    prevBtn.className = 'collocate-time-button';

    var slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '1';
    slider.max = String(collocateTimeBins.length);
    slider.step = '1';
    slider.id = 'collocateTimeRange';
    slider.style.width = '200px';
    slider.style.flex = '1 1 160px';
    slider.style.minWidth = '140px';
    slider.setAttribute('aria-label', 'Collocate time bin');

    var nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.textContent = '›';
    nextBtn.title = 'Next time bin';
    nextBtn.className = 'collocate-time-button';

    var toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.className = 'collocate-time-button';
    toggleBtn.style.border = 'none';
    toggleBtn.style.background = 'none';
    toggleBtn.style.cursor = 'pointer';
    toggleBtn.style.fontSize = '16px';
    toggleBtn.style.lineHeight = '1';
    toggleBtn.style.padding = '2px 6px';
    toggleBtn.style.flex = '0 0 auto';
    toggleBtn.style.minWidth = '32px';
    toggleBtn.style.textAlign = 'center';
    toggleBtn.dataset.clockRole = 'toggle';
    toggleBtn.textContent = '🕒';

    row.appendChild(prevBtn);
    row.appendChild(slider);
    row.appendChild(nextBtn);
    row.appendChild(toggleBtn);
    host.appendChild(row);
    consumeDragEvents(host);
    consumeDragEvents(title);
    consumeDragEvents(row);
    consumeDragEvents(prevBtn);
    consumeDragEvents(nextBtn);
    consumeDragEvents(slider);
    consumeDragEvents(toggleBtn);

    var totalBins = collocateTimeBins.length;
    if (!Number.isFinite(collocateTimeLastActiveIndex) || collocateTimeLastActiveIndex < 1) {
      collocateTimeLastActiveIndex = 1;
    }
    if (collocateTimeLastActiveIndex > totalBins) {
      collocateTimeLastActiveIndex = totalBins || 1;
    }

    var defaultIndex = collocateTimeLastActiveIndex;
    if (!collocateTimeManuallyDisabled && selectedTimeBinKey) {
      var matched = false;
      for (var i = 0; i < totalBins; i++) {
        if (collocateTimeBins[i] && collocateTimeBins[i].key === selectedTimeBinKey) {
          defaultIndex = i + 1;
          collocateTimeLastActiveIndex = defaultIndex;
          matched = true;
          break;
        }
      }
      if (!matched) {
        selectedTimeBinKey = null;
        defaultIndex = collocateTimeLastActiveIndex;
      }
    }

    if (!Number.isFinite(defaultIndex) || defaultIndex < 1) {
      defaultIndex = totalBins ? 1 : 1;
    }
    if (defaultIndex > totalBins) {
      defaultIndex = totalBins;
    }

    slider.value = String(Math.max(1, defaultIndex));

    function updateLabel(emit) {
      if (collocateTimeManuallyDisabled) {
        sliderLabel.textContent = 'All bins';
        selectedTimeBinKey = null;
        if (emit && activeMap) {
          applyTimeFilter(activeMap);
          refreshCollocateSizes(activeMap);
          updateCollocateSummaryLineFromDataset(popupCache);
        }
        return;
      }
      if (totalBins < 1) {
        sliderLabel.textContent = '';
        selectedTimeBinKey = null;
        if (emit && activeMap) {
          applyTimeFilter(activeMap);
          refreshCollocateSizes(activeMap);
          updateCollocateSummaryLineFromDataset(popupCache);
        }
        return;
      }
      var idx = parseInt(slider.value || '1', 10);
      if (!Number.isFinite(idx) || idx < 1) {
        idx = 1;
      }
      if (idx > totalBins) {
        idx = totalBins;
      }
      if (idx < 1) {
        idx = 1;
      }
      slider.value = String(idx);
      var entry = collocateTimeBins[Math.min(totalBins - 1, Math.max(0, idx - 1))];
      if (entry) {
        selectedTimeBinKey = entry.key;
        sliderLabel.textContent = entry.label || entry.key;
        collocateTimeLastActiveIndex = idx;
      } else {
        selectedTimeBinKey = null;
        sliderLabel.textContent = '';
      }
      if (emit && activeMap) {
        applyTimeFilter(activeMap);
        refreshCollocateSizes(activeMap);
        updateCollocateSummaryLineFromDataset(popupCache);
      }
    }

    function updateToggleState() {
      if (collocateTimeManuallyDisabled) {
        toggleBtn.textContent = '🕒';
        toggleBtn.title = 'Time disabled (all bins showing)';
        toggleBtn.setAttribute('aria-pressed', 'true');
        toggleBtn.style.color = '#c53030';
        slider.disabled = true;
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        sliderLabel.style.color = '#c53030';
        toggleBtn.dataset.clockDisabled = '1';
        sliderLabel.textContent = 'All bins';
      } else {
        toggleBtn.textContent = '🕒';
        toggleBtn.title = 'Disable Time';
        toggleBtn.setAttribute('aria-pressed', 'false');
        toggleBtn.style.color = '#2b6cb0';
        slider.disabled = false;
        prevBtn.disabled = false;
        nextBtn.disabled = false;
        sliderLabel.style.color = '#2b6cb0';
        delete toggleBtn.dataset.clockDisabled;
      }
      toggleBtn.setAttribute('aria-label', toggleBtn.title);
    }

    slider.addEventListener('input', function() {
      if (collocateTimeManuallyDisabled) {
        return;
      }
      updateLabel(true);
    });

    function stepSlider(delta) {
      if (collocateTimeManuallyDisabled) {
        return;
      }
      var current = parseInt(slider.value || '1', 10);
      if (!Number.isFinite(current) || current < 1) {
        current = 1;
      }
      var maxVal = parseInt(slider.max || '0', 10);
      if (!Number.isFinite(maxVal) || maxVal < 1) {
        maxVal = collocateTimeBins.length;
      }
      if (maxVal < 1) {
        return;
      }
      var nextVal = current + delta;
      if (nextVal < 1) {
        nextVal = maxVal;
      }
      if (nextVal > maxVal) {
        nextVal = 1;
      }
      slider.value = String(nextVal);
      updateLabel(true);
    }

    prevBtn.addEventListener('click', function() {
      stepSlider(-1);
    });
    nextBtn.addEventListener('click', function() {
      stepSlider(1);
    });

    toggleBtn.addEventListener('click', function() {
      if (!collocateTimeManuallyDisabled) {
        var currentVal = parseInt(slider.value || '1', 10);
        if (Number.isFinite(currentVal) && currentVal >= 1 && currentVal <= totalBins) {
          collocateTimeLastActiveIndex = currentVal;
        }
        collocateTimeManuallyDisabled = true;
        updateToggleState();
        updateLabel(true);
        return;
      }
      collocateTimeManuallyDisabled = false;
      var maxVal = parseInt(slider.max || '0', 10);
      if (!Number.isFinite(maxVal) || maxVal < 1) {
        maxVal = collocateTimeBins.length;
      }
      var restore = parseInt(collocateTimeLastActiveIndex || '1', 10);
      if (!Number.isFinite(restore) || restore < 1) {
        restore = 1;
      }
      if (restore > maxVal) {
        restore = maxVal || 1;
      }
      slider.value = String(restore);
      updateToggleState();
      updateLabel(true);
    });

    updateToggleState();
    updateLabel(false);
  }


  function formatInteger(value) {
    var num = Number(value);
    if (!Number.isFinite(num)) {
      return '0';
    }
    return Math.round(num).toLocaleString();
  }

  function collectCollocateStats(dataset) {
    var articles = 0;
    var cityCount = 0;
    var newspaperSet = new Set();
    if (!dataset) {
      return { articles: 0, newspapers: 0, cities: 0 };
    }
    Object.keys(dataset).forEach(function(key) {
      var data = dataset[key];
      if (!data || !Array.isArray(data.entries) || !data.entries.length) {
        return;
      }
      articles += data.entries.length;
      cityCount += 1;
      data.entries.forEach(function(entry) {
        if (entry && entry.newspaper) {
          newspaperSet.add(entry.newspaper);
        }
      });
    });
    return {
      articles: articles,
      newspapers: newspaperSet.size,
      cities: cityCount,
    };
  }

  function updateCollocateSummaryLineFromDataset(dataset) {
    var line = document.getElementById('collocateSummaryLine');
    if (!line) {
      return;
    }
    var content = line.querySelector('[data-summary-content]');
    if (!content) {
      content = document.createElement('span');
      content.setAttribute('data-summary-content', '1');
      if (line.firstChild) {
        line.insertBefore(content, line.firstChild);
      } else {
        line.appendChild(content);
      }
    }
    if (isTopTermMode()) {
      var stats = collectCollocateStats(dataset);
      var termCounter = {};
      if (dataset) {
        Object.keys(dataset).forEach(function(key) {
          var data = dataset[key];
          if (!data || !data._currentTopTermInfo) {
            return;
          }
          var info = data._currentTopTermInfo;
          if (info && info.term) {
            var t = String(info.term);
            termCounter[t] = (termCounter[t] || 0) + 1;
          }
        });
      }
      var uniqueTerms = Object.keys(termCounter).length;
      var summaryParts = ['Top collocate term per location'];
      if (collocateTimeEnabled) {
        var timeLabel = labelForTimeKey(selectedTimeBinKey);
        if (timeLabel) {
          summaryParts.push('Time: ' + timeLabel);
        }
      }
      summaryParts.push('Locations: ' + formatInteger(stats.cities || 0));
      summaryParts.push('Articles: ' + formatInteger(stats.articles || 0));
      if (uniqueTerms) {
        summaryParts.push('Unique terms: ' + formatInteger(uniqueTerms));
      }
      content.textContent = summaryParts.join(' | ');
      return;
    }
    var term = String(selectedCollocate || '').trim();
    if (!term) {
      content.textContent = 'Collocate term: none selected';
      return;
    }
    var stats = collectCollocateStats(dataset);
    if ((!stats || (!stats.articles && !stats.cities)) && collocateSummary && collocateSummary[term]) {
      var fallback = collocateSummary[term];
      stats = {
        articles: Number(fallback.articles) || 0,
        newspapers: Number(fallback.newspapers) || 0,
        cities: Number(fallback.cities) || 0,
      };
    }
    stats = stats || { articles: 0, newspapers: 0, cities: 0 };
    var summaryText = 'Collocate term "' + term + '": '
      + formatInteger(stats.articles) + ' articles | '
      + formatInteger(stats.newspapers) + ' newspapers | '
      + formatInteger(stats.cities) + ' cities';
    if (collocateTimeEnabled) {
      var timeLabel = labelForTimeKey(selectedTimeBinKey);
      if (timeLabel) {
        summaryText += ' | Time: ' + timeLabel;
      }
    }
    content.textContent = summaryText;
  }

  function interpolateColor(startHex, endHex, t) {
    var clampT = Math.min(1, Math.max(0, t));
    var parseHex = function(hex) {
      var cleaned = String(hex || '').replace('#', '');
      if (cleaned.length === 3) {
        cleaned = cleaned[0] + cleaned[0] + cleaned[1] + cleaned[1] + cleaned[2] + cleaned[2];
      }
      while (cleaned.length < 6) {
        cleaned += '0';
      }
      var r = parseInt(cleaned.slice(0, 2), 16);
      var g = parseInt(cleaned.slice(2, 4), 16);
      var b = parseInt(cleaned.slice(4, 6), 16);
      return { r: r, g: g, b: b };
    };
    var start = parseHex(startHex);
    var end = parseHex(endHex);
    var mix = function(a, b) {
      return Math.round(a + (b - a) * clampT);
    };
    var r = mix(start.r, end.r);
    var g = mix(start.g, end.g);
    var b = mix(start.b, end.b);
    var toHex = function(value) {
      var str = value.toString(16);
      return str.length === 1 ? '0' + str : str;
    };
    return '#' + toHex(r) + toHex(g) + toHex(b);
  }

  function createColorScale(dataset) {
    var emptyColor = '#cbd5e0';
    var thresholds = [1, 5, 10, 20, 40];
    var colors = ['#fed7d7', '#feb2b2', '#fc8181', '#e53e3e', '#9b2c2c'];
    return function(count) {
      var numeric = Number(count);
      if (!Number.isFinite(numeric) || numeric <= 0) {
        return emptyColor;
      }
      for (var i = thresholds.length - 1; i >= 0; i--) {
        if (numeric >= thresholds[i]) {
          return colors[i];
        }
      }
      return colors[0];
    };
  }

  function setBaseEntries(data, entries, articleCount) {
    if (!data) {
      return;
    }
    var base = Array.isArray(entries) ? entries.slice() : [];
    data._baseEntries = base;
    if (Number.isFinite(articleCount)) {
      data._baseArticleCount = Number(articleCount);
    } else if (typeof data._baseArticleCount === 'number' && Number.isFinite(data._baseArticleCount)) {
      // keep existing base article count
    } else {
      data._baseArticleCount = base.length;
    }
  }

  function ensureBaseEntries(data) {
    if (!data) {
      return;
    }
    if (!Array.isArray(data._baseEntries)) {
      var source = [];
      if (Array.isArray(data.entries)) {
        source = data.entries.slice();
      } else if (Array.isArray(data.full_entries)) {
        source = data.full_entries.slice();
      }
      data._baseEntries = source;
      if (!Number.isFinite(data._baseArticleCount)) {
        if (Number.isFinite(data.article_count)) {
          data._baseArticleCount = Number(data.article_count);
        } else if (Number.isFinite(data.full_article_count)) {
          data._baseArticleCount = Number(data.full_article_count);
        } else {
          data._baseArticleCount = source.length;
        }
      }
    }
  }

  function applyCollocateFilterToData(data, overrideTerm, timeKey) {
    if (!data) {
      return;
    }
    ensureBaseEntries(data);
    var baseEntries = Array.isArray(data._baseEntries) ? data._baseEntries : [];
    var baseCount = Number.isFinite(data._baseArticleCount) ? Number(data._baseArticleCount) : baseEntries.length;
    var term = '';
    if (typeof overrideTerm === 'string') {
      term = overrideTerm.trim();
    } else if (overrideTerm) {
      term = String(overrideTerm).trim();
    } else {
      term = String(selectedCollocate || '').trim();
    }
    if (!term) {
      if (isTopTermMode()) {
        data.entries = [];
        data.article_count = 0;
      } else {
        data.entries = baseEntries.slice();
        data.article_count = baseCount;
      }
      return;
    }
    if (!data.collocate_hits || !Object.prototype.hasOwnProperty.call(data.collocate_hits, term)) {
      data.entries = [];
      data.article_count = 0;
      return;
    }
    var hitsList = data.collocate_hits[term];
    if (!Array.isArray(hitsList) || !hitsList.length) {
      data.entries = [];
      data.article_count = 0;
      return;
    }
    var allowed = null;
    if (timeKey) {
      allowed = collectTimeIndexes(data, timeKey);
      if (!allowed || !allowed.size) {
        data.entries = [];
        data.article_count = 0;
        return;
      }
    }
    var allowedHits = new Set();
    hitsList.forEach(function(idx) {
      var num = Number(idx);
      if (!Number.isFinite(num)) {
        return;
      }
      if (allowed && !allowed.has(num)) {
        return;
      }
      allowedHits.add(num);
    });
    if (!allowedHits.size) {
      data.entries = [];
      data.article_count = 0;
      return;
    }
    var filtered = [];
    baseEntries.forEach(function(entry) {
      if (!entry) {
        return;
      }
      var idx = Number(entry.full_index);
      if (Number.isFinite(idx) && allowedHits.has(idx)) {
        filtered.push(entry);
      }
    });
    data.entries = filtered;
    data.article_count = filtered.length;
  }

  function applyCollocateFilterToDataset(dataset, timeKey) {
    if (!dataset) {
      return;
    }
    Object.keys(dataset).forEach(function(key) {
      var data = dataset[key];
      if (!data) {
        return;
      }
      if (isTopTermMode()) {
        var info = computeTopTermInfo(data, timeKey || null);
        if (!info || !info.term) {
          data.entries = [];
          data.article_count = 0;
          return;
        }
        applyCollocateFilterToData(data, info.term, timeKey || null);
      } else {
        applyCollocateFilterToData(data, null, timeKey || null);
      }
    });
  }

  function refreshPopupAfterFilter(mapObj) {
    if (!mapObj) {
      return;
    }
    if (mapObj._popup && typeof mapObj._popup.getElement === 'function') {
      var popupEl = mapObj._popup.getElement();
      if (popupEl) {
        var root = popupEl.querySelector('[data-popup-root="1"]');
        if (root) {
          attach(root);
          attachDockToggle(root, mapObj, mapObj._popup);
        }
      }
    }
    var dockPanel = document.querySelector('[data-dock-panel]');
    if (dockPanel && dockPanel.style.display !== 'none') {
      var dockRoot = dockPanel.querySelector('[data-popup-root="1"]');
      if (dockRoot) {
        attach(dockRoot);
        var dockBtn = dockRoot.querySelector('[data-dock-toggle="1"]');
        updateDockButtonState(dockRoot, dockBtn);
        if (dockState.root === dockRoot) {
          var latNum = dockRoot.dataset && dockRoot.dataset.lat ? Number(dockRoot.dataset.lat) : NaN;
          var lonNum = dockRoot.dataset && dockRoot.dataset.lon ? Number(dockRoot.dataset.lon) : NaN;
          dockState.lat = Number.isFinite(latNum) ? latNum : dockState.lat;
          dockState.lon = Number.isFinite(lonNum) ? lonNum : dockState.lon;
        }
      }
    }
    refreshPinnedEntries();
  }

  function clearHighlight() {
    if (highlightLayer && typeof highlightLayer.remove === 'function') {
      highlightLayer.remove();
    }
    highlightLayer = null;
    currentHighlightId = null;
  }

  function drawHighlight(points) {
    if (!activeMap || !Array.isArray(points) || !points.length) {
      clearHighlight();
      return;
    }
    if (highlightLayer && typeof highlightLayer.remove === 'function') {
      highlightLayer.remove();
    }
    var layer = L.layerGroup();
    points.forEach(function(pt) {
      if (!pt) {
        return;
      }
      var lat = Number(pt.lat);
      var lon = Number(pt.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return;
      }
      var marker = L.circleMarker([lat, lon], {
        radius: 8,
        color: '#ff7e18',
        weight: 2,
        fillColor: '#ffd8a8',
        fillOpacity: 0.6,
        opacity: 1,
        interactive: false,
      });
      layer.addLayer(marker);
    });
    if (!layer.getLayers().length) {
      clearHighlight();
      return;
    }
    highlightLayer = layer.addTo(activeMap);
  }

  function setHighlightForGroup(gid, groupData) {
    if (!activeMap) {
      return;
    }
    var coords = [];
    if (groupData && Array.isArray(groupData.coords) && groupData.coords.length) {
      groupData.coords.forEach(function(pt) {
        if (pt && Number.isFinite(pt.lat) && Number.isFinite(pt.lon)) {
          coords.push({ lat: pt.lat, lon: pt.lon });
        }
      });
    } else if (groupData && Number.isFinite(groupData.lat) && Number.isFinite(groupData.lon)) {
      coords.push({ lat: groupData.lat, lon: groupData.lon });
    }
    if (!coords.length) {
      if (currentHighlightId === gid) {
        clearHighlight();
      }
      return;
    }
    drawHighlight(coords);
    currentHighlightId = gid;
  }

  function updateHighlightFromDataset(dataset) {
    if (currentHighlightId && dataset && dataset[currentHighlightId]) {
      var highlightedData = dataset[currentHighlightId];
      if (highlightedData && highlightedData.entries && highlightedData.entries.length) {
        setHighlightForGroup(currentHighlightId, highlightedData);
        return;
      }
    }
    if (currentHighlightId) {
      clearHighlight();
    }
  }

  function collectNearbyGroupIds(latlng, mapObj, dataset) {
    if (!latlng || !mapObj || !dataset) {
      return [];
    }
    if (!mapObj.latLngToLayerPoint || typeof mapObj.latLngToLayerPoint !== 'function') {
      return [];
    }
    var centerPoint = mapObj.latLngToLayerPoint(latlng);
    if (!centerPoint) {
      return [];
    }
    var radius = Math.max(4, clickRadiusPx || 0);
    var result = [];
    Object.keys(dataset).forEach(function(gid) {
      if (!gid) {
        return;
      }
      var data = dataset[gid];
      if (!data || !data.entries || !data.entries.length) {
        return;
      }
      if (!Number.isFinite(data.lat) || !Number.isFinite(data.lon)) {
        return;
      }
      var groupPoint = mapObj.latLngToLayerPoint(L.latLng(data.lat, data.lon));
      if (!groupPoint) {
        return;
      }
      var dx = centerPoint.x - groupPoint.x;
      var dy = centerPoint.y - groupPoint.y;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= radius) {
        result.push(gid);
      }
    });
    return result;
  }

  function openGroupsPopup(latlng, mapObj) {
    if (!mapObj) {
      return;
    }
    clearHighlight();
    loadData(function(dataset) {
      var ids = collectNearbyGroupIds(latlng, mapObj, dataset);
      if (!ids.length) {
        return;
      }
      var groups = [];
      ids.forEach(function(gid) {
        var data = dataset[gid];
        if (!data || !data.entries || !data.entries.length) {
          return;
        }
        groups.push({ id: gid, data: data });
      });
      if (!groups.length) {
        return;
      }

      if (groups.length > 1) {
        var mergedId = '__multi__';
        var mergedEntries = [];
        var coordsList = [];
        var totalValue = 0;
        var sourceIds = groups.map(function(info) { return info.id; });
        groups.forEach(function(info) {
          var data = info.data;
          if (!data) {
            return;
          }
          if (Array.isArray(data.entries)) {
            data.entries.forEach(function(entry) {
              if (entry) {
                mergedEntries.push(Object.assign({}, entry));
              }
            });
          }
          var valueVal = Number(data.value);
          if (Number.isFinite(valueVal)) {
            totalValue += valueVal;
          }
          if (Array.isArray(data.coords)) {
            data.coords.forEach(function(pt) {
              if (pt && Number.isFinite(pt.lat) && Number.isFinite(pt.lon)) {
                coordsList.push({ lat: pt.lat, lon: pt.lon });
              }
            });
          } else if (Number.isFinite(data.lat) && Number.isFinite(data.lon)) {
            coordsList.push({ lat: data.lat, lon: data.lon });
          }
        });

        if (mergedEntries.length) {
          var baseTemplate = groups[0].data && groups[0].data.template ? groups[0].data.template : '';
          if (baseTemplate) {
            baseTemplate = baseTemplate.replace(/data-group-id="[^"]+"/, 'data-group-id="' + mergedId + '"');
          }
          var mergedTitle = searchTerm ? 'Articles mentioning "' + searchTerm + '" in selected area' : 'Articles in selected area';
          var mergedData = {
            entries: mergedEntries,
            full_entries: mergedEntries.slice(),
            value: totalValue,
            full_value: totalValue,
            article_count: mergedEntries.length,
            full_article_count: mergedEntries.length,
            title: mergedTitle,
            full_title: mergedTitle,
            metric_display: groups[0].data.metric_display,
            metric_normalized_display: groups[0].data.metric_normalized_display,
            normalized: groups[0].data.normalized,
            denominator_label: groups[0].data.denominator_label,
            lat: Number.NaN,
            lon: Number.NaN,
            coords: coordsList,
            template: baseTemplate,
            member_ids: sourceIds,
            location_index: 0,
            location_total: sourceIds.length || coordsList.length || 1,
            location_label: '',
            search_term: searchTerm || '',
            city: '',
            state: '',
            place_label: '',
          };
          dataset[mergedId] = mergedData;
          groups.unshift({ id: mergedId, data: mergedData });
        }
      }
      var actualInfos = [];
      for (var gi = 0; gi < groups.length; gi++) {
        var infoNode = groups[gi];
        if (!infoNode || !infoNode.data) {
          continue;
        }
        if (infoNode.id !== '__multi__') {
          actualInfos.push(infoNode);
        }
      }
      var totalLocations = actualInfos.length;
      if (!totalLocations) {
        totalLocations = groups.length;
      }
      for (var ai = 0; ai < actualInfos.length; ai++) {
        var infoItem = actualInfos[ai];
        if (!infoItem || !infoItem.data) {
          continue;
        }
        infoItem.data.location_index = ai + 1;
        infoItem.data.location_total = totalLocations || 1;
        infoItem.data.location_label = '';
      }
      for (var gi2 = 0; gi2 < groups.length; gi2++) {
        var infoNode2 = groups[gi2];
        if (!infoNode2 || !infoNode2.data) {
          continue;
        }
        if (infoNode2.id === '__multi__') {
          var dataNode = infoNode2.data;
          dataNode.location_index = 0;
          dataNode.location_total = totalLocations || dataNode.location_total || 1;
          if (dataNode.location_total > 1) {
            dataNode.location_label = 'All locations (' + dataNode.location_total + ')';
          } else {
            dataNode.location_label = 'All locations';
          }
        }
      }
      var popup = L.popup({ maxWidth: 360 }).setLatLng(latlng);
      if (groups.length === 1) {
        var singleInfo = groups[0];
        var single = singleInfo ? singleInfo.data : null;
        if (single && single.template) {
          var wrapper = document.createElement('div');
          wrapper.innerHTML = single.template;
          var root = wrapper.firstElementChild;
          if (root) {
            attach(root);
            attachDockToggle(root, mapObj, popup);
            popup.setContent(root);
            popup.openOn(mapObj);
          }
        }
        return;
      }

      var container = document.createElement('div');
      container.style.minWidth = '300px';
      container.style.fontSize = '14px';
      container.style.lineHeight = '1.3';
      var label = document.createElement('div');
      label.style.fontWeight = '600';
      label.textContent = 'Select a location';
      container.appendChild(label);
      var select = document.createElement('select');
      select.style.width = '100%';
      select.style.marginTop = '4px';
      groups.forEach(function(info, idx) {
        var data = info.data;
        var option = document.createElement('option');
        option.value = info.id;
        if (idx === 0) {
          option.selected = true;
        }
        if (info.id === '__multi__') {
          if (data && data.article_count) {
            var allCount = Number(data.article_count);
            var countText = Number.isFinite(allCount) ? allCount + (allCount === 1 ? ' Article' : ' Articles') : 'All Articles';
            var locTotal = Number(data.location_total);
            var locText;
            if (Number.isFinite(locTotal)) {
              locText = locTotal === 1 ? '1 location' : locTotal + ' locations';
            } else {
              locText = 'multiple locations';
            }
            option.textContent = countText + ' across ' + locText;
          } else {
            option.textContent = 'All locations';
          }
          select.appendChild(option);
          return;
        }
        if (!data) {
          option.textContent = info.id;
          select.appendChild(option);
          return;
        }
        var countVal = Number(data.article_count);
        var countLabel;
        if (Number.isFinite(countVal)) {
          countLabel = countVal + ' ' + (countVal === 1 ? 'Article' : 'Articles');
        } else {
          countLabel = 'Articles';
        }
        var termText = (data.search_term || '').toString().trim();
        var locationParts = [];
        if (data.city) {
          locationParts.push(data.city);
        }
        if (data.state) {
          locationParts.push(data.state);
        }
        var locationText = locationParts.filter(function(item) { return item; }).join(', ');
        if (!locationText && data.place_label) {
          locationText = data.place_label;
        }
        if (!locationText && data.title) {
          locationText = data.title;
        }
        var pieces = [countLabel];
        if (termText) {
          pieces.push('w/ ' + termText);
        }
        if (locationText) {
          pieces.push('in ' + locationText);
        }
        option.textContent = pieces.join(' ');
        select.appendChild(option);
      });
      container.appendChild(select);
      var detail = document.createElement('div');
      detail.style.marginTop = '8px';
      container.appendChild(detail);

      function render(gid) {
        detail.innerHTML = '';
        var data = dataset[gid];
        if (!data || !data.template) {
          detail.textContent = 'No data.';
          return;
        }
        var wrapper = document.createElement('div');
        wrapper.innerHTML = data.template;
        var root = wrapper.firstElementChild;
        if (root) {
          detail.appendChild(root);
          attach(root);
        } else {
          detail.textContent = 'No data.';
        }
      }

      select.addEventListener('change', function() {
        render(select.value);
      });

      popup.setContent(container);
      popup.openOn(mapObj);
      render(select.value);
    });
  }

  function applyTimeFilter(mapObj) {
    var currentKeys = [];
    var useAllBins = false;
    if (collocateTimeEnabled) {
      if (!collocateTimeBins.length || !selectedTimeBinKey) {
        useAllBins = true;
      } else {
        currentKeys = resolveTimeKeys(selectedTimeBinKey);
      }
    } else {
      if (!mapObj || !mapObj.timeDimension || typeof mapObj.timeDimension.getCurrentTime !== 'function') {
        return;
      }
      var rawTime = mapObj.timeDimension.getCurrentTime();
      currentKeys = resolveTimeKeys(rawTime);
      if (!currentKeys.length) {
        return;
      }
    }
    loadData(function(dataset) {
      var keys = Object.keys(dataset || {});
      if (!keys.length) {
        return;
      }
      keys.forEach(function(key) {
        var data = dataset[key] || {};
        if (!data.full_entries) {
          data.full_entries = data.entries ? data.entries.slice() : [];
        }
        if (typeof data.full_value === 'undefined') {
          data.full_value = data.value;
        }
        if (typeof data.full_article_count === 'undefined') {
          data.full_article_count = data.article_count;
        }
        if (typeof data.full_title === 'undefined') {
          data.full_title = data.title || '';
        }
        var entriesResult = [];
        var valueResult = data.full_value;
        var countResult = data.full_article_count;
        var titleResult = data.full_title;
        var timeLabelResult = '';
        var hasTimeFilter = false;
        if (data.time_bins && !useAllBins) {
          hasTimeFilter = true;
          var bin = null;
          if (currentKeys && currentKeys.length) {
            for (var ck = 0; ck < currentKeys.length; ck++) {
              var attempt = currentKeys[ck];
              if (!attempt) {
                continue;
              }
              if (Object.prototype.hasOwnProperty.call(data.time_bins, attempt)) {
                bin = data.time_bins[attempt];
                if (bin) {
                  break;
                }
              }
              if (Array.isArray(timeLabels) && timeLabels.length) {
                var idxCandidate = Number(attempt);
                if (Number.isFinite(idxCandidate) && idxCandidate >= 1 && idxCandidate <= timeLabels.length) {
                  var altKey = timeLabels[idxCandidate - 1];
                  if (altKey && Object.prototype.hasOwnProperty.call(data.time_bins, altKey)) {
                    bin = data.time_bins[altKey];
                    if (bin) {
                      break;
                    }
                  }
                }
              }
              if (collocateTimeEnabled && collocateTimeBins.length) {
                for (var tb = 0; tb < collocateTimeBins.length; tb++) {
                  var entry = collocateTimeBins[tb];
                  if (!entry) {
                    continue;
                  }
                  if (entry.key === attempt && entry.iso && Object.prototype.hasOwnProperty.call(data.time_bins, entry.iso)) {
                    bin = data.time_bins[entry.iso];
                    if (bin) {
                      break;
                    }
                  }
                }
                if (bin) {
                  break;
                }
              }
            }
          }
          if (bin && Array.isArray(bin.indexes) && bin.indexes.length) {
            var mapped = [];
            bin.indexes.forEach(function(idx) {
              if (idx >= 0 && idx < data.full_entries.length) {
                var payload = data.full_entries[idx];
                if (payload) {
                  mapped.push(payload);
                }
              }
            });
            entriesResult = mapped;
            valueResult = (typeof bin.value !== 'undefined') ? bin.value : data.full_value;
            countResult = (typeof bin.article_count !== 'undefined') ? bin.article_count : mapped.length;
            titleResult = bin.title || data.full_title;
            timeLabelResult = bin.time_label || '';
          } else {
            entriesResult = [];
            valueResult = 0;
            countResult = 0;
            titleResult = data.full_title;
            timeLabelResult = '';
          }
        } else {
          entriesResult = data.full_entries.slice();
          if (!entriesResult.length && Array.isArray(data.entries)) {
            entriesResult = data.entries.slice();
          }
          valueResult = data.full_value;
          countResult = data.full_article_count;
          titleResult = data.full_title;
          timeLabelResult = '';
        }
        data.entries = entriesResult;
        if (typeof valueResult !== 'undefined') {
          data.value = valueResult;
        }
        data.article_count = Number.isFinite(countResult) ? Number(countResult) : entriesResult.length;
        data.title = titleResult;
        data.time_label = timeLabelResult;
        data._hasTimeFilter = hasTimeFilter;
      });

      keys.forEach(function(key) {
        var data = dataset[key];
        if (!data || !Array.isArray(data.member_ids)) {
          return;
        }
        var combined = [];
        var coords = [];
        var combinedValue = 0;
        data.member_ids.forEach(function(mid) {
          var source = dataset[mid];
          if (!source || !Array.isArray(source.entries) || !source.entries.length) {
            return;
          }
          source.entries.forEach(function(entry) {
            if (entry) {
              combined.push(Object.assign({}, entry));
            }
          });
          var val = Number(source.value);
          if (Number.isFinite(val)) {
            combinedValue += val;
          }
          if (Array.isArray(source.coords)) {
            source.coords.forEach(function(pt) {
              if (pt && Number.isFinite(pt.lat) && Number.isFinite(pt.lon)) {
                coords.push({ lat: pt.lat, lon: pt.lon });
              }
            });
          } else if (Number.isFinite(source.lat) && Number.isFinite(source.lon)) {
            coords.push({ lat: source.lat, lon: source.lon });
          }
        });
        data.entries = combined;
        data.full_entries = combined.slice();
        data.article_count = combined.length;
        data.full_article_count = combined.length;
        data.value = combinedValue;
        data.full_value = combinedValue;
        data.coords = coords;
        var memberCount = Array.isArray(data.member_ids) ? data.member_ids.length : 0;
        if (!Number.isFinite(Number(data.location_total)) || Number(data.location_total) < 1) {
          data.location_total = memberCount || coords.length || 1;
        }
      });

      keys.forEach(function(key) {
        var data = dataset[key];
      if (!data) {
        return;
      }
      setBaseEntries(data, data.entries, data.article_count);
      });
      var filterTimeKey = null;
      if (collocateTimeEnabled) {
        filterTimeKey = selectedTimeBinKey || null;
      } else if (currentKeys && currentKeys.length === 1) {
        filterTimeKey = currentKeys[0];
      }
      applyCollocateFilterToDataset(dataset, filterTimeKey);

      if (typeof mapObj.eachLayer === 'function') {
        mapObj.eachLayer(function(layer) {
          if (!layer || !layer.options || !layer.options.groupId) {
            return;
          }
          var layerId = layer.options.groupId;
          var data = dataset[layerId];
          if (!data || !data._hasTimeFilter) {
            return;
          }
          var visible = data.entries && data.entries.length > 0;
          var baseOpacity = (typeof layer.options.baseOpacity === 'number') ? layer.options.baseOpacity : (typeof layer.options.opacity === 'number' ? layer.options.opacity : 0.5);
          var baseFill = (typeof layer.options.baseFillOpacity === 'number') ? layer.options.baseFillOpacity : (typeof layer.options.fillOpacity === 'number' ? layer.options.fillOpacity : 0.85);
          if (typeof layer.setStyle === 'function') {
            layer.setStyle({
              opacity: visible ? baseOpacity : 0,
              fillOpacity: visible ? baseFill : 0,
            });
          }
          var isGhost = !!layer.options.ghostMarker;
          layer.options.interactive = isGhost ? false : !!visible;
          if (layer._path && layer._path.style) {
            layer._path.style.pointerEvents = (visible && !isGhost) ? 'auto' : 'none';
          }
          if (!visible && typeof layer.closePopup === 'function') {
            layer.closePopup();
          }
          if (!visible && currentHighlightId === layerId) {
            clearHighlight();
          }
        });
      }
      refreshPopupAfterFilter(mapObj);
      if (collocateRanks && (selectedCollocate || isTopTermMode())) {
        refreshCollocateSizes(mapObj);
      } else {
        updateHighlightFromDataset(dataset);
      }
    });
  }
  function updateProgress(root, groupData, index) {
    var progress = root.querySelector('[data-article-progress]');
    if (!progress) return;
    if (!groupData || !groupData.entries || !groupData.entries.length) {
      progress.textContent = '';
      return;
    }
    var total = groupData.entries.length;
    var current = index + 1;
    if (!Number.isFinite(current) || current < 1) current = 1;
    if (current > total) current = total;
    var entry = groupData.entries[index] || null;
    var baseText = 'Article ' + current + ' of ' + total;
    if (!entry) {
      progress.textContent = baseText;
      return;
    }
    var parts = [baseText];
    var yearToken = entry.dataset_year || entry.datasetYear || null;
    if (yearToken) {
      var yearStr = String(yearToken);
      var yearCounts = groupData.year_match_counts || groupData.yearMatchCounts || {};
      var totalMatchesForYear = 0;
      if (yearCounts && Object.prototype.hasOwnProperty.call(yearCounts, yearStr)) {
        totalMatchesForYear = Number(yearCounts[yearStr]) || 0;
      }
      if (totalMatchesForYear > 0) {
        var withinYearIndex = 0;
        for (var i = 0; i <= index; i++) {
          var candidate = groupData.entries[i];
          if (candidate && String(candidate.dataset_year || candidate.datasetYear || '') === yearStr) {
            withinYearIndex += 1;
          }
        }
        if (withinYearIndex < 1) withinYearIndex = 1;
        parts.push('— ' + yearStr + ' match ' + withinYearIndex + ' of ' + totalMatchesForYear);
      } else {
        parts.push('— ' + yearStr);
      }
      var rawYearTotal = entry.dataset_year_total || entry.datasetYearTotal;
      var metricLabel = entry.dataset_metric_label || entry.datasetMetricLabel || 'articles';
      var numericYearTotal = Number(rawYearTotal);
      if (Number.isFinite(numericYearTotal) && numericYearTotal > 0) {
        var localizedTotal = numericYearTotal.toLocaleString();
        var percent = null;
        if (totalMatchesForYear > 0) {
          percent = (totalMatchesForYear / numericYearTotal) * 100;
        }
        if (percent && percent > 0) {
          var rounded = percent >= 0.01 ? percent.toFixed(2) : percent.toFixed(4);
          parts.push('(~' + rounded + '% of ' + localizedTotal + ' ' + metricLabel + ')');
        } else {
          parts.push('(' + localizedTotal + ' ' + metricLabel + ' in dataset)');
        }
      }
    }
    progress.textContent = parts.join(' ');
  }
  function updateDockButtonState(root, btn) {
    if (!root || !btn) {
      return;
    }
    if (root.getAttribute('data-docked') === '1') {
      btn.textContent = '▣';
      btn.title = 'Undock popup';
      btn.style.color = '#2b6cb0';
    } else {
      btn.textContent = '⧉';
      btn.title = dockState.preferred ? 'Dock popup (auto-dock enabled)' : 'Dock popup';
      btn.style.color = dockState.preferred ? '#2b6cb0' : '#2b6cb0';
    }
  }
  function updateLocationProgress(root, groupData) {
    var locationEl = root.querySelector('[data-location-progress]');
    if (!locationEl) return;
    if (!groupData) {
      locationEl.textContent = '';
      return;
    }
    if (groupData.location_label) {
      locationEl.textContent = groupData.location_label;
      return;
    }
    var idx = Number(groupData.location_index);
    var total = Number(groupData.location_total);
    if (!Number.isFinite(idx) || idx < 1) {
      idx = 1;
    }
    if (!Number.isFinite(total) || total < 1) {
      total = 1;
    }
    locationEl.textContent = 'Location ' + idx + ' of ' + total;
  }
  function addScrollGuards(el) {
    if (!el || el.dataset && el.dataset.scrollGuardAttached) {
      return;
    }
    var stop = function(ev) {
      ev.stopPropagation();
    };
    el.addEventListener('wheel', stop, { passive: true });
    el.addEventListener('touchmove', stop, { passive: false });
    if (el.dataset) {
      el.dataset.scrollGuardAttached = '1';
    } else {
      el.setAttribute('data-scroll-guard', '1');
    }
  }

  function enableHoverScroll(container, scrollTarget) {
    if (!container || !scrollTarget) {
      return;
    }
    if (container.__hoverScrollAttached) {
      return;
    }
    container.addEventListener('wheel', function(ev) {
      if (!scrollTarget || ev.defaultPrevented) {
        return;
      }
      if (scrollTarget.contains(ev.target)) {
        return;
      }
      if (scrollTarget.scrollHeight <= scrollTarget.clientHeight) {
        return;
      }
      scrollTarget.scrollTop += ev.deltaY;
      if (typeof ev.preventDefault === 'function') {
        ev.preventDefault();
      }
    }, { passive: false });
    container.__hoverScrollAttached = true;
  }

  function consumeDragEvents(el) {
    if (!el || el.__dragGuardAttached) {
      return;
    }
    var stop = function(ev) {
      ev.stopPropagation();
    };
    ['pointerdown', 'pointermove', 'pointerup', 'mousedown', 'mousemove', 'mouseup', 'touchstart', 'touchmove', 'touchend', 'contextmenu'].forEach(function(evt) {
      el.addEventListener(evt, stop, true);
    });
    el.__dragGuardAttached = true;
  }

  function buildPinSignature(baseKey, groupData, entry, entryIndex, timeLabel, selectionValue) {
    var pieces = [String(baseKey || '')];
    var fullIndex = null;
    if (entry && Object.prototype.hasOwnProperty.call(entry, 'full_index')) {
      fullIndex = entry.full_index;
    }
    if (!Number.isFinite(fullIndex) && entry && Object.prototype.hasOwnProperty.call(entry, 'index')) {
      fullIndex = entry.index;
    }
    if (!Number.isFinite(fullIndex)) {
      fullIndex = Number(entryIndex);
    }
    if (Number.isFinite(fullIndex)) {
      pieces.push('idx:' + fullIndex);
    }
    if (entry && entry.date) {
      pieces.push('date:' + entry.date);
    }
    if (timeLabel) {
      pieces.push('time:' + timeLabel);
    }
    if (selectionValue) {
      pieces.push('sel:' + selectionValue);
    }
    return pieces.join('|');
  }

  function findPinnedBySignature(signature) {
    if (!signature) {
      return null;
    }
    for (var i = 0; i < pinState.entries.length; i++) {
      if (pinState.entries[i] && pinState.entries[i].signature === signature) {
        return pinState.entries[i];
      }
    }
    return null;
  }

  function determineEntryTimeMeta(groupData, entry, fallbackLabel) {
    var result = {
      key: null,
      label: fallbackLabel || '',
      iso: '',
    };
    if (!groupData || !entry || !groupData.time_bins) {
      return result;
    }
    var fullIndex = null;
    if (Object.prototype.hasOwnProperty.call(entry, 'full_index')) {
      fullIndex = entry.full_index;
    }
    if (!Number.isFinite(fullIndex) && Object.prototype.hasOwnProperty.call(entry, 'index')) {
      fullIndex = entry.index;
    }
    if (!Number.isFinite(fullIndex)) {
      fullIndex = null;
    }
    if (fullIndex === null) {
      return result;
    }
    var bins = groupData.time_bins;
    var binKeys = Object.keys(bins || {});
    for (var i = 0; i < binKeys.length; i++) {
      var key = binKeys[i];
      var bin = bins[key];
      if (!bin || !Array.isArray(bin.indexes)) {
        continue;
      }
      if (bin.indexes.indexOf(fullIndex) !== -1) {
        result.key = key;
        result.label = bin.time_label || bin.label || result.label || key;
        if (bin.iso) {
          result.iso = bin.iso;
        }
        break;
      }
    }
    return result;
  }

  function incrementPinCount(baseKey) {
    if (!baseKey) {
      return;
    }
    var current = pinState.counts.get(baseKey) || 0;
    pinState.counts.set(baseKey, current + 1);
    updatePinButtonsByKey(baseKey);
  }

  function decrementPinCount(baseKey) {
    if (!baseKey) {
      return;
    }
    var current = pinState.counts.get(baseKey) || 0;
    var next = Math.max(0, current - 1);
    if (next === 0) {
      pinState.counts.delete(baseKey);
    } else {
      pinState.counts.set(baseKey, next);
    }
    updatePinButtonsByKey(baseKey);
  }

  function updatePinnedOrderIndices() {
    pinState.orderSequence = pinState.orderSequence.filter(function(id) {
      return pinState.lookup.has(id);
    });
    pinState.orderSequence.forEach(function(id, idx) {
      var entry = pinState.lookup.get(id);
      if (entry) {
        entry.orderIndex = idx + 1;
      }
    });
  }

  function updatePinnedHeaderCounter() {
    if (!pinState.headerCounter) {
      return;
    }
    var total = pinState.entries.length;
    if (!total) {
      pinState.headerCounter.textContent = 'Pinned Article 0 of 0';
      return;
    }
    var topEntry = pinState.entries[0];
    var displayIndex = topEntry && Number.isFinite(topEntry.orderIndex) ? topEntry.orderIndex : 1;
    pinState.headerCounter.textContent = 'Pinned Article ' + displayIndex + ' of ' + total;
  }

  function renderPinnedList() {
    updatePinnedOrderIndices();
    var body = getPinBody();
    if (!body) {
      return;
    }
    var frag = document.createDocumentFragment();
    pinState.entries.forEach(function(entry, idx) {
      if (!entry || !entry.wrapper) {
        return;
      }
      var labelIndex = Number.isFinite(entry.orderIndex) ? entry.orderIndex : (idx + 1);
      entry.orderIndex = labelIndex;
      entry.wrapper.dataset.pinDisplayIndex = String(labelIndex);
      var isAlternate = labelIndex % 2 === 0;
      entry.wrapper.style.background = isAlternate ? '#f1f5f9' : 'rgba(255,255,255,0.98)';
      var badge = entry.wrapper.querySelector('[data-pin-badge="1"]');
      if (!badge) {
        badge = document.createElement('div');
        badge.setAttribute('data-pin-badge', '1');
        badge.style.position = 'absolute';
        badge.style.top = '6px';
        badge.style.right = '8px';
        badge.style.background = '#2b6cb0';
        badge.style.color = '#ffffff';
        badge.style.fontSize = '12px';
        badge.style.fontWeight = '600';
        badge.style.padding = '2px 6px';
        badge.style.borderRadius = '999px';
        entry.wrapper.appendChild(badge);
      }
      badge.style.cursor = 'pointer';
      badge.style.userSelect = 'none';
      badge.title = 'Move this pinned article to the top';
      badge.setAttribute('role', 'button');
      badge.setAttribute('tabindex', '0');
      badge.setAttribute('data-pin-entry-id', entry.id);
      if (!badge.dataset.listenerAttached) {
        var handlePinBadgeClick = function(ev) {
          if (ev) {
            if (typeof ev.preventDefault === 'function') {
              ev.preventDefault();
            }
            if (typeof ev.stopPropagation === 'function') {
              ev.stopPropagation();
            }
          }
          var targetId = this.getAttribute('data-pin-entry-id');
          bringPinnedEntryToFront(targetId);
        };
        badge.addEventListener('click', handlePinBadgeClick);
        badge.addEventListener('keydown', function(ev) {
          if (!ev) {
            return;
          }
          if (ev.key === 'Enter' || ev.key === ' ' || ev.key === 'Spacebar') {
            ev.preventDefault();
            ev.stopPropagation();
            bringPinnedEntryToFront(this.getAttribute('data-pin-entry-id'));
          }
        });
        badge.dataset.listenerAttached = '1';
      }
      badge.textContent = '#' + labelIndex;
      badge.setAttribute('aria-label', 'Move pinned article #' + labelIndex + ' to top');
      frag.appendChild(entry.wrapper);
    });
    body.innerHTML = '';
    body.appendChild(frag);
    updatePinnedHeaderCounter();
    var hasEntries = pinState.entries.length > 0;
    if (pinState.navUpBtn) {
      pinState.navUpBtn.disabled = pinState.entries.length <= 1;
    }
    if (pinState.navDownBtn) {
      pinState.navDownBtn.disabled = pinState.entries.length <= 1;
    }
    if (pinState.focusBtn) {
      pinState.focusBtn.disabled = !hasEntries;
    }
  }

  function bringPinnedEntryToFront(entryId) {
    if (!entryId) {
      return;
    }
    var idx = -1;
    for (var i = 0; i < pinState.entries.length; i++) {
      if (pinState.entries[i] && pinState.entries[i].id === entryId) {
        idx = i;
        break;
      }
    }
    if (idx <= 0) {
      return;
    }
    var before = pinState.entries.slice(0, idx);
    var after = pinState.entries.slice(idx);
    pinState.entries = after.concat(before);
    renderPinnedList();
    updatePinPanelVisibility();
  }

  function rotatePinnedEntries(direction) {
    if (pinState.entries.length <= 1) {
      return;
    }
    if (direction < 0) {
      // move first entry to end
      var first = pinState.entries.shift();
      pinState.entries.push(first);
    } else {
      // move last entry to front
      var last = pinState.entries.pop();
      pinState.entries.unshift(last);
    }
    renderPinnedList();
    updatePinPanelVisibility();
  }

  function indexForTimeKey(key) {
    if (!key || !Array.isArray(collocateTimeBins)) {
      return 0;
    }
    for (var i = 0; i < collocateTimeBins.length; i++) {
      if (collocateTimeBins[i] && collocateTimeBins[i].key === key) {
        return i + 1;
      }
    }
    return 0;
  }

  function focusPinnedTopEntry() {
    if (!pinState.entries.length) {
      return;
    }
    var entry = pinState.entries[0];
    if (entry && Number.isFinite(entry.lat) && Number.isFinite(entry.lon) && activeMap) {
      var target = L.latLng(entry.lat, entry.lon);
      var zoom = typeof activeMap.getZoom === 'function' ? activeMap.getZoom() : 8;
      var desiredZoom = Number.isFinite(zoom) ? Math.max(zoom, 8) : 8;
      if (typeof activeMap.flyTo === 'function') {
        activeMap.flyTo(target, desiredZoom);
      } else if (typeof activeMap.setView === 'function') {
        activeMap.setView(target, desiredZoom);
      }
      var highlightData = entry.dataset || (popupCache && entry.gid ? popupCache[entry.gid] : null);
      var highlightId = entry.layerId || entry.gid;
      if (highlightId) {
        setHighlightForGroup(highlightId, highlightData);
      }
      if (typeof openGroupsPopup === 'function' || highlightId) {
        setTimeout(function() {
          if (highlightId) {
            var latestData = entry.dataset || (popupCache && highlightId ? popupCache[highlightId] : null);
            setHighlightForGroup(highlightId, latestData);
          }
          if (typeof openGroupsPopup === 'function') {
            openGroupsPopup(target, activeMap);
          }
        }, 150);
      }
    }
    if (collocateTimeEnabled) {
      if (entry && entry.timeKey) {
        collocateTimeManuallyDisabled = false;
        selectedTimeBinKey = entry.timeKey;
        var idx = indexForTimeKey(entry.timeKey);
        collocateTimeLastActiveIndex = idx || collocateTimeLastActiveIndex || 1;
        initCollocateTimeControls();
        applyTimeFilter(activeMap);
        refreshCollocateSizes(activeMap);
        updateCollocateSummaryLineFromDataset(popupCache);
      } else {
        collocateTimeManuallyDisabled = true;
        selectedTimeBinKey = null;
        initCollocateTimeControls();
        applyTimeFilter(activeMap);
        refreshCollocateSizes(activeMap);
        updateCollocateSummaryLineFromDataset(popupCache);
      }
    } else if (activeMap && activeMap.timeDimension && entry && entry.timeIso) {
      var timeDate = new Date(entry.timeIso);
      if (!Number.isNaN(timeDate.getTime())) {
        activeMap.timeDimension.setCurrentTime(timeDate.getTime());
      }
    }
  }

  function suppressDoubleClick(element) {
    if (!element) {
      return;
    }
    if (element.__dblBlockAttached) {
      return;
    }
    element.addEventListener('dblclick', function(ev) {
      if (typeof ev.preventDefault === 'function') {
        ev.preventDefault();
      }
      ev.stopPropagation();
    });
    element.__dblBlockAttached = true;
  }
  function getPinKey(root) {
    if (!root) {
      return '';
    }
    var gid = root.getAttribute('data-group-id');
    if (gid && gid.trim()) {
      return gid;
    }
    var existing = root.getAttribute('data-pin-key');
    if (existing && existing.trim()) {
      return existing;
    }
    var generated = 'pin-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
    root.setAttribute('data-pin-key', generated);
    return generated;
  }

  var pinState = {
    entries: [],
    lookup: new Map(),
    counts: new Map(),
    orderSequence: [],
    nextSeq: 1,
    headerCounter: null,
    navUpBtn: null,
    navDownBtn: null,
    focusBtn: null,
    body: null,
  };

  function getDockPanel() {
    var panel = document.querySelector('[data-dock-panel]');
    if (panel) {
      addScrollGuards(panel);
      var existingBody = panel.querySelector('[data-dock-body]');
      if (existingBody) {
        addScrollGuards(existingBody);
        enableHoverScroll(panel, existingBody);
        consumeDragEvents(existingBody);
      }
      suppressDoubleClick(panel);
      consumeDragEvents(panel);
      positionDockPanel(panel);
      return panel;
    }
    panel = document.createElement('div');
    panel.setAttribute('data-dock-panel', '1');
    panel.style.position = 'absolute';
    panel.style.top = 'auto';
    panel.style.left = '16px';
    panel.style.right = 'auto';
    panel.style.width = '360px';
    panel.style.minWidth = '260px';
    panel.style.minHeight = '180px';
    panel.style.resize = 'both';
    panel.style.overflow = 'hidden';
    panel.style.zIndex = '410';
    panel.style.background = 'rgba(255,255,255,0.96)';
    panel.style.borderRadius = '6px';
    panel.style.boxShadow = '0 1px 6px rgba(0,0,0,0.25)';
    panel.style.display = 'none';
    panel.style.padding = '0';
    panel.style.boxSizing = 'border-box';
    panel.style.overflow = 'hidden';

    var header = document.createElement('div');
    header.setAttribute('data-dock-header', '1');
    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.justifyContent = 'space-between';
    header.style.padding = '8px 10px';
    header.style.borderBottom = '1px solid rgba(0,0,0,0.1)';
    header.style.fontSize = '13px';
    header.style.fontWeight = '600';
    header.textContent = 'Docked Popup';

    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.textContent = '✕';
    closeBtn.title = 'Close docked popup';
    closeBtn.style.border = 'none';
    closeBtn.style.background = 'none';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.padding = '0';
    closeBtn.style.marginLeft = '8px';
    closeBtn.style.fontSize = '14px';
    closeBtn.addEventListener('click', function() {
      panel.style.display = 'none';
    });
    header.appendChild(closeBtn);

    var body = document.createElement('div');
    body.setAttribute('data-dock-body', '1');
    body.style.padding = '10px';
    body.style.overflowY = 'auto';

    panel.appendChild(header);
    panel.appendChild(body);

    var parent = document.querySelector('.leaflet-container') || document.body;
    parent.appendChild(panel);
    addScrollGuards(panel);
    addScrollGuards(body);
    enableHoverScroll(panel, body);
    suppressDoubleClick(panel);
    suppressDoubleClick(body);
    consumeDragEvents(panel);
    consumeDragEvents(body);
    positionDockPanel(panel);
    return panel;
  }

  function positionDockPanel(panel) {
    if (!panel) {
      return;
    }
    var isHeatmap = mapMode === 'heatmap';
    var topBuffer = isHeatmap ? 200 : 140;
    var bottomBuffer = isHeatmap ? 32 : 16;
    panel.style.top = 'auto';
    panel.style.right = 'auto';
    panel.style.left = '16px';
    panel.style.bottom = bottomBuffer + 'px';
    var combinedBuffer = topBuffer + bottomBuffer;
    panel.style.maxHeight = 'min(65vh, calc(100vh - ' + combinedBuffer + 'px))';
  }

  function refreshDockPanelLayout() {
    var panel = document.querySelector('[data-dock-panel]');
    if (!panel) {
      return;
    }
    positionDockPanel(panel);
  }

  function getDockBody() {
    return getDockPanel().querySelector('[data-dock-body]');
  }

  function showDockPanel() {
    var panel = getDockPanel();
    positionDockPanel(panel);
    panel.style.display = 'block';
  }

  function hideDockPanel() {
    var panel = getDockPanel();
    panel.style.display = 'none';
  }

function getPinPanel() {
    var panel = document.querySelector('[data-pin-panel]');
    if (panel) {
      addScrollGuards(panel);
      var existingBody = panel.querySelector('[data-pin-body]');
      if (existingBody) {
        addScrollGuards(existingBody);
        enableHoverScroll(panel, existingBody);
        consumeDragEvents(existingBody);
        pinState.body = existingBody;
      }
      suppressDoubleClick(panel);
      consumeDragEvents(panel);
      var counterEl = panel.querySelector('[data-pin-counter]');
      if (counterEl) {
        pinState.headerCounter = counterEl;
      }
      var upBtn = panel.querySelector('[data-pin-nav="up"]');
      if (upBtn) {
        pinState.navUpBtn = upBtn;
      }
      var downBtn = panel.querySelector('[data-pin-nav="down"]');
      if (downBtn) {
        pinState.navDownBtn = downBtn;
      }
      var focusBtn = panel.querySelector('[data-pin-nav="focus"]');
      if (focusBtn) {
        pinState.focusBtn = focusBtn;
      }
      return panel;
    }
    panel = document.createElement('div');
    panel.setAttribute('data-pin-panel', '1');
    panel.style.position = 'absolute';
    panel.style.top = '140px';
    panel.style.right = '16px';
    panel.style.width = '340px';
    panel.style.maxHeight = 'calc(100vh - 170px)';
    panel.style.overflow = 'hidden';
    panel.style.zIndex = '410';
    panel.style.background = 'rgba(255,255,255,0.96)';
    panel.style.borderRadius = '6px';
    panel.style.boxShadow = '0 1px 6px rgba(0,0,0,0.25)';
    panel.style.display = 'none';
    panel.style.flexDirection = 'column';
    panel.style.alignItems = 'stretch';
    panel.style.padding = '0';
    panel.style.boxSizing = 'border-box';

    var header = document.createElement('div');
    header.setAttribute('data-pin-header', '1');
    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.justifyContent = 'space-between';
    header.style.padding = '8px 10px';
    header.style.borderBottom = '1px solid rgba(0,0,0,0.1)';
    header.style.fontSize = '13px';
    header.style.fontWeight = '600';

    var headerLeft = document.createElement('div');
    headerLeft.style.display = 'flex';
    headerLeft.style.alignItems = 'center';

    var headerCounter = document.createElement('span');
    headerCounter.setAttribute('data-pin-counter', '1');
    headerCounter.style.fontSize = '11px';
    headerCounter.style.color = '#4a5568';
    headerCounter.textContent = 'Pinned Article 0 of 0';

    headerLeft.appendChild(headerCounter);

    var headerControls = document.createElement('div');
    headerControls.style.display = 'flex';
    headerControls.style.alignItems = 'center';
    headerControls.style.gap = '6px';

    var upBtn = document.createElement('button');
    upBtn.type = 'button';
    upBtn.className = 'collocate-time-button';
    upBtn.textContent = '▲';
    upBtn.title = 'Show previous pinned article';
    upBtn.setAttribute('aria-label', 'Show previous pinned article');
    upBtn.setAttribute('data-pin-nav', 'up');

    var focusBtn = document.createElement('button');
    focusBtn.type = 'button';
    focusBtn.className = 'collocate-time-button';
    focusBtn.textContent = '⌖';
    focusBtn.title = 'Focus map on top pinned article';
    focusBtn.setAttribute('aria-label', 'Focus map on top pinned article');
    focusBtn.setAttribute('data-pin-nav', 'focus');

    var downBtn = document.createElement('button');
    downBtn.type = 'button';
    downBtn.className = 'collocate-time-button';
    downBtn.textContent = '▼';
    downBtn.title = 'Show next pinned article';
    downBtn.setAttribute('aria-label', 'Show next pinned article');
    downBtn.setAttribute('data-pin-nav', 'down');

    headerControls.appendChild(upBtn);
    headerControls.appendChild(focusBtn);
    headerControls.appendChild(downBtn);

    var clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.textContent = 'Clear';
    clearBtn.title = 'Remove all pinned popups';
    clearBtn.style.border = 'none';
    clearBtn.style.background = 'none';
    clearBtn.style.cursor = 'pointer';
    clearBtn.style.padding = '0';
    clearBtn.style.marginLeft = '8px';
    clearBtn.style.fontSize = '12px';
    clearBtn.addEventListener('click', function() {
      clearPinnedEntries();
    });

    header.appendChild(headerLeft);
    header.appendChild(headerControls);
    header.appendChild(clearBtn);

    pinState.headerCounter = headerCounter;
    pinState.navUpBtn = upBtn;
    pinState.navDownBtn = downBtn;
    pinState.focusBtn = focusBtn;

    upBtn.addEventListener('click', function() {
      rotatePinnedEntries(-1);
    });
    downBtn.addEventListener('click', function() {
      rotatePinnedEntries(1);
    });
    focusBtn.addEventListener('click', function() {
      focusPinnedTopEntry();
    });

    var body = document.createElement('div');
    body.setAttribute('data-pin-body', '1');
    body.style.padding = '10px';
    body.style.display = 'flex';
    body.style.flexDirection = 'column';
    body.style.gap = '10px';
    body.style.overflowY = 'auto';
    body.style.flex = '1 1 auto';
    body.style.minHeight = '0';

    panel.appendChild(header);
    panel.appendChild(body);

    var parent = document.querySelector('.leaflet-container') || document.body;
    parent.appendChild(panel);
    addScrollGuards(panel);
    addScrollGuards(body);
    enableHoverScroll(panel, body);
    suppressDoubleClick(panel);
    suppressDoubleClick(body);
    consumeDragEvents(panel);
    consumeDragEvents(body);
    consumeDragEvents(header);
    consumeDragEvents(headerControls);
    consumeDragEvents(upBtn);
    consumeDragEvents(downBtn);
    consumeDragEvents(focusBtn);
    consumeDragEvents(clearBtn);
    pinState.body = body;
    return panel;
  }

  function getPinBody() {
    if (pinState.body && document.body.contains(pinState.body)) {
      return pinState.body;
    }
    var panel = getPinPanel();
    var body = panel.querySelector('[data-pin-body]');
    pinState.body = body;
    return body;
  }

  function updatePinPanelVisibility() {
    var panel = getPinPanel();
    if (!panel) {
      return;
    }
    var hasEntries = pinState.entries.length > 0;
    panel.style.display = hasEntries ? 'flex' : 'none';
    if (hasEntries) {
      var body = getPinBody();
      if (body) {
        addScrollGuards(body);
        enableHoverScroll(panel, body);
        consumeDragEvents(body);
      }
    }
    updatePinnedHeaderCounter();
    if (pinState.navUpBtn) {
      pinState.navUpBtn.disabled = pinState.entries.length <= 1;
    }
    if (pinState.navDownBtn) {
      pinState.navDownBtn.disabled = pinState.entries.length <= 1;
    }
    if (pinState.focusBtn) {
      pinState.focusBtn.disabled = !hasEntries;
    }
  }

  function updatePinButtonsByKey(baseKey) {
    if (!baseKey) {
      return;
    }
    var active = (pinState.counts.get(baseKey) || 0) > 0;
    document.querySelectorAll('[data-pin-toggle="1"][data-pin-key="' + baseKey + '"]').forEach(function(btn) {
      btn.textContent = active ? '📍' : '📌';
      btn.title = active ? 'Unpin popup' : 'Pin popup';
      btn.style.color = active ? '#c53030' : '#2b6cb0';
    });
  }

  function preparePinnedClone(clone, entry) {
    if (!clone || !entry) {
      return;
    }
    clone.setAttribute('data-docked', '0');
    var pinBtn = clone.querySelector('[data-pin-toggle="1"]');
    if (pinBtn) {
      pinBtn.removeAttribute('data-pin-toggle');
      pinBtn.dataset.pinRemove = entry.id;
      pinBtn.textContent = '📍';
      pinBtn.title = 'Unpin this article';
      pinBtn.setAttribute('aria-label', 'Unpin this article');
      pinBtn.style.color = '#c53030';
      pinBtn.style.cursor = 'pointer';
      if (!pinBtn.dataset.pinRemoveAttached) {
        pinBtn.dataset.pinRemoveAttached = '1';
        pinBtn.addEventListener('click', function() {
          removePinnedEntry(entry.id);
        });
      }
    }
    var dockBtn = clone.querySelector('[data-dock-toggle="1"]');
    if (dockBtn) {
      dockBtn.remove();
    }
    var navControls = clone.querySelector('[data-nav-controls]');
    if (navControls && navControls.parentNode) {
      navControls.parentNode.removeChild(navControls);
    }
    var select = clone.querySelector('select[data-map-select]');
    if (select) {
      select.disabled = true;
      select.style.opacity = '0.7';
      select.title = 'Navigation disabled for pinned article';
      consumeDragEvents(select);
    }
    addScrollGuards(clone);
  }

  function removePinnedEntry(id) {
    var entry = pinState.lookup.get(id);
    if (!entry) {
      return;
    }
    if (entry.wrapper && entry.wrapper.parentNode) {
      entry.wrapper.parentNode.removeChild(entry.wrapper);
    }
    pinState.lookup.delete(id);
    pinState.entries = pinState.entries.filter(function(item) {
      return item && item.id !== id;
    });
    pinState.orderSequence = pinState.orderSequence.filter(function(orderId) {
      return orderId !== id;
    });
    updatePinnedOrderIndices();
    decrementPinCount(entry.baseKey);
    renderPinnedList();
    updatePinPanelVisibility();
  }

  function clearPinnedEntries() {
    var affectedKeys = Array.from(pinState.counts.keys());
    pinState.entries.forEach(function(entry) {
      if (entry && entry.wrapper && entry.wrapper.parentNode) {
        entry.wrapper.parentNode.removeChild(entry.wrapper);
      }
    });
    pinState.entries = [];
    pinState.lookup.clear();
    pinState.counts.clear();
    pinState.orderSequence = [];
    pinState.nextSeq = 1;
    updatePinnedOrderIndices();
    var body = getPinBody();
    if (body) {
      body.innerHTML = '';
    }
    affectedKeys.forEach(function(key) {
      updatePinButtonsByKey(key);
    });
    renderPinnedList();
    updatePinPanelVisibility();
  }

  function togglePin(root, mapObj, popupObj) {
    if (!root) {
      return;
    }
    var gid = root.getAttribute('data-group-id');
    if (!gid) {
      return;
    }
    var baseKey = getPinKey(root);
    if (!baseKey) {
      return;
    }
    loadData(function(dataset) {
      var groupData = dataset[gid];
      if (!groupData) {
        return;
      }
      var select = root.querySelector('select[data-map-select]');
      var currentIndex = 0;
      if (select && select.options && select.options.length) {
        currentIndex = select.selectedIndex;
      } else if (typeof groupData._lastIndex === 'number') {
        currentIndex = groupData._lastIndex;
      }
      if (!Number.isFinite(currentIndex) || currentIndex < 0) {
        currentIndex = 0;
      }
      var entry = (groupData.entries && groupData.entries[currentIndex]) || null;
      var selectionValue = '';
      if (select && select.options && select.options[currentIndex]) {
        selectionValue = select.options[currentIndex].value;
      }
      var timeLabel = root.dataset ? root.dataset.timeLabel : '';
      var signature = buildPinSignature(baseKey, groupData, entry, currentIndex, timeLabel, selectionValue);
      var existing = findPinnedBySignature(signature);
      if (existing) {
        removePinnedEntry(existing.id);
        return;
      }

      var panel = getPinPanel();
      var body = getPinBody();
      if (!panel || !body) {
        return;
      }

      var wrapper = document.createElement('div');
      wrapper.style.background = 'rgba(255,255,255,0.98)';
      wrapper.style.border = '1px solid rgba(0,0,0,0.1)';
      wrapper.style.borderRadius = '6px';
      wrapper.style.boxShadow = '0 1px 4px rgba(0,0,0,0.1)';
      wrapper.style.padding = '10px';
      wrapper.style.position = 'relative';
      wrapper.style.maxHeight = 'none';
      wrapper.style.overflow = 'visible';
      consumeDragEvents(wrapper);
      addScrollGuards(wrapper);

      var clone = root.cloneNode(true);
      wrapper.appendChild(clone);
      body.appendChild(wrapper);

      attach(clone);

      var latNum = root.dataset && root.dataset.lat ? Number(root.dataset.lat) : NaN;
      var lonNum = root.dataset && root.dataset.lon ? Number(root.dataset.lon) : NaN;
      var timeMeta = determineEntryTimeMeta(groupData, entry, timeLabel);

      var uniqueId = baseKey + '|pin:' + pinState.nextSeq++;
      var pinEntry = {
        id: uniqueId,
        baseKey: baseKey,
        signature: signature,
        wrapper: wrapper,
        clone: clone,
        gid: gid,
        layerId: groupData && Object.prototype.hasOwnProperty.call(groupData, 'layer_id') ? groupData.layer_id : gid,
        entryIndex: currentIndex,
        selectionValue: selectionValue,
        lat: Number.isFinite(latNum) ? latNum : null,
        lon: Number.isFinite(lonNum) ? lonNum : null,
        timeKey: timeMeta.key,
        timeLabel: timeMeta.label,
        timeIso: timeMeta.iso,
        dataset: groupData,
        orderIndex: 0,
      };

      pinState.lookup.set(uniqueId, pinEntry);
      pinState.orderSequence.push(uniqueId);
      updatePinnedOrderIndices();
      pinState.entries.push(pinEntry);
      incrementPinCount(baseKey);
      preparePinnedClone(clone, pinEntry);
      suppressDoubleClick(wrapper);
      consumeDragEvents(clone);
      renderPinnedList();
      updatePinPanelVisibility();
    });
  }

  function dockPopup(root, mapObj, popupObj) {
    if (!root) {
      return;
    }
    var body = getDockBody();
    dockState.preferred = true;
    dockState.root = root;
    dockState.map = mapObj || dockState.map || activeMap;
    dockState.popup = popupObj || null;
    var lat = root.dataset && root.dataset.lat ? Number(root.dataset.lat) : NaN;
    var lon = root.dataset && root.dataset.lon ? Number(root.dataset.lon) : NaN;
    dockState.lat = Number.isFinite(lat) ? lat : null;
    dockState.lon = Number.isFinite(lon) ? lon : null;
    body.innerHTML = '';
    body.appendChild(root);
    root.setAttribute('data-docked', '1');
    updateDockButtonState(root, root.querySelector('[data-dock-toggle="1"]'));
    addScrollGuards(root);
    addScrollGuards(body);
    showDockPanel();
    var panel = getDockPanel();
    positionDockPanel(panel);
    if (mapObj && typeof mapObj.closePopup === 'function') {
      mapObj.closePopup(popupObj);
    }
  }

  function undockPopup(mapObj) {
    var panel = getDockPanel();
    var root = dockState.root || panel.querySelector('[data-popup-root="1"]');
    var mapRef = mapObj || dockState.map || activeMap;
    dockState.preferred = false;
    if (!root) {
      hideDockPanel();
      return;
    }
    root.setAttribute('data-docked', '0');
    updateDockButtonState(root, root.querySelector('[data-dock-toggle="1"]'));
    dockState.root = null;
    dockState.popup = null;
    dockState.lat = null;
    dockState.lon = null;
    hideDockPanel();
    if (!mapRef || typeof mapRef.openPopup !== 'function') {
      return;
    }
    var popup = L.popup({ maxWidth: 360 });
    var targetLatLng = null;
    if (root.dataset && root.dataset.lat && root.dataset.lon) {
      var latNum = Number(root.dataset.lat);
      var lonNum = Number(root.dataset.lon);
      if (Number.isFinite(latNum) && Number.isFinite(lonNum)) {
        targetLatLng = L.latLng(latNum, lonNum);
      }
    }
    if (!targetLatLng && typeof mapRef.getCenter === 'function') {
      targetLatLng = mapRef.getCenter();
    }
    if (targetLatLng) {
      popup.setLatLng(targetLatLng).setContent(root).openOn(mapRef);
      dockState.popup = popup;
    }
  }

  function refreshPinnedEntries() {
    renderPinnedList();
    updatePinPanelVisibility();
  }

  function attachDockToggle(root, mapObj, popupObj) {
    if (!root) {
      return;
    }
    var pinBtn = root.querySelector('[data-pin-toggle="1"]');
    if (pinBtn && !pinBtn.dataset.listenerAttached) {
      pinBtn.dataset.listenerAttached = '1';
      pinBtn.addEventListener('click', function(ev) {
        ev.preventDefault();
        ev.stopPropagation();
        togglePin(root, mapObj, popupObj);
      });
    }
    if (pinBtn) {
      var key = getPinKey(root);
      pinBtn.dataset.pinKey = key;
      updatePinButtonsByKey(key);
    }
    var dockBtn = root.querySelector('[data-dock-toggle="1"]');
    if (dockBtn && !dockBtn.dataset.listenerAttached) {
      dockBtn.dataset.listenerAttached = '1';
      dockBtn.addEventListener('click', function(ev) {
        ev.preventDefault();
        ev.stopPropagation();
        if (root.getAttribute('data-docked') === '1') {
          undockPopup(mapObj);
        } else {
          dockPopup(root, mapObj, popupObj);
        }
      });
    }
    if (dockBtn) {
      updateDockButtonState(root, dockBtn);
    }
    if (dockState.preferred && mapObj && root.getAttribute('data-docked') !== '1') {
      setTimeout(function() {
        dockPopup(root, mapObj, popupObj);
      }, 0);
    }
    addScrollGuards(root);
  }
function renderEntry(root, groupData, index) {
    var body = root.querySelector('[data-detail-container]');
    if (!body || !groupData || !groupData.entries) {
      if (body) { body.innerHTML = '<div style="color:#999;">No data.</div>'; }
      updateProgress(root, groupData, index || 0);
      updateLocationProgress(root, groupData);
      return;
    }
    var entry = groupData.entries[index];
    if (!entry) {
      body.innerHTML = '<div style="color:#999;">No data.</div>';
      updateProgress(root, groupData, index || 0);
      updateLocationProgress(root, groupData);
      return;
    }
    groupData._lastIndex = index;
    updateLocationProgress(root, groupData);
    var gidForEntry = root.getAttribute('data-group-id') || ('group-' + Date.now());
    var parts = [];
    if (entry.first_line) {
      parts.push('<div style="margin-bottom:4px;"><span style="font-weight:600;">First line:</span> ' + entry.first_line + '</div>');
    }
    if (entry.context) {
      parts.push('<div style="margin-bottom:4px;"><span style="font-weight:600;">Context:</span> ' + entry.context + '</div>');
    }
    var metaParts = [];
    if (entry.date) metaParts.push('Date: ' + entry.date);
    if (entry.newspaper) metaParts.push('Newspaper: ' + entry.newspaper);
    if (entry.place) metaParts.push('Place: ' + entry.place);
    if (entry.page) metaParts.push('Page: ' + entry.page);
    if (metaParts.length) {
      parts.push('<div style="margin-top:4px; font-size:12px; color:#555;">' + metaParts.join(' | ') + '</div>');
    }
    if (entry.pdf_url) {
      parts.push('<div><a href="' + entry.pdf_url + '" target="_blank" rel="noopener">Source Image (PDF)</a></div>');
    }
    var fullId = gidForEntry + '-article-' + index;
    var previewHtml = entry.article_preview || '';
    var canLoadArticle = !!entry.article_html || (entry.attr_row_id && attrTableUrl) || !!previewHtml;
    if (canLoadArticle) {
      parts.push(
        '<div style="margin-top:6px;">'
        + '<button type="button" data-load-text="' + fullId + '" style="padding:4px 8px;">Load text</button>'
        + '</div>'
      );
      parts.push(
        '<div id="' + fullId + '" data-article-full '
        + 'style="display:none; margin-top:6px; max-height:240px; overflow:auto; border-top:1px solid #ddd; padding-top:6px;"></div>'
      );
    }
    body.innerHTML = parts.join('');

    if (canLoadArticle) {
      var loadBtn = body.querySelector('button[data-load-text]');
      var target = loadBtn ? document.getElementById(fullId) : null;
      if (loadBtn && target) {
        loadBtn.addEventListener('click', function() {
          var isVisible = target.style.display !== 'none';
          if (isVisible) {
            target.style.display = 'none';
            loadBtn.textContent = 'Load text';
            return;
          }
          if (target.dataset.loaded === '1') {
            target.style.display = 'block';
            loadBtn.textContent = 'Hide text';
            return;
          }
          if (entry.article_html) {
            target.innerHTML = entry.article_html;
            target.dataset.loaded = '1';
            target.style.display = 'block';
            loadBtn.textContent = 'Hide text';
            return;
          }
          if (entry.attr_row_id && attrTableUrl) {
            loadBtn.disabled = true;
            loadBtn.textContent = 'Loading…';
            articleLoader.get(entry.attr_row_id, function(rawText) {
              loadBtn.disabled = false;
              if (typeof rawText === 'string' && rawText.trim()) {
                target.innerHTML = highlightAndEscape(rawText);
                target.dataset.loaded = '1';
                target.style.display = 'block';
                loadBtn.textContent = 'Hide text';
              } else if (previewHtml) {
                target.innerHTML = previewHtml;
                target.dataset.loaded = '1';
                target.style.display = 'block';
                loadBtn.textContent = 'Hide text';
              } else if (rawText === '') {
                target.innerHTML = '<div style="color:#999;">No text available.</div>';
                target.dataset.loaded = '1';
                target.style.display = 'block';
                loadBtn.textContent = 'Hide text';
              } else {
                target.innerHTML = '<div style="color:#999;">Unable to load text from attribute table.</div>';
                target.dataset.loaded = 'error';
                target.style.display = 'block';
                loadBtn.textContent = 'Load text';
              }
            });
            return;
          }
          if (previewHtml) {
            target.innerHTML = previewHtml;
            target.dataset.loaded = '1';
            target.style.display = 'block';
            loadBtn.textContent = 'Hide text';
            return;
          }
          target.innerHTML = '<div style="color:#999;">No text available.</div>';
          target.dataset.loaded = '1';
          target.style.display = 'block';
          loadBtn.textContent = 'Hide text';
        });
      }
    }
    updateProgress(root, groupData, index);
  }
function attach(root) {
    var gid = root.getAttribute('data-group-id');
    if (!gid) return;
    loadData(function(data) {
      var groupData = data[gid];
      if (!groupData) return;
      if (root.dataset) {
        var latVal = '';
        var lonVal = '';
        if (Number.isFinite(Number(groupData.lat)) && Number.isFinite(Number(groupData.lon))) {
          latVal = String(groupData.lat);
          lonVal = String(groupData.lon);
        } else if (Array.isArray(groupData.coords) && groupData.coords.length) {
          var coord = groupData.coords.find(function(pt) {
            return pt && Number.isFinite(pt.lat) && Number.isFinite(pt.lon);
          });
          if (coord) {
            latVal = String(coord.lat);
            lonVal = String(coord.lon);
          }
        }
        root.dataset.lat = latVal;
        root.dataset.lon = lonVal;
        if (groupData.time_label) {
          root.dataset.timeLabel = groupData.time_label;
        } else {
          delete root.dataset.timeLabel;
        }
        if (groupData.location_index && groupData.location_total) {
          root.dataset.locationIndex = String(groupData.location_index);
          root.dataset.locationTotal = String(groupData.location_total);
        } else {
          delete root.dataset.locationIndex;
          delete root.dataset.locationTotal;
        }
      }
      addScrollGuards(root);
      suppressDoubleClick(root);
      consumeDragEvents(root);
      updateDockButtonState(root, root.querySelector('[data-dock-toggle="1"]'));
      var header = root.querySelector('[data-popup-header]');
      if (header && typeof groupData.title === 'string') {
        header.textContent = groupData.title;
      }
      var select = root.querySelector('select[data-map-select]');
      var startIndex = 0;
      if (select) {
        var desiredIndex = (typeof groupData._lastIndex === 'number') ? groupData._lastIndex : 0;
        var existing = select.options.length;
        select.innerHTML = '';
        if (groupData.entries && groupData.entries.length) {
          var frag = document.createDocumentFragment();
          groupData.entries.forEach(function(entry, idx) {
            var option = document.createElement('option');
            option.value = String(idx);
            option.textContent = entry.label || entry.date || ('Entry ' + (idx + 1));
            frag.appendChild(option);
          });
          select.appendChild(frag);
        }
        if (select.options.length) {
          if (desiredIndex < 0 || desiredIndex >= select.options.length) {
            desiredIndex = 0;
          }
          select.selectedIndex = desiredIndex;
          startIndex = desiredIndex;
        } else {
          select.selectedIndex = -1;
          startIndex = 0;
          renderEntry(root, groupData, 0);
        }
        if (!select.dataset.listenerAttached) {
          select.addEventListener('change', function() {
            groupData._lastIndex = select.selectedIndex;
            renderEntry(root, groupData, select.selectedIndex);
          });
          select.dataset.listenerAttached = '1';
        }
      }
      var buttons = root.querySelectorAll('button[data-step]');
      var hasOptions = !!(select && select.options && select.options.length);
      buttons.forEach(function(btn) {
        btn.disabled = !hasOptions;
        if (btn.dataset.listenerAttached) return;
        btn.addEventListener('click', function() {
          if (!select || !select.options.length) return;
          var step = parseInt(btn.getAttribute('data-step') || '0', 10);
          var total = select.options.length;
          var idx = select.selectedIndex + step;
          idx = (idx % total + total) % total;
          select.selectedIndex = idx;
          groupData._lastIndex = idx;
          renderEntry(root, groupData, idx);
        });
        btn.dataset.listenerAttached = '1';
      });
      if (!hasOptions) {
        groupData._lastIndex = 0;
      } else if (typeof groupData._lastIndex !== 'number') {
        groupData._lastIndex = startIndex;
      }
      renderEntry(root, groupData, startIndex);
      setHighlightForGroup(gid, groupData);
      attachDockToggle(root, activeMap, activeMap && activeMap._popup ? activeMap._popup : null);
    });
  }
  var mapVarName = "${map_var}";
  function whenLayerReady(layerName, callback, attempt) {
    if (!layerName || typeof callback !== 'function') {
      return;
    }
    var tryCount = typeof attempt === 'number' ? attempt : 0;
    var layerRef = window[layerName];
    if (layerRef && typeof layerRef.on === 'function') {
      callback(layerRef);
      return;
    }
    if (tryCount > 200) {
      console.warn('Layer not ready for cluster popups:', layerName);
      return;
    }
    setTimeout(function() { whenLayerReady(layerName, callback, tryCount + 1); }, 25);
  }
  function whenMapReady(callback, attempt) {
    var tryCount = typeof attempt === 'number' ? attempt : 0;
    var mapRef = window[mapVarName];
    if (mapRef && typeof mapRef.on === 'function') {
      callback(mapRef);
      return;
    }
    if (tryCount > 200) {
      console.warn('Map instance not ready for popups:', mapVarName);
      return;
    }
    setTimeout(function() { whenMapReady(callback, tryCount + 1); }, 20);
  }
  function init() {
    whenMapReady(function(mapObj) {
      activeMap = mapObj;
      initCollocateTimeControls();
      applyTimeFilter(mapObj);
      // Initialize collocate selector default and first sizing
      (function initCollocateSelector(){
        if (isTopTermMode()) {
          refreshCollocateSizes(activeMap);
          updateCollocateSummaryLineFromDataset(popupCache);
          return;
        }
        if (!collocateRanks || !collocateTerms || !collocateTerms.length) return;
        var sel = document.getElementById('collocateTermSelect');
        if (!sel) return;
        if (selectedCollocate) {
          var matchedIndex = -1;
          for (var i = 0; i < sel.options.length; i++) {
            if (String(sel.options[i].value).trim() === selectedCollocate) {
              matchedIndex = i;
              break;
            }
          }
          if (matchedIndex >= 0) {
            sel.selectedIndex = matchedIndex;
          } else {
            selectedCollocate = String(sel.value || collocateTerms[0] || '').trim();
          }
        } else {
          selectedCollocate = String(sel.value || collocateTerms[0] || '').trim();
        }
        sel.addEventListener('change', function(){
          selectedCollocate = String(this.value||'').trim();
          refreshCollocateSizes(activeMap);
          updateCollocateSummaryLineFromDataset(popupCache);
        });
        refreshCollocateSizes(activeMap);
        updateCollocateSummaryLineFromDataset(popupCache);
      })();
      if (mapObj.timeDimension && typeof mapObj.timeDimension.on === 'function') {
        mapObj.timeDimension.on('timeload', function() { applyTimeFilter(mapObj); });
        mapObj.timeDimension.on('timechange', function() { applyTimeFilter(mapObj); });
      }
      if (mapMode === 'heatmap') {
        mapObj.on('click', function(e) {
          if (!e || !e.latlng) {
            return;
          }
          if (e.originalEvent && e.originalEvent.target) {
            var target = e.originalEvent.target;
            if (typeof target.closest === 'function' && target.closest('.leaflet-popup')) {
              return;
            }
          }
          openGroupsPopup(e.latlng, mapObj);
        });
      }
      mapObj.on('popupopen', function(e) {
        var root = e.popup.getElement();
        if (!root) return;
        var container = root.querySelector('[data-popup-root="1"]');
        if (container) {
          attach(container);
          attachDockToggle(container, mapObj, e.popup);
          addScrollGuards(container);
        }
      });
      mapObj.on('popupclose', function() {
        clearHighlight();
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  if (typeof window !== 'undefined' && window && typeof window.addEventListener === 'function') {
    window.addEventListener('resize', refreshDockPanelLayout);
  }
${cluster_block}
})();
""")
    script_html = script_template.substitute(map_var=map_var, cluster_block=cluster_block)
    m.get_root().script.add_child(folium.Element(script_html))

    m.save(out_html)
    collocate_csv_path: Optional[str] = None
    if collocate_export_csv and popup_dataset:
        csv_filename = f"{base_name}_{suffix}_collocates.csv"
        export_path = os.path.join(out_dir, csv_filename)
        try:
            collocate_csv_path = _export_collocate_csv(
                export_path,
                groups=groups,
                popup_dataset=popup_dataset,
                summary=summary,
                collocate_map_variant=collocate_map_variant,
                rank_index=rank_index,
                collocate_hits_by_city=collocate_hits_by_city,
                collocate_terms_list=collocate_terms_list,
                time_index=time_index,
                range_start=range_start,
                range_end=range_end,
                collocate_rank_term_scope=collocate_rank_term_scope,
                collocate_rank_time_key=collocate_rank_time_key,
            )
        except Exception:
            collocate_csv_path = None

    metadata_paths: Dict[str, str] = {}
    metadata_common = {
        'tool': 'create_map',
        'parameters': {
            'mode': mode,
            'time_unit': time_unit,
            'time_step': time_step,
            'linger_unit': linger_unit,
            'linger_step': linger_step,
            'disable_time': disable_time,
            'metric': metric,
            'normalize': normalize,
            'normalize_denominator': normalize_denominator,
            'lightweight': lightweight,
            'table_mode': table_mode,
            'table_row_limit': table_row_limit,
            'heat_radius': heat_radius,
            'heat_value': heat_value,
            'grad_min_radius': grad_min_radius,
            'grad_max_radius': grad_max_radius,
            'collocate_rank_mode': collocate_rank_mode,
            'collocate_drop_stopwords': collocate_drop_stopwords,
            'collocate_window': collocate_window,
            'collocate_rank_top_n': collocate_rank_top_n,
            'collocate_rank_term_scope': collocate_rank_term_scope,
            'collocate_rank_time_key': collocate_rank_time_key,
            'collocate_rank_focus': collocate_rank_focus,
            'collocate_rank_focus_city': collocate_rank_focus_city,
            'collocate_rank_focus_state': collocate_rank_focus_state,
            'collocate_rank_time_label': collocate_rank_time_label,
            'collocate_rank_focus_label': collocate_rank_focus_label,
            'collocate_rank_colorize': collocate_rank_colorize,
            'collocate_time_slider': collocate_time_slider_enabled,
            'collocate_export_csv': collocate_export_csv,
        },
        'inputs': {
            'geojson_path': geojson_path,
        },
        'summary': summary,
        'config': config_payload,
        'collocate_term_stats': collocate_term_stats,
    }

    meta_payload_map = dict(metadata_common)
    meta_payload_map.update({'output_type': 'map_html'})
    meta_path = write_metadata_file(project_dir, out_html, meta_payload_map, enabled=metadata_enabled)
    if meta_path:
        metadata_paths['map_html'] = meta_path
    if attr_file:
        attr_meta = dict(metadata_common)
        attr_meta.update({'output_type': 'attribute_table_html'})
        meta_path = write_metadata_file(project_dir, attr_file, attr_meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths['attribute_table'] = meta_path

    if collocate_csv_path:
        csv_meta = dict(metadata_common)
        csv_meta.update({'output_type': 'collocate_csv'})
        meta_path = write_metadata_file(project_dir, collocate_csv_path, csv_meta, enabled=metadata_enabled)
        if meta_path:
            metadata_paths['collocate_csv'] = meta_path

    result: Dict[str, Optional[str]] = {"map_path": out_html, 'summary': summary}
    if attr_file:
        result["attribute_table"] = attr_file
    result['settings'] = {
        'mode': mode,
        'metric': metric_key,
        'normalize': normalize_flag,
        'normalize_denominator': denominator_key,
        'lightweight': lightweight,
        'table_mode': table_mode_norm,
        'table_row_limit': row_limit_val or 0,
        'collocate_time_slider': collocate_time_slider_enabled,
        'collocate_export_csv': collocate_export_csv,
    }
    if metadata_paths:
        result['metadata'] = metadata_paths
    if collocate_csv_path:
        result['collocate_csv'] = collocate_csv_path
    return result
