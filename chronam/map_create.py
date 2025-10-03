import html
import json
import os
import re
import uuid
from string import Template as StrTemplate
from jinja2 import Template as JinjaTemplate
from branca.element import MacroElement
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Callable

import folium
from folium import Html, Popup
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster


# ----------------------------
# Helpers: dates and formatting
# ----------------------------

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

def _extract_points(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract and validate points/features from GeoJSON features.
    Returns list of dicts with: lat, lon, props(dict), date(dt)
    """
    out: List[Dict[str, Any]] = []
    for feat in features:
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
) -> Dict[str, Any]:
    props = entry.get('props') or {}
    article_text = props.get('article') or ''
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
        'context': snippet_html or '',
        'pdf_url': _esc(pdf_url) if pdf_url else '',
        'date': _esc(date_val),
        'newspaper': _esc(newspaper_val),
        'place': _esc(place_val),
        'page': _esc((props.get('page') or '').strip()),
        'article_html': _article_excerpt(article_text, search_term, max_chars=3000)
        if (embed_article and article_text)
        else '',
        'article_preview': _article_excerpt(article_text, search_term, max_chars=600)
        if article_text
        else '',
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
        '<div data-nav-controls style="display:flex; align-items:center; gap:4px;">'
        '<button type="button" data-step="-1" style="padding:2px 6px;">&#9664;</button>'
        '<button type="button" data-step="1" style="padding:2px 6px;">&#9654;</button>'
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

    metric_text_html = ' | '.join(_esc(line) for line in metric_lines) if metric_lines else ''
    metrics_html = (
        f'<div data-popup-metrics style="margin-top:2px; font-size:12px; color:#555;">{metric_text_html}</div>'
        if metric_text_html
        else '<div data-popup-metrics style="margin-top:2px; font-size:12px; color:#555; display:none;"></div>'
    )

    summary_html = (
        f'<div data-popup-summary style="font-size:12px; color:#555; margin-bottom:4px;">Articles: {article_count:,}</div>'
        if article_count is not None else '<div data-popup-summary style="display:none;"></div>'
    )

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

    header_html = f'<div data-popup-header style="margin-bottom:4px; font-weight:600;">{_esc(title_text)}</div>'

    container_html = '<div data-detail-container style="margin-top:6px; font-size:14px; line-height:1.3; min-height:120px;">Loading…</div>'

    return (
        f'<div data-popup-root="1" data-group-id="{_esc(group_id)}" '
        f'data-total-entries="{len(entries)}" '
        'style="font-size:14px; line-height:1.25; min-width:260px;">'
        f'{header_html}'
        f'{summary_html}'
        f'{metrics_html}'
        f'{select_block}'
        f'{container_html}'
        f'{footer_html}'
        '</div>'
    )


def _group_points(points: List[Dict[str, Any]], precision: int = 6) -> List[Dict[str, Any]]:
    """Group point dictionaries by rounded lat/lon for shared popups."""
    grouped_map: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for pt in points:
        key = (round(pt["lat"], precision), round(pt["lon"], precision))
        grouped_map.setdefault(key, []).append(pt)

    groups: List[Dict[str, Any]] = []
    for idx, (key, entries) in enumerate(grouped_map.items()):
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
            if dt <= t:
                insert_i = i
                break
        else:
            insert_i = len(time_index) - 1

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
        if dt <= t:
            insert_i = i
            break
    else:
        insert_i = len(time_index) - 1

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
        for key in columns[2:]:
            value = props.get(key, '')
            cell_html: str
            if key == 'article':
                value = _truncate_plain_text(value, max_chars=420)
                cell_html = _esc(value)
                cells.append(_td(cell_html, [('data-column', 'article')]))
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

    Returns:
        dict with 'map_path' and optional 'attribute_table'.
    """

    permitted_modes = {"points", "heatmap", "graduated", "cluster"}
    mode_normalized = (mode or "points").strip().lower()
    if mode_normalized not in permitted_modes:
        mode_normalized = "points"
    mode = mode_normalized

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features") or []
    if not isinstance(features, list):
        raise ValueError("GeoJSON does not contain a valid 'features' list.")

    pts = _extract_points(features)
    for idx, entry in enumerate(pts):
        entry['_popup_row_id'] = _sanitize_element_id(f'feature-row-{idx}')
    groups = _group_points(pts)
    search_term = _detect_search_term(geojson_path, data)

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
    start_meta = metadata.get('start_date') or metadata.get('StartDate')
    end_meta = metadata.get('end_date') or metadata.get('EndDate')

    if dates_dt:
        min_dt = min(dates_dt)
        max_dt = max(dates_dt)
    else:
        min_dt = max_dt = None

    dated_pts = [p for p in pts if p.get("date") is not None]
    use_time_slider = bool(dated_pts) and time_enabled
    time_index: List[datetime] = []
    time_labels: List[str] = []
    if use_time_slider and min_dt and max_dt:
        time_index = _build_time_index(min_dt, max_dt, time_unit, max(1, int(time_step or 1)))
        time_labels = [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in time_index]

    start_str = start_meta or (min_dt.strftime('%Y-%m-%d') if min_dt else '')
    end_str = end_meta or (max_dt.strftime('%Y-%m-%d') if max_dt else '')
    if not start_str and end_str:
        start_str = end_str
    if not end_str and start_str:
        end_str = start_str
    date_range = (start_str, end_str) if start_str or end_str else ()

    popup_width = 320 if lightweight else 360

    embed_articles = not lightweight
    values: List[float] = []
    popup_dataset: Dict[str, Any] = {}
    for group in groups:
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

        entry_payloads = [
            _entry_payload(entry, search_term, embed_article=embed_articles)
            for entry in entries
        ]

        for entry in entries:
            entry["value"] = value

        first_props = entries[0].get('props', {}) if entries else {}
        city_raw = str(first_props.get('City') or '').strip()
        state_raw = str(first_props.get('State') or '').strip()
        place_label = ', '.join([p for p in (city_raw, state_raw) if p])

        title_text, article_count, _ = _group_header(entries, stats, search_term)
        dataset_entry: Dict[str, Any] = {
            'entries': entry_payloads,
            'full_entries': entry_payloads,
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

        if time_enabled and use_time_slider and time_index:
            time_bins: Dict[str, Dict[str, Any]] = {}
            for idx_entry, entry in enumerate(entries):
                dt = entry.get('date')
                bin_indices = _assign_time_bins(dt, time_index, linger_unit, int(linger_step or 0))
                for bin_idx in bin_indices:
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
    start_meta = metadata.get('start_date') or metadata.get('StartDate')
    end_meta = metadata.get('end_date') or metadata.get('EndDate')

    if dates_dt:
        min_dt = min(dates_dt)
        max_dt = max(dates_dt)
    else:
        min_dt = max_dt = None

    dated_pts = [p for p in pts if p.get("date") is not None]
    use_time_slider = bool(dated_pts) and time_enabled
    time_index: List[datetime] = []
    time_labels: List[str] = []
    if use_time_slider and min_dt and max_dt:
        time_index = _build_time_index(min_dt, max_dt, time_unit, max(1, int(time_step or 1)))
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
    }

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix_parts = [mode, metric_key]
    if normalize_flag:
        suffix_parts.append('norm')
    if lightweight:
        suffix_parts.append('lite')
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
        table_kwargs['include_columns'] = ['date', 'Title', 'City', 'State', 'lccn', 'page', 'url']
        table_kwargs['omit_article'] = True

    max_rows = row_limit_val
    if lightweight:
        light_limit = min(len(pts), 1000) if pts else None
        if light_limit:
            max_rows = light_limit if max_rows is None else min(max_rows, light_limit)
        if table_mode_norm == 'full':
            table_kwargs.setdefault('omit_article', True)
            table_kwargs.setdefault('include_columns', [
                'date',
                'Title',
                'headline',
                'Headline',
                'City',
                'State',
                'page',
                'url',
            ])

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

    header_lines = [f'<div><strong>File:</strong> {_esc(summary["geojson_name"])}</div>']
    if summary.get('term'):
        header_lines.append(f'<div><strong>Term:</strong> {_esc(summary["term"])}</div>')
    if summary.get('date_range'):
        start, end = summary['date_range']
        date_text = start if start == end else ' – '.join([s for s in (start, end) if s])
        if date_text:
            header_lines.append(f'<div><strong>Date range:</strong> {_esc(date_text)}</div>')
    counts_line = f"{articles_count:,} articles | {len(newspaper_ids):,} newspapers | {len(city_set):,} cities"
    header_lines.append(f'<div><strong>Counts:</strong> {_esc(counts_line)}</div>')
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
    if attr_file:
        link_name = os.path.basename(attr_file)
        header_lines.append(
            f'<div><a href="{html.escape(link_name)}" target="_blank" rel="noopener">Open attribute table</a></div>'
        )
    if lightweight:
        header_lines.append(
            '<div style="font-size:12px; color:#555;">Lightweight mode: popups and table trimmed for size.</div>'
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

    config_payload = {
        'attribute_table': os.path.basename(attr_file) if attr_file else '',
        'search_term': search_term or '',
        'inline_articles': bool(embed_articles),
        'time_labels': time_labels if use_time_slider and time_labels else [],
        'map_mode': mode,
        'click_radius_px': heat_radius_val,
    }
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
  var activeMap = null;
  var highlightLayer = null;
  var currentHighlightId = null;

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
    if (typeof config.click_radius_px === 'number' && Number.isFinite(config.click_radius_px)) {
      clickRadiusPx = Math.max(4, Number(config.click_radius_px));
    }
  })();

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
    if (typeof timeValue === 'number' && Array.isArray(timeLabels) && timeLabels.length) {
      var idx = Math.round(timeValue);
      if (Number.isFinite(idx) && idx >= 1 && idx <= timeLabels.length) {
        var label = timeLabels[idx - 1];
        if (label && keys.indexOf(label) === -1) {
          keys.push(label);
        }
      }
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
    if (!mapObj || !mapObj.timeDimension || typeof mapObj.timeDimension.getCurrentTime !== 'function') {
      return;
    }
    var rawTime = mapObj.timeDimension.getCurrentTime();
    var currentKeys = resolveTimeKeys(rawTime);
    if (!currentKeys.length) {
      return;
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
        if (!data.time_bins) {
          data.entries = data.full_entries.slice();
          data.value = data.full_value;
          data.article_count = data.full_article_count;
          data.title = data.full_title;
          data.time_label = '';
          data._hasTimeFilter = false;
          return;
        }
        data._hasTimeFilter = true;
        var bin = null;
        for (var ck = 0; ck < currentKeys.length; ck++) {
          var attempt = currentKeys[ck];
          if (!attempt) {
            continue;
          }
          if (data.time_bins && Object.prototype.hasOwnProperty.call(data.time_bins, attempt)) {
            bin = data.time_bins[attempt];
            if (bin) {
              break;
            }
          }
          if (Array.isArray(timeLabels) && timeLabels.length) {
            var idxCandidate = Number(attempt);
            if (Number.isFinite(idxCandidate) && idxCandidate >= 1 && idxCandidate <= timeLabels.length) {
              var altKey = timeLabels[idxCandidate - 1];
              if (altKey && data.time_bins && Object.prototype.hasOwnProperty.call(data.time_bins, altKey)) {
                bin = data.time_bins[altKey];
                if (bin) {
                  break;
                }
              }
            }
          }
        }
        if (bin && Array.isArray(bin.indexes) && bin.indexes.length) {
          var mapped = [];
          bin.indexes.forEach(function(idx) {
            if (idx >= 0 && idx < data.full_entries.length) {
              var payload = data.full_entries[idx];
              if (payload) mapped.push(payload);
            }
          });
          data.entries = mapped;
          data.value = (typeof bin.value !== 'undefined') ? bin.value : data.full_value;
          data.article_count = (typeof bin.article_count !== 'undefined') ? bin.article_count : mapped.length;
          data.title = bin.title || data.full_title;
          data.time_label = bin.time_label || '';
        } else {
          data.entries = [];
          data.value = 0;
          data.article_count = 0;
          data.title = data.full_title;
          data.time_label = '';
        }
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
          if (!source || !source.entries || !source.entries.length) {
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
      if (mapObj._popup && typeof mapObj._popup.getElement === 'function') {
        var popupEl = mapObj._popup.getElement();
        if (popupEl) {
          var root = popupEl.querySelector('[data-popup-root="1"]');
          if (root) {
            attach(root);
          }
        }
      }
      if (currentHighlightId && dataset[currentHighlightId]) {
        var highlightedData = dataset[currentHighlightId];
        if (highlightedData && highlightedData.entries && highlightedData.entries.length) {
          setHighlightForGroup(currentHighlightId, highlightedData);
        } else {
          clearHighlight();
        }
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
    progress.textContent = 'Article ' + current + ' of ' + total;
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
      var header = root.querySelector('[data-popup-header]');
      if (header && typeof groupData.title === 'string') {
        header.textContent = groupData.title;
      }
      var summary = root.querySelector('[data-popup-summary]');
      if (summary) {
        if (typeof groupData.article_count === 'number') {
          summary.textContent = 'Articles: ' + Number(groupData.article_count).toLocaleString();
          summary.style.display = 'block';
        } else {
          summary.textContent = '';
          summary.style.display = 'none';
        }
      }
      var metricsEl = root.querySelector('[data-popup-metrics]');
      if (metricsEl) {
        var metricParts = [];
        if (groupData.metric_display && typeof groupData.value === 'number') {
          var metricValue = Number(groupData.value);
          if (Number.isFinite(metricValue)) {
            var metricText = groupData.normalized ? metricValue.toFixed(4) : Math.round(metricValue).toLocaleString();
            metricParts.push(groupData.metric_display + ': ' + metricText);
          }
        }
        if (groupData.normalized && groupData.denominator_label) {
          metricParts.push('Normalized by ' + groupData.denominator_label);
        }
        if (metricParts.length) {
          metricsEl.innerHTML = metricParts.join(' | ');
          metricsEl.style.display = 'block';
        } else {
          metricsEl.innerHTML = '';
          metricsEl.style.display = 'none';
        }
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
      applyTimeFilter(mapObj);
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
${cluster_block}
})();
""")
    script_html = script_template.substitute(map_var=map_var, cluster_block=cluster_block)
    m.get_root().script.add_child(folium.Element(script_html))

    m.save(out_html)
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
    }
    return result
