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
from folium.plugins import HeatMap, HeatMapWithTime


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


def _group_popup_html(
    group: Dict[str, Any],
    search_term: Optional[str],
    group_id: str,
    *,
    lightweight: bool = False,
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

    first_props = entries[0].get('props', {}) if entries else {}
    city_name = (first_props.get('City') or '').strip() or 'this location'
    term_text = (search_term or '').strip()
    article_count = int(stats.get('article_count', len(entries))) if stats else len(entries)

    if term_text:
        title_text = f'Articles mentioning "{term_text}" in {city_name}'
    else:
        title_text = f'Articles in {city_name}'

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

    metric_text_html = ''
    if metric_lines:
        metric_text_html = ' | '.join(_esc(line) for line in metric_lines)
    metrics_html = (
        f'<div style="margin-top:2px; font-size:12px; color:#555;">{metric_text_html}</div>'
        if metric_text_html else ''
    )

    select_block = f'<div style="margin-top:6px;">{select_html}</div>'
    footer_html = (
        '<div style="margin-top:6px; display:flex; justify-content:flex-end; align-items:center; gap:6px;">'
        '<span data-article-progress style="font-size:12px; color:#555;"></span>'
        f'{nav_buttons_html}'
        '</div>'
    )

    header_html = f'<div style="margin-bottom:4px; font-weight:600;">{_esc(title_text)}</div>'

    container_html = '<div data-detail-container style="margin-top:6px; font-size:14px; line-height:1.3; min-height:120px;">Loading…</div>'

    return (
        f'<div data-popup-root="1" data-group-id="{_esc(group_id)}" '
        f'data-total-entries="{len(entries)}" '
        'style="font-size:14px; line-height:1.25; min-width:260px;">'
        f'{header_html}'
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
) -> None:
    """Add grouped point markers with selection popups to the map."""
    for idx, group in enumerate(groups):
        entries = group.get('entries') or []
        if not entries:
            continue
        group_id = group.get('id') or f'group-{idx}'
        popup_html = Html(
            _group_popup_html(group, search_term, group_id, lightweight=lightweight),
            script=True,
        )
        popup_obj = Popup(popup_html, max_width=popup_width)
        lat, lon = group.get('location', (entries[0]['lat'], entries[0]['lon']))
        radius = radius_func(group) if callable(radius_func) else 4.0
        try:
            radius_value = max(1.0, float(radius))
        except (TypeError, ValueError):
            radius_value = 4.0
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius_value,
            weight=0,
            fill=True,
            fill_opacity=0.85,
            color="#2b6cb0",
            fill_color="#2b6cb0",
            popup=popup_obj,
        ).add_to(map_obj)


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
        cells = [f"<td>{lat_cell}</td>", f"<td>{lon_cell}</td>"]
        for key in columns[2:]:
            value = props.get(key, '')
            cell_html: str
            if key == 'article':
                value = _truncate_plain_text(value, max_chars=420)
                cell_html = _esc(value)
            elif key in link_set and value:
                href = html.escape(str(value), quote=True)
                text = html.escape(str(value))
                cell_html = f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'
            else:
                cell_html = _esc(value)
            cells.append(f"<td>{cell_html}</td>")
        rows.append('<tr>' + ''.join(cells) + '</tr>')

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

    permitted_modes = {"points", "heatmap", "graduated"}
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

    popup_width = 320 if lightweight else 360

    values: List[float] = []
    popup_dataset: Dict[str, Any] = {}
    for group in groups:
        stats = _compute_group_stats(group.get("entries", []), search_term)
        group["stats"] = stats
        value = _compute_group_value(stats, metric_key, normalize_flag, denominator_key)
        group["value"] = value
        group["metric_key"] = metric_key
        group["metric_display"] = metric_display
        group["metric_normalized_display"] = metric_normalized_display
        group["normalized"] = normalize_flag
        group["denominator_label"] = denom_label if normalize_flag else ''

        entry_payloads: List[Dict[str, Any]] = []
        for entry in group.get("entries", []):
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
            }
            label_value = _feature_label(props)
            if label_value:
                payload['label'] = label_value

            entry_payloads.append(payload)

            entry["value"] = value

        popup_dataset[group['id']] = {'entries': entry_payloads}
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

    start_str = start_meta or (min_dt.strftime('%Y-%m-%d') if min_dt else '')
    end_str = end_meta or (max_dt.strftime('%Y-%m-%d') if max_dt else '')
    if not start_str and end_str:
        start_str = end_str
    if not end_str and start_str:
        end_str = start_str
    date_range = (start_str, end_str) if start_str or end_str else ()

    time_enabled = mode == 'heatmap' and not disable_time
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
        dated_pts = [p for p in pts if p.get("date") is not None]
        use_time_slider = bool(dated_pts) and not disable_time

        if use_time_slider and dated_pts:
            min_dt = min(p["date"] for p in dated_pts)
            max_dt = max(p["date"] for p in dated_pts)
            idx = _build_time_index(min_dt, max_dt, time_unit, max(1, int(time_step or 1)))
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
            index_labels = [t.strftime("%Y-%m-%dT00:00:00Z") for t in idx]
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
        )
    else:
        _add_point_markers(
            m,
            groups,
            search_term,
            point_radius,
            popup_width=popup_width,
            lightweight=lightweight,
        )


    attr_file = _write_attribute_table(pts, attr_path, **table_kwargs)
    if attr_file:
        summary['attribute_table'] = attr_file

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

    popup_json_js = json.dumps(popup_dataset, ensure_ascii=False).replace('</', '<\\/')
    data_script = (
        '<script id="popup-data" type="application/json">'
        f"{popup_json_js}"
        '</script>'
    )
    m.get_root().html.add_child(folium.Element(data_script))

    map_var = m.get_name()
    script_template = StrTemplate("""
(function() {
  var popupCache = null;
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
  function renderEntry(root, groupData, index) {
    var body = root.querySelector('[data-detail-container]');
    if (!body || !groupData || !groupData.entries) {
      if (body) { body.innerHTML = '<div style="color:#999;">No data.</div>'; }
      updateProgress(root, groupData, index || 0);
      return;
    }
    var entry = groupData.entries[index];
    if (!entry) {
      body.innerHTML = '<div style="color:#999;">No data.</div>';
      updateProgress(root, groupData, index || 0);
      return;
    }
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
      parts.push('<div><a href="' + entry.pdf_url + '" target="_blank" rel="noopener">Source Image</a></div>');
    }
    body.innerHTML = parts.join('');
    updateProgress(root, groupData, index);
  }
  function attach(root) {
    var gid = root.getAttribute('data-group-id');
    if (!gid) return;
    loadData(function(data) {
      var groupData = data[gid];
      if (!groupData) return;
      var select = root.querySelector('select[data-map-select]');
      if (select) {
        if (select.dataset.optionsLoaded !== '1') {
          if (select.dataset.optionsSource === 'json') {
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
          }
          if (select.options.length && select.selectedIndex < 0) {
            select.selectedIndex = 0;
          }
          select.dataset.optionsLoaded = '1';
        }
        if (!select.dataset.listenerAttached) {
          select.addEventListener('change', function() {
            renderEntry(root, groupData, select.selectedIndex);
          });
          select.dataset.listenerAttached = '1';
        }
      }
      var buttons = root.querySelectorAll('button[data-step]');
      var hasOptions = !!(select && select.options && select.options.length);
      buttons.forEach(function(btn) {
        if (!hasOptions) {
          btn.disabled = true;
        } else if (btn.disabled) {
          btn.disabled = false;
        }
        if (btn.dataset.listenerAttached) return;
        btn.addEventListener('click', function() {
          var step = parseInt(btn.getAttribute('data-step') || '0', 10);
          var sel = root.querySelector('select[data-map-select]');
          if (!sel || !sel.options.length) return;
          var total = sel.options.length;
          var idx = sel.selectedIndex + step;
          idx = (idx % total + total) % total;
          sel.selectedIndex = idx;
          renderEntry(root, groupData, idx);
        });
        btn.dataset.listenerAttached = '1';
      });
      var startIndex = 0;
      if (select && select.options.length) {
        if (select.selectedIndex < 0) {
          select.selectedIndex = 0;
        }
        startIndex = select.selectedIndex;
      }
      renderEntry(root, groupData, startIndex);
    });
  }
  var mapVarName = "${map_var}";
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
      mapObj.on('popupopen', function(e) {
        var root = e.popup.getElement();
        if (!root) return;
        var container = root.querySelector('[data-popup-root="1"]');
        if (container) {
          attach(container);
        }
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
""")
    script_html = script_template.substitute(map_var=map_var)
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
