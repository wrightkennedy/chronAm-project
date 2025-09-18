import html
import json
import os
import re
import uuid
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
    return html.escape(str(value))


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


def _article_excerpt(text: Any, term: Any, max_chars: int = 600) -> str:
    if not text:
        return ""
    term_str = str(term).strip() if term else ""
    text_str = re.sub(r"\s+", " ", str(text)).strip()
    if max_chars and len(text_str) > max_chars:
        text_str = text_str[:max_chars].rstrip() + '…'
    return _highlight_term(text_str, term_str)


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


def _popup_html(props: Dict[str, Any], search_term: Optional[str], popup_id: str) -> str:
    summary = _first_line_excerpt(props.get("article") or "", 120)
    snippet_html = _keyword_snippet(props.get("article"), search_term)
    url = (props.get("url") or "").strip()
    pdf_url = url.replace(".jp2", ".pdf") if url else ""

    date_str = _format_date_str(props.get("date") or "")
    newspaper = (props.get("newspaper_name") or props.get("Title") or "").strip()
    city = (props.get("City") or "").strip()
    state = (props.get("State") or "").strip()
    location = ", ".join([p for p in (city, state) if p])
    page = (props.get("page") or "").strip()

    article_html = _article_excerpt(props.get("article"), search_term, max_chars=700)
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
    if snippet_html:
        lines.append(
            '<div style="margin-bottom:4px;"><span style="font-weight:600;">Context:</span> '
            f'{snippet_html}</div>'
        )
    if summary:
        lines.append(f'<div style="margin-bottom:4px;">{_esc(summary)}</div>')
    if pdf_url:
        lines.append(
            f'<div><a href="{_esc(pdf_url)}" target="_blank" rel="noopener">Source Image</a></div>'
        )

    if article_html or meta_rows:
        button = (
            f"<button type=\"button\" style=\"margin-top:6px;\" "
            f"onclick=\"var el=document.getElementById('{show_more_id}');"
            "if(!el){return;}var hidden=el.style.display==='none';"
            "el.style.display=hidden?'block':'none';"
            "this.textContent=hidden?'Show less':'Show more';\">Show more</button>"
        )
        lines.append(button)
        hidden_parts = ''.join(meta_rows)
        if article_html:
            hidden_parts += f'<div style="margin-top:6px;">{article_html}</div>'
        lines.append(
            f'<div id="{show_more_id}" style="display:none; margin-top:6px; max-height:260px; overflow:auto;">'
            f'{hidden_parts}</div>'
        )

    if not lines:
        lines.append('<div>No additional information available.</div>')

    return '<div style="font-size:14px; line-height:1.25;">' + "\n".join(lines) + "</div>"


def _feature_label(props: Dict[str, Any]) -> str:
    date_str = _format_date_str(props.get("date") or "")
    newspaper = (props.get("newspaper_name") or props.get("Title") or "").strip()
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


def _group_popup_html(entries: List[Dict[str, Any]], search_term: Optional[str], group_id: str) -> str:
    if not entries:
        return '<div style="font-size:14px;">No data available.</div>'

    options = []
    details = []
    for idx, entry in enumerate(entries):
        label = _feature_label(entry["props"])
        options.append(f'<option value="{idx}">{_esc(label)}</option>')
        detail_id = _sanitize_element_id(f"{group_id}-{idx}")
        detail_html = _popup_html(entry["props"], search_term, detail_id)
        display = 'block' if idx == 0 else 'none'
        details.append(
            f'<div data-detail="{idx}" style="display:{display}; margin-top:6px;">{detail_html}</div>'
        )

    header = f'Articles at this location ({len(entries)})'
    onchange_js = (
        "var root=this.closest('[data-popup-root]');"
        "if(!root){return;}"
        "var value=this.value;"
        "root.querySelectorAll('[data-detail]').forEach(function(el){"
        "el.style.display = el.getAttribute('data-detail') === value ? 'block' : 'none';"
        "});"
    )
    select_html = (
        f'<select data-map-select style="width:100%;" onchange="{onchange_js}">' +
        "".join(options) +
        '</select>'
    )

    return (
        '<div data-popup-root="1" style="font-size:14px; line-height:1.25; min-width:240px;">'
        f'<div style="margin-bottom:4px; font-weight:600;">{_esc(header)}</div>'
        f'{select_html}'
        + "".join(details) +
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
) -> None:
    """Add grouped point markers with selection popups to the map."""
    for idx, group in enumerate(groups):
        entries = group.get('entries') or []
        if not entries:
            continue
        group_id = group.get('id') or f'group-{idx}'
        popup_html = Html(_group_popup_html(entries, search_term, group_id), script=True)
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


def _write_attribute_table(points: List[Dict[str, Any]], out_path: str) -> Optional[str]:
    """Render a simple HTML attribute table for the supplied points."""

    def _format_coord(value: Any) -> str:
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return ""

    columns: List[str] = ["Latitude", "Longitude"]
    seen = {"Latitude", "Longitude"}
    for entry in points:
        props = entry.get("props") if isinstance(entry, dict) else None
        if not isinstance(props, dict):
            continue
        for key in props.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                columns.append(key_str)

    rows: List[str] = []
    for entry in points:
        props = entry.get("props") if isinstance(entry, dict) else {}
        if not isinstance(props, dict):
            props = {}
        lat_cell = _esc(_format_coord(entry.get("lat")))
        lon_cell = _esc(_format_coord(entry.get("lon")))
        cells = [f"<td>{lat_cell}</td>", f"<td>{lon_cell}</td>"]
        for key in columns[2:]:
            cells.append(f"<td>{_esc(props.get(key, ''))}</td>")
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    if not rows:
        rows.append(
            '<tr><td colspan="{}">{}</td></tr>'.format(
                len(columns),
                _esc("No point data available."),
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

    allowed_denominators = {"word_count", "article_count", "page_count"}
    denom_key = (normalize_denominator or "").strip().lower()
    normalize_flag = bool(normalize) and denom_key in allowed_denominators
    denominator_key = denom_key if normalize_flag else None

    values: List[float] = []
    for group in groups:
        stats = _compute_group_stats(group.get("entries", []), search_term)
        group["stats"] = stats
        value = _compute_group_value(stats, metric_key, normalize_flag, denominator_key)
        group["value"] = value
        for entry in group.get("entries", []):
            entry["value"] = value
        values.append(value)

    if groups and not any(v > 0 for v in values):
        for group in groups:
            group["value"] = 1.0
            for entry in group.get("entries", []):
                entry["value"] = 1.0

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

        _add_point_markers(m, groups, search_term, point_radius)
    elif mode == "graduated":
        resolver = _graduated_radius_resolver(groups, float(grad_min_val), float(grad_max_val))
        _add_point_markers(m, groups, search_term, resolver)
    else:
        _add_point_markers(m, groups, search_term, point_radius)

    fname = os.path.basename(geojson_path)
    label_text = f"GeoJSON: {fname}"
    caption_html = (
        '<div style="position: fixed; top: 5px; left: 5px; z-index:9999; '
        'background-color: rgba(255,255,255,0.8); padding: 2px;">'
        f"{_esc(label_text)}</div>"
    )
    m.get_root().html.add_child(folium.Element(caption_html))

    base = os.path.splitext(os.path.basename(geojson_path))[0]
    out_dir = os.path.dirname(geojson_path)
    attr_path = os.path.join(out_dir, f"{base}_attributes.html")
    attr_file = _write_attribute_table(pts, attr_path)
    if attr_file:
        link_name = os.path.basename(attr_file)
        table_button = (
            '<div style="position: fixed; top: 5px; right: 5px; z-index:9999;">'
            f'<a href="{html.escape(link_name)}" target="_blank" rel="noopener" '
            'style="background: rgba(255,255,255,0.85); padding: 6px 10px; '
            'border-radius: 4px; text-decoration: none; font-weight: 600; color: #2b6cb0;">'
            'Attribute Table</a></div>'
        )
        m.get_root().html.add_child(folium.Element(table_button))

    suffix = "_heatmap" if mode == "heatmap" else "_graduated" if mode == "graduated" else "_points"
    out_html = os.path.join(out_dir, f"{base}{suffix}.html")
    m.save(out_html)
    result: Dict[str, Optional[str]] = {"map_path": out_html}
    if attr_file:
        result["attribute_table"] = attr_file
    return result
