import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

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


def _popup_html(props: Dict[str, Any]) -> str:
    city = (props.get("City") or "").strip()
    state = (props.get("State") or "").strip()
    newspaper = (props.get("newspaper_name") or props.get("Title") or "").strip()
    date_str = _format_date_str(props.get("date") or "")
    first_line = _first_line_excerpt(props.get("article") or "", 75)
    url = (props.get("url") or "").strip()
    pdf_url = url.replace(".jp2", ".pdf") if url else ""

    lines = []
    line_city_state = ", ".join([p for p in [city, state] if p])
    if line_city_state:
        lines.append(f'<div style="margin-bottom:2px;">{line_city_state}</div>')
    if newspaper:
        lines.append(f'<div style="margin-bottom:2px;">{newspaper}</div>')
    if date_str:
        lines.append(f'<div style="margin-bottom:2px;">{date_str}</div>')
    if first_line:
        lines.append(f'<div style="margin-bottom:4px;">{first_line}</div>')
    if pdf_url:
        lines.append(f'<div><a href="{pdf_url}" target="_blank" rel="noopener">Source Image</a></div>')

    return '<div style="font-size:14px; line-height:1.25;">' + "\n".join(lines) + "</div>"


def _heat_slices(
    points: List[Dict[str, Any]],
    time_index: List[datetime],
    linger_unit: str,
    linger_step: int
) -> List[List[Tuple[float, float]]]:
    """
    Build HeatMapWithTime slices: list where each entry is a list of [lat, lon] for that time slice.
    We include a point in slices from its date slice up to linger length.
    """
    slices: List[List[Tuple[float, float]]] = [[] for _ in time_index]

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
            slices[j].append((p["lat"], p["lon"]))

    return slices


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
) -> str:
    """
    Create a leaflet map next to the GeoJSON.

    Modes:
      - "points": static points (CircleMarker dots) with popups.
      - "heatmap": time slider + heat density (HeatMapWithTime). Also adds a static dot layer
                   with popups so the user can click while scrubbing the slider.

    Time slider parameters (heatmap mode only):
      - time_unit: 'day' | 'week' | 'month' | 'year'   (default 'month')
      - time_step: integer step for the slider increments (default 1)
      - linger_unit: same options as time_unit (default 'week')
      - linger_step: how long (in linger_unit) a point remains visible after its date (default 0)

    Returns: path to generated HTML.
    """
    mode = (mode or "points").strip().lower()
    if mode not in ("points", "heatmap"):
        mode = "points"

    # Load features
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features") or []
    if not isinstance(features, list):
        raise ValueError("GeoJSON does not contain a valid 'features' list.")

    # Extract validated points with parsed dates
    pts = _extract_points(features)

    # Base map centered on US
    m = folium.Map(location=[37.8, -96.0], zoom_start=4)

    if mode == "heatmap":
        dated_pts = [p for p in pts if p["date"] is not None]
        if dated_pts:
            min_dt = min(p["date"] for p in dated_pts)
            max_dt = max(p["date"] for p in dated_pts)
        else:
            # no dates: fallback to static heat
            min_dt = max_dt = None

        if min_dt and max_dt:
            # Build a datetime index and heat slices
            idx = _build_time_index(min_dt, max_dt, time_unit, int(time_step or 1))
            slices = _heat_slices(dated_pts, idx, linger_unit, int(linger_step or 0))
            heat_data = [[[lat, lon] for (lat, lon) in frame] for frame in slices]

            # Convert to ISO strings for display in the control
            index_labels = [t.strftime("%Y-%m-%dT00:00:00Z") for t in idx]

            # Add the heatmap-with-time
            HeatMapWithTime(
                heat_data,
                index=index_labels,
                auto_play=False,
                max_opacity=0.7,
                radius=15,
            ).add_to(m)

            # Add a STATIC dot layer with popups (always clickable)
            for p in pts:
                html = Html(_popup_html(p["props"]), script=True)
                popup_obj = Popup(html, max_width=300)
                folium.CircleMarker(
                    location=[p["lat"], p["lon"]],
                    radius=3,
                    weight=0,
                    fill=True,
                    fill_opacity=0.85,
                    color="#2b6cb0",
                    fill_color="#2b6cb0",
                    popup=popup_obj,
                ).add_to(m)
        else:
            # fallback: no valid dates -> single static heat layer + static dots with popups
            coords = [[p["lat"], p["lon"]] for p in pts]
            if coords:
                HeatMap(coords, max_opacity=0.7, radius=15).add_to(m)
            for p in pts:
                html = Html(_popup_html(p["props"]), script=True)
                popup_obj = Popup(html, max_width=300)
                folium.CircleMarker(
                    location=[p["lat"], p["lon"]],
                    radius=3,
                    weight=0,
                    fill=True,
                    fill_opacity=0.85,
                    color="#2b6cb0",
                    fill_color="#2b6cb0",
                    popup=popup_obj,
                ).add_to(m)

    else:
        # Static points mode with popups
        for p in pts:
            html = Html(_popup_html(p["props"]), script=True)
            popup_obj = Popup(html, max_width=300)
            folium.CircleMarker(
                location=[p["lat"], p["lon"]],
                radius=3,
                weight=0,
                fill=True,
                fill_opacity=0.85,
                color="#2b6cb0",
                fill_color="#2b6cb0",
                popup=popup_obj,
            ).add_to(m)

    # Add filename caption
    fname = os.path.basename(geojson_path)
    label_text = f"GeoJSON: {fname}"
    caption_html = (
        '<div style="position: fixed; top: 5px; left: 5px; z-index:9999; '
        'background-color: rgba(255,255,255,0.8); padding: 2px;">'
        f"{label_text}</div>"
    )
    m.get_root().html.add_child(folium.Element(caption_html))

    # Save alongside the GeoJSON
    base = os.path.splitext(os.path.basename(geojson_path))[0]
    suffix = "_heatmap" if mode == "heatmap" else "_points"
    out_html = os.path.join(os.path.dirname(geojson_path), f"{base}{suffix}.html")
    m.save(out_html)
    return out_html