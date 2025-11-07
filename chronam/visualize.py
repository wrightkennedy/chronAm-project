
"""
visualize.py
Plotting utilities for collocation analysis.
- plot_bar: horizontal bar chart for collocate frequencies.
- plot_rank_changes: bump chart showing rank changes across time bins.

These functions are GUI-agnostic and can be called from PyQt handlers.
"""

from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import matplotlib


_PLOT_BACKEND_INITIALIZED = False


def _ensure_pyplot():
    """Return matplotlib.pyplot with a Qt-friendly backend, falling back to Agg."""
    global _PLOT_BACKEND_INITIALIZED
    if not _PLOT_BACKEND_INITIALIZED:
        try:
            matplotlib.use("Qt5Agg", force=True)
        except Exception:
            try:
                matplotlib.use("Agg", force=True)
            except Exception:
                # If even Agg failed we still attempt to import pyplot; it may already be set up.
                pass
        _PLOT_BACKEND_INITIALIZED = True
    import matplotlib.pyplot as plt  # local import to avoid early backend init
    return plt


def _load_df(obj: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, str):
        if obj.lower().endswith(".csv"):
            return pd.read_csv(obj)
        if obj.lower().endswith(".json"):
            return pd.read_json(obj)
        raise ValueError("Unsupported file type. Provide .csv, .json, or a DataFrame.")
    raise ValueError("Provide a path or a pandas DataFrame.")


def plot_bar(collocation_results: Union[str, pd.DataFrame], output_path: Optional[str] = None, top_n: int = 20):
    """Show or save a bar chart of the top-N collocates by frequency."""
    plt = _ensure_pyplot()
    df = _load_df(collocation_results)
    if df.empty:
        raise ValueError("No data to plot.")
    if "collocate_term" not in df.columns or "frequency" not in df.columns:
        raise ValueError("DataFrame must contain 'collocate_term' and 'frequency'.")
    df = df.sort_values(["frequency","collocate_term"], ascending=[False, True]).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.barh(df["collocate_term"][::-1], df["frequency"][::-1])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Collocate Term")
    ax.set_title("Top Collocates")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return fig
    else:
        plt.show(block=False)
        return fig


def plot_rank_changes(df_or_path: Union[str, pd.DataFrame],
                     output_path: Optional[str] = None,
                     top_n: Optional[int] = None,
                     home_bin_index: Optional[int] = None,
                     legend_order: Optional[List[str]] = None,
                     show_term_labels: bool = False,
                     enable_hover: bool = True,
                     settings_text: Optional[str] = None,
                     use_log_scale: bool = False):
    """
    Build a bump chart of rank (1=top) vs time_bin for a subset of terms.

    When use_log_scale is True the y-axis uses a logarithmic scale (still inverted so
    rank 1 appears at the top of the chart).

    If top_n and home_bin_index are provided, the set of terms displayed is taken
    from the top-N terms in the specified bin index (1-based).
    """
    plt = _ensure_pyplot()
    df = _load_df(df_or_path)
    required = {"time_bin","collocate_term","ordinal_rank"}
    if not required.issubset(df.columns):
        raise ValueError("Data must contain columns: time_bin, collocate_term, ordinal_rank")

    # Order bins chronologically
    try:
        bins_ordered = sorted(df["time_bin"].unique(), key=lambda x: pd.to_datetime(str(x), errors="coerce"))
    except Exception:
        bins_ordered = list(df["time_bin"].unique())

    if top_n is not None and home_bin_index is not None and legend_order is None:
        hb = max(1, min(home_bin_index, len(bins_ordered)))
        home_label = bins_ordered[hb-1]
        subset = df[df["time_bin"] == home_label].sort_values("ordinal_rank").head(top_n)
        terms = subset["collocate_term"].unique().tolist()
        df = df[df["collocate_term"].isin(terms)]
        legend_order = terms

    # Pivot to wide for plotting
    pivot = df.pivot_table(index="time_bin", columns="collocate_term", values="ordinal_rank", aggfunc="min")
    pivot = pivot.reindex(bins_ordered)

    if legend_order:
        ordered_terms = [term for term in legend_order if term in pivot.columns]
    else:
        ordered_terms = list(pivot.columns)
    pivot = pivot[ordered_terms]

    fig, ax = plt.subplots(figsize=(14, 7))
    positions = np.arange(len(bins_ordered))
    lines = []
    scatter_points = []
    for idx, term in enumerate(ordered_terms, start=1):
        series = pivot[term].to_numpy(dtype=float)
        mask = ~np.isnan(series)
        xs = positions[mask]
        ys = series[mask]
        label_text = f"({idx}) {term}"
        line, = ax.plot(positions, series, marker='o', label=label_text)
        lines.append(line)
        scatter_points.append((term, xs, ys))
        if show_term_labels:
            for x_val, y_val in zip(xs, ys):
                ax.text(x_val, y_val, term, fontsize=8, ha='center', va='bottom')

    ax.set_xticks(positions)
    ax.set_xticklabels([str(b) for b in bins_ordered], rotation=45 if len(bins_ordered) > 6 else 0)
    if use_log_scale:
        ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Ordinal Rank (1 = top)")
    ax.set_title("Collocate Rank Changes Over Time")
    ax.legend(lines, ordered_terms, title="Term", bbox_to_anchor=(1.02, 1), loc="upper left")
    if settings_text:
        wrapped = '\n'.join(textwrap.fill(part, 110) for part in str(settings_text).splitlines())
        fig.text(0.5, 0.98, wrapped, ha='center', va='top', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
    else:
        plt.tight_layout()

    if enable_hover and scatter_points:
        annot = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.8), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(term: str, x_val: float, y_val: float):
            annot.xy = (x_val, y_val)
            year_idx = int(round(x_val))
            if 0 <= year_idx < len(bins_ordered):
                year_label = bins_ordered[year_idx]
            else:
                year_label = ''
            annot.set_text(f"{term}\nYear: {year_label}\nRank: {int(y_val)}")
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return
            tolerance = 0.25
            for term, xs, ys in scatter_points:
                if len(xs) == 0:
                    continue
                dist = np.hypot(xs - event.xdata, ys - event.ydata)
                idx = dist.argmin()
                if dist[idx] <= tolerance:
                    update_annot(term, xs[idx], ys[idx])
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return fig
    else:
        plt.show(block=False)
        return fig


def _load_topic_terms_from_path(path_str: str) -> Dict[int, str]:
    """Best-effort load of topic-term strings located next to a by-time CSV."""
    terms: Dict[int, str] = {}
    try:
        path = Path(path_str)
    except (TypeError, ValueError):
        return terms
    if not path.exists():
        return terms
    name = path.name
    candidates: List[Path] = []
    if "topics_by_time" in name:
        candidates.append(path.with_name(name.replace("topics_by_time", "topics", 1)))
    if name.startswith("topics_by_time_"):
        candidates.append(path.with_name(name.replace("topics_by_time_", "topics_", 1)))
    if not candidates:
        candidates.append(path.with_name(name.replace("_by_time", "")))
    seen: set = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_key = str(candidate.resolve())
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        try:
            df_topics = pd.read_csv(candidate)
        except Exception:
            continue
        required = {"topic_id", "top_terms"}
        if not required.issubset(df_topics.columns):
            continue
        for row in df_topics.itertuples(index=False):
            raw_topic_id = getattr(row, "topic_id", None)
            if pd.isna(raw_topic_id):
                continue
            try:
                topic_id = int(raw_topic_id)
            except Exception:
                continue
            raw_terms = getattr(row, "top_terms", "")
            if isinstance(raw_terms, str):
                terms[topic_id] = raw_terms
            elif isinstance(raw_terms, (list, tuple)):
                terms[topic_id] = ", ".join(str(part) for part in raw_terms)
            else:
                terms[topic_id] = str(raw_terms)
        if terms:
            break
    return terms


def plot_topics_over_time(
    df_or_path: Union[str, pd.DataFrame],
    output_path: Optional[str] = None,
    top_n: Optional[int] = 10,
    metric: str = "weight_sum",
    show_legend: bool = True,
    label_points: bool = False,
    log_scale: bool = False,
    enable_hover: bool = True,
    settings_text: Optional[str] = None,
):
    """Plot topic statistics across time bins."""
    plt = _ensure_pyplot()
    df = _load_df(df_or_path)
    if df.empty:
        raise ValueError("No topic data to plot.")

    def _adjust_canvas(
        fig,
        legend_obj,
        *,
        left: float,
        bottom: float,
        top: float,
        default_right: float = 0.98,
    ) -> None:
        """Shrink or expand margins so the legend stays inside the canvas."""
        if legend_obj is None:
            fig.subplots_adjust(left=left, right=default_right, bottom=bottom, top=top)
            return
        try:
            fig.canvas.draw()
        except Exception:
            pass
        renderer = getattr(fig.canvas, "get_renderer", lambda: None)()
        fig_bbox = fig.bbox
        if renderer is None or fig_bbox.width <= 0:
            fig.subplots_adjust(left=left, right=default_right, bottom=bottom, top=top)
            return
        legend_bbox = legend_obj.get_window_extent(renderer=renderer)
        fig_width = max(fig_bbox.width, 1.0)
        horizontal_pad = 0.02
        legend_width_frac = legend_bbox.width / fig_width
        reserved = min(legend_width_frac + horizontal_pad, 0.45)
        right = max(left + 0.3, 1 - reserved)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        try:
            fig.canvas.draw()
        except Exception:
            return
        renderer = getattr(fig.canvas, "get_renderer", lambda: None)()
        if renderer is None:
            return
        legend_bbox = legend_obj.get_window_extent(renderer=renderer)
        overflow = max(0.0, (legend_bbox.x1 - fig_bbox.x1) / fig_width)
        if overflow > 0:
            right = max(left + 0.25, right - overflow - horizontal_pad)
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    metric = (metric or "weight_sum").strip().lower()
    metric_map = {
        "weight_sum": {
            "column": "weight_sum",
            "label": "Topic Weight",
            "title": "Topic Weight over Time",
            "ascending": False,
            "invert": False,
        },
        "ordinal_rank": {
            "column": "ordinal_rank",
            "label": "Topic Rank (1 = top)",
            "title": "Topic Rank over Time",
            "ascending": True,
            "invert": True,
        },
        "doc_count": {
            "column": "doc_count",
            "label": "Article Count",
            "title": "Topic Article Counts over Time",
            "ascending": False,
            "invert": False,
        },
    }
    if metric not in metric_map:
        valid = ", ".join(metric_map.keys())
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {valid}")
    metric_info = metric_map[metric]
    metric_field = metric_info["column"]

    required = {"time_bin", "topic_id", metric_field}
    if not required.issubset(df.columns):
        raise ValueError(f"Data must contain columns: {', '.join(sorted(required))}")

    df = df.copy()
    df["topic_id"] = df["topic_id"].astype(int)
    if "topic_label" in df.columns:
        df["topic_label"] = df["topic_label"].fillna("").astype(str)
    else:
        df["topic_label"] = df["topic_id"].map(lambda tid: f"Topic {tid}")

    try:
        bins_ordered = sorted(
            df["time_bin"].dropna().unique(),
            key=lambda x: pd.to_datetime(str(x), errors="coerce"),
        )
    except Exception:
        bins_ordered = list(df["time_bin"].dropna().unique())
    if not bins_ordered:
       bins_ordered = list(df["time_bin"].unique())

    df = df[df["time_bin"].isin(bins_ordered)].copy()
    if df.empty:
        raise ValueError("Time bin data is empty after filtering.")

    def _shorten(text: str, limit: int) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    topic_terms_map: Dict[int, str] = {}
    if isinstance(df_or_path, str):
        topic_terms_map = _load_topic_terms_from_path(df_or_path)
    if "topic_terms" in df.columns:
        series_terms = df.dropna(subset=["topic_terms"]).groupby("topic_id")["topic_terms"].first()
        for tid, terms_val in series_terms.items():
            try:
                tid_int = int(tid)
            except Exception:
                continue
            topic_terms_map[tid_int] = str(terms_val)

    grouped = df.groupby(["topic_id", "topic_label"], as_index=False)[metric_field]
    if metric == "ordinal_rank":
        totals = grouped.mean().sort_values(metric_field, ascending=True)
    else:
        totals = grouped.sum().sort_values(metric_field, ascending=metric_info["ascending"])

    if top_n is not None and top_n > 0:
        top_topics = totals.head(top_n)["topic_id"].tolist()
    else:
        top_topics = totals["topic_id"].tolist()

    df = df[df["topic_id"].isin(top_topics)]
    if df.empty:
        raise ValueError("No topic data remains after filtering top topics.")

    label_map = df.groupby("topic_id")["topic_label"].first().to_dict()

    pivot = df.pivot_table(
        index="time_bin",
        columns="topic_id",
        values=metric_field,
        aggfunc="first",
    ).reindex(bins_ordered)
    if metric in ("weight_sum", "doc_count"):
        pivot = pivot.fillna(0.0)

    ordered_topics = [tid for tid in top_topics if tid in pivot.columns]
    if not ordered_topics:
        ordered_topics = [col for col in pivot.columns if col in totals["topic_id"].tolist()]
    if not ordered_topics:
        raise ValueError("No topics available to plot.")

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    positions = np.arange(len(bins_ordered))
    handles: List = []
    legend_labels: List[str] = []
    legend_topic_ids: List[int] = []
    scatter_points: List[Tuple[int, np.ndarray, np.ndarray, List[str]]] = []

    for topic_id in ordered_topics:
        series = pivot[topic_id].to_numpy(dtype=float)
        mask = ~np.isnan(series)
        if not mask.any():
            continue
        xs = positions[mask]
        ys = series[mask]
        bin_idx = np.flatnonzero(mask)
        bin_labels = [str(bins_ordered[idx]) for idx in bin_idx]
        line, = ax.plot(positions, series, marker="o")
        topic_label = label_map.get(topic_id) or f"Topic {topic_id}"
        legend_label = f"{topic_id}: {_shorten(topic_label, 20) or f'Topic {topic_id}'}"
        if show_legend:
            handles.append(line)
            legend_labels.append(legend_label)
            legend_topic_ids.append(topic_id)
        if enable_hover:
            scatter_points.append((topic_id, xs, ys, bin_labels))
        if label_points and xs.size:
            final_x = xs[-1]
            final_y = ys[-1]
            point_label = _shorten(topic_label, 12) or f"Topic {topic_id}"
            ax.text(final_x + 0.1, final_y, point_label, fontsize=8, ha="left", va="bottom")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(b) for b in bins_ordered], rotation=45 if len(bins_ordered) > 6 else 0)
    if log_scale:
        ax.set_yscale("log")
    if metric_info["invert"]:
        ax.invert_yaxis()
    ax.set_xlabel("Time Bin")
    ax.set_ylabel(metric_info["label"])
    ax.set_title(metric_info["title"])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    legend = None
    legend_text_entries: List[Tuple] = []
    legend_handle_entries: List[Tuple] = []
    legend_hover_display = None
    if show_legend and handles:
        legend_cols = 1
        if len(legend_labels) > 14:
            legend_cols = 2
        if len(legend_labels) > 28:
            legend_cols = 3
        legend = ax.legend(
            handles,
            legend_labels,
            title="Topic",
            bbox_to_anchor=(1, 1),
            loc="upper left",
            ncol=legend_cols,
            borderaxespad=0.6,
            columnspacing=0.9,
            handlelength=2.6,
        )
        legend_text_entries = list(zip(legend.get_texts(), legend_topic_ids))
        legend_handles_candidate = getattr(legend, "legendHandles", None)
        if legend_handles_candidate is None:
            legend_handles_candidate = getattr(legend, "legend_handles", None)
        legend_handles_list = list(legend_handles_candidate) if legend_handles_candidate is not None else []
        legend_handle_entries = list(zip(legend_handles_list, legend_topic_ids))
        legend_hover_display = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9, color="dimgray")

    layout_bottom = 0.16 if legend_hover_display is not None else 0.12
    layout_top = 0.94
    if settings_text:
        wrapped = "\n".join(textwrap.fill(str(part), 110) for part in str(settings_text).splitlines())
        fig.text(0.5, 0.97, wrapped, ha="center", va="top", fontsize=10)
        layout_top = 0.9
    _adjust_canvas(fig, legend, left=0.08, bottom=layout_bottom, top=layout_top)

    if enable_hover and scatter_points:
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.8),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)

        current_legend_topic = {"value": None}

        def _topic_terms(tid: int) -> str:
            base = topic_terms_map.get(tid) or label_map.get(tid) or f"Topic {tid}"
            return str(base)

        def _format_value(val: float) -> str:
            if metric == "ordinal_rank":
                return f"{int(round(val))}"
            if metric == "doc_count":
                return f"{int(round(val)):,}"
            return f"{val:,.3f}".rstrip("0").rstrip(".")

        def update_annot(topic_id: int, x_val: float, y_val: float, date_label: str):
            annot.xy = (x_val, y_val)
            terms_preview = _shorten(_topic_terms(topic_id), 20)
            annot.set_text(
                f"Topic {topic_id}: {terms_preview}\nTime: {date_label}\n{metric_info['label']}: {_format_value(y_val)}"
            )
            annot.get_bbox_patch().set_alpha(0.85)

        def update_legend_hover(topic_id: Optional[int]):
            if legend_hover_display is None:
                return
            if current_legend_topic["value"] == topic_id:
                return
            current_legend_topic["value"] = topic_id
            if topic_id is None:
                if legend_hover_display.get_text():
                    legend_hover_display.set_text("")
                    fig.canvas.draw_idle()
                return
            terms_full = _topic_terms(topic_id)
            legend_hover_display.set_text(textwrap.fill(f"Topic {topic_id}: {terms_full}", 110))
            fig.canvas.draw_idle()

        def hover(event):
            point_shown = False
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                tolerance = 0.35
                for topic_id, xs_vals, ys_vals, labels in scatter_points:
                    if len(xs_vals) == 0:
                        continue
                    distances = np.hypot(xs_vals - event.xdata, ys_vals - event.ydata)
                    idx = distances.argmin()
                    if distances[idx] <= tolerance:
                        update_annot(topic_id, xs_vals[idx], ys_vals[idx], labels[idx])
                        if not annot.get_visible():
                            annot.set_visible(True)
                        fig.canvas.draw_idle()
                        point_shown = True
                        update_legend_hover(None)
                        break
            if not point_shown and annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

            hovered_topic = None
            for text_obj, tid in legend_text_entries:
                contains, _ = text_obj.contains(event)
                if contains:
                    hovered_topic = tid
                    break
            if hovered_topic is None:
                for handle_obj, tid in legend_handle_entries:
                    contains, _ = handle_obj.contains(event)
                    if contains:
                        hovered_topic = tid
                        break
            update_legend_hover(hovered_topic)

        fig.canvas.mpl_connect("motion_notify_event", hover)

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return fig
    plt.show(block=False)
    return fig


def plot_articles_by_year(data: Union[str, pd.DataFrame, Dict[str, int]],
                          output_path: Optional[str] = None,
                          title: Optional[str] = None):
    """Plot a simple line chart of article counts by year."""
    plt = _ensure_pyplot()
    if isinstance(data, dict):
        items = sorted(
            [(int(year), int(count)) for year, count in data.items() if str(year).isdigit()],
            key=lambda pair: pair[0]
        )
        df = pd.DataFrame(items, columns=['year', 'article_count'])
    else:
        df = _load_df(data)
        if 'year' not in df.columns or 'article_count' not in df.columns:
            raise ValueError("Data must contain 'year' and 'article_count' columns.")
        df = df[['year', 'article_count']].copy()
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['article_count'] = pd.to_numeric(df['article_count'], errors='coerce')
        df = df.dropna(subset=['year', 'article_count'])
        df.sort_values('year', inplace=True)

    if df.empty:
        raise ValueError("No yearly data to plot.")

    fig, ax = plt.subplots()
    ax.plot(df['year'], df['article_count'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Articles')
    ax.set_title(title or 'Articles per Year')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    if len(df) > 12:
        ax.set_xticks(df['year'][:: max(1, len(df)//12)])
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return fig
    else:
        plt.show(block=False)
        return fig
