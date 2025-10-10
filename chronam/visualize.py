
"""
visualize.py
Plotting utilities for collocation analysis.
- plot_bar: horizontal bar chart for collocate frequencies.
- plot_rank_changes: bump chart showing rank changes across time bins.

These functions are GUI-agnostic and can be called from PyQt handlers.
"""

from typing import Optional, Union, List, Dict
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

    fig, ax = plt.subplots()
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

    fig, ax = plt.subplots()
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
