
"""
visualize.py
Plotting utilities for collocation analysis.
- plot_bar: horizontal bar chart for collocate frequencies.
- plot_rank_changes: bump chart showing rank changes across time bins.

These functions are GUI-agnostic and can be called from PyQt handlers.
"""

from typing import Optional, Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
                     enable_hover: bool = True):
    """
    Build a bump chart of rank (1=top) vs time_bin for a subset of terms.

    If top_n and home_bin_index are provided, the set of terms displayed is taken
    from the top-N terms in the specified bin index (1-based).
    """
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
    for term in ordered_terms:
        series = pivot[term].to_numpy(dtype=float)
        mask = ~np.isnan(series)
        xs = positions[mask]
        ys = series[mask]
        line, = ax.plot(positions, series, marker='o', label=term)
        lines.append(line)
        scatter_points.append((term, xs, ys))
        if show_term_labels:
            for x_val, y_val in zip(xs, ys):
                ax.text(x_val, y_val, term, fontsize=8, ha='center', va='bottom')

    ax.set_xticks(positions)
    ax.set_xticklabels([str(b) for b in bins_ordered], rotation=45 if len(bins_ordered) > 6 else 0)
    ax.invert_yaxis()
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Ordinal Rank (1 = top)")
    ax.set_title("Collocate Rank Changes Over Time")
    ax.legend(lines, ordered_terms, title="Term", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if enable_hover and scatter_points:
        annot = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.8), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(term: str, x_val: float, y_val: float):
            annot.xy = (x_val, y_val)
            annot.set_text(f"{term}\nRank: {int(y_val)}")
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
