from __future__ import annotations

import itertools
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
from dr_plotter import FigureManager, consts
from dr_plotter.configs import PlotConfig
from dr_plotter.scripting.utils import show_or_save_plot
from dr_plotter.theme import BUMP_PLOT_THEME, Theme

from datadec import DataDecide
from datadec.scripting.utils import format_perplexity

BUMP_COLS = ["time", "category", "score"]
MIN_TIME_POINTS = 2
EXTENDED_COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d3",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#5254a3",
    "#8ca252",
    "#bd9e39",
    "#ad494a",
    "#a55194",
    "#6b6ecf",
    "#b5cf6b",
    "#e7ba52",
    "#d6616b",
    "#ce6dbd",
    "#de9ed6",
    "#31a354",
    "#756bb1",
    "#636363",
    "#969696",
]


def prepare_datadecide_bump_data(
    kwargs: dict[str, Any],
    transformations: list[Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]],
    time_col: str,
    category_col: str,
    **attrs: Any,
) -> pd.DataFrame:
    metric = kwargs.get("metric", "pile-valppl")
    dd = DataDecide()
    step_lims = (kwargs.get("min_step"), kwargs.get("max_step"))
    df = dd.prepare_plot_data(
        params=dd.select_params("all"),
        data=dd.select_data("all"),
        metrics=[metric],
        aggregate_seeds=True,
        auto_filter=True,
        melt=False,
        step_lims=step_lims,
    )
    for transform_func in transformations:
        df = transform_func(df, kwargs)
    melted_df = dd.melt_for_plotting(df, metrics=[metric], include_seeds=False)
    bump_data = prepare_bump_ranking_data(melted_df, time_col, category_col, "value")
    bump_data.attrs["metric"] = metric
    for key, value in attrs.items():
        bump_data.attrs[key] = value
    return bump_data


def create_extended_color_palette() -> list[str]:
    return EXTENDED_COLOR_PALETTE


def create_bump_theme_with_colors(
    num_categories: int, theme_name: str = "bump_custom"
) -> Theme:
    extended_colors = create_extended_color_palette()
    colors_to_use = extended_colors[: max(num_categories, len(extended_colors))]
    return Theme(
        name=theme_name,
        parent=BUMP_PLOT_THEME,
        **{
            consts.get_cycle_key("hue"): itertools.cycle(colors_to_use),
        },
    )


def prepare_bump_ranking_data(
    df: pd.DataFrame,
    time_col: str,
    category_col: str,
    value_col: str,
    invert_score: bool = True,
) -> pd.DataFrame:
    bump_data = df.rename(
        columns={k: v for k, v in zip([time_col, category_col, value_col], BUMP_COLS)}
    )[BUMP_COLS]
    bump_data["original_value"] = bump_data["score"].copy()
    if invert_score:
        bump_data["score"] = -bump_data["score"]
    return bump_data


def _draw_labels(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_pos: float,
    ha: str,
    color: str,
    edge_color: str,
    transform: plt.transform,
    fontsize: int,
) -> None:
    for _, row in data.iterrows():
        ax.text(
            x_pos,
            row["rank"],
            f"{row['rank']}. {row['category']}",
            transform=transform,
            fontsize=fontsize,
            ha=ha,
            va="center",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": color,
                "alpha": 0.7,
                "edgecolor": edge_color,
            },
        )


def _get_ranked_data(bump_data: pd.DataFrame, time_val: float) -> pd.DataFrame:
    data = bump_data[bump_data["time"] == time_val].copy()
    data = data.sort_values("score", ascending=False)
    data["rank"] = range(1, len(data) + 1)
    return data


def add_bump_ranking_labels(
    ax: plt.Axes,
    bump_data: pd.DataFrame,
    left_color: str = "lightblue",
    right_color: str = "lightgreen",
    left_edge_color: str = "navy",
    right_edge_color: str = "darkgreen",
    fontsize: int = 9,
) -> None:
    time_points = sorted(bump_data["time"].unique())
    if len(time_points) < MIN_TIME_POINTS:
        return
    first_time, last_time = time_points[0], time_points[-1]
    first_data = _get_ranked_data(bump_data, first_time)
    last_data = _get_ranked_data(bump_data, last_time)
    if hasattr(ax, "get_yaxis_transform"):
        left_x, right_x = -0.02, 0.98
        left_transform = right_transform = ax.get_yaxis_transform()
    else:
        left_x, right_x = -0.15, len(time_points) - 1 + 0.15
        left_transform = right_transform = ax.transData
    _draw_labels(
        ax, first_data, left_x, "right", left_color, left_edge_color, left_transform
    )
    _draw_labels(
        ax, last_data, right_x, "left", right_color, right_edge_color, right_transform
    )


def add_bump_value_annotations(
    ax: plt.Axes,
    bump_data: pd.DataFrame,
    formatter_func: Any = format_perplexity,
    fontsize: int = 8,
    offset_x: int = 5,
    offset_y: int = 8,
) -> None:
    ranked_data = (
        bump_data.copy()
        .sort_values(["time", "score"], ascending=[True, False])
        .assign(
            rank=lambda df: df.groupby("time")["score"]
            .rank("first", ascending=False)
            .astype(int)
        )
    )
    for _, row in ranked_data.iterrows():
        value_text = formatter_func(row["original_value"])
        ax.annotate(
            value_text,
            xy=(row["time"], row["rank"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
            },
            arrowprops=None,
        )


def render_bump_plot(
    bump_data: pd.DataFrame,
    plot_config: PlotConfig,
    output_params: dict[str, Any],
    custom_axis_formatter: Any = None,
) -> None:
    num_categories = len(bump_data["category"].unique())
    custom_theme = create_bump_theme_with_colors(num_categories)
    plot_config.style.theme = custom_theme

    metric_str = bump_data.attrs["metric"].replace("_", " ").replace("-", " ").title()
    if "x_label" in bump_data.attrs:
        x_label = bump_data.attrs["x_label"]
        title = f"Recipe Rankings by {x_label} ({metric_str})"
    else:
        title = f"Recipe Rankings by Model Size ({metric_str})"

    with FigureManager(plot_config) as fm:
        fm.plot(
            bump_data,
            "bump",
            time_col="time",
            value_col="score",
            category_col="category",
            marker="o",
            linewidth=2,
            title=title,
        )
        ax = fm.get_axes(0, 0)
        add_bump_ranking_labels(ax, bump_data)
        add_bump_value_annotations(ax, bump_data)
        if custom_axis_formatter:
            custom_axis_formatter(ax, bump_data)

    show_or_save_plot(
        fm.fig,
        output_params.get("save_dir"),
        output_params.get("pause"),
        output_params.get("filename_prefix", "bump_plot"),
    )
