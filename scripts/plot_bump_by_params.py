from __future__ import annotations

from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from dr_plotter import FigureManager
from dr_plotter.scripting import (
    CLIWorkflowConfig,
    dimensional_plotting_cli,
    execute_cli_workflow,
)
from dr_plotter.scripting.utils import show_or_save_plot

from datadec import DataDecide
from datadec.script_utils import (
    convert_domain_args_to_faceting,
    create_bump_theme_with_colors,
    format_perplexity,
    numerical_sort_key,
)

FIXED_PARAMS = {
    "x_axis": "params",
    "hue_by": "data",
    "final_step_only": True,
}
EXPECTED_DOMAIN_PARAMS = {"metric", "params", "data", "exclude_params", "exclude_data"}


def prepare_data(kwargs: dict[str, Any]) -> pd.DataFrame:
    metric = kwargs.get("metric", "pile-valppl")

    print("Bump Analysis by Params Configuration:")
    print(f"  Metric: {metric}")

    dd = DataDecide()

    # Use "all" as defaults - filtering handled by faceting system
    params_list = dd.select_params("all")
    data_list = dd.select_data("all")

    print(f"Using params: {params_list}")
    print(f"Using data: {data_list}")

    df = dd.prepare_plot_data(
        params=params_list,
        data=data_list,
        metrics=[metric],
        aggregate_seeds=True,
        auto_filter=True,
        melt=True,
    )

    print(f"Base data shape: {df.shape}")

    # Get final performance for each (params, data) combination
    final_rows = []
    for params_size in df["params"].unique():
        params_df = df[df["params"] == params_size]
        max_step_for_params = params_df["step"].max()
        final_step_data = params_df[params_df["step"] == max_step_for_params]
        final_rows.append(final_step_data)

    final_df = pd.concat(final_rows, ignore_index=True)
    print(f"Final step data shape: {final_df.shape}")

    # Create bump plot data
    bump_data = final_df.rename(
        columns={
            "params": "time",
            "data": "category",
            "value": "score",
        }
    )[["time", "category", "score"]]

    # Keep original perplexity values for labeling
    bump_data["original_ppl"] = bump_data["score"].copy()

    # Map params to numeric positions for plotting
    unique_params = sorted(bump_data["time"].unique(), key=numerical_sort_key)
    param_to_pos = {param: idx for idx, param in enumerate(unique_params)}
    bump_data["time"] = bump_data["time"].map(param_to_pos)

    # Invert scores for ranking (higher score = better rank)
    bump_data["score"] = -bump_data["score"]

    # Store metadata for plotting
    bump_data.attrs["original_param_names"] = unique_params
    bump_data.attrs["metric"] = metric

    print(f"Bump data shape: {bump_data.shape}")
    print(f"Categories: {len(bump_data['category'].unique())}")
    print(f"Time points: {len(bump_data['time'].unique())}")

    return bump_data


def add_ranking_labels(ax: plt.Axes, bump_data: pd.DataFrame) -> None:
    time_points = sorted(bump_data["time"].unique())

    if len(time_points) < 2:
        return

    first_time = time_points[0]
    last_time = time_points[-1]

    # Left side labels (first time point)
    first_data = bump_data[bump_data["time"] == first_time].copy()
    first_data = first_data.sort_values("score", ascending=False)
    first_data["rank"] = range(1, len(first_data) + 1)

    for _, row in first_data.iterrows():
        ax.text(
            -0.15,
            row["rank"],
            f"{row['rank']}. {row['category']}",
            transform=ax.transData,
            fontsize=9,
            ha="right",
            va="center",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightblue",
                "alpha": 0.7,
                "edgecolor": "navy",
            },
        )

    # Right side labels (last time point)
    last_data = bump_data[bump_data["time"] == last_time].copy()
    last_data = last_data.sort_values("score", ascending=False)
    last_data["rank"] = range(1, len(last_data) + 1)

    for _, row in last_data.iterrows():
        ax.text(
            len(time_points) - 1 + 0.15,
            row["rank"],
            f"{row['rank']}. {row['category']}",
            transform=ax.transData,
            fontsize=9,
            ha="left",
            va="center",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightgreen",
                "alpha": 0.7,
                "edgecolor": "darkgreen",
            },
        )


def add_value_annotations(ax: plt.Axes, bump_data: pd.DataFrame) -> None:
    ranked_data = []
    for time_point in bump_data["time"].unique():
        time_data = bump_data[bump_data["time"] == time_point].copy()
        time_data = time_data.sort_values("score", ascending=False)
        time_data["rank"] = range(1, len(time_data) + 1)
        ranked_data.append(time_data)

    all_ranked_data = pd.concat(ranked_data, ignore_index=True)

    for _, row in all_ranked_data.iterrows():
        ppl_text = format_perplexity(row["original_ppl"])
        ax.annotate(
            ppl_text,
            xy=(row["time"], row["rank"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
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


@click.command()
@click.option(
    "--metric", default="pile-valppl", help="Metric to plot for ranking comparison"
)
@click.option(
    "--params",
    multiple=True,
    default=["all"],
    help="Model parameter sizes (e.g., 150M 300M 1B) or 'all'",
)
@click.option(
    "--data", multiple=True, default=["all"], help="Data recipes or named groups"
)
@click.option(
    "--exclude-params",
    multiple=True,
    default=[],
    help="Model parameter sizes to exclude when using 'all'",
)
@click.option(
    "--exclude-data",
    multiple=True,
    default=[],
    help="Data recipes to exclude when using 'all'",
)
@dimensional_plotting_cli()
def main(**kwargs):
    """
    Bump plot showing recipe rankings across model parameter sizes (final step only).

    Replaces plot_bump.py functionality.

    Example:
    python scripts/plot_bump_by_params.py --metric pile-valppl --data base
    """
    # Convert domain args to faceting format
    faceting_params = convert_domain_args_to_faceting(kwargs)

    bump_data, plot_config = execute_cli_workflow(
        kwargs,
        CLIWorkflowConfig(
            data_loader=prepare_data,
            fixed_params={**FIXED_PARAMS, **faceting_params},
            allowed_unused=EXPECTED_DOMAIN_PARAMS,
        ),
    )

    # Create custom theme with extended colors
    num_categories = len(bump_data["category"].unique())
    custom_theme = create_bump_theme_with_colors(num_categories)

    # Override theme in plot config
    plot_config.style.theme = custom_theme

    # Create plot
    metric_str = bump_data.attrs["metric"].replace("_", " ").replace("-", " ").title()
    title = f"Recipe Rankings by Model Size ({metric_str})"

    with FigureManager(plot_config) as fm:
        fm.plot(
            "bump",
            0,
            0,
            bump_data,
            time_col="time",
            value_col="score",
            category_col="category",
            marker="o",
            linewidth=2,
            title=title,
        )

        # Add annotations
        ax = fm.get_axes(0, 0)
        add_ranking_labels(ax, bump_data)
        add_value_annotations(ax, bump_data)

        # Format x-axis with original param names
        original_param_names = bump_data.attrs["original_param_names"]
        ax.set_xticks(range(len(original_param_names)))
        ax.set_xticklabels(original_param_names)
        ax.set_xlabel("Model Size")

    show_or_save_plot(
        fm.fig,
        kwargs.get("save_dir"),
        kwargs.get("pause"),
        "bump_by_params",
    )
    click.echo("✅ Bump analysis by params completed!")


if __name__ == "__main__":
    main()
