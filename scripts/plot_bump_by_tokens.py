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
from matplotlib import ticker

from datadec import DataDecide
from datadec.script_utils import (
    convert_domain_args_to_faceting,
    create_bump_theme_with_colors,
    format_perplexity,
    format_step_label,
    format_token_count,
)

FIXED_PARAMS = {
    "x": "time",
    "y": "score",
    "hue_by": "category",
    "rows": 1,
    "cols": 1,
}
EXPECTED_DOMAIN_PARAMS = {
    "metric",
    "params",
    "data",
    "exclude_params",
    "exclude_data",
    "x_axis",
    "min_step",
    "max_step",
}


def prepare_data(kwargs: dict[str, Any]) -> pd.DataFrame:
    metric = kwargs.get("metric", "pile-valppl")
    x_axis = kwargs.get("x_axis", "tokens")
    min_step = kwargs.get("min_step")
    max_step = kwargs.get("max_step")

    dd = DataDecide()

    # Extract faceting selections - these will be automatically applied by the workflow
    # Use "all" as default, filtering handled by faceting system
    df = dd.prepare_plot_data(
        params=dd.select_params("all"),
        data=dd.select_data("all"),
        metrics=[metric],
        aggregate_seeds=True,
        auto_filter=True,
        melt=True,
    )

    if x_axis == "tokens":
        if "tokens" not in df.columns:
            token_info = dd.full_eval[
                ["params", "data", "step", "tokens"]
            ].drop_duplicates()
            df = df.merge(token_info, on=["params", "data", "step"], how="left")

    if min_step is not None:
        df = df[df["step"] >= min_step]
    if max_step is not None:
        df = df[df["step"] <= max_step]

    assert not df.empty, "No data found after applying step filters"

    df["param_data_combo"] = df["params"].astype(str) + "-" + df["data"].astype(str)
    x_col = "tokens" if x_axis == "tokens" else "step"
    min_times_per_combo = df.groupby("param_data_combo")[x_col].min()
    common_start_time = min_times_per_combo.max()
    df = df[df[x_col] >= common_start_time].copy()
    if x_axis == "tokens":
        time_col = "tokens"
        x_label = "Token Count"
    else:
        time_col = "step"
        x_label = "Training Steps"

    category_col = "param_data_combo"
    bump_data = df.rename(
        columns={
            time_col: "time",
            category_col: "category",
            "value": "score",
        }
    )[["time", "category", "score"]]
    bump_data["original_ppl"] = bump_data["score"].copy()
    bump_data["score"] = -bump_data["score"]
    bump_data.attrs["x_label"] = x_label
    bump_data.attrs["x_axis"] = x_axis
    bump_data.attrs["metric"] = metric
    return bump_data


def add_ranking_labels(ax: plt.Axes, bump_data: pd.DataFrame) -> None:
    time_points = sorted(bump_data["time"].unique())
    assert len(time_points) >= 2, "Not enough time points to add ranking labels"

    first_time = time_points[0]
    last_time = time_points[-1]
    first_data = bump_data[bump_data["time"] == first_time].copy()
    first_data = first_data.sort_values("score", ascending=False)
    first_data["rank"] = range(1, len(first_data) + 1)
    for _, row in first_data.iterrows():
        ax.text(
            -0.02,
            row["rank"],
            f"{row['rank']}. {row['category']}",
            transform=ax.get_yaxis_transform(),
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
    last_data = bump_data[bump_data["time"] == last_time].copy()
    last_data = last_data.sort_values("score", ascending=False)
    last_data["rank"] = range(1, len(last_data) + 1)
    for _, row in last_data.iterrows():
        ax.text(
            0.98,
            row["rank"],
            f"{row['rank']}. {row['category']}",
            transform=ax.get_yaxis_transform(),
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
@click.option(
    "--x-axis",
    type=click.Choice(["steps", "tokens"]),
    default="tokens",
    help="X-axis dimension: 'steps' for training steps, 'tokens' for token count",
)
@click.option("--min-step", type=float, help="Minimum training step to include")
@click.option("--max-step", type=float, help="Maximum training step to include")
@dimensional_plotting_cli()
def main(**kwargs):
    faceting_params = convert_domain_args_to_faceting(kwargs)
    bump_data, plot_config = execute_cli_workflow(
        kwargs,
        CLIWorkflowConfig(
            data_loader=prepare_data,
            fixed_params={**FIXED_PARAMS, **faceting_params},
            allowed_unused=EXPECTED_DOMAIN_PARAMS,
        ),
    )
    num_categories = len(bump_data["category"].unique())
    custom_theme = create_bump_theme_with_colors(num_categories)
    plot_config.style.theme = custom_theme
    metric_str = bump_data.attrs["metric"].replace("_", " ").replace("-", " ").title()
    x_label = bump_data.attrs["x_label"]
    title = f"Recipe Rankings by {x_label} ({metric_str})"

    with FigureManager(plot_config) as fm:
        fm.plot_faceted(
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
        add_ranking_labels(ax, bump_data)
        add_value_annotations(ax, bump_data)

        x_axis = bump_data.attrs["x_axis"]
        if x_axis == "tokens":
            ax.set_xscale("log")
            ax.set_xlabel("Token Count (log scale)")
            ax.set_xlim(2e8, 1.2e11)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: format_token_count(x))
            )
        else:
            x_values = sorted(bump_data["time"].unique())
            ax.set_xticks(x_values[:: max(1, len(x_values) // 8)])
            ax.set_xticklabels([format_step_label(s) for s in ax.get_xticks()])
            ax.set_xlabel("Training Steps")

    show_or_save_plot(
        fm.fig,
        kwargs.get("save_dir"),
        kwargs.get("pause"),
        f"bump_by_{x_axis}",
    )
    click.echo("✅ Bump analysis by tokens/steps completed!")


if __name__ == "__main__":
    main()
