from __future__ import annotations

from typing import Any

import click
import pandas as pd
from dr_plotter.scripting import (
    CLIWorkflowConfig,
    dimensional_plotting_cli,
    execute_cli_workflow,
)

from datadec.scripting import (
    convert_domain_args_to_faceting,
    numerical_sort_key,
)
from datadec.scripting.bump_utils import (
    prepare_datadecide_bump_data,
    render_bump_plot,
)

FIXED_PARAMS = {
    "x_axis": "params",
    "hue_by": "data",
    "final_step_only": True,
}
EXPECTED_DOMAIN_PARAMS = {"metric", "params", "data", "exclude_params", "exclude_data"}


def extract_final_steps(df: pd.DataFrame, kwargs: dict[str, Any]) -> pd.DataFrame:
    final_rows = []
    for params_size in df["params"].unique():
        params_df = df[df["params"] == params_size]
        max_step_for_params = params_df["step"].max()
        final_step_data = params_df[params_df["step"] == max_step_for_params]
        final_rows.append(final_step_data)
    return pd.concat(final_rows, ignore_index=True)


def map_params_to_positions(
    bump_data: pd.DataFrame, kwargs: dict[str, Any]
) -> pd.DataFrame:
    unique_params = sorted(bump_data["time"].unique(), key=numerical_sort_key)
    param_to_pos = {param: idx for idx, param in enumerate(unique_params)}
    bump_data["time"] = bump_data["time"].map(param_to_pos)
    bump_data.attrs["original_param_names"] = unique_params
    return bump_data


def prepare_data(kwargs: dict[str, Any]) -> pd.DataFrame:
    bump_data = prepare_datadecide_bump_data(
        kwargs,
        transformations=[extract_final_steps],
        time_col="params",
        category_col="data",
    )
    return map_params_to_positions(bump_data, kwargs)


def format_params_axis(ax, bump_data):
    original_param_names = bump_data.attrs["original_param_names"]
    ax.set_xticks(range(len(original_param_names)))
    ax.set_xticklabels(original_param_names)
    ax.set_xlabel("Model Size")


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
    faceting_params = convert_domain_args_to_faceting(kwargs)
    bump_data, plot_config = execute_cli_workflow(
        kwargs,
        CLIWorkflowConfig(
            data_loader=prepare_data,
            fixed_params={**FIXED_PARAMS, **faceting_params},
            allowed_unused=EXPECTED_DOMAIN_PARAMS,
        ),
    )
    render_bump_plot(
        bump_data,
        plot_config,
        output_params={
            "save_dir": kwargs.get("save_dir"),
            "pause": kwargs.get("pause"),
            "filename_prefix": "bump_by_params",
        },
        custom_axis_formatter=format_params_axis,
    )


if __name__ == "__main__":
    main()
