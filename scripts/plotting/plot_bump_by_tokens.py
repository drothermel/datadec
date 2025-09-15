from __future__ import annotations

from typing import Any

import click
import pandas as pd
from dr_plotter.scripting import (
    CLIWorkflowConfig,
    dimensional_plotting_cli,
    execute_cli_workflow,
)
from matplotlib import ticker

from datadec.scripting import (
    align_to_common_start_point,
    convert_domain_args_to_faceting,
    format_step_label,
    format_token_count,
)
from datadec.scripting.bump_utils import (
    prepare_datadecide_bump_data,
    render_bump_plot,
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
    x_axis = kwargs.get("x_axis", "tokens")
    time_col = "tokens" if x_axis == "tokens" else "step"
    x_label = "Token Count" if x_axis == "tokens" else "Training Steps"
    return prepare_datadecide_bump_data(
        kwargs,
        transformations=[align_to_common_start_point],
        time_col=time_col,
        category_col="param_data_combo",
        x_label=x_label,
        x_axis=x_axis,
    )


def format_tokens_axis(ax, bump_data):
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
    x_axis = bump_data.attrs["x_axis"]
    render_bump_plot(
        bump_data,
        plot_config,
        output_params={
            "save_dir": kwargs.get("save_dir"),
            "pause": kwargs.get("pause"),
            "filename_prefix": f"bump_by_{x_axis}",
        },
        custom_axis_formatter=format_tokens_axis,
    )


if __name__ == "__main__":
    main()
