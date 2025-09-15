from __future__ import annotations

from typing import Any

import click
import pandas as pd
from dr_plotter import FigureManager
from dr_plotter.scripting import (
    CLIWorkflowConfig,
    dimensional_plotting_cli,
    execute_cli_workflow,
)
from dr_plotter.scripting.utils import show_or_save_plot

from datadec import DataDecide

SCALE_ROW = ["lin-lin", "lin-log", "log-lin", "log-log"]
FIXED_PARAMS = {
    "x": "x_value",
    "rows_by": "x_type",
    "cols_by": "scale_type",
    "hue_by": "params",
    "xyscale": [SCALE_ROW for _ in range(2)],  # Same scale pattern for both rows
}
EXPECTED_DOMAIN_PARAMS = {"params", "data_source", "metric"}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    idx_min = df.groupby(["params", "data"])["tokens"].idxmin()
    idx_max = df.groupby(["params", "data"])["tokens"].idxmax()
    df_pd_min = (
        df.loc[idx_min]
        .reset_index(drop=True)
        .rename(
            columns={
                "tokens": "min_tokens",
                "value": "min_step_value",
                "step": "min_step",
            }
        )
    )
    df_pd_max = (
        df.loc[idx_max]
        .reset_index(drop=True)
        .rename(
            columns={
                "tokens": "max_tokens",
                "value": "max_step_value",
                "step": "max_step",
            }
        )
    )
    df = df.merge(df_pd_min, on=["params", "data"], how="left")
    df = df.merge(df_pd_max, on=["params", "data"], how="left")
    df["normed_value"] = df["value"] / df["min_step_value"]
    df["normed_centered_value"] = 1 - (df["value"] / df["min_step_value"])
    df["normed_x"] = df["tokens"] / df["max_tokens"]
    return df


def prepare_faceted_data(df: pd.DataFrame) -> pd.DataFrame:
    df_tokens = df.copy()
    df_tokens["x_type"] = "tokens"
    df_tokens["x_value"] = df_tokens["tokens"]
    df_tokens["x_label"] = "Tokens"
    df_normed = df.copy()
    df_normed["x_type"] = "% training"
    df_normed["x_value"] = df_normed["normed_x"]
    df_normed["x_label"] = "% of training tokens"
    faceted_df = pd.concat([df_tokens, df_normed], ignore_index=True)

    # Create DataFrame with scale types and cross join for all combinations
    scale_df = pd.DataFrame({"scale_type": SCALE_ROW})
    return pd.merge(faceted_df, scale_df, how="cross")


def prepare_data(kwargs: dict[str, Any]) -> pd.DataFrame:
    dd = DataDecide()
    df = dd.prepare_plot_data(
        params=dd.select_params(kwargs["params"]),
        data=dd.select_data(kwargs["data_source"]),
        metrics=[kwargs["metric"]],
        aggregate_seeds=True,
    )
    df = normalize_df(df)
    return prepare_faceted_data(df)


def apply_fixed_params(merged_args: dict[str, Any]) -> dict[str, Any]:
    for param, value in FIXED_PARAMS.items():
        assert param not in merged_args, (
            f"Param: {param} is fixed and cannot be overridden"
        )
        merged_args[param] = value
    return merged_args


@click.command()
@click.option(
    "--params",
    multiple=True,
    default=["20M", "60M", "90M", "530M"],
    help="Model parameters to include",
)
@click.option(
    "--data-source",
    default="Dolma1.7",
    help="Data source to analyze",
)
@click.option(
    "--metric",
    default="pile-valppl",
    help="Metric to plot",
)
@click.option(
    "--y",
    type=click.Choice(["normed_centered_value", "value", "normed_value"]),
    default="normed_centered_value",
    help="Y-axis value to plot",
)
@dimensional_plotting_cli(skip_fields={"y"})
def main(**kwargs):
    click.echo("Loading DataDec data...")
    df, plot_config = execute_cli_workflow(
        kwargs,
        CLIWorkflowConfig(
            data_loader=prepare_data,
            fixed_params=FIXED_PARAMS,
            allow_unused=EXPECTED_DOMAIN_PARAMS,
        ),
    )
    with FigureManager(plot_config) as fm:
        fm.plot(df, "line")
    show_or_save_plot(
        fm.fig,
        kwargs.get("save_dir"),
        kwargs.get("pause"),
        "rescale_training_curve_exp",
    )


if __name__ == "__main__":
    main()
