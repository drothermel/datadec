#!/usr/bin/env python3
"""
Match across parameter lines analysis using dr_plotter CLI framework.

Analyzes model performance across different parameter sizes with multiple
scaling views (tokens vs % training) and scale combinations (lin/log).
Uses dimensional plotting to replace complex manual grid layouts.
"""

from __future__ import annotations

import click
import pandas as pd

from dr_plotter import FigureManager
from dr_plotter.scripting import (
    CLIConfig,
    dimensional_plotting_cli,
    validate_layout_options,
    build_faceting_config,
    build_plot_config,
)
from dr_plotter.scripting.utils import show_or_save_plot
from dr_plotter.theme import BASE_THEME, Theme, FigureStyles

from datadec import DataDecide


# Custom theme for parameter matching analysis with automatic scale handling
PARAM_MATCH_THEME = Theme(
    name="param_match_analysis",
    parent=BASE_THEME,
    figure_styles=FigureStyles(
        legend_position=(0.5, 0.02),
        subplot_width=4.0,
        subplot_height=3.5,
        legend_frameon=True,
        legend_tight_layout_rect=(0, 0.08, 1, 1),
    ),
)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data by minimum and maximum tokens for each (params, data) group."""
    # Get min and max values for normalization
    idx_min = df.groupby(["params", "data"])["tokens"].idxmin()
    idx_max = df.groupby(["params", "data"])["tokens"].idxmax()
    
    df_pd_min = (
        df.loc[idx_min]
        .reset_index(drop=True)
        .rename(columns={
            "tokens": "min_tokens",
            "value": "min_step_value",
            "step": "min_step",
        })
    )
    df_pd_max = (
        df.loc[idx_max]
        .reset_index(drop=True)
        .rename(columns={
            "tokens": "max_tokens",
            "value": "max_step_value", 
            "step": "max_step",
        })
    )
    
    df = df.merge(df_pd_min, on=["params", "data"], how="left")
    df = df.merge(df_pd_max, on=["params", "data"], how="left")
    
    # Create normalized values
    df["normed_value"] = df["value"] / df["min_step_value"]
    df["normed_centered_value"] = 1 - (df["value"] / df["min_step_value"])
    df["normed_x"] = df["tokens"] / df["max_tokens"]
    
    return df


def prepare_faceted_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape data for dimensional plotting with x_type and scale_type dimensions."""
    
    # Create two versions of the data - one for each x-axis type
    df_tokens = df.copy()
    df_tokens["x_type"] = "tokens"
    df_tokens["x_value"] = df_tokens["tokens"]
    df_tokens["x_label"] = "Tokens"
    
    df_normed = df.copy()
    df_normed["x_type"] = "% training" 
    df_normed["x_value"] = df_normed["normed_x"]
    df_normed["x_label"] = "% of training tokens"
    
    # Combine both x-axis types
    faceted_df = pd.concat([df_tokens, df_normed], ignore_index=True)
    
    # Create scale type combinations by duplicating data
    scale_combinations = [
        ("lin-lin", False, False),
        ("lin-log", False, True), 
        ("log-lin", True, False),
        ("log-log", True, True),
    ]
    
    final_dfs = []
    for scale_name, x_log, y_log in scale_combinations:
        scale_df = faceted_df.copy()
        scale_df["scale_type"] = scale_name
        scale_df["x_log"] = x_log
        scale_df["y_log"] = y_log
        final_dfs.append(scale_df)
    
    return pd.concat(final_dfs, ignore_index=True)


@click.command()
@dimensional_plotting_cli(["x_type", "scale_type", "params", "data", "metric"])
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
    "--y-column",
    type=click.Choice(["normed_centered_value", "value", "normed_value"]),
    default="normed_centered_value",
    help="Y-axis value to plot",
)
def main(**kwargs):
    """Generate parameter matching analysis with dimensional faceting.
    
    Creates a comprehensive view of model performance across different
    parameter sizes, scales, and normalizations using dr_plotter's faceting system.
    """
    
    # Load configuration
    config = CLIConfig()
    if kwargs.get("config"):
        try:
            config = CLIConfig.from_yaml(kwargs["config"])
            click.echo(f"✅ Loaded configuration from {kwargs['config']}")
        except Exception as e:
            click.echo(f"❌ Error loading config: {e}")
            return

    # Remove config file path from kwargs
    cli_kwargs = {k: v for k, v in kwargs.items() if k != "config"}
    
    # Merge config with CLI args for validation
    merged_args = config.merge_with_cli_args(cli_kwargs)
    
    # Validate layout with merged arguments  
    validate_layout_options(click.get_current_context(), **merged_args)

    # Load and prepare data
    click.echo("Loading DataDec data...")
    dd = DataDecide()
    
    params = list(kwargs["params"])
    data_source = kwargs["data_source"]
    metric = kwargs["metric"]
    
    df = dd.prepare_plot_data(
        params=dd.select_params(params),
        data=dd.select_data(data_source), 
        metrics=[metric],
        aggregate_seeds=True
    )
    
    click.echo(f"Normalizing data...")
    df = normalize_df(df)
    
    click.echo(f"Preparing faceted data...")
    faceted_df = prepare_faceted_data(df)
    
    click.echo(f"Generated {len(faceted_df)} data points for faceting")

    # Create faceting configuration - this replaces the entire manual grid!
    faceting_config = build_faceting_config(
        config,
        x="x_value",
        y=kwargs["y_column"],
        exterior_x_label="X-Axis (varies by scale)",
        exterior_y_label="Normalized Centered Perplexity",
        **cli_kwargs,
    )

    # Create plot configuration
    plot_config = build_plot_config(config, theme=PARAM_MATCH_THEME, **cli_kwargs)

    # Generate the entire analysis with a single plotting call!
    click.echo("Creating dimensional parameter matching analysis...")
    with FigureManager(plot_config) as fm:
        fm.plot_faceted(faceted_df, "line", faceting=faceting_config, linewidth=2)
        
        # Apply log scales based on scale_type data
        # This is the one piece we still need to do manually, but much cleaner
        for i, x_type in enumerate(["tokens", "% training"]):
            for j, (scale_name, x_log, y_log) in enumerate([
                ("lin-lin", False, False),
                ("lin-log", False, True),
                ("log-lin", True, False), 
                ("log-log", True, True),
            ]):
                ax = fm.get_axes(i, j)
                if x_log:
                    ax.set_xscale("log")
                if y_log:
                    ax.set_yscale("log")
        
        # Set figure title
        fm.fig.suptitle(f"{metric} Analysis: {data_source} ({', '.join(params)})", fontsize=16)

    # Handle output
    class Args:
        save_dir = kwargs["save_dir"]
        pause = kwargs["pause"]

    show_or_save_plot(fm.fig, Args(), "param_match_analysis")
    click.echo("✅ Parameter matching analysis completed!")


if __name__ == "__main__":
    main()