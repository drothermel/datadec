#!/usr/bin/env python3
"""
DataDec plotting using dr_plotter CLI framework.

This demonstrates the extension pattern for domain-specific applications
using the reusable CLI framework from dr_plotter.
"""

from __future__ import annotations

import click

from dr_plotter import FigureManager
from dr_plotter.scripting import (
    CLIConfig,
    dimensional_plotting_cli,
    validate_layout_options,
    build_faceting_config,
    build_plot_config,
    validate_dimensions_interactive,
)
from dr_plotter.scripting.utils import show_or_save_plot
from dr_plotter.theme import BASE_THEME, FigureStyles, Theme

from datadec import DataDecide

# 🎨 DATADEC THEME - Custom theme for scaling analysis
DATADEC_THEME = Theme(
    name="datadec_analysis",
    parent=BASE_THEME,
    figure_styles=FigureStyles(
        legend_position=(0.5, 0.02),
        multi_legend_positions=[(0.3, 0.02), (0.7, 0.02)],
        subplot_width=3.5,
        subplot_height=3.0,
        row_title_rotation=90,
        legend_frameon=True,
        suptitle_y=0.98,
        legend_tight_layout_rect=(0, 0.08, 1, 1),
    ),
)


@click.command()
@dimensional_plotting_cli(["params", "data", "metric", "seed"])
@click.option(
    "--aggregate-seeds",
    is_flag=True,
    help="Aggregate seeds to show mean values",
)
@click.option(
    "--metrics",
    multiple=True,
    default=["pile-valppl", "wikitext_103-valppl"],
    help="Metrics to plot (can specify multiple)",
)
def main(**kwargs):
    """Plot DataDec training curves with dimensional faceting."""

    # Load configuration
    config = CLIConfig()
    if kwargs.get("config"):
        try:
            config = CLIConfig.from_yaml(kwargs["config"])
            click.echo(f"✅ Loaded configuration from {kwargs['config']}")
        except Exception as e:
            click.echo(f"❌ Error loading config: {e}")
            return

    # Remove config file path from kwargs since we pass CLIConfig object separately
    cli_kwargs = {k: v for k, v in kwargs.items() if k != "config"}

    # Merge config with CLI args for validation
    merged_args = config.merge_with_cli_args(cli_kwargs)

    # Validate layout with merged arguments
    validate_layout_options(click.get_current_context(), **merged_args)

    # Load DataDec data
    click.echo("Loading DataDec data...")
    dd = DataDecide()

    df = dd.prepare_plot_data(
        params=dd.select_params("all"),
        data=dd.select_data("all"),
        metrics=list(kwargs["metrics"]),
        aggregate_seeds=kwargs["aggregate_seeds"],
        auto_filter=True,
        melt=True,
    )
    click.echo(f"Loaded {len(df)} data points")

    # Create faceting configuration using framework
    faceting_config = build_faceting_config(
        config,
        x="step",
        y="value",
        exterior_x_label="Training Steps",
        exterior_y_label="Perplexity",
        **cli_kwargs,
    )

    # Validate dimensional usage
    if not validate_dimensions_interactive(df, faceting_config):
        return

    # Create plot configuration using framework
    plot_config = build_plot_config(config, theme=DATADEC_THEME, **cli_kwargs)

    # Generate plot
    click.echo("Creating dimensional plot...")
    with FigureManager(plot_config) as fm:
        fm.plot_faceted(df, "line", faceting=faceting_config, linewidth=1.5)

    # Handle output
    class Args:
        save_dir = kwargs["save_dir"]
        pause = kwargs["pause"]

    show_or_save_plot(fm.fig, Args(), "datadec_scaling")
    click.echo("✅ DataDec plot completed!")


if __name__ == "__main__":
    main()
