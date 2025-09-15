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
from dr_plotter.theme import BASE_THEME, FigureStyles, Theme

from datadec import DataDecide

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

FIXED_PARAMS = {
    "x": "step",
    "y": "value",
    "exterior_x_label": "Training Steps",
    "exterior_y_label": "Perplexity",
}
EXPECTED_DOMAIN_PARAMS = {"aggregate_seeds", "metrics"}


def prepare_data(kwargs: dict[str, Any]) -> pd.DataFrame:
    dd = DataDecide()
    return dd.prepare_plot_data(
        params=dd.select_params("all"),
        data=dd.select_data("all"),
        metrics=list(kwargs["metrics"]),
        aggregate_seeds=kwargs["aggregate_seeds"],
        auto_filter=True,
        melt=True,
    )


@click.command()
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
@dimensional_plotting_cli()
def main(**kwargs):
    click.echo("Loading DataDec data...")
    df, plot_config = execute_cli_workflow(
        kwargs,
        CLIWorkflowConfig(
            data_loader=prepare_data,
            fixed_params=FIXED_PARAMS,
            allow_unused=EXPECTED_DOMAIN_PARAMS,
            theme=DATADEC_THEME,
        ),
    )
    with FigureManager(plot_config) as fm:
        fm.plot(df, "line", linewidth=1.5)

    show_or_save_plot(
        fm.fig,
        kwargs.get("save_dir"),
        kwargs.get("pause"),
        "datadec_scaling",
    )
    click.echo("✅ DataDec plot completed!")


if __name__ == "__main__":
    main()
