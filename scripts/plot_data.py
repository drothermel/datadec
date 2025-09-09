#!/usr/bin/env python3
"""
Simplified plot_data.py using enhanced FacetingConfig system.

This demonstrates the power of the new dimensional plotting framework.
"""

from __future__ import annotations

import argparse

from dr_plotter import FigureManager
from dr_plotter.configs import (
    FacetingConfig,
    LayoutConfig,
    LegendConfig,
    PlotConfig,
    StyleConfig,
)
from dr_plotter.scripting.utils import parse_key_value_args, show_or_save_plot
from dr_plotter.theme import BASE_THEME, FigureStyles, Theme

from datadec import DataDecide

VALID_DIMENSIONS = {"params", "data", "metric", "seed"}

# 🎨 CUSTOM THEME EXAMPLE - How to fine-tune visual settings
# Copy this pattern to customize detailed visual aspects
CUSTOM_THEME = Theme(
    name="scaling_analysis",
    parent=BASE_THEME,
    figure_styles=FigureStyles(
        # Custom legend positioning
        legend_position=(0.5, 0.02),  # Center, slightly higher
        multi_legend_positions=[
            (0.3, 0.02),
            (0.7, 0.02),
        ],  # Grouped legends spread wider
        # Custom subplot spacing
        subplot_width=3.5,  # Slightly smaller than default 4.0
        subplot_height=3.0,  # Compact height for overview plots
        # Row title styling
        row_title_rotation=90,  # Vertical labels (default)
        # row_title_rotation=45,  # Angled labels (uncomment to try)
        # row_title_rotation=0,   # Horizontal labels (uncomment to try)
        # Legend styling
        legend_frameon=True,  # Borders on legends
        # Layout spacing
        suptitle_y=0.98,  # Title position
        legend_tight_layout_rect=(0, 0.08, 1, 1),  # More space below for legends
    ),
    # You could also customize colors, line styles, etc. here
)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot training curves with enhanced dimensional control"
    )

    # 🎯 SIMPLIFIED FACETING - Support both rows AND cols
    parser.add_argument(
        "--rows",
        choices=list(VALID_DIMENSIONS),
        help="Dimension to use for row faceting",
    )
    parser.add_argument(
        "--cols",
        choices=list(VALID_DIMENSIONS),
        help="Dimension to use for column faceting",
    )
    parser.add_argument(
        "--rows-and-cols",
        choices=list(VALID_DIMENSIONS),
        help="Dimension to wrap across rows and columns",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=4,
        help="Maximum columns for wrapping layout (default: 4)",
    )

    # Visual channels
    parser.add_argument(
        "--hue-by",
        choices=list(VALID_DIMENSIONS),
        help="Dimension for color/line grouping",
    )
    parser.add_argument(
        "--alpha-by",
        choices=list(VALID_DIMENSIONS),
        help="Dimension for transparency grouping",
    )
    parser.add_argument(
        "--size-by",
        choices=list(VALID_DIMENSIONS),
        help="Dimension for size grouping",
    )

    # 🎯 DIMENSIONAL CONTROL - Much simpler than before!
    parser.add_argument(
        "--fixed",
        nargs="+",
        help="Fixed dimensions: key=value format (e.g., --fixed seed=0 metric=pile-valppl)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Exclude values: key=value1,value2 format (e.g., --exclude params=1B,2B)",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        help="Order values: key=value1,value2,... format (e.g., --order params=7B,30B,70B)",
    )

    # 🎯 AUTOMATIC BEHAVIORS
    parser.add_argument(
        "--subplot-width",
        type=float,
        default=4.0,
        help="Width of each subplot (default: 4.0)",
    )
    parser.add_argument(
        "--subplot-height",
        type=float,
        default=4.0,
        help="Height of each subplot (default: 4.0)",
    )
    parser.add_argument(
        "--no-auto-titles",
        action="store_false",
        dest="auto_titles",
        help="Disable automatic descriptive titles (default: enabled)",
    )
    parser.set_defaults(auto_titles=True)

    # Data control
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate seeds to show mean values",
    )

    # Legend
    parser.add_argument(
        "--legend-strategy",
        choices=["subplot", "figure", "grouped", "none"],
        default="subplot",
        help="Legend placement strategy (default: subplot)",
    )

    # Output
    parser.add_argument(
        "--save-dir",
        help="Directory to save plot",
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=5,
        help="Display duration in seconds",
    )

    return parser


def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()

    # Validate layout specification
    layout_options = [args.rows, args.cols, args.rows_and_cols]
    specified_layouts = [opt for opt in layout_options if opt is not None]
    if len(specified_layouts) == 0:
        parser.error("Must specify one of: --rows, --cols, or --rows-and-cols")
    if len(specified_layouts) > 1:
        parser.error("Specify only one layout option")

    # Parse dimensional control arguments
    fixed_dimensions = parse_key_value_args(args.fixed)
    exclude_dimensions = parse_key_value_args(args.exclude)
    ordered_dimensions = parse_key_value_args(args.order)

    # Load data
    dd = DataDecide()

    # 🎯 DRAMATIC SIMPLIFICATION - Let DataDecide handle preparation
    # Load all available data, filtering will be handled by FacetingConfig
    df = dd.prepare_plot_data(
        params=dd.select_params("all"),  # Get all available params
        data=dd.select_data("all"),  # Get all available data
        metrics=["pile-valppl", "wikitext_103-valppl"],  # Use your specified metrics
        aggregate_seeds=args.aggregate_seeds,
        auto_filter=True,
        melt=True,
    )

    # 🎯 CREATE FACETING CONFIG - This is where the magic happens!
    faceting_config = FacetingConfig(
        # Core plotting
        x="step",
        y="value",
        # Layout strategy
        rows=args.rows,
        cols=args.cols,
        rows_and_cols=args.rows_and_cols,
        max_cols=args.max_cols if args.rows_and_cols else None,
        # Visual channels
        hue_by=args.hue_by,
        alpha_by=args.alpha_by,
        size_by=args.size_by,
        # 🎯 DIMENSIONAL CONTROL - The new superpowers!
        fixed_dimensions=fixed_dimensions if fixed_dimensions else None,
        exclude_dimensions=exclude_dimensions if exclude_dimensions else None,
        ordered_dimensions=ordered_dimensions if ordered_dimensions else None,
        # 🎯 AUTOMATIC BEHAVIORS
        subplot_width=args.subplot_width,
        subplot_height=args.subplot_height,
        auto_titles=args.auto_titles,
        row_titles=args.auto_titles if args.rows else False,
        col_titles=args.auto_titles if args.cols else False,
        # Enhanced UX
        exterior_x_label="Training Steps",
        exterior_y_label="Perplexity",  # More meaningful than generic "Value"
    )

    # 🎯 MINIMAL SETUP - Most complexity handled automatically!
    plot_config = PlotConfig(
        layout=LayoutConfig(
            rows=1,
            cols=1,  # Will be auto-calculated by FacetingConfig
            tight_layout=True,
            tight_layout_pad=1.0,
        ),
        legend=LegendConfig(strategy=args.legend_strategy),
        style=StyleConfig(theme=CUSTOM_THEME),  # Pass custom theme through style config
    )

    # 🎯 SINGLE PLOTTING CALL - Everything else is automatic!
    with FigureManager(plot_config) as fm:
        fm.plot_faceted(df, "line", faceting=faceting_config, linewidth=1.5)

    show_or_save_plot(fm.fig, args, "plot_data_new")
    print("✅ Plot generated successfully!")


if __name__ == "__main__":
    main()
