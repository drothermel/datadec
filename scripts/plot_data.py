from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from dr_plotter import FigureManager
from dr_plotter.configs import PlotConfig, PositioningConfig

from datadec import DataDecide

VALID_DIMENSIONS = {"params", "data", "metrics", "seed"}


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot mean training curves with faceted layout for DataDecide eval"
    )

    # Faceting structure (mutually exclusive)
    facet_group = parser.add_mutually_exclusive_group(required=True)
    facet_group.add_argument(
        "--row",
        choices=["params", "data", "metrics", "seed"],
        help="Dimension to use for row faceting",
    )
    facet_group.add_argument(
        "--col",
        choices=["params", "data", "metrics", "seed"],
        help="Dimension to use for column faceting",
    )

    parser.add_argument(
        "--lines",
        choices=["params", "data", "metrics", "seed"],
        required=True,
        help="Dimension to use for line grouping within each subplot",
    )

    parser.add_argument(
        "--alpha",
        choices=["params", "data", "metrics", "seed"],
        help="Dimension to use for alpha channel (transparency) grouping",
    )

    # Value selection
    parser.add_argument(
        "--row_values",
        nargs="+",
        help="Values for row dimension (or 'all' for all available)",
    )
    parser.add_argument(
        "--col_values",
        nargs="+",
        help="Values for column dimension (or 'all' for all available)",
    )
    parser.add_argument(
        "--line_values",
        nargs="+",
        required=True,
        help="Values for line dimension (or 'all' for all available)",
    )
    parser.add_argument(
        "--alpha_values",
        nargs="+",
        help="Values for alpha dimension (or 'all' for all available)",
    )

    # Fixed dimensions (for dimensions not used in plotting)
    parser.add_argument(
        "--fixed-values",
        nargs="+",
        help="Fixed dimension values in key=value format (e.g., --fixed-values metrics=pile-valppl seed=0)",
    )

    # Aggregation control
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate seeds to show mean values instead of individual seeds",
    )

    # Data filtering (derived from dimensional arguments)
    parser.add_argument(
        "--exclude-params",
        nargs="+",
        default=[],
        help="Model parameter sizes to exclude when using 'all'",
    )
    parser.add_argument(
        "--exclude-data",
        nargs="+",
        default=[],
        help="Data recipes to exclude when using 'all'",
    )

    # Legend (reused from plot_seeds)
    parser.add_argument(
        "--legend",
        choices=["subplot", "grouped", "figure"],
        default="subplot",
        help="Legend strategy: subplot (per-axes), grouped (by-channel), figure",
    )

    # Output (reused from plot_seeds)
    parser.add_argument("--save", type=str, help="Save plot to file (specify path)")
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display plot interactively"
    )

    # Layout (reused from plot_seeds)
    parser.add_argument(
        "--figsize-per-subplot",
        type=float,
        default=4.0,
        help="Figure size per subplot (default: 4.0)",
    )
    parser.add_argument(
        "--no-sharex",
        action="store_true",
        help="Disable x-axis sharing across subplots",
    )
    parser.add_argument(
        "--no-sharey",
        action="store_true",
        help="Disable y-axis sharing across subplots",
    )

    # Axis configuration (reused from plot_seeds)
    parser.add_argument(
        "--xlog", action="store_true", help="Use logarithmic scale for x-axis"
    )
    parser.add_argument(
        "--ylog", action="store_true", help="Use logarithmic scale for y-axis"
    )

    # Axis limits (new)
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="X-axis limits (e.g., --xlim 0 1000)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Y-axis limits (e.g., --ylim 2.5 4.0)",
    )

    return parser


def process_metrics_input(metrics_input: list[str] | None) -> list[str]:
    metrics = []
    if not metrics_input:
        return metrics
    for item in metrics_input:
        metrics.extend([m.strip() for m in item.split(",")])
    return metrics


def parse_fixed_values(fixed_values: list[str] | None) -> dict[str, list[str]]:
    fixed_dict = {}
    if not fixed_values:
        return fixed_dict

    for item in fixed_values:
        if "=" not in item:
            raise ValueError(f"Fixed values must be in key=value format, got: {item}")

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        assert key in VALID_DIMENSIONS, (
            f"Invalid fixed dimension: {key}. Must be one of: {VALID_DIMENSIONS}"
        )
        if key not in fixed_dict:
            fixed_dict[key] = []
        fixed_dict[key].extend([v.strip() for v in value.split(",")])
    return fixed_dict


def resolve_dimension_values(
    dimension: str,
    values: list[str] | None,
    params: list[str] = [],
    data: list[str] = [],
    metrics: list[str] = [],
    seeds: list[str] = [],
) -> list[str]:
    dim_map = {
        "params": params,
        "data": data,
        "metrics": metrics,
        "seed": seeds,
    }

    if dimension not in dim_map:
        raise ValueError(f"Unknown dimension: {dimension}")

    if values is None or (len(values) == 1 and values[0] == "all"):
        return dim_map[dimension]
    return values


# TODO: Refactor this function - it's overly complex (86 statements, 28 branches)
# Consider breaking into smaller functions for data preparation, plotting, and format
def plot_means(  # noqa: C901, PLR0912, PLR0915
    row: str | None = None,
    col: str | None = None,
    lines: str | None = None,
    alpha: str | None = None,
    row_values: list[str] | None = None,
    col_values: list[str] | None = None,
    line_values: list[str] | None = None,
    alpha_values: list[str] | None = None,
    fixed_values: list[str] | None = None,
    aggregate_seeds: bool = True,
    exclude_params: list[str] | None = None,
    exclude_data: list[str] | None = None,
    legend_strategy: str = "subplot",
    save_path: str | None = None,
    show_plot: bool = True,
    figsize_per_subplot: float = 4.0,
    sharex: bool = True,
    sharey: bool = True,
    xlog: bool = False,
    ylog: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    dd = DataDecide()

    exclude_params = exclude_params or []
    exclude_data = exclude_data or []

    # Parse fixed values from key=value format
    fixed_dict = parse_fixed_values(fixed_values)

    # Map dimension names to DataFrame column names
    dim_to_col = {
        "params": "params",
        "data": "data",
        "metrics": "metric",
        "seed": "seed",
    }

    # Determine which dimensions we need and collect all required values
    facet_dim = row if row else col
    dimensions_used = {facet_dim, lines}
    if alpha:
        dimensions_used.add(alpha)
    if fixed_dict:
        dimensions_used.update(fixed_dict.keys())

    # Validate that all 4 dimensions are accounted for
    unaccounted = VALID_DIMENSIONS - dimensions_used
    if unaccounted:
        raise ValueError(
            f"All dimensions must be assigned to row/col/lines/alpha or fixed. Missing: {unaccounted}"
        )

    # Collect all values needed for each dimension
    all_params = []
    all_data = []
    all_metrics = []
    all_seeds = []

    # Handle explicit dimension assignments
    for dim in ["params", "data", "metrics", "seed"]:
        if dim == facet_dim:
            values = row_values if row else col_values
        elif dim == lines:
            values = line_values
        elif dim == alpha:
            values = alpha_values
        elif dim in fixed_dict:
            values = fixed_dict[dim]
        else:
            values = None

        if dim == "params" and values:
            all_params = resolve_dimension_values("params", values)
        elif dim == "data" and values:
            all_data = resolve_dimension_values("data", values)
        elif dim == "metrics" and values:
            processed_values = process_metrics_input(values)
            all_metrics = resolve_dimension_values("metrics", processed_values)
        elif dim == "seed" and values:
            all_seeds = resolve_dimension_values("seed", values)

    # Use "all" for dimensions not explicitly specified
    if not all_params:
        all_params = dd.select_params("all", exclude=exclude_params)
    if not all_data:
        all_data = dd.select_data("all", exclude=exclude_data)
    if not all_metrics:
        all_metrics = []
    if not all_seeds:
        all_seeds = []

    # Resolve final dimension values for plotting
    facet_values = resolve_dimension_values(
        dimension=facet_dim,
        values=row_values if row else col_values,
        params=all_params,
        data=all_data,
        metrics=all_metrics,
        seeds=all_seeds,
    )
    line_values_resolved = resolve_dimension_values(
        lines,
        line_values,
        params=all_params,
        data=all_data,
        metrics=all_metrics,
        seeds=all_seeds,
    )
    df = dd.prepare_plot_data(
        params=all_params,
        data=all_data,
        metrics=all_metrics,
        aggregate_seeds=aggregate_seeds,
        auto_filter=True,
        melt=True,
    )
    nfacets = len(facet_values)
    if row:
        nrows, ncols = nfacets, 1
        figsize = (figsize_per_subplot * ncols, figsize_per_subplot * nrows)
    else:
        nrows, ncols = 1, nfacets
        figsize = (figsize_per_subplot * ncols, figsize_per_subplot * nrows)
    if legend_strategy == "figure":
        tight_layout_rect = (0.01, 0.15, 0.99, 0.97)
        positioning_config = PositioningConfig(legend_y_offset_factor=0.02)
        legend_config = {
            "strategy": legend_strategy,
            "position": "lower center",
            "channel_titles": {lines: lines.title()},
            "positioning_config": positioning_config,
        }
    else:
        tight_layout_rect = (0.01, 0.01, 0.99, 0.97)
        legend_config = {
            "strategy": legend_strategy,
            "position": "best",
            "channel_titles": {lines: lines.title()},
        }
    fixed_parts = []
    for dim, values in fixed_dict.items():
        values_str = ", ".join(values)
        fixed_parts.append(f"{dim.title()}: {values_str}")
    title_parts = [
        f"({'; '.join(fixed_parts)})" if fixed_parts else "",
        f"{facet_dim.title()} x {lines.title()}",
    ]
    title_parts = [part for part in title_parts if part]  # Remove empty parts
    layout_config = {
        "rows": nrows,
        "cols": ncols,
        "figsize": figsize,
        "tight_layout_pad": 0.5,
        "tight_layout_rect": tight_layout_rect,
        "subplot_kwargs": {"sharex": sharex, "sharey": sharey},
        "figure_title": f"{' '.join(title_parts)}",
    }
    if xlog:
        layout_config["xscale"] = "log"
    if ylog:
        layout_config["yscale"] = "log"
    with FigureManager(
        PlotConfig(
            layout=layout_config,
            legend=legend_config,
            kwargs={"suptitle_y": 0.98},
        )
    ) as fm:
        plot_kwargs = {
            "data": df,
            "plot_type": "line",
            "rows": dim_to_col[facet_dim] if row else None,
            "cols": dim_to_col[facet_dim] if col else None,
            "lines": dim_to_col[lines],
            "x": "step",
            "y": "value",
            "linewidth": 1.5,
            "marker": None,
            "row_order": facet_values if row else None,
            "col_order": facet_values if col else None,
            "lines_order": line_values_resolved,
            "row_titles": bool(row),
            "col_titles": bool(col),
            "exterior_x_label": "Training Steps",
        }

        if alpha:
            plot_kwargs["alpha_by"] = dim_to_col[alpha]
        else:
            plot_kwargs["alpha"] = 0.8

        fm.plot_faceted(**plot_kwargs)

        # Apply axis limits if specified
        if xlim or ylim:
            for facet_idx in range(nfacets):
                ax = fm.get_axes(facet_idx, 0) if row else fm.get_axes(0, facet_idx)
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()

    if not show_plot and not save_path:
        print("Warning: Plot not saved or displayed. Use --save or remove --no-show")


def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()

    show_plot = not args.no_show
    sharex = not args.no_sharex
    sharey = not args.no_sharey
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None

    plot_means(
        row=args.row,
        col=args.col,
        lines=args.lines,
        alpha=args.alpha,
        row_values=args.row_values,
        col_values=args.col_values,
        line_values=args.line_values,
        alpha_values=args.alpha_values,
        fixed_values=getattr(args, "fixed_values", None),
        aggregate_seeds=args.aggregate_seeds,
        exclude_params=args.exclude_params,
        exclude_data=args.exclude_data,
        legend_strategy=args.legend,
        save_path=args.save,
        show_plot=show_plot,
        figsize_per_subplot=args.figsize_per_subplot,
        sharex=sharex,
        sharey=sharey,
        xlog=args.xlog,
        ylog=args.ylog,
        xlim=xlim,
        ylim=ylim,
    )


if __name__ == "__main__":
    main()
