"""
Builder for scaling curve plots.
"""

from typing import Optional, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from dr_plotter.figure import FigureManager

from .base import BasePlotBuilder


class ScalingPlotBuilder(BasePlotBuilder):
    """Builder for scaling curve plots with flexible configuration."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the scaling plot builder.

        Args:
            df: DataFrame containing the data to plot
        """
        super().__init__(df)

        # Scaling-specific defaults
        self.config.update(
            {
                "y_col": "pile-valppl",
                "line_col": "params",  # What creates different colored lines
                "subplot_col": "data",  # What creates different subplots
                "style_col": None,  # Optional line style encoding
                "title_prefix": "Scaling Curves",
                "params_filter": None,  # Convenience storage
                "subplot_filter": None,  # Convenience storage
            }
        )

    def with_params(self, params_list: Optional[List[str]]):
        """
        Convenience method to set params_filter and apply filtering.

        Args:
            params_list: List of parameter values to keep

        Returns:
            self for method chaining
        """
        self.config["params_filter"] = params_list
        return self.filter_params(params_list)

    def with_data(self, data_list: Optional[List[str]]):
        """
        Convenience method to set subplot_filter and apply filtering.

        Args:
            data_list: List of data recipe values to keep

        Returns:
            self for method chaining
        """
        self.config["subplot_filter"] = data_list
        return self.filter_data(data_list)

    def build(self) -> Tuple[plt.Figure, FigureManager]:
        """
        Build the scaling curve plot.

        Returns:
            Tuple of (matplotlib figure, FigureManager instance)
        """
        # Calculate layout using base class method
        subplot_values, nrows, ncols, figsize, fig_kwargs = self._calculate_layout(
            self.config["subplot_col"]
        )

        # Create figure manager with matplotlib kwargs
        self.fm = FigureManager(rows=nrows, cols=ncols, figsize=figsize, **fig_kwargs)

        # Set overall title
        title = f"{self.config['title_prefix']}: {self.config['y_col']} vs {self.config['x_col']}"
        self.fm.fig.suptitle(title, fontsize=16)

        # FigureManager automatically coordinates colors across subplots

        # Plot each subplot
        for i, subplot_val in enumerate(subplot_values):
            row = i // ncols
            col = i % ncols

            # Filter data for this subplot
            subset = self.plot_df[
                self.plot_df[self.config["subplot_col"]] == subplot_val
            ]

            if len(subset) == 0:
                continue

            # Sort by params if needed for consistent legend ordering
            subset = self._sort_data_for_consistency(subset, self.config["line_col"])

            # Prepare plot kwargs
            plot_kwargs = self.config.get("plot_kwargs", {}).copy()

            # Use sequential colormap if specified
            if (
                self.config.get("colormap")
                or (
                    self.config.get("two_color_start")
                    and self.config.get("two_color_end")
                )
                or self.config.get("multi_color_sequence")
            ):
                # Get unique values for this subplot, preserving original filter order if available
                hue_values = subset[self.config["line_col"]].unique().tolist()

                # Preserve original filter order if this column has a filter applied
                if self.config["line_col"] == "params" and self.config.get(
                    "params_filter"
                ):
                    # For params as line_col, use params_filter order
                    original_order = self.config["params_filter"]
                    hue_values = [val for val in original_order if val in hue_values]
                elif self.config["line_col"] == "data" and self.config.get(
                    "subplot_filter"
                ):
                    # For data as line_col, use subplot_filter order (from with_data)
                    original_order = self.config["subplot_filter"]
                    hue_values = [val for val in original_order if val in hue_values]
                color_map = self._generate_sequential_colors(
                    hue_values,
                    self.config.get("colormap", "plasma"),
                    color_range_min=self.config.get("color_range_min", 0.1),
                    color_range_max=self.config.get("color_range_max", 1.0),
                    two_color_start=self.config.get("two_color_start"),
                    two_color_end=self.config.get("two_color_end"),
                    multi_color_sequence=self.config.get("multi_color_sequence"),
                    is_params=(self.config["line_col"] == "params"),
                )

                # Override the FigureManager's shared style cycles with our sequential colors
                import itertools
                from datadec.model_utils import param_to_numeric

                # Sort values to ensure proper progression (preserve order for non-params)
                if self.config["line_col"] == "params":
                    sorted_values = sorted(hue_values, key=param_to_numeric)
                else:
                    # For non-parameters, preserve the input order (intentional data recipe quality order)
                    sorted_values = hue_values
                color_values = [color_map[val] for val in sorted_values]

                # Override the shared style cycles that the StyleEngine uses
                if (
                    not hasattr(self.fm, "_shared_style_cycles")
                    or self.fm._shared_style_cycles is None
                ):
                    # Initialize if not already done
                    self.fm._get_shared_style_cycles()

                # Replace the color cycle with our sequential colors
                self.fm._shared_style_cycles["color"] = itertools.cycle(color_values)

            # Use sequential line styles if specified and style_col is params
            if (
                self.config.get("linestyle_sequence")
                and self.config.get("style_col") == "params"
            ):
                # Get unique parameter values for this subplot for line styles
                style_values = subset[self.config["style_col"]].unique().tolist()
                linestyle_map = self._generate_sequential_linestyles(
                    style_values,
                    linestyle_sequence=self.config.get("linestyle_sequence"),
                )

                # Override the FigureManager's line style cycle
                sorted_styles = sorted(style_values, key=param_to_numeric)
                linestyle_values = [linestyle_map[val] for val in sorted_styles]
                self.fm._shared_style_cycles["linestyle"] = itertools.cycle(
                    linestyle_values
                )

            # Use FigureManager's plot method for automatic color coordination
            self.fm.plot(
                "line",
                row=row,
                col=col,
                data=subset,
                x=self.config["x_col"],
                y=self.config["y_col"],
                hue_by=self.config["line_col"],
                style_by=self.config.get("style_col"),
                title=f"{self.config['subplot_col']}={subplot_val}",
                **plot_kwargs,
            )

            # Apply log scale if configured
            self._apply_log_scale(self.fm.get_axes(row, col))

        # Hide unused subplots
        self._hide_unused_subplots(len(subplot_values), nrows * ncols)

        # Apply tight layout
        plt.tight_layout()

        return self.fm.fig, self.fm


# Convenience function for backward compatibility
def plot_scaling_curves(
    df: pd.DataFrame,
    x_col: str = "tokens",
    y_col: str = "pile-valppl",
    line_col: str = "params",
    subplot_col: str = "data",
    style_col: Optional[str] = None,
    params_filter: Optional[List] = None,
    subplot_filter: Optional[List] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 2,
    title_prefix: str = "Scaling Curves",
    log_scale: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, FigureManager]:
    """
    Plot scaling curves with flexible configuration using the builder pattern.

    This is a convenience function that maintains backward compatibility
    with the original API while using the new builder pattern internally.

    Args:
        df: DataFrame containing the data
        x_col: Column for x-axis (typically "tokens")
        y_col: Column for y-axis (metric to plot)
        line_col: Column that creates different colored lines
        subplot_col: Column that creates different subplots
        style_col: Optional column for line style encoding
        params_filter: List of values to filter line_col by
        subplot_filter: List of values to filter subplot_col by
        figsize: Figure size tuple
        ncols: Number of columns in subplot grid
        title_prefix: Prefix for overall figure title
        log_scale: Use log/log scale for both x and y axes
        **kwargs: Additional arguments passed to dr_plotter

    Returns:
        Tuple of (figure, FigureManager)
    """
    # Use the builder pattern
    builder = ScalingPlotBuilder(df)

    # Configure the builder
    builder.configure(
        x_col=x_col,
        y_col=y_col,
        line_col=line_col,
        subplot_col=subplot_col,
        style_col=style_col,
        figsize=figsize,
        ncols=ncols,
        title_prefix=title_prefix,
        log_scale=log_scale,
        plot_kwargs=kwargs,
    )

    # Apply filters
    if params_filter is not None:
        builder.with_params(params_filter)
    if subplot_filter is not None:
        builder.with_data(subplot_filter)

    # Build and return
    return builder.build()
