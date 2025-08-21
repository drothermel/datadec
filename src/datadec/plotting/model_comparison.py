"""
Builder for multi-metric model comparison plots.
"""

from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dr_plotter.figure import FigureManager

from .base import BasePlotBuilder, _sort_params_values


class ModelComparisonBuilder(BasePlotBuilder):
    """Builder for multi-metric model comparison plots."""

    def __init__(self, df: pd.DataFrame, metrics: List[str]):
        """
        Initialize the model comparison builder.

        Args:
            df: DataFrame containing the data to plot
            metrics: List of metric columns to plot
        """
        super().__init__(df)
        self.metrics = metrics

        # Model comparison specific defaults
        self.config.update(
            {
                "line_col": "params",  # Default: params as lines
                "style_col": None,  # Optional style encoding
                "subplot_col": "data",  # For filtering purposes
                "title_prefix": "Model Comparison",
                "params_filter": None,
                "data_filter": None,
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
        Convenience method to set data_filter and apply filtering.

        Args:
            data_list: List of data recipe values to keep

        Returns:
            self for method chaining
        """
        self.config["data_filter"] = data_list
        return self.filter_data(data_list)

    def build(self) -> Tuple[plt.Figure, FigureManager]:
        """
        Build the model comparison plot.

        Returns:
            Tuple of (matplotlib figure, FigureManager instance)
        """
        # Check if we're in "stacked" mode (multiple metrics with subplot structure)
        stacked_mode = self.config.get("stacked_subplots", False)

        if stacked_mode:
            # In stacked mode: calculate layout based on subplot_col values and metrics
            subplot_col = self.config.get("subplot_col", "data")
            subplot_values = self.plot_df[subplot_col].unique()

            # Sort subplot values based on column type and available filters
            if "params" in subplot_col:
                subplot_values = _sort_params_values(subplot_values)
            elif subplot_col == "data" and self.config.get("subplot_filter"):
                # Use the filter order if available
                filter_order = self.config["subplot_filter"]
                subplot_values = [val for val in filter_order if val in subplot_values]
            else:
                subplot_values = sorted(subplot_values)

            # Layout: metrics as rows, subplot_col values as columns
            n_metrics = len(self.metrics)
            ncols = len(subplot_values)  # One column per data/param value
            nrows = n_metrics  # One row per metric
        else:
            # Original mode: calculate layout based on metrics only
            n_metrics = len(self.metrics)
            ncols = self.config.get(
                "ncols", min(3, n_metrics)
            )  # Use config ncols or default
            nrows = int(np.ceil(n_metrics / ncols))

        # Calculate figure size
        figsize = self.config.get("figsize")
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        # Extract matplotlib kwargs for FigureManager
        matplotlib_kwargs = ["sharey", "sharex", "squeeze", "subplot_kw", "gridspec_kw"]
        fig_kwargs = {k: self.config[k] for k in matplotlib_kwargs if k in self.config}

        # Create figure manager with matplotlib kwargs
        self.fm = FigureManager(rows=nrows, cols=ncols, figsize=figsize, **fig_kwargs)

        # Set overall title
        title = (
            f"{self.config['title_prefix']}: Multiple Metrics vs {self.config['x_col']}"
        )
        self.fm.fig.suptitle(title, fontsize=16)

        if stacked_mode:
            # Stacked mode: plot each metric in its own row with multiple subplots per metric
            self._plot_stacked_mode(subplot_values, ncols, nrows)
        else:
            # Original mode: plot each metric in its own subplot
            self._plot_original_mode(ncols)

        # Hide unused subplots if needed
        if not stacked_mode:
            self._hide_unused_subplots(len(self.metrics), nrows * ncols)

        # Apply tight layout
        plt.tight_layout()

        return self.fm.fig, self.fm

    def _plot_original_mode(self, ncols):
        """Plot each metric in its own subplot (original behavior)."""
        for i, metric in enumerate(self.metrics):
            row = i // ncols
            col = i % ncols

            # Build list of columns we need for this metric
            key_columns = [self.config["x_col"], metric, self.config["line_col"]]
            if self.config.get("style_col"):
                key_columns.append(self.config["style_col"])

            # Only keep columns that exist in the dataframe
            available_columns = [c for c in key_columns if c in self.plot_df.columns]

            # Filter data for this metric (remove NaN values)
            metric_data = self.plot_df[available_columns].dropna()

            if len(metric_data) == 0:
                continue

            # Sort data for consistent ordering
            if self.config.get("style_col"):
                metric_data = self._sort_data_for_consistency(
                    metric_data, self.config["style_col"]
                )
            metric_data = self._sort_data_for_consistency(
                metric_data, self.config["line_col"]
            )

            # Apply sequential styling before plotting (only on first iteration to avoid resetting)
            if i == 0:  # Only set up cycles once for the entire figure
                self._apply_sequential_styling()

            # Use FigureManager's plot method for color coordination
            self.fm.plot(
                "line",
                row=row,
                col=col,
                data=metric_data,
                x=self.config["x_col"],
                y=metric,
                hue_by=self.config["line_col"],
                style_by=self.config.get("style_col"),
                title=metric,
                **self.config.get("plot_kwargs", {}),
            )

            # Apply log scale if configured
            if self.config.get("log_scale"):
                ax = self.fm.get_axes(row, col)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(f"{self.config['x_col']} (log scale)")
                ax.set_ylabel(f"{metric} (log scale)")

    def _plot_stacked_mode(self, subplot_values, ncols, nrows):
        """Plot metrics stacked: each metric gets its own row of subplots."""
        # Apply sequential styling once for the entire figure
        self._apply_sequential_styling()

        subplot_col = self.config.get("subplot_col", "data")

        # Plot each metric in its own row
        for metric_idx, metric in enumerate(self.metrics):
            # Plot each subplot value in its own column
            for subplot_idx, subplot_val in enumerate(subplot_values):
                row = metric_idx  # Row based on metric
                col = subplot_idx  # Column based on subplot value

                # Build list of columns we need for this metric
                key_columns = [self.config["x_col"], metric, self.config["line_col"]]
                if self.config.get("style_col"):
                    key_columns.append(self.config["style_col"])

                # Filter for this subplot value
                subset = self.plot_df[self.plot_df[subplot_col] == subplot_val]

                # Only keep columns that exist and remove NaN values
                available_columns = [c for c in key_columns if c in subset.columns]
                subset = subset[available_columns].dropna()

                if len(subset) == 0:
                    continue

                # Sort data for consistent ordering
                if self.config.get("style_col"):
                    subset = self._sort_data_for_consistency(
                        subset, self.config["style_col"]
                    )
                subset = self._sort_data_for_consistency(
                    subset, self.config["line_col"]
                )

                # Create subplot title (show subplot value for top row only)
                title = f"{subplot_col}={subplot_val}" if metric_idx == 0 else ""

                # Use FigureManager's plot method for color coordination
                self.fm.plot(
                    "line",
                    row=row,
                    col=col,
                    data=subset,
                    x=self.config["x_col"],
                    y=metric,
                    hue_by=self.config["line_col"],
                    style_by=self.config.get("style_col"),
                    title=title,
                    **self.config.get("plot_kwargs", {}),
                )

                # Remove individual subplot legends since we use unified legend
                ax = self.fm.get_axes(row, col)
                legend = ax.get_legend()
                if legend:
                    legend.remove()

                # Set ylabel for leftmost column only
                if col == 0:
                    ax.set_ylabel(
                        f"{metric} (log scale)"
                        if self.config.get("log_scale")
                        else metric
                    )

                # Apply log scale if configured
                if self.config.get("log_scale"):
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    if row == nrows - 1:  # Bottom row
                        ax.set_xlabel(f"{self.config['x_col']} (log scale)")

        # Share y-axis across each row (each metric gets its own shared y-axis)
        self._share_y_axis_per_row(nrows, ncols)

        # Share x-axis across each row if requested
        if self.config.get("sharex_per_row", False):
            self._share_x_axis_per_row(nrows, ncols)

    def _apply_sequential_styling(self):
        """Apply sequential color and line style styling."""
        # Use sequential colormap if specified
        if (
            self.config.get("colormap")
            or (self.config.get("two_color_start") and self.config.get("two_color_end"))
            or self.config.get("multi_color_sequence")
        ):
            # Get unique values for this line_col, preserving original filter order if available
            hue_values = self.plot_df[self.config["line_col"]].unique().tolist()

            # Preserve original filter order if this column has a filter applied
            if self.config["line_col"] == "params" and self.config.get("params_filter"):
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

            # Ensure _shared_style_cycles is properly initialized
            if self.fm._shared_style_cycles is None:
                self.fm._shared_style_cycles = {}

            # Replace the color cycle with our sequential colors
            self.fm._shared_style_cycles["color"] = itertools.cycle(color_values)

        # Use sequential line styles if specified and style_col is params
        if (
            self.config.get("linestyle_sequence")
            and self.config.get("style_col") == "params"
        ):
            # Get unique parameter values for line style mapping
            style_values = self.plot_df[self.config["style_col"]].unique().tolist()
            linestyle_map = self._generate_sequential_linestyles(
                style_values, linestyle_sequence=self.config.get("linestyle_sequence")
            )

            # Override the FigureManager's line style cycle
            import itertools
            from datadec.model_utils import param_to_numeric

            # Ensure shared style cycles are initialized
            if (
                not hasattr(self.fm, "_shared_style_cycles")
                or self.fm._shared_style_cycles is None
            ):
                self.fm._get_shared_style_cycles()
            if self.fm._shared_style_cycles is None:
                self.fm._shared_style_cycles = {}

            sorted_styles = sorted(style_values, key=param_to_numeric)
            linestyle_values = [linestyle_map[val] for val in sorted_styles]

            # Override both shared cycles and theme to ensure StyleEngine uses our sequence
            self.fm._shared_style_cycles["linestyle"] = itertools.cycle(
                linestyle_values
            )

            # Also override the BASE_THEME's linestyle_cycle that StyleEngine uses
            from dr_plotter.theme import BASE_THEME

            BASE_THEME.styles["linestyle_cycle"] = itertools.cycle(linestyle_values)

    def _share_y_axis_per_row(self, nrows: int, ncols: int):
        """Share y-axis across each row (each metric gets its own shared y-axis)."""
        for row in range(nrows):
            # Get all axes in this row
            row_axes = []
            for col in range(ncols):
                ax = self.fm.get_axes(row, col)
                if ax and ax.get_visible():
                    row_axes.append(ax)

            # Calculate combined y-axis range for all axes in this row
            if len(row_axes) > 1:
                # Collect all y-limits from axes in this row
                all_y_mins = []
                all_y_maxs = []

                for ax in row_axes:
                    y_min, y_max = ax.get_ylim()
                    all_y_mins.append(y_min)
                    all_y_maxs.append(y_max)

                # Calculate the combined range that encompasses all data
                combined_y_min = min(all_y_mins)
                combined_y_max = max(all_y_maxs)

                # Apply the combined range to all axes in this row
                ref_ax = row_axes[0]
                ref_ax.set_ylim(combined_y_min, combined_y_max)

                for ax in row_axes[1:]:
                    # Share y-axis with the reference axis
                    ax.sharey(ref_ax)
                    # Remove y-axis labels and ticks from non-leftmost subplots
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

    def _share_x_axis_per_row(self, nrows: int, ncols: int):
        """Share x-axis across each row (each metric gets its own shared x-axis range)."""
        for row in range(nrows):
            # Get all axes in this row
            row_axes = []
            for col in range(ncols):
                ax = self.fm.get_axes(row, col)
                if ax and ax.get_visible():
                    row_axes.append(ax)

            # Calculate combined x-axis range for all axes in this row
            if len(row_axes) > 1:
                # Collect all x-limits from axes in this row
                all_x_mins = []
                all_x_maxs = []

                for ax in row_axes:
                    x_min, x_max = ax.get_xlim()
                    all_x_mins.append(x_min)
                    all_x_maxs.append(x_max)

                # Calculate the combined range that encompasses all data
                combined_x_min = min(all_x_mins)
                combined_x_max = max(all_x_maxs)

                # Apply the combined range to all axes in this row
                for ax in row_axes:
                    ax.set_xlim(combined_x_min, combined_x_max)


# Convenience function for backward compatibility
def plot_model_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    x_col: str = "tokens",
    line_col: str = "params",
    style_col: Optional[str] = None,
    subplot_col: str = "data",
    params_filter: Optional[List] = None,
    subplot_filter: Optional[List] = None,
    figsize: Optional[Tuple[int, int]] = None,
    log_scale: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, FigureManager]:
    """
    Plot multiple metrics for model comparison, with each metric in its own subplot.

    This is a convenience function that maintains backward compatibility
    with the original API while using the new builder pattern internally.

    Args:
        df: DataFrame containing the data
        metrics: List of metric columns to plot
        x_col: Column for x-axis
        line_col: Column that creates different colored lines (hue)
        style_col: Optional column for line style encoding
        subplot_col: Column that determines subplot filtering when used
        params_filter: Filter to specific param values
        subplot_filter: Filter to specific values (e.g., data recipes)
        figsize: Figure size tuple
        log_scale: Use log/log scale for both x and y axes
        **kwargs: Additional arguments passed to dr_plotter

    Returns:
        Tuple of (figure, FigureManager)
    """
    # Use the builder pattern
    builder = ModelComparisonBuilder(df, metrics)

    # Configure the builder
    builder.configure(
        x_col=x_col,
        line_col=line_col,
        style_col=style_col,
        subplot_col=subplot_col,
        figsize=figsize,
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
