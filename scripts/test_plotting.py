#!/usr/bin/env python3
"""
Test script for datadec plotting functionality.
Creates various configurations of scaling curve plots to validate the implementation.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add src to path so we can import datadec
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from datadec import DataDecide
from datadec.plotting import ScalingPlotBuilder, ModelComparisonBuilder
from datadec.model_utils import param_to_numeric


def fix_sharey_labels(builder, fm):
    """
    Fix y-axis labels for shared y-axis plots by removing redundant labels.

    Args:
        builder: The plot builder instance (to access config)
        fm: The FigureManager instance (to access axes)
    """
    if builder.config.get("sharey"):
        ncols = builder.config["ncols"]
        nrows = 1  # Assuming single row layouts for now

        for row in range(nrows):
            for col in range(1, ncols):  # Skip leftmost (col=0)
                ax = fm.get_axes(row=row, col=col)
                if ax and ax.get_visible():
                    ax.set_ylabel("")


def add_unified_legend_below(builder, fm, adjust_layout=True, legend_ncol=None):
    """
    Create a unified legend below all subplots, removing individual subplot legends.

    Args:
        builder: The plot builder instance (to access config)
        fm: The FigureManager instance (to access figure and axes)
        adjust_layout: Whether to adjust subplot layout to make room for legend
        legend_ncol: Number of columns in legend. If None, uses len(all_labels) for single row
    """
    ncols = builder.config.get("ncols", 1)
    nrows = 1  # Assuming single row layouts for now

    # Collect all unique legend handles and labels from subplots
    handle_map = {}  # Map from label to handle

    for row in range(nrows):
        for col in range(ncols):
            ax = fm.get_axes(row=row, col=col)
            if ax and ax.get_visible():
                # Get legend handles and labels from this subplot
                handles, labels = ax.get_legend_handles_labels()

                # Store unique labels with their handles
                for handle, label in zip(handles, labels):
                    if label not in handle_map:
                        handle_map[label] = handle

                # Remove individual subplot legend
                legend = ax.get_legend()
                if legend:
                    legend.remove()

    # Order the labels according to the builder's configuration
    line_col = builder.config.get("line_col")
    all_labels = []
    all_handles = []

    # Determine the ordering based on which column is the line_col and available filters
    if line_col == "params" and builder.config.get("params_filter"):
        # Use params_filter order
        ordered_values = builder.config["params_filter"]
    elif line_col == "data" and builder.config.get("subplot_filter"):
        # Use subplot_filter order (from with_data method)
        ordered_values = builder.config["subplot_filter"]
    else:
        # Fallback to the order they appear in handle_map
        ordered_values = list(handle_map.keys())

    # Build the ordered lists based on the configuration order
    for value in ordered_values:
        if value in handle_map:
            all_labels.append(value)
            all_handles.append(handle_map[value])

    # Create unified legend below subplots if we have any labels
    if all_handles and all_labels:
        # Determine number of columns for legend
        if legend_ncol is None:
            # Default: single row (all labels in one row)
            legend_cols = len(all_labels)
        else:
            # Use specified number of columns
            legend_cols = min(legend_ncol, len(all_labels))

        # Adjust bottom margin based on number of legend rows needed
        if adjust_layout:
            legend_rows = (
                len(all_labels) + legend_cols - 1
            ) // legend_cols  # Ceiling division
            bottom_margin = (
                0.1 + (legend_rows - 1) * 0.05
            )  # More space for multi-row legends
            plt.subplots_adjust(bottom=bottom_margin)

        # Create figure-level legend below all subplots
        fm.fig.legend(
            all_handles,
            all_labels,
            loc="upper center",  # Legend anchor point
            bbox_to_anchor=(0.5, 0.02),  # Position below subplots
            ncol=legend_cols,  # Configurable columns
            frameon=True,  # Show legend frame
            fancybox=True,  # Rounded corners
            shadow=True,  # Drop shadow
        )


def add_grouped_legends_below(
    builder,
    fm,
    adjust_layout=True,
    line_col_ncol=None,
    style_col_ncol=None,
    legend_spacing=0.1,
    legend_y_pos=0.02,
    bottom_margin=None,
):
    """
    Create two separate legends below subplots - one for colors (line_col)
    and one for line styles (style_col), positioned side by side.

    Args:
        builder: The plot builder instance (to access config)
        fm: The FigureManager instance (to access figure and axes)
        adjust_layout: Whether to adjust subplot layout to make room for legends
        line_col_ncol: Number of columns for line_col legend. If None, uses single row
        style_col_ncol: Number of columns for style_col legend. If None, uses single row
        legend_spacing: Horizontal spacing between legend edges (0.0-1.0)
        legend_y_pos: Vertical position of legends (0.0=bottom, 1.0=top)
        bottom_margin: Manual bottom margin override. If None, calculates automatically
    """
    ncols = builder.config.get("ncols", 1)
    nrows = 1  # Assuming single row layouts for now

    # Get column mappings from builder config
    line_col = builder.config.get("line_col", "params")  # Controls colors
    style_col = builder.config.get("style_col", "data")  # Controls line styles

    # Collect all handles and labels from subplots to analyze
    all_handles = []
    all_labels = []

    for row in range(nrows):
        for col in range(ncols):
            ax = fm.get_axes(row=row, col=col)
            if ax and ax.get_visible():
                # Get legend handles and labels from this subplot
                handles, labels = ax.get_legend_handles_labels()
                all_handles.extend(handles)
                all_labels.extend(labels)

                # Remove individual subplot legend
                legend = ax.get_legend()
                if legend:
                    legend.remove()

    if not all_handles or not all_labels:
        return

    # Extract unique values for each grouping
    # Parse labels to extract the grouping values
    # Labels are typically in format like "data_value, params_value"
    line_col_values = set()
    style_col_values = set()
    handle_map = {}  # Map (line_col_val, style_col_val) -> handle

    for handle, label in zip(all_handles, all_labels):
        # Parse label to extract values and remove key prefixes
        # Labels from dr_plotter are typically like "data=DCLM-Baseline, params=10M"
        parts = [part.strip() for part in label.split(",")]
        if len(parts) == 2:
            # Extract values by removing the key= prefix
            part1_clean = parts[0].split("=")[-1] if "=" in parts[0] else parts[0]
            part2_clean = parts[1].split("=")[-1] if "=" in parts[1] else parts[1]

            if line_col == "data":
                line_col_val, style_col_val = part1_clean, part2_clean
            else:  # line_col == "params"
                style_col_val, line_col_val = part1_clean, part2_clean

            line_col_values.add(line_col_val)
            style_col_values.add(style_col_val)
            handle_map[(line_col_val, style_col_val)] = handle

    # Convert to lists for consistent ordering
    sorted_line_col_values = list(line_col_values)
    sorted_style_col_values = list(style_col_values)

    # Create custom Line2D elements for each group
    line_col_handles = []
    line_col_labels = []

    # For line_col group (colors), use first available style_col to get color
    first_style_val = sorted_style_col_values[0] if sorted_style_col_values else ""
    for line_val in sorted_line_col_values:
        # Find a handle that has this line_col value to extract its color
        sample_handle = None
        for (l_val, s_val), handle in handle_map.items():
            if l_val == line_val:
                sample_handle = handle
                break

        if sample_handle:
            # Create Line2D with same color, but standard line style
            custom_handle = Line2D(
                [0], [0], color=sample_handle.get_color(), linewidth=2, linestyle="-"
            )
            line_col_handles.append(custom_handle)
            # Use clean value without key prefix
            line_col_labels.append(line_val)

    # For style_col group (line styles), use first available line_col to get style
    style_col_handles = []
    style_col_labels = []

    first_line_val = sorted_line_col_values[0] if sorted_line_col_values else ""
    for style_val in sorted_style_col_values:
        # Find a handle that has this style_col value to extract its line style
        sample_handle = None
        for (l_val, s_val), handle in handle_map.items():
            if s_val == style_val:
                sample_handle = handle
                break

        if sample_handle:
            # Create Line2D with same line style, but black color for clarity
            custom_handle = Line2D(
                [0],
                [0],
                color="black",
                linewidth=2,
                linestyle=sample_handle.get_linestyle(),
            )
            style_col_handles.append(custom_handle)
            # Use clean value without key prefix
            style_col_labels.append(style_val)

    # Calculate number of columns for each legend
    line_col_cols = (
        line_col_ncol if line_col_ncol is not None else len(line_col_handles)
    )
    style_col_cols = (
        style_col_ncol if style_col_ncol is not None else len(style_col_handles)
    )

    # Calculate required rows for each legend
    line_col_rows = (
        (len(line_col_handles) + line_col_cols - 1) // line_col_cols
        if line_col_handles
        else 0
    )
    style_col_rows = (
        (len(style_col_handles) + style_col_cols - 1) // style_col_cols
        if style_col_handles
        else 0
    )
    max_legend_rows = max(line_col_rows, style_col_rows)

    # Calculate adaptive positioning for centered legends with edge-based spacing
    center_x = 0.5  # Center of figure

    # Estimate legend width (approximate, since we can't measure before creation)
    # Assume each legend takes about 0.15-0.2 of figure width
    estimated_legend_width = 0.18

    # Position legends so their edges are separated by legend_spacing
    # Left legend: right edge at (center - spacing/2)
    # Right legend: left edge at (center + spacing/2)
    left_legend_x = center_x - (legend_spacing / 2) - (estimated_legend_width / 2)
    right_legend_x = center_x + (legend_spacing / 2) + (estimated_legend_width / 2)

    # Adjust bottom margin based on maximum legend rows
    if adjust_layout:
        if bottom_margin is not None:
            # Use manual override
            calculated_margin = bottom_margin
        else:
            # Calculate based on legend rows, but more compact
            calculated_margin = 0.08 + (max_legend_rows - 1) * 0.03  # Reduced padding
        plt.subplots_adjust(bottom=calculated_margin)

    # Create first legend (line_col group) - positioned left of center
    if line_col_handles:
        legend1 = fm.fig.legend(
            line_col_handles,
            line_col_labels,
            loc="upper center",
            bbox_to_anchor=(left_legend_x, legend_y_pos),  # Adaptive left position
            ncol=line_col_cols,  # Configurable columns
            frameon=True,
            fancybox=True,
            shadow=True,
            title=line_col.title(),  # e.g., "Data" or "Params"
        )
        # Add first legend as artist to preserve it
        fm.fig.add_artist(legend1)

    # Create second legend (style_col group) - positioned right of center
    if style_col_handles:
        legend2 = fm.fig.legend(
            style_col_handles,
            style_col_labels,
            loc="upper center",
            bbox_to_anchor=(right_legend_x, legend_y_pos),  # Adaptive right position
            ncol=style_col_cols,  # Configurable columns
            frameon=True,
            fancybox=True,
            shadow=True,
            title=style_col.title(),  # e.g., "Data" or "Params"
        )


def main():
    # Load data
    print("Loading DataDecide data...")
    data_dir = repo_root / "outputs" / "example_data"

    if not data_dir.exists():
        print(
            f"Data directory {data_dir} does not exist. Please run data pipeline first."
        )
        return

    dd = DataDecide(data_dir=str(data_dir), verbose=True)

    # Use mean_eval for averaged data across seeds
    df = dd.load_dataframe("mean_eval")
    print(f"Loaded mean_eval dataframe with shape: {df.shape}")
    print(f"Available columns: {list(df.columns)[:20]}...")  # Truncate long list
    print(f"Unique params: {sorted(df['params'].unique())}")
    print(f"Unique data: {sorted(df['data'].unique())[:10]}...")  # Truncate long list

    # Drop rows with NaN values in key plotting columns
    key_columns = [
        "tokens",
        "pile-valppl",
        "mmlu_average_correct_prob",
        "params",
        "data",
    ]
    available_key_columns = [col for col in key_columns if col in df.columns]
    print(f"Filtering NaN values in columns: {available_key_columns}")
    initial_shape = df.shape
    df = df.dropna(subset=available_key_columns)
    print(
        f"After NaN filtering: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)"
    )

    # Create output directory for plots
    plots_dir = repo_root / "plots" / "test_plotting"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    # Use the requested params and data values
    test_params = ["10M", "20M", "60M", "90M"]
    test_data = [
        "Dolma1.7",
        "DCLM-Baseline 25% / Dolma 75%",
        "DCLM-Baseline 50% / Dolma 50%",
        "DCLM-Baseline 75% / Dolma 25%",
        "DCLM-Baseline",
    ]

    print(
        f"Using {len(test_params)} params and {len(test_data)} data recipes for dynamic ncols"
    )

    # Configuration 1: params as lines, data as subplots (your main use case)
    print("\n=== Configuration 1: params as lines, data as subplots ===")
    try:
        # Using the new builder pattern
        builder1 = ScalingPlotBuilder(df)
        builder1.clean_data(
            key_columns=available_key_columns
        )  # Data already cleaned above, but showing pattern
        builder1.with_params(test_params)
        builder1.with_data(test_data)
        builder1.configure(
            x_col="tokens",
            y_col="pile-valppl",
            line_col="params",
            subplot_col="data",
            title_prefix="Config 1",
            ncols=len(test_data),  # Single row matching number of data subplots
            figsize=(len(test_data) * 5, 5),  # Wider figure for single row
            sharey=True,  # Share y-axis across subplots
            multi_color_sequence=["darkred", "lightcoral", "plum", "lightblue", "darkblue"],  # 5-color progression
            color_range_min=0.0,  # Use full range for 5-color progression
            color_range_max=1.0,
        )
        fig1, fm1 = builder1.build()
        fix_sharey_labels(builder1, fm1)
        add_unified_legend_below(builder1, fm1)

        fig1.savefig(
            plots_dir / "config1_params_lines_data_subplots.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("✓ Saved config1_params_lines_data_subplots.png")
    except Exception as e:
        print(f"✗ Error in config 1: {e}")

    # Configuration 2: data as lines, params as subplots (swapped)
    print("\n=== Configuration 2: data as lines, params as subplots ===")
    try:
        # Sort the DataFrame by data column according to test_data order before plotting
        # Create a mapping of data values to their position in test_data list
        data_order_map = {data_val: i for i, data_val in enumerate(test_data)}

        # Add a temporary sort column and sort the DataFrame
        df_sorted = df.copy()
        df_sorted["_temp_data_order"] = df_sorted["data"].map(data_order_map)
        df_sorted = df_sorted.sort_values("_temp_data_order").drop(
            columns=["_temp_data_order"]
        )

        print(f"Data order in test_data: {test_data}")
        print(
            f"Data order after sorting DataFrame: {df_sorted['data'].unique().tolist()}"
        )

        # More concise builder usage
        builder2 = (
            ScalingPlotBuilder(df_sorted)  # Use sorted DataFrame
            .with_params(test_params)
            .with_data(test_data)
            .configure(
                x_col="tokens",
                y_col="pile-valppl",
                line_col="data",
                subplot_col="params",
                title_prefix="Config 2",
                ncols=len(test_params),  # Single row matching number of param subplots
                figsize=(len(test_params) * 5, 5),  # Wider figure for single row
                sharey=True,  # Share y-axis across subplots
                multi_color_sequence=["darkred", "lightcoral", "plum", "lightblue", "darkblue"],  # Same 5-color progression as Config 1
                color_range_min=0.0,  # Use full range for 5-color progression
                color_range_max=1.0,
            )
        )
        fig2, fm2 = builder2.build()
        fix_sharey_labels(builder2, fm2)
        add_unified_legend_below(builder2, fm2)

        fig2.savefig(
            plots_dir / "config2_data_lines_params_subplots.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("✓ Saved config2_data_lines_params_subplots.png")
    except Exception as e:
        print(f"✗ Error in config 2: {e}")

    # Configuration 3: Different metric (MMLU instead of perplexity)
    print("\n=== Configuration 3: MMLU metric ===")
    try:
        # Sort the DataFrame by data column according to test_data order before plotting (same as Config 2)
        # Create a mapping of data values to their position in test_data list
        data_order_map = {data_val: i for i, data_val in enumerate(test_data)}
        
        # Add a temporary sort column and sort the DataFrame
        df_sorted_config3 = df.copy()
        df_sorted_config3["_temp_data_order"] = df_sorted_config3["data"].map(data_order_map)
        df_sorted_config3 = df_sorted_config3.sort_values("_temp_data_order").drop(
            columns=["_temp_data_order"]
        )
        
        # Another concise example showing different metric
        builder3 = (
            ScalingPlotBuilder(df_sorted_config3)  # Use sorted DataFrame
            .with_params(test_params)
            .with_data(test_data)
            .configure(
                y_col="mmlu_average_correct_prob",
                title_prefix="Config 3",
                ncols=len(test_data),  # Single row matching number of data subplots
                sharey=True,  # Share y-axis for this metric comparison
                multi_color_sequence=["darkred", "lightcoral", "plum", "lightblue", "darkblue"],  # Same 5-color progression as Config 1
                color_range_min=0.0,  # Use full range for 5-color progression
                color_range_max=1.0,
            )
        )
        fig3, fm3 = builder3.build()
        fix_sharey_labels(builder3, fm3)
        add_unified_legend_below(builder3, fm3)

        fig3.savefig(
            plots_dir / "config3_mmlu_metric.png", dpi=150, bbox_inches="tight"
        )
        print("✓ Saved config3_mmlu_metric.png")
    except Exception as e:
        print(f"✗ Error in config 3: {e}")

    # Configuration 4: Multi-metric comparison
    print("\n=== Configuration 4: Multi-metric comparison ===")
    try:
        test_metrics = ["pile-valppl", "mmlu_average_correct_prob"]
        # Check which metrics actually exist in the data
        available_metrics = [m for m in test_metrics if m in df.columns]
        print(f"Available metrics: {available_metrics}")

        if available_metrics:
            # Using ModelComparisonBuilder
            builder4 = (
                ModelComparisonBuilder(df, available_metrics)
                .with_params(test_params)
                .with_data(test_data)
                .configure(
                    x_col="tokens",
                    line_col="data",  # Data recipes as different colors (hue)
                    style_col="params",  # Model sizes as different line styles
                    ncols=2,  # Single row with 2 columns (for 2 metrics)
                    figsize=(10, 5),  # Wider figure for single row
                    sharey=False,  # Don't share y-axis (different metrics have different scales)
                    multi_color_sequence=["darkred", "lightcoral", "plum", "lightblue", "darkblue"],  # Same 5-color progression for data recipes
                    linestyle_sequence=[
                        "-",
                        "--",
                        "-.",
                        ":",
                    ],  # Solid to dotted progression for model sizes
                )
            )
            fig4, fm4 = builder4.build()

            # Note: No sharey fix needed since sharey=False
            # Use grouped legends to separate data (colors) from params (line styles)
            # Use single column for each legend with minimal spacing and compact layout
            add_grouped_legends_below(
                builder4,
                fm4,
                line_col_ncol=1,
                style_col_ncol=1,
                legend_spacing=0.02,  # Minimal spacing between legend edges
                legend_y_pos=0.01,  # Lower position
                bottom_margin=0.12,  # Compact bottom margin
            )

            fig4.savefig(
                plots_dir / "config4_multi_metric.png", dpi=150, bbox_inches="tight"
            )
            print("✓ Saved config4_multi_metric.png")
        else:
            print("✗ No valid metrics found for multi-metric plot")
    except Exception as e:
        print(f"✗ Error in config 4: {e}")

    # Configuration 5: Single data recipe, more params
    print("\n=== Configuration 5: Single data recipe, more params ===")
    try:
        # Use the same test_params as other configurations, but sorted numerically
        all_params = sorted(test_params, key=param_to_numeric)
        single_data = [test_data[0]] if test_data else [sorted(df["data"].unique())[0]]

        # Example using from_datadecide constructor if we had a DataDecide instance
        # builder5 = ScalingPlotBuilder.from_datadecide(dd, 'mean_eval', verbose=True)

        fig5, fm5 = (
            ScalingPlotBuilder(df)
            .with_params(all_params)
            .with_data(single_data)
            .configure(
                title_prefix="Config 5",
                ncols=1,  # Single column (1 data recipe = 1 subplot)
                colormap="Purples",  # Use single-color purple progression for model size
                color_range_min=0.1,  # Start from 30% of colormap (more visible)
                color_range_max=1.0,  # End at 90% of colormap (not too dark)
            )
            .build()
        )

        fig5.savefig(
            plots_dir / "config5_single_data_more_params.png",
            dpi=150,
            bbox_inches="tight",
        )
        print("✓ Saved config5_single_data_more_params.png")
    except Exception as e:
        print(f"✗ Error in config 5: {e}")

    print(f"\nPlotting test complete! Check plots in: {plots_dir}")
    print("Use your image viewer to inspect the generated plots.")


if __name__ == "__main__":
    main()
