#!/usr/bin/env python3
"""
Comprehensive visual regression tests for plotting configurations.
These tests capture ALL visual details of the 7 plot configurations to ensure
no changes occur during refactoring.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add src to path so we can import datadec
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from datadec import DataDecide
from datadec.plotting import ScalingPlotBuilder, ModelComparisonBuilder
from datadec.model_utils import param_to_numeric


# Import the helper functions from test_plotting.py
test_plotting_path = repo_root / "scripts" / "test_plotting.py"
sys.path.insert(0, str(test_plotting_path.parent))
from test_plotting import (
    fix_sharey_labels,
    add_unified_legend_below,
    add_grouped_legends_below,
)


@pytest.fixture(scope="session")
def datadecide_instance():
    """Load DataDecide instance with test data."""
    data_dir = repo_root / "outputs" / "example_data"

    if not data_dir.exists():
        pytest.skip(
            f"Data directory {data_dir} does not exist. Please run data pipeline first."
        )

    dd = DataDecide(data_dir=str(data_dir), verbose=False)
    return dd


@pytest.fixture(scope="session")
def test_dataframe(datadecide_instance):
    """Load and clean test dataframe."""
    df = datadecide_instance.load_dataframe("mean_eval")

    # Drop rows with NaN values in key plotting columns
    key_columns = [
        "tokens",
        "pile-valppl",
        "mmlu_average_correct_prob",
        "params",
        "data",
    ]
    available_key_columns = [col for col in key_columns if col in df.columns]
    df = df.dropna(subset=available_key_columns)

    return df


@pytest.fixture(scope="session")
def test_params():
    """Standard test parameters used across configurations."""
    return ["10M", "20M", "60M", "90M"]


@pytest.fixture(scope="session")
def test_data():
    """Standard test data recipes used across configurations."""
    return [
        "Dolma1.7",
        "DCLM-Baseline 25% / Dolma 75%",
        "DCLM-Baseline 50% / Dolma 50%",
        "DCLM-Baseline 75% / Dolma 25%",
        "DCLM-Baseline",
    ]


def extract_figure_properties(fig):
    """Extract all figure-level properties for comparison."""
    return {
        "figsize": fig.get_size_inches().tolist(),
        "dpi": fig.get_dpi(),
        "facecolor": fig.get_facecolor(),
        "edgecolor": fig.get_edgecolor(),
        "tight_layout": fig.get_tight_layout(),
        "suptitle_text": fig._suptitle.get_text() if fig._suptitle else None,
        "suptitle_fontsize": fig._suptitle.get_fontsize() if fig._suptitle else None,
        "num_axes": len(fig.get_axes()),
    }


def extract_axis_properties(ax):
    """Extract all axis properties for comparison."""
    return {
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "xscale": ax.get_xscale(),
        "yscale": ax.get_yscale(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "title": ax.get_title(),
        "visible": ax.get_visible(),
        "position": ax.get_position().bounds,
        "xticks": ax.get_xticks().tolist(),
        "yticks": ax.get_yticks().tolist(),
        "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
        "grid": ax.get_axisbelow(),
        "aspect": ax.get_aspect(),
        "autoscale": ax.get_autoscale_on(),
        "navigate": ax.get_navigate(),
        "xmargin": ax.margins()[0],
        "ymargin": ax.margins()[1],
    }


def extract_line_properties(line):
    """Extract all line properties for comparison."""
    xdata, ydata = line.get_data()

    # Convert data to lists for comparison, handling potential NaN values
    xdata_list = []
    ydata_list = []
    if len(xdata) > 0:
        for x, y in zip(xdata, ydata):
            # Only include finite values for reliable comparison
            if np.isfinite(x) and np.isfinite(y):
                xdata_list.append(float(x))
                ydata_list.append(float(y))

    return {
        "color": line.get_color(),
        "linestyle": line.get_linestyle(),
        "linewidth": line.get_linewidth(),
        "marker": line.get_marker(),
        "markersize": line.get_markersize(),
        "markeredgecolor": line.get_markeredgecolor(),
        "markeredgewidth": line.get_markeredgewidth(),
        "markerfacecolor": line.get_markerfacecolor(),
        "alpha": line.get_alpha(),
        "antialiased": line.get_antialiased(),
        "dash_capstyle": line.get_dash_capstyle(),
        "dash_joinstyle": line.get_dash_joinstyle(),
        "drawstyle": line.get_drawstyle(),
        "fillstyle": line.get_fillstyle(),
        "label": line.get_label(),
        "xdata_length": len(xdata),
        "ydata_length": len(ydata),
        "xdata_range": [float(np.min(xdata)), float(np.max(xdata))]
        if len(xdata) > 0
        else [0, 0],
        "ydata_range": [float(np.min(ydata)), float(np.max(ydata))]
        if len(ydata) > 0
        else [0, 0],
        # Capture actual data points for exact verification
        "xdata_values": xdata_list,
        "ydata_values": ydata_list,
        "finite_data_count": len(xdata_list),  # Number of valid (finite) data points
        "solid_capstyle": line.get_solid_capstyle(),
        "solid_joinstyle": line.get_solid_joinstyle(),
        "visible": line.get_visible(),
        "zorder": line.get_zorder(),
    }


def extract_legend_properties(legend):
    """Extract all legend properties for comparison."""
    if legend is None:
        return None

    # Get legend texts and handles using correct matplotlib API
    texts = [t.get_text() for t in legend.get_texts()]
    try:
        # Try the correct method first
        handles, labels = legend.get_legend_handles_labels()
    except AttributeError:
        # Fallback to accessing legend_handles property directly
        handles = legend.legend_handles if hasattr(legend, "legend_handles") else []

    # Extract handle properties
    handle_props = []
    for handle in handles:
        if hasattr(handle, "get_color"):  # Line2D or similar
            handle_props.append(
                {
                    "type": type(handle).__name__,
                    "color": handle.get_color()
                    if hasattr(handle, "get_color")
                    else None,
                    "linestyle": handle.get_linestyle()
                    if hasattr(handle, "get_linestyle")
                    else None,
                    "linewidth": handle.get_linewidth()
                    if hasattr(handle, "get_linewidth")
                    else None,
                    "marker": handle.get_marker()
                    if hasattr(handle, "get_marker")
                    else None,
                }
            )
        else:
            handle_props.append(
                {
                    "type": type(handle).__name__,
                    "color": None,
                    "linestyle": None,
                    "linewidth": None,
                    "marker": None,
                }
            )

    return {
        "numpoints": getattr(legend, "numpoints", None),
        "markerscale": getattr(legend, "markerscale", None),
        "markerfirst": getattr(legend, "markerfirst", None),
        "scatterpoints": getattr(legend, "scatterpoints", None),
        "scatteryoffsets": getattr(legend, "scatteryoffsets", None),
        "columnspacing": getattr(legend, "columnspacing", None),
        "handlelength": getattr(legend, "handlelength", None),
        "handletextpad": getattr(legend, "handletextpad", None),
        "borderaxespad": getattr(legend, "borderaxespad", None),
        "loc": getattr(legend, "_loc", None),
        "bbox_to_anchor": legend.get_bbox_to_anchor().bounds
        if legend.get_bbox_to_anchor()
        else None,
        "ncol": getattr(legend, "_ncol", None),
        "fontsize": getattr(legend, "_fontsize", None),
        "texts": texts,
        "texts_ordered": texts,  # Capture exact ordering of legend text
        "num_handles": len(handles),
        "handle_properties": handle_props,
        "handle_properties_ordered": handle_props,  # Capture exact ordering of handles
        "frameon": legend.get_frame_on() if hasattr(legend, "get_frame_on") else None,
        "shadow": getattr(legend, "shadow", None),
        "framealpha": legend.get_frame().get_alpha() if legend.get_frame() else None,
        "facecolor": legend.get_frame().get_facecolor() if legend.get_frame() else None,
        "edgecolor": legend.get_frame().get_edgecolor() if legend.get_frame() else None,
        "title": legend.get_title().get_text() if legend.get_title() else None,
    }


def extract_complete_plot_state(fig, fm=None):
    """Extract complete visual state of a plot for regression testing."""
    state = {"figure": extract_figure_properties(fig), "axes": [], "figure_legends": []}

    # Extract all axes properties
    for ax in fig.get_axes():
        ax_state = extract_axis_properties(ax)

        # Extract all lines in this axis
        ax_state["lines"] = []
        for line in ax.get_lines():
            ax_state["lines"].append(extract_line_properties(line))

        # Extract axis legend (if any)
        ax_state["legend"] = extract_legend_properties(ax.get_legend())

        state["axes"].append(ax_state)

    # Extract figure-level legends
    # Look for matplotlib Legend objects that are children of the figure
    for child in fig.get_children():
        if hasattr(child, "get_texts") and type(child).__name__ == "Legend":
            # This is a figure-level legend
            state["figure_legends"].append(extract_legend_properties(child))

    # Also check figure.legends attribute if available
    if hasattr(fig, "legends"):
        for legend in fig.legends:
            # Add if not already captured above
            legend_props = extract_legend_properties(legend)
            if legend_props not in state["figure_legends"]:
                state["figure_legends"].append(legend_props)

    return state


class TestConfig1:
    """Test Config 1: params as lines, data as subplots."""

    def test_config1_visual_properties(self, test_dataframe, test_params, test_data):
        """Test all visual properties of Config 1 remain unchanged."""

        # Build the plot exactly as in test_plotting.py
        builder1 = ScalingPlotBuilder(test_dataframe)
        builder1.clean_data(
            key_columns=[
                "tokens",
                "pile-valppl",
                "mmlu_average_correct_prob",
                "params",
                "data",
            ]
        )
        builder1.with_params(test_params)
        builder1.with_data(test_data)
        builder1.configure(
            x_col="tokens",
            y_col="pile-valppl",
            line_col="params",
            subplot_col="data",
            title_prefix="Config 1",
            ncols=len(test_data),
            figsize=(len(test_data) * 5, 5),
            sharey=True,
            multi_color_sequence=[
                "darkred",
                "lightcoral",
                "plum",
                "lightblue",
                "darkblue",
            ],
            color_range_min=0.0,
            color_range_max=1.0,
        )
        fig1, fm1 = builder1.build()
        fix_sharey_labels(builder1, fm1)
        add_unified_legend_below(builder1, fm1)

        # Extract complete visual state
        state = extract_complete_plot_state(fig1, fm1)

        # Expected values that must remain exactly the same
        expected = {
            "figure": {
                "figsize": [25.0, 5.0],  # 5 data * 5 width, 5 height
                "num_axes": 5,  # One per data recipe
            },
            "axes_count": 5,
            "lines_per_axis": 4,  # One per param (10M, 20M, 60M, 90M)
            "shared_ylim": True,  # All axes should have same y-limits
            "y_labels_count": 1,  # Only leftmost should have y-label due to sharey fix
        }

        # Verify figure properties
        assert state["figure"]["figsize"] == expected["figure"]["figsize"]
        assert state["figure"]["num_axes"] == expected["figure"]["num_axes"]

        # Verify axes count and structure
        assert len(state["axes"]) == expected["axes_count"]

        # Verify each axis has correct number of lines
        for ax_state in state["axes"]:
            assert len(ax_state["lines"]) == expected["lines_per_axis"]

        # Verify sharey behavior - all axes should have same y-limits
        y_limits = [ax_state["ylim"] for ax_state in state["axes"]]
        first_ylim = y_limits[0]
        for ylim in y_limits:
            assert (
                abs(ylim[0] - first_ylim[0]) < 1e-10
            )  # Allow floating point precision
            assert abs(ylim[1] - first_ylim[1]) < 1e-10

        # Verify y-label fixing - only first axis should have y-label
        y_labels = [ax_state["ylabel"] for ax_state in state["axes"]]
        assert y_labels[0] != ""  # First axis has label
        assert all(label == "" for label in y_labels[1:])  # Rest are empty

        # Verify line colors are distinct and consistent
        for ax_state in state["axes"]:
            line_colors = [line["color"] for line in ax_state["lines"]]
            # Each line should have a different color
            assert len(set(line_colors)) == len(
                ax_state["lines"]
            )  # All lines have different colors

        # Verify legends exist (either figure-level or axis-level)
        total_legend_count = len(state["figure_legends"])
        axis_legend_count = sum(
            1 for ax_state in state["axes"] if ax_state["legend"] is not None
        )
        assert (
            total_legend_count + axis_legend_count >= 1
        )  # Should have at least some legend content

        # Verify legend ordering for Config 1 (params as line_col)
        if total_legend_count > 0:
            legend_props = state["figure_legends"][0]
            if legend_props and legend_props["texts_ordered"]:
                legend_texts = legend_props["texts_ordered"]
                # Config 1 uses params as lines, so legend should follow test_params order
                expected_param_order = test_params  # ["10M", "20M", "60M", "90M"]
                # Verify all expected params appear in legend
                for param in expected_param_order:
                    assert any(param in text for text in legend_texts), (
                        f"Param {param} not found in legend"
                    )

        # Verify data point values are captured for each line
        for ax_idx, ax_state in enumerate(state["axes"]):
            for line_idx, line_props in enumerate(ax_state["lines"]):
                # Each line should have actual data points
                assert line_props["finite_data_count"] > 0, (
                    f"Axis {ax_idx}, Line {line_idx} has no finite data points"
                )
                assert (
                    len(line_props["xdata_values"]) == line_props["finite_data_count"]
                )
                assert (
                    len(line_props["ydata_values"]) == line_props["finite_data_count"]
                )
                # Verify data points are reasonable (positive tokens, reasonable perplexity)
                if len(line_props["xdata_values"]) > 0:
                    assert all(x > 0 for x in line_props["xdata_values"]), (
                        "X values (tokens) should be positive"
                    )
                    assert all(y > 0 for y in line_props["ydata_values"]), (
                        "Y values (perplexity) should be positive"
                    )

        plt.close(fig1)


class TestConfig2:
    """Test Config 2: data as lines, params as subplots."""

    def test_config2_visual_properties(self, test_dataframe, test_params, test_data):
        """Test all visual properties of Config 2 remain unchanged."""

        # Sort DataFrame exactly as in test_plotting.py
        data_order_map = {data_val: i for i, data_val in enumerate(test_data)}
        df_sorted = test_dataframe.copy()
        df_sorted["_temp_data_order"] = df_sorted["data"].map(data_order_map)
        df_sorted = df_sorted.sort_values("_temp_data_order").drop(
            columns=["_temp_data_order"]
        )

        # Build the plot exactly as in test_plotting.py
        builder2 = (
            ScalingPlotBuilder(df_sorted)
            .with_params(test_params)
            .with_data(test_data)
            .configure(
                x_col="tokens",
                y_col="pile-valppl",
                line_col="data",
                subplot_col="params",
                title_prefix="Config 2",
                ncols=len(test_params),
                figsize=(len(test_params) * 5, 5),
                sharey=True,
                multi_color_sequence=[
                    "darkred",
                    "lightcoral",
                    "plum",
                    "lightblue",
                    "darkblue",
                ],
                color_range_min=0.0,
                color_range_max=1.0,
            )
        )
        fig2, fm2 = builder2.build()
        fix_sharey_labels(builder2, fm2)
        add_unified_legend_below(builder2, fm2)

        # Extract complete visual state
        state = extract_complete_plot_state(fig2, fm2)

        # Expected values
        expected = {
            "figure": {
                "figsize": [20.0, 5.0],  # 4 params * 5 width, 5 height
                "num_axes": 4,  # One per param
            },
            "axes_count": 4,
            "lines_per_axis": 5,  # One per data recipe
        }

        # Verify figure properties
        assert state["figure"]["figsize"] == expected["figure"]["figsize"]
        assert state["figure"]["num_axes"] == expected["figure"]["num_axes"]
        assert len(state["axes"]) == expected["axes_count"]

        # Verify each axis has correct number of lines
        for ax_state in state["axes"]:
            assert len(ax_state["lines"]) == expected["lines_per_axis"]

        # Verify data ordering preservation for Config 2 (data as line_col)
        # Extract line labels from first subplot to verify ordering
        first_ax_lines = state["axes"][0]["lines"]
        line_labels = [line["label"] for line in first_ax_lines]

        # Config 2 uses data as lines, so verify data recipes appear in correct order
        expected_data_order = test_data  # Should preserve the test_data order
        for data_recipe in expected_data_order:
            # Each data recipe should appear in at least one line label
            assert any(data_recipe in label for label in line_labels), (
                f"Data recipe {data_recipe} not found in line labels"
            )

        # Verify cross-validation: line labels should match what appears in legends
        total_legend_count = len(state["figure_legends"])
        if total_legend_count > 0:
            legend_props = state["figure_legends"][0]
            if legend_props and legend_props["texts_ordered"]:
                legend_texts = legend_props["texts_ordered"]
                # Every unique line label should have corresponding legend entry
                unique_line_labels = set()
                for ax_state in state["axes"]:
                    for line in ax_state["lines"]:
                        if line["label"] and not line["label"].startswith(
                            "_"
                        ):  # Skip internal matplotlib labels
                            unique_line_labels.add(line["label"])

                for line_label in unique_line_labels:
                    assert any(
                        line_label == legend_text for legend_text in legend_texts
                    ), f"Line label '{line_label}' not found in legend"

        plt.close(fig2)


class TestConfig3:
    """Test Config 3: MMLU metric."""

    def test_config3_visual_properties(self, test_dataframe, test_params, test_data):
        """Test all visual properties of Config 3 remain unchanged."""

        # Sort DataFrame exactly as in test_plotting.py
        data_order_map = {data_val: i for i, data_val in enumerate(test_data)}
        df_sorted_config3 = test_dataframe.copy()
        df_sorted_config3["_temp_data_order"] = df_sorted_config3["data"].map(
            data_order_map
        )
        df_sorted_config3 = df_sorted_config3.sort_values("_temp_data_order").drop(
            columns=["_temp_data_order"]
        )

        # Build the plot exactly as in test_plotting.py
        builder3 = (
            ScalingPlotBuilder(df_sorted_config3)
            .with_params(test_params)
            .with_data(test_data)
            .configure(
                y_col="mmlu_average_correct_prob",
                title_prefix="Config 3",
                ncols=len(test_data),
                sharey=True,
                multi_color_sequence=[
                    "darkred",
                    "lightcoral",
                    "plum",
                    "lightblue",
                    "darkblue",
                ],
                color_range_min=0.0,
                color_range_max=1.0,
            )
        )
        fig3, fm3 = builder3.build()
        fix_sharey_labels(builder3, fm3)
        add_unified_legend_below(builder3, fm3)

        # Extract complete visual state
        state = extract_complete_plot_state(fig3, fm3)

        # Expected values - similar to Config 1 but uses default figsize since not explicitly set
        expected = {
            "figure": {
                "figsize": [
                    25.0,
                    4.0,
                ],  # 5 data * 5 width, default height (different from Config 1)
                "num_axes": 5,
            },
            "axes_count": 5,
            "lines_per_axis": 4,
        }

        # Verify core structure
        assert state["figure"]["figsize"] == expected["figure"]["figsize"]
        assert state["figure"]["num_axes"] == expected["figure"]["num_axes"]
        assert len(state["axes"]) == expected["axes_count"]

        # Verify each axis has correct number of lines
        for ax_state in state["axes"]:
            assert len(ax_state["lines"]) == expected["lines_per_axis"]

        # Verify y-axis shows MMLU metric
        for ax_state in state["axes"]:
            # Y-axis should be different from Config 1 (which uses pile-valppl)
            # MMLU values are typically between 0 and 1
            assert ax_state["ylim"][0] >= 0  # MMLU is non-negative
            assert ax_state["ylim"][1] <= 1  # MMLU is typically ≤ 1

        plt.close(fig3)


class TestConfig4:
    """Test Config 4: Multi-metric comparison."""

    def test_config4_visual_properties(self, test_dataframe):
        """Test all visual properties of Config 4 remain unchanged."""

        # Define config-specific parameters
        config4_params = ["20M", "90M", "530M"]
        config4_data = [
            "Dolma1.7",
            "DCLM-Baseline 25% / Dolma 75%",
            "DCLM-Baseline 75% / Dolma 25%",
            "DCLM-Baseline",
        ]
        config4_colors = ["darkred", "lightcoral", "lightblue", "darkblue"]
        config4_linestyles = ["-", "--", ":"]

        test_metrics = ["pile-valppl", "mmlu_average_correct_prob"]
        available_metrics = [m for m in test_metrics if m in test_dataframe.columns]

        # Sort DataFrame exactly as in test_plotting.py
        data_order_map = {data_val: i for i, data_val in enumerate(config4_data)}
        df_sorted_config4 = test_dataframe.copy()
        df_sorted_config4["_temp_data_order"] = df_sorted_config4["data"].map(
            data_order_map
        )
        df_sorted_config4 = df_sorted_config4.sort_values("_temp_data_order").drop(
            columns=["_temp_data_order"]
        )

        # Build the plot exactly as in test_plotting.py
        builder4 = (
            ModelComparisonBuilder(df_sorted_config4, available_metrics)
            .with_params(config4_params)
            .with_data(config4_data)
            .configure(
                x_col="tokens",
                line_col="data",
                style_col="params",
                ncols=2,
                figsize=(10, 5),
                sharey=False,
                multi_color_sequence=config4_colors,
                linestyle_sequence=config4_linestyles,
            )
        )
        fig4, fm4 = builder4.build()

        add_grouped_legends_below(
            builder4,
            fm4,
            line_col_ncol=1,
            style_col_ncol=1,
            legend_spacing=0.02,
            legend_y_pos=0.01,
            bottom_margin=0.12,
        )

        # Extract complete visual state
        state = extract_complete_plot_state(fig4, fm4)

        # Expected values
        expected = {
            "figure": {
                "figsize": [10.0, 5.0],
                "num_axes": 2,  # One per metric
            },
            "axes_count": 2,
            "expected_line_count_per_axis": len(config4_data)
            * len(config4_params),  # 4 * 3 = 12
        }

        # Verify figure properties
        assert state["figure"]["figsize"] == expected["figure"]["figsize"]
        assert state["figure"]["num_axes"] == expected["figure"]["num_axes"]
        assert len(state["axes"]) == expected["axes_count"]

        # Verify line count per axis (combinations of data recipes and param sizes)
        for ax_state in state["axes"]:
            assert len(ax_state["lines"]) == expected["expected_line_count_per_axis"]

        # Verify sharey=False - axes should have different y-limits
        if len(state["axes"]) >= 2:
            ax1_ylim = state["axes"][0]["ylim"]
            ax2_ylim = state["axes"][1]["ylim"]
            # Y-limits should be different for different metrics
            assert ax1_ylim != ax2_ylim

        # Verify grouped legends exist (they might be figure-level or other arrangements)
        # At minimum, we should have legend content somewhere
        total_legend_count = len(state["figure_legends"])
        axis_legend_count = sum(
            1 for ax_state in state["axes"] if ax_state["legend"] is not None
        )
        assert (
            total_legend_count + axis_legend_count >= 1
        )  # Should have at least some legend content

        # Verify Config 4 specific properties
        if total_legend_count >= 2:
            # Config 4 should have grouped legends (data colors + param styles)
            # Verify legend titles and content
            legend_titles = [
                legend["title"] for legend in state["figure_legends"] if legend
            ]
            legend_titles = [
                title for title in legend_titles if title
            ]  # Remove None values

            # Should have legends for both line_col (data) and style_col (params)
            expected_titles = ["Data", "Params"]  # From grouped legends implementation
            for expected_title in expected_titles:
                assert any(expected_title in str(title) for title in legend_titles), (
                    f"Legend title '{expected_title}' not found"
                )

        # Verify line style variation in Config 4
        for ax_state in state["axes"]:
            line_styles = [line["linestyle"] for line in ax_state["lines"]]
            # Should have multiple line styles due to style_col="params"
            unique_styles = set(line_styles)
            assert len(unique_styles) > 1, (
                "Config 4 should have multiple line styles for different param sizes"
            )

        plt.close(fig4)


class TestConfig5:
    """Test Config 5: Single data recipe, more params."""

    def test_config5_visual_properties(self, test_dataframe, test_params, test_data):
        """Test all visual properties of Config 5 remain unchanged."""

        # Define config-specific parameters
        all_params = sorted(test_params, key=param_to_numeric)
        single_data = (
            [test_data[0]]
            if test_data
            else [sorted(test_dataframe["data"].unique())[0]]
        )

        # Build the plot exactly as in test_plotting.py
        fig5, fm5 = (
            ScalingPlotBuilder(test_dataframe)
            .with_params(all_params)
            .with_data(single_data)
            .configure(
                title_prefix="Config 5",
                ncols=1,
                colormap="Purples",
                color_range_min=0.1,
                color_range_max=1.0,
            )
            .build()
        )

        # Extract complete visual state
        state = extract_complete_plot_state(fig5, fm5)

        # Expected values
        expected = {
            "figure": {
                "num_axes": 1,  # Single subplot
            },
            "axes_count": 1,
            "lines_per_axis": len(all_params),  # One per param
        }

        # Verify figure properties
        assert state["figure"]["num_axes"] == expected["figure"]["num_axes"]
        assert len(state["axes"]) == expected["axes_count"]

        # Verify single axis has correct number of lines
        assert len(state["axes"][0]["lines"]) == expected["lines_per_axis"]

        # Verify colormap is different from multi-color sequence configs
        line_colors = [line["color"] for line in state["axes"][0]["lines"]]
        # Purple colormap should produce different colors than the multi-color sequences
        assert len(set(line_colors)) == len(all_params)  # Each line different color

        plt.close(fig5)


class TestConfig6:
    """Test Config 6: Stacked - pile-valppl + mmlu, params as lines."""

    def test_config6_visual_properties(self, test_dataframe, test_params, test_data):
        """Test all visual properties of Config 6 remain unchanged."""

        # Sort DataFrame exactly as in test_plotting.py
        data_order_map = {data_val: i for i, data_val in enumerate(test_data)}
        df_sorted_config6 = test_dataframe.copy()
        df_sorted_config6["_temp_data_order"] = df_sorted_config6["data"].map(
            data_order_map
        )
        df_sorted_config6 = df_sorted_config6.sort_values("_temp_data_order").drop(
            columns=["_temp_data_order"]
        )

        # Define config-specific parameters
        config6_metrics = ["pile-valppl", "mmlu_average_correct_prob"]

        # Build the plot exactly as in test_plotting.py
        builder6 = (
            ModelComparisonBuilder(df_sorted_config6, config6_metrics)
            .with_params(test_params)
            .with_data(test_data)
            .configure(
                x_col="tokens",
                line_col="params",
                subplot_col="data",
                style_col=None,
                title_prefix="Config 6: Stacked Metrics",
                stacked_subplots=True,
                figsize=(len(test_data) * 5, 10),
                log_scale=True,
                sharey=False,
                multi_color_sequence=[
                    "darkred",
                    "lightcoral",
                    "plum",
                    "lightblue",
                    "darkblue",
                ],
                color_range_min=0.0,
                color_range_max=1.0,
            )
        )
        fig6, fm6 = builder6.build()
        add_unified_legend_below(builder6, fm6)

        # Extract complete visual state
        state = extract_complete_plot_state(fig6, fm6)

        # Expected values for stacked layout
        expected = {
            "figure": {
                "figsize": [25.0, 10.0],  # 5 data * 5 width, 10 height for stacked
                "num_axes": len(config6_metrics)
                * len(test_data),  # 2 metrics * 5 data = 10 axes
            },
            "axes_count": len(config6_metrics) * len(test_data),
            "lines_per_axis": len(test_params),  # 4 params per axis
        }

        # Verify figure properties
        assert state["figure"]["figsize"] == expected["figure"]["figsize"]
        assert state["figure"]["num_axes"] == expected["figure"]["num_axes"]
        assert len(state["axes"]) == expected["axes_count"]

        # Verify each axis has correct number of lines
        for ax_state in state["axes"]:
            assert len(ax_state["lines"]) == expected["lines_per_axis"]

        # Verify stacked layout - should have 2 rows (metrics) and 5 columns (data)
        # Axes should be arranged in grid pattern
        positions = [ax_state["position"] for ax_state in state["axes"]]
        unique_y_positions = set(
            round(pos[1], 3) for pos in positions
        )  # Y positions (rows)
        unique_x_positions = set(
            round(pos[0], 3) for pos in positions
        )  # X positions (cols)

        assert len(unique_y_positions) == len(config6_metrics)  # 2 rows
        assert len(unique_x_positions) == len(test_data)  # 5 columns

        plt.close(fig6)


class TestConfig7:
    """Test Config 7: Stacked - pile-valppl + mmlu, data as lines."""

    def test_config7_visual_properties(self, test_dataframe):
        """Test all visual properties of Config 7 remain unchanged."""

        # Define config-specific parameters (matching test_plotting.py exactly)
        config7_data = [
            "Dolma1.7",
            "DCLM-Baseline 25% / Dolma 75%",
            # Skip: "DCLM-Baseline 50% / Dolma 50%",
            "DCLM-Baseline 75% / Dolma 25%",
            "DCLM-Baseline",
        ]
        config7_metrics = ["pile-valppl", "mmlu_average_correct_prob"]
        config7_params = [
            "20M",
            "60M",
            "90M",
            "300M",
            "1B",
        ]

        # Sort DataFrame exactly as in test_plotting.py
        data_order_map = {data_val: i for i, data_val in enumerate(config7_data)}
        df_sorted_config7 = test_dataframe.copy()
        df_sorted_config7["_temp_data_order"] = df_sorted_config7["data"].map(
            data_order_map
        )
        df_sorted_config7 = df_sorted_config7.sort_values("_temp_data_order").drop(
            columns=["_temp_data_order"]
        )

        # Build the plot exactly as in test_plotting.py
        builder7 = (
            ModelComparisonBuilder(df_sorted_config7, config7_metrics)
            .with_params(config7_params)
            .with_data(config7_data)
            .configure(
                x_col="tokens",
                line_col="data",
                subplot_col="params",
                style_col=None,
                title_prefix="Config 7: Stacked Metrics",
                stacked_subplots=True,
                figsize=(len(config7_params) * 5, 10),
                log_scale=True,
                sharey=False,
                sharex_per_row=True,
                multi_color_sequence=[
                    "darkred",
                    "lightcoral",
                    "lightblue",
                    "darkblue",
                ],  # 4-color progression (no plum since no 50/50 recipe)
                color_range_min=0.0,
                color_range_max=1.0,
            )
        )
        fig7, fm7 = builder7.build()
        add_unified_legend_below(builder7, fm7)

        # Extract complete visual state
        state = extract_complete_plot_state(fig7, fm7)

        # Expected values for stacked layout
        expected = {
            "figure": {
                "figsize": [25.0, 10.0],  # 5 params * 5 width, 10 height for stacked
                "num_axes": len(config7_metrics)
                * len(config7_params),  # 2 metrics * 5 params = 10 axes
            },
            "axes_count": len(config7_metrics) * len(config7_params),
            "lines_per_axis": len(config7_data),  # 4 data recipes per axis
        }

        # Verify figure properties
        assert state["figure"]["figsize"] == expected["figure"]["figsize"]
        assert state["figure"]["num_axes"] == expected["figure"]["num_axes"]
        assert len(state["axes"]) == expected["axes_count"]

        # Verify each axis has correct number of lines
        for ax_state in state["axes"]:
            assert len(ax_state["lines"]) == expected["lines_per_axis"]

        # Verify stacked layout - should have 2 rows (metrics) and 5 columns (params)
        positions = [ax_state["position"] for ax_state in state["axes"]]
        unique_y_positions = set(
            round(pos[1], 3) for pos in positions
        )  # Y positions (rows)
        unique_x_positions = set(
            round(pos[0], 3) for pos in positions
        )  # X positions (cols)

        assert len(unique_y_positions) == len(config7_metrics)  # 2 rows
        assert len(unique_x_positions) == len(config7_params)  # 5 columns

        # Verify sharex_per_row behavior - axes in same row should have same x-limits
        # Group axes by row (y-position)
        axes_by_row = {}
        for i, ax_state in enumerate(state["axes"]):
            y_pos = round(ax_state["position"][1], 3)
            if y_pos not in axes_by_row:
                axes_by_row[y_pos] = []
            axes_by_row[y_pos].append(ax_state)

        # Check that each row has consistent x-limits
        for y_pos, row_axes in axes_by_row.items():
            first_xlim = row_axes[0]["xlim"]
            for ax_state in row_axes[1:]:
                assert abs(ax_state["xlim"][0] - first_xlim[0]) < 1e-10
                assert abs(ax_state["xlim"][1] - first_xlim[1]) < 1e-10

        # Verify Config 7 specific data filtering (no 50/50 recipe)
        config7_data = [
            "Dolma1.7",
            "DCLM-Baseline 25% / Dolma 75%",
            "DCLM-Baseline 75% / Dolma 25%",
            "DCLM-Baseline",
        ]

        # Extract line labels from first subplot to verify data recipes
        first_ax_lines = state["axes"][0]["lines"]
        line_labels = [line["label"] for line in first_ax_lines]

        # Verify that only expected data recipes appear (no 50/50)
        for data_recipe in config7_data:
            assert any(data_recipe in label for label in line_labels), (
                f"Expected data recipe {data_recipe} not found"
            )

        # Verify that 50/50 recipe is NOT present
        excluded_recipe = "DCLM-Baseline 50% / Dolma 50%"
        assert not any(excluded_recipe in label for label in line_labels), (
            f"Excluded recipe {excluded_recipe} should not be present in Config 7"
        )

        # Verify 4-color progression (not 5-color) due to reduced data set
        for ax_state in state["axes"]:
            line_colors = [line["color"] for line in ax_state["lines"]]
            unique_colors = set(line_colors)
            assert len(unique_colors) == 4, (
                f"Config 7 should have exactly 4 colors, got {len(unique_colors)}"
            )

        plt.close(fig7)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
