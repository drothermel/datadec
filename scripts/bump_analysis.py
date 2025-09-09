#!/usr/bin/env python3
"""
Unified bump analysis using dr_plotter CLI framework.

Consolidates plot_bump.py and plot_bump_timesteps.py functionality into a single
configurable script that handles both final-step and time-series bump plots through
dimensional choices.
"""

from __future__ import annotations

import click
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import itertools

from dr_plotter import FigureManager, consts
from dr_plotter.scripting import (
    CLIConfig,
    dimensional_plotting_cli,
    validate_layout_options,
    build_faceting_config,
    build_plot_config,
)
from dr_plotter.scripting.utils import show_or_save_plot
from dr_plotter.theme import BUMP_PLOT_THEME, Theme

from datadec import DataDecide
from datadec.constants import (
    BASE_AND_QC,
    BASE_RECIPES,
    CUSTOM_RECIPE_FAMILIES,
    OLMES_PERFORMANCE_RECIPE_CHUNKS,
    PPL_PERFORMANCE_RECIPE_CHUNKS,
    RECIPES_WITHOUT_ABLATIONS,
)
from datadec.model_utils import param_to_numeric


# Constants
THOUSAND = 1000
MILLION = 1e6
BILLION = 1e9
MIN_POINTS_FOR_SAMPLING = 2


def create_extended_color_palette() -> list[str]:
    """Create extended color palette for many categories."""
    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194",
        "#6b6ecf", "#b5cf6b", "#e7ba52", "#d6616b", "#ce6dbd",
        "#de9ed6", "#31a354", "#756bb1", "#636363", "#969696",
    ]


def create_bump_theme_with_colors(num_categories: int) -> Theme:
    """Create custom theme with extended color palette for bump plots."""
    extended_colors = create_extended_color_palette()
    colors_to_use = extended_colors[:max(num_categories, len(extended_colors))]
    
    return Theme(
        name="bump_unified",
        parent=BUMP_PLOT_THEME,
        **{
            consts.get_cycle_key("hue"): itertools.cycle(colors_to_use),
        },
    )


def resolve_data_groups(data_args: list[str]) -> list[str]:
    """Resolve named data groups to actual recipe lists."""
    named_groups = {
        "base": BASE_RECIPES,
        "base_qc": BASE_AND_QC,
        "no_ablations": RECIPES_WITHOUT_ABLATIONS,
        **CUSTOM_RECIPE_FAMILIES,
        **{
            f"{k.replace('_performance', '')}": v
            for k, v in PPL_PERFORMANCE_RECIPE_CHUNKS.items()
        },
        **{
            f"{k.replace('_performance', '')}": v
            for k, v in OLMES_PERFORMANCE_RECIPE_CHUNKS.items()
        },
    }

    resolved_recipes = []
    for arg in data_args:
        if arg in named_groups:
            resolved_recipes.extend(named_groups[arg])
        elif arg == "all":
            return data_args
        else:
            resolved_recipes.append(arg)

    return list(dict.fromkeys(resolved_recipes))


def numerical_sort_key(param_size: str) -> float:
    """Convert parameter size string to numerical value for proper sorting."""
    if param_size.endswith("M"):
        return float(param_size[:-1])
    elif param_size.endswith("B"):
        return float(param_size[:-1]) * 1000
    else:
        return float(param_size)


def format_perplexity(ppl_value: float) -> str:
    return f"{ppl_value:.2f}"


def format_step_label(step: float) -> str:
    if step >= THOUSAND:
        return f"{step / THOUSAND:.1f}k"
    else:
        return f"{int(step)}"


def format_token_count(token_count: float) -> str:
    if token_count >= BILLION:
        return f"{token_count / BILLION:.1f}B"
    elif token_count >= MILLION:
        return f"{token_count / MILLION:.0f}M"
    elif token_count >= THOUSAND:
        return f"{token_count / THOUSAND:.0f}K"
    else:
        return f"{int(token_count)}"


def prepare_final_step_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for final-step bump plot (replaces plot_bump.py)."""
    # Get final performance for each (params, data) combination
    final_rows = []
    for params_size in df["params"].unique():
        params_df = df[df["params"] == params_size]
        max_step_for_params = params_df["step"].max()
        final_step_data = params_df[params_df["step"] == max_step_for_params]
        final_rows.append(final_step_data)

    final_step_df = pd.concat(final_rows, ignore_index=True)
    print(f"Final step data shape: {final_step_df.shape}")
    return final_step_df


def prepare_timestep_data(df: pd.DataFrame, x_axis: str, min_step: float | None, max_step: float | None) -> pd.DataFrame:
    """Prepare data for timestep bump plot (replaces plot_bump_timesteps.py)."""
    # Add token information if x_axis is "tokens"
    if x_axis == "tokens":
        # Check if tokens column already exists
        if "tokens" not in df.columns:
            dd = DataDecide()
            token_info = dd.full_eval[["params", "data", "step", "tokens"]].drop_duplicates()
            df = df.merge(token_info, on=["params", "data", "step"], how="left")
        
        print(f"Token range: {df['tokens'].min():.0f} to {df['tokens'].max():.0f}")

    # Filter by step range if specified
    if min_step is not None:
        df = df[df["step"] >= min_step]
        print(f"After min_step filter: {df.shape}")

    if max_step is not None:
        df = df[df["step"] <= max_step]
        print(f"After max_step filter: {df.shape}")

    if df.empty:
        raise ValueError("No data found after applying step filters")

    # Create param x data combinations as categories for time series
    df["param_data_combo"] = df["params"].astype(str) + "-" + df["data"].astype(str)

    # Find common start point across all trajectories
    x_col = "tokens" if x_axis == "tokens" else "step"
    min_times_per_combo = df.groupby("param_data_combo")[x_col].min()
    common_start_time = min_times_per_combo.max()

    # Filter all trajectories to start from the common start point
    original_shape = df.shape[0]
    df = df[df[x_col] >= common_start_time].copy()
    filtered_shape = df.shape[0]

    x_label = "token" if x_axis == "tokens" else "step"
    print(f"Aligned all trajectories to common start {x_label}: {common_start_time}")
    print(f"Filtered from {original_shape} to {filtered_shape} points for alignment")

    return df


def create_bump_data(df: pd.DataFrame, x_axis: str, hue_by: str, final_step_only: bool) -> tuple[pd.DataFrame, str, list[str] | None]:
    """Create bump plot data with appropriate axis mapping."""
    if final_step_only:
        # Final step mode: X=params, Lines=data
        time_col = "params"
        category_col = "data"
        x_label = "Model Size"
    else:
        # Time series mode: X=steps/tokens, Lines=param_data_combo or other grouping
        if x_axis == "tokens":
            time_col = "tokens"
            x_label = "Token Count"
        else:
            time_col = "step" 
            x_label = "Training Steps"
            
        if hue_by == "param_data_combo":
            category_col = "param_data_combo"
        elif hue_by == "data":
            category_col = "data"
        elif hue_by == "params":
            category_col = "params"
        else:
            raise ValueError(f"Unknown hue_by value: {hue_by}")

    # Create bump plot data
    bump_data = df.rename(columns={
        time_col: "time",
        category_col: "category", 
        "value": "score",
    })[["time", "category", "score"]]

    # Keep original perplexity values for labeling (before inversion)
    bump_data["original_ppl"] = bump_data["score"].copy()

    # Initialize param names for return
    original_param_names = None

    # Ensure numeric sorting for any param-based time axis
    if time_col == "params" or (not final_step_only and "param" in category_col):
        # For param-based categories, sort them numerically in category names
        if "param" in category_col:
            # Extract param from param_data_combo and sort categories by param size
            def sort_param_combo(combo):
                param_part = combo.split("-")[0]  # Get "150M" from "150M-C4"
                return numerical_sort_key(param_part)
            
            unique_categories = sorted(bump_data["category"].unique(), key=sort_param_combo)
            category_to_pos = {cat: idx for idx, cat in enumerate(unique_categories)}
            # Don't remap categories, just ensure they're processed in numeric order
        
        # For param-based time axis (final step mode), map to numeric positions
        if time_col == "params":
            unique_params = sorted(bump_data["time"].unique(), key=numerical_sort_key)
            param_to_numeric_pos = {param: idx for idx, param in enumerate(unique_params)}
            bump_data["time"] = bump_data["time"].map(param_to_numeric_pos)
            # Store original param names for x-axis labeling
            original_param_names = unique_params

    # Invert scores for ranking (higher score = better rank)
    bump_data["score"] = -bump_data["score"]
    
    return bump_data, x_label, original_param_names


def add_ranking_labels(ax: plt.Axes, bump_data: pd.DataFrame, final_step_only: bool) -> None:
    """Add ranking labels appropriate for the plot type."""
    # Time points are already numeric, so just sort them normally
    time_points = sorted(bump_data["time"].unique())
    
    if len(time_points) < 2:
        return
        
    first_time = time_points[0]
    last_time = time_points[-1]

    # Left side labels (first time point)
    first_data = bump_data[bump_data["time"] == first_time].copy()
    first_data = first_data.sort_values("score", ascending=False)
    first_data["rank"] = range(1, len(first_data) + 1)

    for _, row in first_data.iterrows():
        category_name = row["category"]
        rank = row["rank"]
        ax.text(
            -0.15 if final_step_only else -0.02,
            rank,
            f"{rank}. {category_name}",
            transform=ax.transData if final_step_only else ax.get_yaxis_transform(),
            fontsize=9,
            ha="right",
            va="center",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightblue",
                "alpha": 0.7,
                "edgecolor": "navy",
            },
        )

    # Right side labels (last time point)  
    last_data = bump_data[bump_data["time"] == last_time].copy()
    last_data = last_data.sort_values("score", ascending=False)
    last_data["rank"] = range(1, len(last_data) + 1)

    for _, row in last_data.iterrows():
        category_name = row["category"]
        rank = row["rank"]
        
        if final_step_only:
            x_pos = len(time_points) - 1 + 0.15
            transform = ax.transData
        else:
            x_pos = 0.98
            transform = ax.get_yaxis_transform()
            
        ax.text(
            x_pos,
            rank,
            f"{rank}. {category_name}",
            transform=transform,
            fontsize=9,
            ha="left",
            va="center",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightgreen", 
                "alpha": 0.7,
                "edgecolor": "darkgreen",
            },
        )


def add_value_annotations(ax: plt.Axes, bump_data: pd.DataFrame, final_step_only: bool) -> None:
    """Add perplexity value annotations to data points."""
    # Time values are already processed appropriately, use them directly
    time_to_x = {time_point: time_point for time_point in bump_data["time"].unique()}

    ranked_data = []
    for time_point in bump_data["time"].unique():
        time_data = bump_data[bump_data["time"] == time_point].copy()
        time_data = time_data.sort_values("score", ascending=False)
        time_data["rank"] = range(1, len(time_data) + 1)
        ranked_data.append(time_data)

    all_ranked_data = pd.concat(ranked_data, ignore_index=True)

    for _, row in all_ranked_data.iterrows():
        x_pos = time_to_x[row["time"]]
        y_pos = row["rank"]
        ppl_text = format_perplexity(row["original_ppl"])

        ax.annotate(
            ppl_text,
            xy=(x_pos, y_pos),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
            },
            arrowprops=None,
        )


@click.command()
@dimensional_plotting_cli([])  # No fixed dimensions - let CLI handle validation
@click.option("--metric", default="pile-valppl", 
              help="Metric to plot for ranking comparison")
@click.option("--params", multiple=True, default=["all"], 
              help="Model parameter sizes (e.g., 150M 300M 1B) or 'all'")
@click.option("--data", multiple=True, default=["all"],
              help="Data recipes or named groups") 
@click.option("--exclude-params", multiple=True, default=[], 
              help="Model parameter sizes to exclude when using 'all'")
@click.option("--exclude-data", multiple=True, default=[],
              help="Data recipes to exclude when using 'all'")
@click.option("--x-axis", type=click.Choice(["params", "steps", "tokens"]), default="params",
              help="X-axis dimension: 'params' for model sizes, 'steps' for training steps, 'tokens' for token count")
@click.option("--final-step-only", is_flag=True, 
              help="Use only final step performance (replaces plot_bump.py mode)")
@click.option("--min-step", type=float, help="Minimum training step to include")
@click.option("--max-step", type=float, help="Maximum training step to include")
@click.option("--figsize", nargs=2, type=float, default=(12, 8),
              help="Figure size width height")
def main(**kwargs):
    """
    Unified bump analysis tool.
    
    Examples:
    
    # Replaces plot_bump.py (final step, X=params, Lines=data):
    python scripts/bump_analysis.py --x-axis params --hue-by data --final-step-only
    
    # Replaces plot_bump_timesteps.py (time series, X=steps, Lines=param×data):
    python scripts/bump_analysis.py --x-axis steps --hue-by param_data_combo
    
    # New possibilities:
    python scripts/bump_analysis.py --x-axis steps --hue-by data --rows params
    """
    # Validate CLI arguments
    ctx = click.get_current_context()
    validate_layout_options(ctx, **kwargs)
    
    # Build configuration
    config = CLIConfig.from_yaml(kwargs["config"]) if kwargs.get("config") else CLIConfig()
    
    # Extract CLI arguments for merging
    cli_kwargs = {k: v for k, v in kwargs.items() 
                  if k not in ["config"] and v is not None and v != () and v != []}
    
    # Merge CLI args into config 
    merged_config = config.merge_with_cli_args(cli_kwargs)
    
    # Extract specific parameters
    metric = merged_config.get("metric", "pile-valppl")
    params_list = list(merged_config.get("params", ["all"]))
    data_list = list(merged_config.get("data", ["all"]))
    exclude_params = list(merged_config.get("exclude_params", []))
    exclude_data = list(merged_config.get("exclude_data", []))
    x_axis = merged_config.get("x_axis", "params")
    hue_by = merged_config.get("hue_by", "data")  # From CLI framework
    final_step_only = merged_config.get("final_step_only", False)
    min_step = merged_config.get("min_step")
    max_step = merged_config.get("max_step")
    figsize = tuple(merged_config.get("figsize", (12, 8)))
    
    print(f"Bump Analysis Configuration:")
    print(f"  Metric: {metric}")
    print(f"  X-axis: {x_axis}")
    print(f"  Hue by: {hue_by}")
    print(f"  Final step only: {final_step_only}")
    print(f"  Model sizes: {params_list}")
    print(f"  Data recipes: {data_list}")

    # Initialize DataDecide
    dd = DataDecide()
    
    # Handle "all" values and exclusions for params
    if params_list == ["all"]:
        params_list = dd.select_params("all", exclude=exclude_params)
    
    # Resolve named data groups and handle "all"
    resolved_data = resolve_data_groups(data_list)
    if resolved_data == ["all"]:
        data_list = dd.select_data("all", exclude=exclude_data)
    else:
        data_list = [d for d in resolved_data if d not in exclude_data]

    print(f"Resolved params: {params_list}")
    print(f"Resolved data: {data_list}")

    # Prepare plot data
    df = dd.prepare_plot_data(
        params=params_list,
        data=data_list,
        metrics=[metric],
        aggregate_seeds=True,
        auto_filter=True,
        melt=True,
    )
    
    print(f"Base data shape: {df.shape}")

    # Prepare data based on mode
    if final_step_only:
        df = prepare_final_step_data(df)
    else:
        df = prepare_timestep_data(df, x_axis, min_step, max_step)

    # Create bump plot data
    bump_data, x_label, original_param_names = create_bump_data(df, x_axis, hue_by, final_step_only)
    
    print(f"Bump data shape: {bump_data.shape}")
    print(f"Categories: {len(bump_data['category'].unique())}")
    print(f"Time points: {len(bump_data['time'].unique())}")

    # Create custom theme with extended colors
    num_categories = len(bump_data["category"].unique())
    custom_theme = create_bump_theme_with_colors(num_categories)

    # Create plot
    metric_str = metric.replace("_", " ").replace("-", " ").title()
    title = f"Recipe Rankings by {x_label} ({metric_str})"
    
    from dr_plotter.configs import PlotConfig
    
    with FigureManager(
        PlotConfig(
            layout={"rows": 1, "cols": 1, "figsize": figsize},
            style={"theme": custom_theme},
        )
    ) as fm:
        fm.plot(
            "bump",
            0, 0,
            bump_data,
            time_col="time",
            value_col="score", 
            category_col="category",
            marker="o",
            linewidth=2,
            title=title,
        )

        # Add annotations
        ax = fm.get_axes(0, 0)
        add_ranking_labels(ax, bump_data, final_step_only)
        add_value_annotations(ax, bump_data, final_step_only)

        # Format x-axis appropriately
        if final_step_only and original_param_names is not None:
            # For final step mode with params, set x-axis labels to original param names
            ax.set_xticks(range(len(original_param_names)))
            ax.set_xticklabels(original_param_names)
            ax.set_xlabel("Model Size")
        elif not final_step_only:
            if x_axis == "tokens":
                ax.set_xscale("log")
                ax.set_xlabel("Token Count (log scale)")
                ax.set_xlim(2e8, 1.2e11)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: format_token_count(x))
                )
            else:
                x_values = sorted(bump_data["time"].unique())
                ax.set_xticks(x_values[::max(1, len(x_values) // 8)])
                ax.set_xticklabels([format_step_label(s) for s in ax.get_xticks()])
                ax.set_xlabel("Training Step")

    # Handle output
    save_dir = merged_config.get("save_dir")
    if save_dir:
        import os
        save_path = os.path.join(save_dir, f"bump_analysis_{x_axis}_{hue_by}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()