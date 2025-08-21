"""
Plotting functionality for datadec using dr_plotter.
"""

from typing import Optional, List, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    import dr_plotter.api as drp
    from dr_plotter.figure import FigureManager
except ImportError:
    raise ImportError(
        "dr_plotter is required for plotting functionality. "
        "Please install it from the dr_plotter repository."
    )

from .model_utils import param_to_numeric


def _sort_params_values(values: List[str]) -> List[str]:
    """Sort parameter values by numeric value for proper ordering."""
    return sorted(values, key=param_to_numeric)


def plot_scaling_curves(
    df: pd.DataFrame,
    x_col: str = "tokens",
    y_col: str = "pile-valppl", 
    line_col: str = "params",        # Creates different colored lines (hue)
    subplot_col: str = "data",       # Creates different subplots
    style_col: Optional[str] = None, # Optional line style encoding
    params_filter: Optional[List] = None,    # Filter to specific param values
    subplot_filter: Optional[List] = None,   # Filter to specific data recipes  
    figsize: Optional[Tuple[int, int]] = None,
    ncols: int = 2,  # Subplot layout
    title_prefix: str = "Scaling Curves",
    log_scale: bool = True,          # Use log/log scale for both axes
    **kwargs
) -> Tuple[plt.Figure, FigureManager]:
    """
    Plot scaling curves with flexible configuration using dr_plotter.
    
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
        **kwargs: Additional arguments passed to dr_plotter.line()
        
    Returns:
        Tuple of (figure, FigureManager)
    """
    # Filter data
    plot_df = df.copy()
    
    # Apply filters based on column names, not parameter names
    if params_filter is not None:
        # params_filter should apply to whichever column contains "params" values
        if "params" in line_col:
            plot_df = plot_df[plot_df[line_col].isin(params_filter)]
        elif "params" in subplot_col:
            plot_df = plot_df[plot_df[subplot_col].isin(params_filter)]
    
    if subplot_filter is not None:
        # subplot_filter should apply to whichever column contains "data" values
        if "data" in line_col:
            plot_df = plot_df[plot_df[line_col].isin(subplot_filter)]
        elif "data" in subplot_col:
            plot_df = plot_df[plot_df[subplot_col].isin(subplot_filter)]
    
    # Get unique subplot values and calculate grid layout
    subplot_values = plot_df[subplot_col].unique()
    # Sort params values numerically if subplot_col contains params
    if "params" in subplot_col:
        subplot_values = _sort_params_values(subplot_values)
    else:
        subplot_values = sorted(subplot_values)
    n_subplots = len(subplot_values)
    nrows = int(np.ceil(n_subplots / ncols))
    
    # Set default figure size if not provided
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    # Create figure manager
    fm = FigureManager(rows=nrows, cols=ncols, figsize=figsize)
    
    # Set overall title
    fm.fig.suptitle(f"{title_prefix}: {y_col} vs {x_col}", fontsize=16)
    
    # FigureManager will automatically coordinate colors across subplots!
    
    # Create plots for each subplot value using FigureManager for color coordination
    for i, subplot_val in enumerate(subplot_values):
        row = i // ncols
        col = i % ncols
        
        # Filter data for this subplot
        subplot_data = plot_df[plot_df[subplot_col] == subplot_val]
        
        if len(subplot_data) == 0:
            continue
        
        # Sort data by line_col values if it contains params for consistent legend ordering
        if "params" in line_col:
            line_values = _sort_params_values(subplot_data[line_col].unique())
            # Reorder data to match sorted line values
            subplot_data = subplot_data.set_index(line_col).loc[line_values].reset_index()
            
        # Use FigureManager's plot method for automatic color coordination!
        fm.plot(
            "line",
            row=row,
            col=col,
            data=subplot_data,
            x=x_col,
            y=y_col,
            hue_by=line_col,
            style_by=style_col,
            title=f"{subplot_col}={subplot_val}",
            **kwargs
        )
        
        # Apply log scale and update labels if requested
        if log_scale:
            ax = fm.get_axes(row=row, col=col)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f"{x_col} (log scale)")
            ax.set_ylabel(f"{y_col} (log scale)")
    
    # Hide unused subplots
    for i in range(n_subplots, nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = fm.get_axes(row=row, col=col)
        ax.set_visible(False)
    
    plt.tight_layout()
    return fm.fig, fm


def plot_model_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    x_col: str = "tokens",
    line_col: str = "params",
    style_col: Optional[str] = None,  # NEW: Optional line style encoding
    subplot_col: str = "data",  # What creates subplots (when used)
    params_filter: Optional[List] = None,
    subplot_filter: Optional[List] = None,  # Filter for subplot values
    figsize: Optional[Tuple[int, int]] = None,
    log_scale: bool = True,
    **kwargs
) -> Tuple[plt.Figure, FigureManager]:
    """
    Plot multiple metrics for model comparison, with each metric in its own subplot.
    
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
        **kwargs: Additional arguments passed to dr_plotter.line()
        
    Returns:
        Tuple of (figure, FigureManager)
    """
    # Apply same filtering logic as plot_scaling_curves
    plot_df = df.copy()
    
    # Apply filters based on column names, not parameter names
    if params_filter is not None:
        # params_filter should apply to whichever column contains "params" values
        if "params" in line_col:
            plot_df = plot_df[plot_df[line_col].isin(params_filter)]
        elif style_col and "params" in style_col:
            plot_df = plot_df[plot_df[style_col].isin(params_filter)]
        elif "params" in subplot_col:
            plot_df = plot_df[plot_df[subplot_col].isin(params_filter)]
    
    if subplot_filter is not None:
        # subplot_filter should apply to whichever column contains "data" values
        if "data" in line_col:
            plot_df = plot_df[plot_df[line_col].isin(subplot_filter)]
        elif "data" in subplot_col:
            plot_df = plot_df[plot_df[subplot_col].isin(subplot_filter)]
    
    # Calculate grid layout
    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    
    # Set default figure size if not provided
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    # Create figure manager
    fm = FigureManager(rows=nrows, cols=ncols, figsize=figsize)
    fm.fig.suptitle(f"Model Comparison: Multiple Metrics vs {x_col}", fontsize=16)
    
    # Create plots for each metric
    for i, metric in enumerate(metrics):
        row = i // ncols
        col = i % ncols
        
        # Filter data for this specific metric (remove NaN values)
        key_columns = [x_col, metric, line_col]
        if style_col:
            key_columns.append(style_col)
        if subplot_col and subplot_col not in key_columns:
            key_columns.append(subplot_col)
        available_columns = [c for c in key_columns if c in plot_df.columns]
        metric_data = plot_df[available_columns].dropna()
        
        if len(metric_data) == 0:
            continue
        
        # Sort data by params column for consistent ordering (whether in line_col or style_col)
        if style_col and "params" in style_col:
            # If params is in style_col, sort by that
            params_values = _sort_params_values(metric_data[style_col].unique())
            metric_data = metric_data.set_index(style_col).loc[params_values].reset_index()
        elif "params" in line_col:
            # If params is in line_col, sort by that
            line_values = _sort_params_values(metric_data[line_col].unique())
            metric_data = metric_data.set_index(line_col).loc[line_values].reset_index()
            
        # Use FigureManager's plot method for color coordination
        fm.plot(
            "line",
            row=row,
            col=col,
            data=metric_data,
            x=x_col,
            y=metric,
            hue_by=line_col,
            style_by=style_col,
            title=metric,
            **kwargs
        )
        
        # Apply log scale and update labels if requested
        if log_scale:
            ax = fm.get_axes(row=row, col=col)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f"{x_col} (log scale)")
            ax.set_ylabel(f"{metric} (log scale)")
    
    # Hide unused subplots
    for i in range(n_metrics, nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = fm.get_axes(row=row, col=col)
        ax.set_visible(False)
    
    plt.tight_layout()
    return fm.fig, fm