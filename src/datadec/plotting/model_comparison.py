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
        self.config.update({
            'line_col': 'params',           # Default: params as lines
            'style_col': None,               # Optional style encoding
            'subplot_col': 'data',           # For filtering purposes
            'title_prefix': 'Model Comparison',
            'params_filter': None,
            'data_filter': None,
        })
    
    def with_params(self, params_list: Optional[List[str]]):
        """
        Convenience method to set params_filter and apply filtering.
        
        Args:
            params_list: List of parameter values to keep
            
        Returns:
            self for method chaining
        """
        self.config['params_filter'] = params_list
        return self.filter_params(params_list)
    
    def with_data(self, data_list: Optional[List[str]]):
        """
        Convenience method to set data_filter and apply filtering.
        
        Args:
            data_list: List of data recipe values to keep
            
        Returns:
            self for method chaining
        """
        self.config['data_filter'] = data_list
        return self.filter_data(data_list)
    
    def build(self) -> Tuple[plt.Figure, FigureManager]:
        """
        Build the model comparison plot.
        
        Returns:
            Tuple of (matplotlib figure, FigureManager instance)
        """
        # Calculate layout based on metrics instead of data values
        n_metrics = len(self.metrics)
        ncols = self.config.get('ncols', min(3, n_metrics))  # Use config ncols or default
        nrows = int(np.ceil(n_metrics / ncols))
        
        # Calculate figure size
        figsize = self.config.get('figsize')
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        
        # Extract matplotlib kwargs for FigureManager
        matplotlib_kwargs = ['sharey', 'sharex', 'squeeze', 'subplot_kw', 'gridspec_kw']
        fig_kwargs = {k: self.config[k] for k in matplotlib_kwargs if k in self.config}
        
        # Create figure manager with matplotlib kwargs
        self.fm = FigureManager(rows=nrows, cols=ncols, figsize=figsize, **fig_kwargs)
        
        # Set overall title
        title = f"{self.config['title_prefix']}: Multiple Metrics vs {self.config['x_col']}"
        self.fm.fig.suptitle(title, fontsize=16)
        
        # Plot each metric
        for i, metric in enumerate(self.metrics):
            row = i // ncols
            col = i % ncols
            
            # Build list of columns we need for this metric
            key_columns = [self.config['x_col'], metric, self.config['line_col']]
            if self.config.get('style_col'):
                key_columns.append(self.config['style_col'])
            
            # Only keep columns that exist in the dataframe
            available_columns = [c for c in key_columns if c in self.plot_df.columns]
            
            # Filter data for this metric (remove NaN values)
            metric_data = self.plot_df[available_columns].dropna()
            
            if len(metric_data) == 0:
                continue
            
            # Sort data for consistent ordering
            if self.config.get('style_col'):
                metric_data = self._sort_data_for_consistency(metric_data, self.config['style_col'])
            metric_data = self._sort_data_for_consistency(metric_data, self.config['line_col'])
            
            # Apply sequential styling before plotting (only on first iteration to avoid resetting)
            if i == 0:  # Only set up cycles once for the entire figure
                # Use sequential colormap if specified and line_col is params
                if (self.config.get('colormap') and 
                    self.config['line_col'] == 'params'):
                    
                    # Get unique parameter values for color mapping
                    hue_values = self.plot_df[self.config['line_col']].unique().tolist()
                    color_map = self._generate_sequential_colors(
                        hue_values, 
                        self.config['colormap'],
                        color_range_min=self.config.get('color_range_min', 0.1),
                        color_range_max=self.config.get('color_range_max', 1.0)
                    )
                    
                    # Override the FigureManager's shared style cycles with our sequential colors
                    import itertools
                    from datadec.model_utils import param_to_numeric
                    
                    # Sort parameter values numerically to ensure proper progression
                    sorted_params = sorted(hue_values, key=param_to_numeric)
                    color_values = [color_map[val] for val in sorted_params]
                    
                    # Override the shared style cycles that the StyleEngine uses
                    if not hasattr(self.fm, '_shared_style_cycles') or self.fm._shared_style_cycles is None:
                        # Initialize if not already done
                        self.fm._get_shared_style_cycles()
                    
                    # Ensure _shared_style_cycles is properly initialized
                    if self.fm._shared_style_cycles is None:
                        self.fm._shared_style_cycles = {}
                    
                    # Replace the color cycle with our sequential colors
                    self.fm._shared_style_cycles['color'] = itertools.cycle(color_values)
                
                # Use sequential line styles if specified and style_col is params
                if (self.config.get('linestyle_sequence') and 
                    self.config.get('style_col') == 'params'):
                    
                    # Get unique parameter values for line style mapping
                    style_values = self.plot_df[self.config['style_col']].unique().tolist()
                    linestyle_map = self._generate_sequential_linestyles(
                        style_values,
                        linestyle_sequence=self.config.get('linestyle_sequence')
                    )
                    
                    # Override the FigureManager's line style cycle  
                    import itertools
                    from datadec.model_utils import param_to_numeric
                    
                    # Ensure shared style cycles are initialized 
                    if not hasattr(self.fm, '_shared_style_cycles') or self.fm._shared_style_cycles is None:
                        self.fm._get_shared_style_cycles()
                    if self.fm._shared_style_cycles is None:
                        self.fm._shared_style_cycles = {}
                    
                    sorted_styles = sorted(style_values, key=param_to_numeric)
                    linestyle_values = [linestyle_map[val] for val in sorted_styles]
                    
                    # Override both shared cycles and theme to ensure StyleEngine uses our sequence
                    self.fm._shared_style_cycles['linestyle'] = itertools.cycle(linestyle_values)
                    
                    # Also override the BASE_THEME's linestyle_cycle that StyleEngine uses
                    from dr_plotter.theme import BASE_THEME
                    BASE_THEME.styles['linestyle_cycle'] = itertools.cycle(linestyle_values)
                    
            
            # Use FigureManager's plot method for color coordination
            self.fm.plot(
                'line',
                row=row,
                col=col,
                data=metric_data,
                x=self.config['x_col'],
                y=metric,
                hue_by=self.config['line_col'],
                style_by=self.config.get('style_col'),
                title=metric,
                **self.config.get('plot_kwargs', {})
            )
            
            # Apply log scale if configured
            if self.config.get('log_scale'):
                ax = self.fm.get_axes(row, col)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel(f"{self.config['x_col']} (log scale)")
                ax.set_ylabel(f"{metric} (log scale)")
        
        # Hide unused subplots
        self._hide_unused_subplots(n_metrics, nrows * ncols)
        
        # Apply tight layout
        plt.tight_layout()
        
        return self.fm.fig, self.fm


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
    **kwargs
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
        plot_kwargs=kwargs
    )
    
    # Apply filters
    if params_filter is not None:
        builder.with_params(params_filter)
    if subplot_filter is not None:
        builder.with_data(subplot_filter)
    
    # Build and return
    return builder.build()