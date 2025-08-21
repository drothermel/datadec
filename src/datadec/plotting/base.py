"""
Base class for all plot builders with common functionality.
"""

from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from dr_plotter.figure import FigureManager
except ImportError:
    raise ImportError(
        "dr_plotter is required for plotting functionality. "
        "Please install it from the dr_plotter repository."
    )

from ..model_utils import param_to_numeric


def _sort_params_values(values: List[str]) -> List[str]:
    """Sort parameter values by numeric value for proper ordering."""
    return sorted(values, key=param_to_numeric)


class BasePlotBuilder:
    """Base class for all plot builders with common functionality."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the base plot builder.
        
        Args:
            df: DataFrame containing the data to plot
        """
        self.df = df
        self.plot_df = df.copy()
        self.fm = None
        
        # Common default configuration
        self.config = {
            'x_col': 'tokens',
            'ncols': 2,
            'log_scale': True,
            'figsize_per_subplot': (5, 4),
            'figsize': None  # Can override calculated size
        }
    
    def clean_data(self, key_columns: Optional[List[str]] = None, dropna: bool = True):
        """
        Clean data by removing NaN values in key columns.
        
        Args:
            key_columns: List of columns to check for NaN values
            dropna: Whether to drop NaN values
            
        Returns:
            self for method chaining
        """
        if dropna and key_columns:
            available = [c for c in key_columns if c in self.plot_df.columns]
            if available:
                initial_shape = self.plot_df.shape
                self.plot_df = self.plot_df.dropna(subset=available)
                if self.config.get('verbose'):
                    print(f"After NaN filtering: {self.plot_df.shape} (removed {initial_shape[0] - self.plot_df.shape[0]} rows)")
        return self
    
    def filter_params(self, params_list: Optional[List[str]]):
        """
        Smart filtering that checks both line_col and subplot_col for params.
        
        Args:
            params_list: List of parameter values to keep
            
        Returns:
            self for method chaining
        """
        if params_list and 'line_col' in self.config and 'subplot_col' in self.config:
            if 'params' in self.config['line_col']:
                self.plot_df = self.plot_df[self.plot_df[self.config['line_col']].isin(params_list)]
            elif 'params' in self.config['subplot_col']:
                self.plot_df = self.plot_df[self.plot_df[self.config['subplot_col']].isin(params_list)]
        return self
    
    def filter_data(self, data_list: Optional[List[str]]):
        """
        Smart filtering for data recipes.
        
        Args:
            data_list: List of data recipe values to keep
            
        Returns:
            self for method chaining
        """
        if data_list and 'line_col' in self.config and 'subplot_col' in self.config:
            if 'data' in self.config['line_col']:
                self.plot_df = self.plot_df[self.plot_df[self.config['line_col']].isin(data_list)]
            elif 'data' in self.config['subplot_col']:
                self.plot_df = self.plot_df[self.plot_df[self.config['subplot_col']].isin(data_list)]
        return self
    
    def configure(self, **kwargs):
        """
        Update configuration with custom values.
        
        Args:
            **kwargs: Configuration key-value pairs to update
            
        Returns:
            self for method chaining
        """
        self.config.update(kwargs)
        return self
    
    def validate_data(self, verbose: bool = False):
        """
        Validate and display data information.
        
        Args:
            verbose: Whether to print validation information
            
        Returns:
            self for method chaining
        """
        if verbose:
            print(f"Data shape: {self.plot_df.shape}")
            if 'params' in self.plot_df.columns:
                print(f"Unique params: {sorted(self.plot_df['params'].unique())}")
            if 'data' in self.plot_df.columns:
                data_values = sorted(self.plot_df['data'].unique())
                print(f"Unique data: {data_values[:10]}..." if len(data_values) > 10 else f"Unique data: {data_values}")
        return self
    
    def _calculate_layout(self, subplot_col: str) -> Tuple[List, int, int, Tuple[int, int], dict]:
        """
        Common layout calculation.
        
        Args:
            subplot_col: Column to use for creating subplots
            
        Returns:
            Tuple of (subplot_values, nrows, ncols, figsize, fig_kwargs)
        """
        subplot_values = self.plot_df[subplot_col].unique()
        
        # Sort params values if needed
        if 'params' in subplot_col:
            subplot_values = _sort_params_values(subplot_values)
        else:
            subplot_values = sorted(subplot_values)
        
        n_subplots = len(subplot_values)
        ncols = self.config['ncols']
        nrows = int(np.ceil(n_subplots / ncols))
        
        # Calculate figure size if not explicitly set
        figsize = self.config.get('figsize')
        if figsize is None:
            figsize = (
                self.config['figsize_per_subplot'][0] * ncols,
                self.config['figsize_per_subplot'][1] * nrows
            )
        
        # Extract matplotlib kwargs for FigureManager
        matplotlib_kwargs = ['sharey', 'sharex', 'squeeze', 'subplot_kw', 'gridspec_kw']
        fig_kwargs = {k: self.config[k] for k in matplotlib_kwargs if k in self.config}
        
        return subplot_values, nrows, ncols, figsize, fig_kwargs
    
    def _apply_log_scale(self, ax):
        """
        Apply log scale to axes if configured.
        
        Args:
            ax: Matplotlib axes object
        """
        if self.config.get('log_scale'):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(f"{self.config.get('x_col', 'x')} (log scale)")
            
            # Get the y label - might be from config or from the specific implementation
            y_label = self.config.get('y_col', 'y')
            ax.set_ylabel(f"{y_label} (log scale)")
    
    def _hide_unused_subplots(self, n_used: int, total_subplots: int):
        """
        Hide unused subplots in grid.
        
        Args:
            n_used: Number of subplots actually used
            total_subplots: Total number of subplots in grid
        """
        for i in range(n_used, total_subplots):
            row = i // self.config['ncols']
            col = i % self.config['ncols']
            ax = self.fm.get_axes(row, col)
            ax.set_visible(False)
    
    def _sort_data_for_consistency(self, subset: pd.DataFrame, sort_col: str) -> pd.DataFrame:
        """
        Sort data by params values if the column contains params for consistent legend ordering.
        
        Args:
            subset: DataFrame subset to sort
            sort_col: Column to check and potentially sort by
            
        Returns:
            Sorted DataFrame
        """
        if 'params' in sort_col and sort_col in subset.columns:
            unique_values = subset[sort_col].unique()
            sorted_values = _sort_params_values(unique_values)
            # Only reorder if we have the values to reorder by
            if len(sorted_values) > 0:
                try:
                    subset = subset.set_index(sort_col).loc[sorted_values].reset_index()
                except KeyError:
                    # Some values might be missing, just return original
                    pass
        return subset
    
    def build(self):
        """
        Build the plot. Must be implemented by subclasses.
        
        Returns:
            Tuple of (figure, FigureManager)
        """
        raise NotImplementedError("Subclasses must implement build()")
    
    @classmethod
    def from_datadecide(cls, dd, dataframe_name: str = 'mean_eval', verbose: bool = False):
        """
        Constructor that integrates with DataDecide.
        
        Args:
            dd: DataDecide instance
            dataframe_name: Name of dataframe to load
            verbose: Whether to print loading information
            
        Returns:
            New instance of the builder
        """
        df = dd.load_dataframe(dataframe_name)
        if verbose:
            print(f"Loaded {dataframe_name} with shape: {df.shape}")
        return cls(df)