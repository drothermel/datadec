"""
Plotting functionality for datadec using dr_plotter.

This module provides builders for creating various types of plots:
- ScalingPlotBuilder: For scaling curve plots with subplots
- ModelComparisonBuilder: For multi-metric comparison plots
"""

from .scaling import ScalingPlotBuilder, plot_scaling_curves
from .model_comparison import ModelComparisonBuilder, plot_model_comparison
from .base import BasePlotBuilder

__all__ = [
    # Builders (recommended)
    'ScalingPlotBuilder',
    'ModelComparisonBuilder',
    'BasePlotBuilder',
    
    # Convenience functions (backward compatibility)
    'plot_scaling_curves',
    'plot_model_comparison',
]