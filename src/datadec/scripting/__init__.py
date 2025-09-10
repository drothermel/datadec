"""
DataDecide scripting utilities.

Provides utilities for CLI scripts, visualization helpers, and plot-specific functionality.
Organized into general utilities and specialized modules for different plot types.
"""

# Import key utilities for easy access
from .utils import (
    align_to_common_start_point,
    convert_domain_args_to_faceting,
    resolve_data_groups,
    numerical_sort_key,
    format_perplexity,
    format_step_label,
    format_token_count,
)

from .bump_utils import (
    add_bump_ranking_labels,
    add_bump_value_annotations,
    prepare_bump_ranking_data,
    prepare_datadecide_bump_data,
    create_bump_theme_with_colors,
    render_bump_plot,
)

__all__ = [
    # General utilities
    "align_to_common_start_point",
    "convert_domain_args_to_faceting",
    "resolve_data_groups",
    "numerical_sort_key",
    "format_perplexity",
    "format_step_label",
    "format_token_count",
    # Bump utilities
    "add_bump_ranking_labels",
    "add_bump_value_annotations",
    "prepare_bump_ranking_data",
    "prepare_datadecide_bump_data",
    "create_bump_theme_with_colors",
    "render_bump_plot",
]
