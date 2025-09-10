"""
Shared utilities for datadec plotting scripts.

Provides common functionality for CLI argument processing, data group resolution,
formatting, and theme creation across different plotting scripts.
"""

from __future__ import annotations

from typing import Any
import itertools

from dr_plotter import consts
from dr_plotter.theme import BUMP_PLOT_THEME, Theme

from datadec.constants import (
    BASE_AND_QC,
    BASE_RECIPES,
    BILLION,
    CUSTOM_RECIPE_FAMILIES,
    MILLION,
    OLMES_PERFORMANCE_RECIPE_CHUNKS,
    PPL_PERFORMANCE_RECIPE_CHUNKS,
    RECIPES_WITHOUT_ABLATIONS,
    THOUSAND,
)


def convert_domain_args_to_faceting(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert domain-specific CLI args to faceting system format.

    Maps intuitive domain args (--params, --data, --exclude-params, --exclude-data)
    to the standard dr_plotter faceting system's order/exclude parameters.

    Args:
        kwargs: CLI arguments dictionary

    Returns:
        Dictionary with faceting parameters (order, exclude) for use in FIXED_PARAMS
    """
    faceting_params = {}

    # Handle params selection
    params_list = list(kwargs.get("params", ["all"]))
    exclude_params = list(kwargs.get("exclude_params", []))

    if params_list != ["all"]:
        if "order" not in faceting_params:
            faceting_params["order"] = {}
        faceting_params["order"]["params"] = params_list

    if exclude_params:
        if "exclude" not in faceting_params:
            faceting_params["exclude"] = {}
        faceting_params["exclude"]["params"] = exclude_params

    # Handle data selection
    data_list = list(kwargs.get("data", ["all"]))
    exclude_data = list(kwargs.get("exclude_data", []))

    # Resolve named data groups
    if data_list != ["all"]:
        resolved_data = resolve_data_groups(data_list)
        if resolved_data != ["all"]:
            if "order" not in faceting_params:
                faceting_params["order"] = {}
            faceting_params["order"]["data"] = resolved_data

    if exclude_data:
        if "exclude" not in faceting_params:
            faceting_params["exclude"] = {}
        faceting_params["exclude"]["data"] = exclude_data

    return faceting_params


def resolve_data_groups(data_args: list[str]) -> list[str]:
    """
    Resolve named data groups to actual recipe lists.

    Expands named groups like "base", "base_qc" to their constituent recipes
    while preserving individual recipe names and the special "all" value.

    Args:
        data_args: List of data group names or individual recipes

    Returns:
        List of resolved recipe names, with duplicates removed
    """
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
            return data_args  # Return early to preserve "all"
        else:
            resolved_recipes.append(arg)

    return list(
        dict.fromkeys(resolved_recipes)
    )  # Remove duplicates while preserving order


def numerical_sort_key(param_size: str) -> float:
    """
    Convert parameter size string to numerical value for proper sorting.

    Handles sizes like "150M", "1.3B" by converting to float values where
    B (billion) = 1000 * M (million) for consistent ordering.

    Args:
        param_size: Parameter size string (e.g., "150M", "1.3B")

    Returns:
        Numerical value for sorting
    """
    if param_size.endswith("M"):
        return float(param_size[:-1])
    elif param_size.endswith("B"):
        return float(param_size[:-1]) * 1000
    else:
        return float(param_size)


def create_extended_color_palette() -> list[str]:
    """
    Create extended color palette for plots with many categories.

    Provides 40 distinct colors for cases where the default matplotlib
    color cycle isn't sufficient (e.g., bump plots with many data recipes).

    Returns:
        List of hex color codes
    """
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d3",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
        "#5254a3",
        "#8ca252",
        "#bd9e39",
        "#ad494a",
        "#a55194",
        "#6b6ecf",
        "#b5cf6b",
        "#e7ba52",
        "#d6616b",
        "#ce6dbd",
        "#de9ed6",
        "#31a354",
        "#756bb1",
        "#636363",
        "#969696",
    ]


def create_bump_theme_with_colors(
    num_categories: int, theme_name: str = "bump_custom"
) -> Theme:
    """
    Create custom bump plot theme with extended color palette.

    Uses the extended color palette to ensure sufficient colors for
    plots with many categories (e.g., many data recipes or model sizes).

    Args:
        num_categories: Number of categories that need distinct colors
        theme_name: Name for the custom theme

    Returns:
        Custom theme with extended color cycle
    """
    extended_colors = create_extended_color_palette()
    colors_to_use = extended_colors[: max(num_categories, len(extended_colors))]

    return Theme(
        name=theme_name,
        parent=BUMP_PLOT_THEME,
        **{
            consts.get_cycle_key("hue"): itertools.cycle(colors_to_use),
        },
    )


# Formatting utilities


def format_perplexity(ppl_value: float) -> str:
    """Format perplexity values for display."""
    return f"{ppl_value:.2f}"


def format_step_label(step: float) -> str:
    """Format training step values for axis labels."""
    if step >= THOUSAND:
        return f"{step / THOUSAND:.1f}k"
    else:
        return f"{int(step)}"


def format_token_count(token_count: float) -> str:
    """Format token count values for axis labels."""
    if token_count >= BILLION:
        return f"{token_count / BILLION:.1f}B"
    elif token_count >= MILLION:
        return f"{token_count / MILLION:.0f}M"
    elif token_count >= THOUSAND:
        return f"{token_count / THOUSAND:.0f}K"
    else:
        return f"{int(token_count)}"
