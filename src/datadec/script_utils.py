"""Scripting utilities for DataDecide.

Provides convenient functions for parameter and data recipe selection in scripts,
with support for validation, "all" keyword, exclusion lists, and proper sorting.
"""

from typing import Callable, List, Optional, Union

from datadec import constants as consts
from datadec.model_utils import param_to_numeric


def select_choices(
    choices: Union[str, List[str]],
    valid_options: List[str],
    name: str,
    exclude: Optional[List[str]] = None,
    sort_key: Optional[Callable] = None,
) -> List[str]:
    """Generic choice selection with validation, "all" support, and exclusion.

    Args:
        choices: Single choice, list of choices, or "all" for all options
        valid_options: List of valid options to choose from
        name: Name of the choice type for error messages
        exclude: List of choices to exclude from final selection
        sort_key: Optional function for sorting results

    Returns:
        List of validated, deduplicated, and sorted choices

    Raises:
        ValueError: If any choice is invalid

    Examples:
        >>> select_choices("option1", ["option1", "option2"], "test")
        ["option1"]
        >>> select_choices("all", ["option1", "option2"], "test", exclude=["option1"])
        ["option2"]
    """
    exclude = exclude or []

    # Handle "all" keyword
    if choices == "all":
        selected = valid_options.copy()
    else:
        # Normalize to list
        if isinstance(choices, str):
            choices = [choices]

        # Validate each choice
        selected = []
        for choice in choices:
            if choice == "all":
                selected.extend(valid_options)
            elif choice in valid_options:
                selected.append(choice)
            else:
                raise ValueError(
                    f"Invalid {name} '{choice}'. Available: {valid_options}"
                )

    # Apply exclusions
    selected = [choice for choice in selected if choice not in exclude]

    # Deduplicate while preserving order
    seen = set()
    deduplicated = []
    for choice in selected:
        if choice not in seen:
            deduplicated.append(choice)
            seen.add(choice)

    # Sort if sort_key provided
    if sort_key:
        deduplicated.sort(key=sort_key)

    return deduplicated


def select_params(
    params: Union[str, List[str]] = "all",
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Select and validate model parameter sizes.

    Args:
        params: Parameter size(s) to select, or "all" for all available
        exclude: Parameter sizes to exclude from selection

    Returns:
        List of validated parameter sizes, sorted by numeric value

    Examples:
        >>> select_params("150M")
        ["150M"]
        >>> select_params(["150M", "1B"])
        ["150M", "1B"]
        >>> select_params("all", exclude=["4M", "6M"])
        ["8M", "10M", "14M", ...]
    """
    return select_choices(
        choices=params,
        valid_options=consts.ALL_MODEL_SIZE_STRS,
        name="parameter size",
        exclude=exclude,
        sort_key=param_to_numeric,
    )


def select_data(
    data: Union[str, List[str]] = "all",
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Select and validate data recipe names.

    Args:
        data: Data recipe name(s) to select, or "all" for all available
        exclude: Data recipes to exclude from selection

    Returns:
        List of validated data recipe names, in original order

    Examples:
        >>> select_data("C4")
        ["C4"]
        >>> select_data(["C4", "Dolma1.7"])
        ["C4", "Dolma1.7"]
        >>> select_data("all", exclude=["C4"])
        ["DCLM-Baseline", "Dolma1.7", ...]
    """
    return select_choices(
        choices=data,
        valid_options=consts.ALL_DATA_NAMES,
        name="data recipe",
        exclude=exclude,
        sort_key=None,  # Keep original order for data recipes
    )


def validate_param_data_combination(
    param: str,
    data: str,
    available_combinations: Optional[List[tuple]] = None,
) -> bool:
    """Validate that a parameter-data combination exists in the dataset.

    Args:
        param: Parameter size to validate
        data: Data recipe to validate
        available_combinations: Optional list of (param, data) tuples that exist

    Returns:
        True if combination is valid, False otherwise

    Note:
        If available_combinations is None, only validates individual param/data
        existence, not their combination.
    """
    # Validate individual components first
    try:
        select_params(param)
        select_data(data)
    except ValueError:
        return False

    # If combination list provided, check specific combination
    if available_combinations is not None:
        return (param, data) in available_combinations

    return True
