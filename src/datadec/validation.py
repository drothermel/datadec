from typing import List, Optional, Union

from datadec import constants as consts

FILTER_TYPES = ["max_steps", "ppl", "olmes"]
METRIC_TYPES = ["ppl", "olmes"]


def _choices_valid(choices: Union[str, List[str]], valid_options: List[str]) -> bool:
    if choices == "all":
        return True
    choice_list = [choices] if isinstance(choices, str) else choices
    return all(choice in valid_options for choice in choice_list)


def _select_choices(
    choices: Union[str, List[str]],
    valid_options: List[str],
    exclude: Optional[List[str]] = None,
) -> List[str]:
    exclude = exclude or []
    if choices == "all":
        selected = [choice for choice in valid_options if choice not in exclude]
    else:
        choice_list = set([choices] if isinstance(choices, str) else choices)
        selected = [choice for choice in choice_list if choice not in exclude]
    return selected


def validate_filter_types(filter_types: List[str]) -> None:
    """Validate filter types against known options."""
    assert all(filter_type in FILTER_TYPES for filter_type in filter_types), (
        f"Invalid filter types: {filter_types}. Available: {FILTER_TYPES}"
    )


def validate_metric_type(metric_type: Optional[str]) -> None:
    """Validate metric type against known options."""
    if metric_type is not None:
        assert metric_type in METRIC_TYPES, (
            f"Unknown metric_type '{metric_type}'. Available: {METRIC_TYPES}"
        )


def validate_metrics(metrics: List[str]) -> None:
    """Validate metric names against known options."""
    assert all(metric in consts.ALL_KNOWN_METRICS for metric in metrics), (
        f"Unknown metrics: {metrics}. Available: {consts.ALL_KNOWN_METRICS}"
    )


def select_params(
    params: Union[str, List[str]] = "all",
    exclude: Optional[List[str]] = None,
) -> List[str]:
    assert _choices_valid(params, consts.ALL_MODEL_SIZE_STRS), (
        f"Invalid parameter size. Available: {consts.ALL_MODEL_SIZE_STRS}"
    )
    return _select_choices(
        choices=params,
        valid_options=consts.ALL_MODEL_SIZE_STRS,
        exclude=exclude,
    )


def select_data(
    data: Union[str, List[str]] = "all",
    exclude: Optional[List[str]] = None,
) -> List[str]:
    assert _choices_valid(data, consts.ALL_DATA_NAMES), (
        f"Invalid data recipe. Available: {consts.ALL_DATA_NAMES}"
    )
    return _select_choices(
        choices=data, valid_options=consts.ALL_DATA_NAMES, exclude=exclude
    )
