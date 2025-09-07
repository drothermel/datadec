from typing import List, Optional, Union

from datadec import constants as consts


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
