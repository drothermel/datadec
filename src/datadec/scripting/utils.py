from __future__ import annotations

from typing import Any

import pandas as pd

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
    faceting_params = {}
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
    data_list = list(kwargs.get("data", ["all"]))
    exclude_data = list(kwargs.get("exclude_data", []))
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


def align_to_common_start_point(
    df: pd.DataFrame, kwargs: dict[str, Any]
) -> pd.DataFrame:
    x_axis = kwargs.get("x_axis", "tokens")
    df["param_data_combo"] = df["params"].astype(str) + "-" + df["data"].astype(str)
    x_col = "tokens" if x_axis == "tokens" else "step"
    min_times_per_combo = df.groupby("param_data_combo")[x_col].min()
    common_start_time = min_times_per_combo.max()
    return df[df[x_col] >= common_start_time].copy()
