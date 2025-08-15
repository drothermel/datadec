from typing import List, Optional, Tuple, Union

import pandas as pd

from datadec import constants as consts


def print_shape(df: pd.DataFrame, msg: str = "", verbose: bool = False):
    if verbose:
        print(f"{msg} shape: {df.shape[0]:,} rows x {df.shape[1]:,} cols")


def filter_by_max_step_to_use(df: pd.DataFrame) -> pd.DataFrame:
    max_step_filter = df["params"].map(consts.MAX_STEP_TO_USE)
    return df[df["step"] <= max_step_filter]


def select_by_data_param_combos(
    df: pd.DataFrame,
    data_names: Optional[Union[str, List[str]]] = None,
    param_strs: Optional[Union[str, List[str]]] = None,
    data_param_combos: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    data_names = consts.ALL_DATA_NAMES if data_names is None else data_names
    data_names = [data_names] if isinstance(data_names, str) else data_names
    param_strs = consts.ALL_PARAM_STRS if param_strs is None else param_strs
    param_strs = [param_strs] if isinstance(param_strs, str) else param_strs

    if data_param_combos:
        combined_filter = pd.Series([False] * len(df), index=df.index)
        for data, param in data_param_combos:
            combo_filter = (df["data"] == data) & (df["params"] == param)
            combined_filter = combined_filter | combo_filter
        return df[combined_filter]
    return df[df["data"].isin(data_names) & df["params"].isin(param_strs)]


def create_mean_std_df(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["params", "data", "step"]

    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col not in group_cols]

    mean_df = merged_df.groupby(group_cols)[agg_cols].mean().reset_index()
    std_df = merged_df.groupby(group_cols)[agg_cols].std().reset_index()

    return mean_df, std_df
