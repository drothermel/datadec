from typing import List, Optional, Tuple

import pandas as pd

from datadec import constants as consts
from datadec import validation


def print_shape(df: pd.DataFrame, msg: str = "", verbose: bool = False):
    if verbose:
        print(f"{msg} shape: {df.shape[0]:,} rows x {df.shape[1]:,} cols")


def filter_by_max_step_to_use(df: pd.DataFrame) -> pd.DataFrame:
    max_step_filter = df["params"].map(consts.MAX_STEP_TO_USE)
    return df[df["step"] <= max_step_filter]


def filter_ppl_rows(df: pd.DataFrame) -> pd.DataFrame:
    ppl_columns = [col for col in df.columns if col in consts.PPL_TYPES]
    assert len(ppl_columns) > 0, (
        f"No perplexity columns found in dataframe. Expected: {consts.PPL_TYPES}"
    )
    filtered_df = df.dropna(subset=ppl_columns, how="all")
    return filtered_df


def filter_olmes_rows(df: pd.DataFrame) -> pd.DataFrame:
    olmes_columns = [col for col in df.columns if col in consts.OLMES_METRICS]
    assert len(olmes_columns) > 0, (
        f"No OLMES metric columns found in dataframe. Expected tasks: {consts.OLMES_TASKS}"
    )
    filtered_df = df.dropna(subset=olmes_columns, how="all")
    return filtered_df


def select_by_data_param_combos(
    df: pd.DataFrame,
    data: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    data_param_combos: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    data = data or consts.ALL_DATA_NAMES
    params = params or consts.ALL_MODEL_SIZE_STRS
    if data_param_combos:
        combined_filter = pd.Series([False] * len(df), index=df.index)
        for data_name, param_name in data_param_combos:
            combo_filter = (df["data"] == data_name) & (df["params"] == param_name)
            combined_filter = combined_filter | combo_filter
        return df[combined_filter]
    return df[df["data"].isin(data) & df["params"].isin(params)]


def create_mean_std_df(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = consts.MEAN_ID_COLUMNS
    exclude_cols = consts.FULL_ID_COLUMNS
    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col not in exclude_cols]
    mean_df = merged_df.groupby(group_cols)[agg_cols].mean().reset_index()
    std_df = merged_df.groupby(group_cols)[agg_cols].std().reset_index()
    return mean_df, std_df


def melt_for_plotting(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    include_seeds: bool = True,
    drop_na: bool = True,
    id_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    id_columns = consts.FULL_ID_COLUMNS if id_columns is None else id_columns
    id_columns = id_columns if include_seeds else consts.MEAN_ID_COLUMNS
    if metrics is None:
        metrics = [col for col in df.columns if col in consts.ALL_KNOWN_METRICS]
    validation.validate_metrics(metrics, df_cols=list(df.columns))
    id_cols = (
        id_columns if include_seeds else [col for col in id_columns if col != "seed"]
    )
    assert all(col in df.columns for col in id_cols), (
        f"Invalid id_columns: {id_cols}. Available: {df.columns}"
    )
    melted_df = df.melt(
        id_vars=id_cols,
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    if drop_na:
        melted_df = melted_df.dropna(subset=["value"])
    return melted_df
