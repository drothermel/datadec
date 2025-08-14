from typing import Tuple

import pandas as pd

from datadec import constants as consts


def filter_by_max_step_to_use(df: pd.DataFrame) -> pd.DataFrame:
    max_step_filter = df["params"].map(consts.MAX_STEP_TO_USE)
    return df[df["step"] <= max_step_filter]


def merge_in_ds_and_model_details(
    merged_df: pd.DataFrame,
    dataset_details: pd.DataFrame,
    model_details: pd.DataFrame,
) -> pd.DataFrame:
    result = merged_df.merge(dataset_details, on="data", how="left")
    result = result.merge(model_details, on="params", how="left")
    return result


def get_max_ppl_vals(df: pd.DataFrame) -> pd.DataFrame:
    ppl_cols = [col for col in df.columns if col.endswith("ppl")]
    
    return df.groupby(["params", "data"])[ppl_cols].max().reset_index()


def set_step_val_to_max_ppl_val(df: pd.DataFrame, step: int = 0) -> pd.DataFrame:
    result = df.copy()
    result.loc[result["step"] == step, "value"] = result.loc[
        result["step"] == step
    ].apply(lambda row: row[f"{row['data']}-valppl"], axis=1)
    return result


def select_by_data_param_combos(
    df: pd.DataFrame,
    data_names: list[str] = None,
    param_strs: list[str] = None,
    data_param_combos: list[tuple] = None,
) -> pd.DataFrame:
    if data_param_combos:
        combined_filter = pd.Series([False] * len(df), index=df.index)
        for data, param in data_param_combos:
            combo_filter = (df["data"] == data) & (df["params"] == param)
            combined_filter = combined_filter | combo_filter
        return df[combined_filter]
    
    if data_names is None:
        data_names = consts.ALL_DATA_NAMES
    if param_strs is None:
        param_strs = consts.ALL_PARAM_STRS
    
    return df[df["data"].isin(data_names) & df["params"].isin(param_strs)]


def create_mean_std_df(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["params", "data", "step"]
    
    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col not in group_cols]
    
    mean_df = merged_df.groupby(group_cols)[agg_cols].mean().reset_index()
    std_df = merged_df.groupby(group_cols)[agg_cols].std().reset_index()
    
    return mean_df, std_df


def merge_ppl_and_dwn_dfs(ppl_df: pd.DataFrame, dwn_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([ppl_df, dwn_df], ignore_index=True)