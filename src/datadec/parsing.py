import json
from typing import List

import pandas as pd

from datadec import constants as consts


def list_col_to_columns(orig_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    json_data = orig_df[col_name].str.replace("'", '"')
    df = pd.json_normalize(json_data.apply(json.loads))
    df = pd.concat([orig_df.drop(col_name, axis=1), df], axis=1)
    return df


def reorder_df_cols(df: pd.DataFrame, prefix_order: List[str]) -> pd.DataFrame:
    df = df.copy()
    return df[prefix_order + [col for col in df.columns if col not in prefix_order]]


def make_step_to_token_compute_df(dwn_df: pd.DataFrame) -> pd.DataFrame:
    step_data = dwn_df[["params", "step", "tokens", "compute"]].drop_duplicates()
    step_data = step_data.reset_index(drop=True)
    return step_data


def parse_perplexity_dataframe(ppl_df: pd.DataFrame) -> pd.DataFrame:
    df = ppl_df.copy()

    df["seed"] = df["seed"].map(consts.SEED_MAP)

    for old_name, new_name in consts.PPL_NAME_MAP.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    df = df.drop(columns=consts.PPL_DROP_COLS, errors="ignore")
    return df


def expand_downstream_metrics(dwn_df: pd.DataFrame) -> pd.DataFrame:
    print("Expanding downstream metrics column (this may take 2-5 minutes)...")
    df = list_col_to_columns(dwn_df, "metrics")
    return df


def complete_downstream_parsing(metrics_expanded_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_expanded_df.copy()

    df["seed"] = df["seed"].map(consts.SEED_MAP)

    df = df.drop(columns=consts.DROP_METRICS, errors="ignore")
    df = df.drop(columns=consts.DWN_DROP_COLS, errors="ignore")

    df = average_mmlu_metrics(df)
    df = pivot_task_metrics_to_columns(df)

    return df


def parse_downstream_dataframe(dwn_df: pd.DataFrame) -> pd.DataFrame:
    metrics_expanded = expand_downstream_metrics(dwn_df)
    return complete_downstream_parsing(metrics_expanded)


def average_mmlu_metrics(df: pd.DataFrame) -> pd.DataFrame:
    mmlu_cols = [col for col in df.columns if col.startswith("mmlu_") and col != "mmlu_average"]
    
    if mmlu_cols:
        df["mmlu_average"] = df[mmlu_cols].mean(axis=1)
    
    return df


def pivot_task_metrics_to_columns(dwn_df: pd.DataFrame) -> pd.DataFrame:
    id_vars = [col for col in consts.KEY_COLS if col in dwn_df.columns]
    value_vars = [col for col in dwn_df.columns if col not in consts.EXCLUDE_COLS]
    
    df_melted = dwn_df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="task_metric",
        value_name="value"
    )
    
    df_pivoted = df_melted.pivot_table(
        index=id_vars,
        columns="task_metric",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    df_pivoted.columns.name = None
    
    return df_pivoted