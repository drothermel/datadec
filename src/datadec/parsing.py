"""Data parsing and transformation functions for DataDecide datasets.

This module contains all functions for parsing and transforming raw DataDecide datasets
into analysis-ready formats, including perplexity data, downstream evaluation data,
and various data manipulation utilities.
"""

import json
from typing import List

import pandas as pd

from datadec import constants as consts


def list_col_to_columns(orig_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Convert a column containing list/dict strings to separate columns.
    
    Args:
        orig_df: DataFrame with a column containing JSON-like strings
        col_name: Name of the column to expand
        
    Returns:
        DataFrame with the specified column expanded into separate columns
    """
    json_data = orig_df[col_name].str.replace("'", '"')  # Single to double quotes
    df = pd.json_normalize(json_data.apply(json.loads))
    df = pd.concat([orig_df.drop(col_name, axis=1), df], axis=1)
    return df


def reorder_df_cols(df: pd.DataFrame, prefix_order: List[str]) -> pd.DataFrame:
    """Reorder DataFrame columns with specified columns first.
    
    Args:
        df: DataFrame to reorder
        prefix_order: List of column names to place at the beginning
        
    Returns:
        DataFrame with reordered columns
    """
    df = df.copy()
    return df[prefix_order + [col for col in df.columns if col not in prefix_order]]


def make_step_to_token_compute_df(dwn_df: pd.DataFrame) -> pd.DataFrame:
    """Create mapping from training steps to tokens and compute.
    
    Args:
        dwn_df: Downstream evaluation DataFrame with params, step, tokens, compute columns
        
    Returns:
        DataFrame with params, tokens_per_step, compute_per_step columns
    """
    assert all(
        [col in dwn_df.columns for col in ["params", "step", "tokens", "compute"]]
    )
    step_map = dwn_df[dwn_df["step"] > 0].copy()
    step_map["tokens_per_step"] = step_map["tokens"] / step_map["step"]
    step_map["compute_per_step"] = step_map["compute"] / step_map["step"]
    return step_map[["params", "tokens_per_step", "compute_per_step"]].drop_duplicates()


def parse_perplexity_dataframe(ppl_df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw perplexity evaluation DataFrame.
    
    Args:
        ppl_df: Raw perplexity DataFrame from DataDecide dataset
        
    Returns:
        Parsed DataFrame with renamed columns and mapped seeds
    """
    df = ppl_df.copy()
    df = df.drop(columns=consts.PPL_DROP_COLS)
    df = df.rename(columns=consts.PPL_NAME_MAP)
    df = reorder_df_cols(df, consts.KEY_COLS)
    df["seed"] = df["seed"].map(consts.SEED_MAP)
    return df


def parse_downstream_dataframe(dwn_df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw downstream evaluation DataFrame.
    
    Args:
        dwn_df: Raw downstream evaluation DataFrame from DataDecide dataset
        
    Returns:
        Parsed DataFrame with expanded metrics, averaged MMLU, and pivoted tasks
    """
    df = dwn_df.copy()
    df = df.drop(columns=consts.DWN_DROP_COLS)
    df = list_col_to_columns(df, "metrics")
    df = df.drop(columns=consts.DROP_METRICS)
    df = average_mmlu_metrics(df)
    df = pivot_task_metrics_to_columns(df)
    df = reorder_df_cols(df, consts.KEY_COLS)
    df["seed"] = df["seed"].map(consts.SEED_MAP)
    return df


def average_mmlu_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average MMLU metrics and add as a new task.
    
    Args:
        df: DataFrame with task-level evaluation metrics
        
    Returns:
        DataFrame with added mmlu_average task containing averaged metrics
    """
    mmlu_tasks = [
        task for task in df["task"].unique() if "mmlu" in task.lower()
    ]
    mmlu_df = df[df["task"].isin(mmlu_tasks)].drop(columns=["task"])
    metric_names = [
        col
        for col in df.columns
        if col not in consts.EXCLUDE_COLS and col not in consts.DROP_METRICS
    ]
    mmlu_avg = mmlu_df.groupby(consts.KEY_COLS).agg("mean").reset_index()
    mmlu_avg["task"] = "mmlu_average"
    return pd.concat([df, mmlu_avg], ignore_index=True)


def pivot_task_metrics_to_columns(dwn_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot task metrics from rows to columns.
    
    Args:
        dwn_df: DataFrame with tasks in rows and metrics as columns
        
    Returns:
        DataFrame with task-metric combinations as columns
    """
    pivoted_metrics = []
    metric_names = [
        col
        for col in dwn_df.columns
        if col not in consts.EXCLUDE_COLS and col not in consts.DROP_METRICS
    ]
    for metric_col in metric_names:
        pivoted = dwn_df.pivot_table(
            index=consts.KEY_COLS,
            columns="task",
            values=metric_col,
            aggfunc="first",
        )
        pivoted.columns = [f"{task}_{metric_col}" for task in pivoted.columns]
        pivoted_metrics.append(pivoted)
    new_dwn_df = pd.concat(pivoted_metrics, axis=1).reset_index()
    return new_dwn_df