"""DataFrame utility functions for DataDecide.

This module contains pure functions for common DataFrame operations including
filtering, merging, aggregation, and data manipulation tasks.
"""

from typing import Tuple

import pandas as pd

from datadec import constants as consts


def filter_by_max_step_to_use(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame by maximum step to use for each parameter size.

    Args:
        df: DataFrame with 'params' and 'step' columns

    Returns:
        Filtered DataFrame containing only rows where step <= max_step_to_use
    """
    df = df.copy()
    df["max_step_to_use"] = df["params"].map(consts.MAX_STEP_TO_USE)
    return df[df["step"] <= df["max_step_to_use"]]


def merge_in_ds_and_model_details(
    input_df: pd.DataFrame, ds_details_df: pd.DataFrame, model_details_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge dataset and model details into the input DataFrame.

    Args:
        input_df: Input DataFrame to merge details into
        ds_details_df: Dataset details DataFrame (joined on 'data' column)
        model_details_df: Model details DataFrame (joined on 'params' column)

    Returns:
        DataFrame with merged dataset and model details
    """
    return input_df.merge(
        ds_details_df,
        on="data",
        how="left",
    ).merge(
        model_details_df,
        on="params",
        how="left",
    )


def get_max_ppl_vals(df: pd.DataFrame) -> pd.DataFrame:
    """Get maximum perplexity values across all perplexity columns.

    Args:
        df: DataFrame containing perplexity columns

    Returns:
        DataFrame with maximum values for each perplexity column
    """
    ppl_cols = consts.PPL_TYPES
    return df[ppl_cols].max().reset_index()


def set_step_val_to_max_ppl_val(df: pd.DataFrame, step: int = 0) -> pd.DataFrame:
    """Set perplexity values at a specific step to maximum values for NaN entries.

    This function is used to handle missing perplexity values by filling them
    with the maximum observed values for each perplexity metric.

    Args:
        df: DataFrame with perplexity columns
        step: Step value to target for filling NaN values

    Returns:
        DataFrame with NaN perplexity values filled at the specified step
    """
    ppl_cols = consts.PPL_TYPES
    max_ppl_vals = get_max_ppl_vals(df)
    df = df.copy()
    step_mask = df["step"] == step

    for col in ppl_cols:
        na_mask = df[col].isna()
        df.loc[step_mask & na_mask, col] = max_ppl_vals[col][0]

    return df


def select_by_data_param_combos(
    df: pd.DataFrame,
    data_param_combos: list[tuple[str, str]],
    just_params: bool = False,
    just_data: bool = False,
) -> pd.DataFrame:
    """Filter DataFrame by specific data/parameter combinations.

    This function creates a mask for filtering DataFrames based on combinations
    of data recipes and model parameters, with options to filter by just one dimension.

    Args:
        df: DataFrame to filter
        data_param_combos: List of (data, params) tuples to match
        just_params: If True, filter only by params (ignore data)
        just_data: If True, filter only by data (ignore params)

    Returns:
        Filtered DataFrame containing only rows matching the specified combinations

    Raises:
        AssertionError: If both just_params and just_data are True
    """
    # Create a mask for each specific (data, params) combination
    mask = pd.Series([False] * len(df), index=df.index)

    for data, params in data_param_combos:
        if just_params:
            assert not just_data, "Cannot specify both just_params and just_data"
            combo_mask = df["params"] == params
        elif just_data:
            assert not just_params, "Cannot specify both just_params and just_data"
            combo_mask = df["data"] == data
        else:
            combo_mask = (df["data"] == data) & (df["params"] == params)
        mask = mask | combo_mask

    return df[mask]


def create_mean_std_df(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create mean and standard deviation DataFrames by averaging across seeds.

    This function groups the data by all key columns except 'seed' and computes
    mean and standard deviation across random seeds for all numeric columns.

    Args:
        merged_df: Input DataFrame with evaluation results across multiple seeds

    Returns:
        Tuple of (mean_df, std_df) containing aggregated statistics
    """
    group_cols_no_seed = [c for c in consts.KEY_COLS if c != "seed"]

    mean_df = (
        merged_df.drop(columns=["seed"])
        .groupby(group_cols_no_seed)
        .mean(numeric_only=True)
        .reset_index()
    )

    std_df = (
        merged_df.drop(columns=["seed"])
        .groupby(group_cols_no_seed)
        .std(numeric_only=True)
        .reset_index()
    )

    return mean_df, std_df
