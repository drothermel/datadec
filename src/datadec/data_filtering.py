from typing import List, Union, Tuple, Optional
import pandas as pd
from datadec import DataDecide
from datadec.constants import PPL_TYPES, OLMES_TASKS, METRIC_NAMES


def get_filtered_data_split_by_params(
    dd: DataDecide,
    holdout_param: str,
    metric: str,
    exclude_data: Optional[List[str]] = None,
    exclude_params: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_filtered_df, all_validated_params, _ = get_filtered_data(
        dd,
        "all",
        "all",
        metric,
        exclude_params=exclude_params,
        exclude_data=exclude_data,
    )
    train_params = [p for p in all_validated_params if p not in [holdout_param]]
    train_df = all_filtered_df[all_filtered_df["params"].isin(train_params)]
    eval_df = all_filtered_df[all_filtered_df["params"] == holdout_param]
    assert len(train_df) > 0, f"No train data when excluding {exclude_params}"
    assert len(eval_df) > 0, f"No eval data for holdout param {holdout_param}"
    return train_df, eval_df


def get_all_possible_metric_columns() -> List[str]:
    all_metrics = PPL_TYPES.copy()

    for task in OLMES_TASKS:
        for metric_type in METRIC_NAMES:
            all_metrics.append(f"{task}_{metric_type}")

    return all_metrics


def filter_to_specific_metrics(
    df: pd.DataFrame, keep_metrics: Union[str, List[str]]
) -> pd.DataFrame:
    if isinstance(keep_metrics, str):
        keep_metrics = [keep_metrics]

    all_possible_metrics = get_all_possible_metric_columns()
    basic_cols = [col for col in df.columns if col not in all_possible_metrics]

    cols_to_keep = basic_cols + [
        metric for metric in keep_metrics if metric in df.columns
    ]

    return df[cols_to_keep]


def validate_metric(metric: str) -> str:
    if metric in PPL_TYPES:
        return "ppl"

    for task in OLMES_TASKS:
        for metric_type in METRIC_NAMES:
            olmes_metric = f"{task}_{metric_type}"
            if metric == olmes_metric:
                return "olmes"

    available_ppl = PPL_TYPES
    available_olmes_examples = [
        f"{task}_{metric_type}"
        for task in OLMES_TASKS[:3]
        for metric_type in METRIC_NAMES[:3]
    ]
    raise ValueError(
        f"Invalid metric '{metric}'. Available PPL metrics: {available_ppl}. "
        f"OLMES metrics follow pattern 'task_metric_type', examples: {available_olmes_examples}"
    )


def better_easy_index_df(df, validated_data, validated_params):
    return df[df["params"].isin(validated_params) & df["data"].isin(validated_data)]


def get_filtered_data(
    dd: DataDecide,
    param_size: str,
    data_recipe: str,
    metric: str,
    exclude_data: Union[str, List[str], None] = None,
    exclude_params: Union[str, List[str], None] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    metric_type = validate_metric(metric)
    validated_params = dd.select_params(param_size, exclude=exclude_params)
    validated_data = dd.select_data(data_recipe, exclude=exclude_data)
    filtered_df = dd.get_filtered_df(
        filter_types=["max_steps", metric_type],
        return_means=True,
        verbose=False,
    )
    final_df = better_easy_index_df(filtered_df, validated_data, validated_params)
    final_df = filter_to_specific_metrics(final_df, metric)

    return final_df, validated_params, validated_data


def print_dataset_summary(
    filtered_df: pd.DataFrame,
    validated_params: List[str],
    validated_data: List[str],
    metric: str,
    metric_type: str,
) -> None:
    print("\n=== Dataset Summary ===")
    print(f"Parameters: {validated_params}, Data: {validated_data}")
    print(f"Metric: {metric} (type: {metric_type})")
    print(f"Filtered dataset size: {len(filtered_df)} points")

    print("\nDataset composition:")
    print(filtered_df.groupby(["params", "data"]).size().reset_index(name="count"))

    print(
        f"\nMetric range: {filtered_df[metric].min():.4f} - {filtered_df[metric].max():.4f}"
    )
    print(
        f"Step range: {filtered_df['step'].min():.0f} - {filtered_df['step'].max():.0f}"
    )
