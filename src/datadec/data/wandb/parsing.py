from __future__ import annotations

import json

import pandas as pd

from datadec.data.wandb import wandb_constants as wconsts
from datadec.data.wandb.wandb_transforms import (
    add_datadecide_columns,
    convert_objects_and_normalize_dtypes,
    drop_wandb_constant_ignored_cols,
    filter_broken_initial_testing_runs,
    filter_dpo_test_runs,
)


def split_oe_cols_vs_rest(remaining_cols: list[str]) -> tuple[list[str], list[str]]:
    oe_cols = [col for col in remaining_cols if col.startswith("oe_eval_metrics/")]
    rest_cols = [col for col in remaining_cols if col not in oe_cols]
    return oe_cols, rest_cols


def split_pretrain_eval_cols_vs_rest(
    remaining_cols: list[str],
) -> tuple[list[str], list[str]]:
    pretrain_cols = [col for col in remaining_cols if col.startswith("pretrain_eval")]
    rest_cols = [col for col in remaining_cols if col not in pretrain_cols]
    return pretrain_cols, rest_cols


def split_obj_vs_nonobj_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    object_columns = []
    nonobject_columns = []
    for col in df.columns:
        if df[col].dtype == "object":
            object_columns.append(col)
        else:
            nonobject_columns.append(col)
    return object_columns, nonobject_columns


def filter_constant_and_nanconstant_cols(df: pd.DataFrame) -> dict[str, list[str]]:
    all_nan_columns = []
    all_constant_columns = []
    constant_or_nan_columns = []
    other_columns = []

    for col in df.columns:
        if df[col].dtype == "object":
            assert False, "Filter object columns before finding constants"

        nunique_with_nan = df[col].nunique(dropna=False)
        has_nan = df[col].isna().any()

        if nunique_with_nan == 0:
            all_nan_columns.append(col)
        elif nunique_with_nan == 1:
            if has_nan:
                all_nan_columns.append(col)
            else:
                all_constant_columns.append(col)
        elif nunique_with_nan == 2 and has_nan:
            constant_or_nan_columns.append(col)
        else:
            other_columns.append(col)

    return {
        "all_nan": all_nan_columns,
        "all_constant": all_constant_columns,
        "constant_or_nan": constant_or_nan_columns,
        "other": other_columns,
    }


def filter_pretrain_metric_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if "pretrain_eval" in col]


def extract_oe_eval_metrics_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    oe_eval_cols = [col for col in df.columns if col.startswith("oe_eval_metrics/")]
    if oe_eval_cols:
        remaining_df = df.drop(columns=oe_eval_cols)
        return remaining_df, oe_eval_cols
    return df, []


def assert_nan_only_columns(df: pd.DataFrame) -> None:
    for col in wconsts.KEY_SETS["nan_only_cols"]:
        if col in df.columns:
            assert df[col].isna().all(), (
                f"Column {col} expected to be all NaN but contains non-NaN values"
            )


def assert_constant_or_nan_columns(df: pd.DataFrame) -> None:
    for col in wconsts.CONSTANT_OR_NAN_COLS:
        if col in df.columns:
            nunique = df[col].nunique(dropna=False)
            assert nunique <= 2, (
                f"Column {col} expected to be constant or NaN but has {nunique} unique values"
            )
            if nunique == 2:
                assert df[col].isna().any(), (
                    f"Column {col} has 2 values but no NaN - should be constant or NaN"
                )


def assert_exact_match_columns(df: pd.DataFrame) -> None:
    for exact_group in wconsts.EXACT_MATCH_COLS:
        present_cols = [col for col in exact_group if col in df.columns]
        if len(present_cols) >= 2:
            base_col = present_cols[0]
            for other_col in present_cols[1:]:
                mask = df[base_col].notna() & df[other_col].notna()
                if mask.any():
                    matches = (df[base_col] == df[other_col]) & mask
                    assert matches[mask].all(), (
                        f"Exact match columns {base_col} ↔ {other_col} do not match exactly"
                    )


def assert_no_remaining_noise_columns(col_categories: dict[str, list[str]]) -> None:
    assert len(col_categories["all_nan"]) == 0, (
        f"Found {len(col_categories['all_nan'])} unexpected all-NaN columns: {col_categories['all_nan']}"
    )
    assert len(col_categories["all_constant"]) == 0, (
        f"Found {len(col_categories['all_constant'])} unexpected all-constant columns: {col_categories['all_constant']}"
    )
    assert len(col_categories["constant_or_nan"]) == 0, (
        f"Found {len(col_categories['constant_or_nan'])} unexpected constant-or-NaN columns: {col_categories['constant_or_nan']}"
    )


def categorize_columns_by_key_sets(
    all_cols: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    categorized: dict[str, list[str]] = {name: [] for name in wconsts.KEY_SETS.keys()}
    remaining_cols = list(all_cols)
    for category_name, key_list in wconsts.KEY_SETS.items():
        matched_cols = [col for col in remaining_cols if col in key_list]
        categorized[category_name] = matched_cols
        remaining_cols = [col for col in remaining_cols if col not in matched_cols]
    return categorized, remaining_cols


def parse_and_clean_runs_df(
    runs_df: pd.DataFrame,
) -> dict[str, list[str] | pd.DataFrame]:
    filtered_df = filter_broken_initial_testing_runs(runs_df)
    filtered_df = filter_dpo_test_runs(filtered_df)
    assert_nan_only_columns(filtered_df)
    assert_constant_or_nan_columns(filtered_df)
    filtered_df = drop_wandb_constant_ignored_cols(filtered_df)
    filtered_df = convert_objects_and_normalize_dtypes(filtered_df)
    assert_exact_match_columns(filtered_df)
    rest_cols = filtered_df.columns.tolist()
    oe_cols, rest_cols = split_oe_cols_vs_rest(rest_cols)
    pretrain_cols, rest_cols = split_pretrain_eval_cols_vs_rest(rest_cols)
    categorized_cols, rest_cols = categorize_columns_by_key_sets(rest_cols)
    object_cols, nonobject_cols = split_obj_vs_nonobj_cols(filtered_df[rest_cols])
    assert len(object_cols) == 0, (
        f"Found {len(object_cols)} unexpected object columns after conversion: {object_cols}"
    )
    if nonobject_cols:
        nonobj_df = filtered_df[nonobject_cols]
        col_categories = filter_constant_and_nanconstant_cols(nonobj_df)
        truly_uncategorized = col_categories["other"]
    else:
        col_categories = {
            "all_nan": [],
            "all_constant": [],
            "constant_or_nan": [],
            "other": [],
        }
        truly_uncategorized = []

    assert_no_remaining_noise_columns(col_categories)
    assert len(truly_uncategorized) == 0, (
        f"Found {len(truly_uncategorized)} truly uncategorized columns: {truly_uncategorized}"
    )
    return {
        "filtered_df": filtered_df,
        "object_cols": object_cols,
        "all_nan_cols": col_categories["all_nan"],
        "all_constant_cols": col_categories["all_constant"],
        "constant_or_nan_cols": col_categories["constant_or_nan"],
        "pretrain_cols": pretrain_cols,
        "oe_cols": oe_cols,
        "truly_uncategorized": truly_uncategorized,
        **categorized_cols,
    }


def preprocess_object_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df, oe_eval_cols = extract_oe_eval_metrics_cols(df)
    df = convert_objects_and_normalize_dtypes(df)
    return df, {"oe_eval_cols": oe_eval_cols}


def parse_wandb_tags(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if "wandb_tags" not in filtered_df.columns:
        return pd.DataFrame(index=filtered_df.index)
    all_tags: set[str] = set()
    tags_series = filtered_df["wandb_tags"].dropna()
    for tag_string in tags_series:
        if isinstance(tag_string, str):
            tags = [tag.strip() for tag in tag_string.split(",")]
            all_tags.update(tags)
    tag_df = pd.DataFrame(index=filtered_df.index)
    for tag in sorted(all_tags):
        col_name = f"{tag}_tag"
        tag_df[col_name] = False

        for idx, tag_string in filtered_df["wandb_tags"].items():
            if pd.notna(tag_string) and isinstance(tag_string, str):
                tags = [tag.strip() for tag in tag_string.split(",")]
                if tag in tags:
                    tag_df.loc[idx, col_name] = True

    return tag_df


def parse_oe_eval_metrics(
    filtered_df: pd.DataFrame, oe_cols: list[str]
) -> pd.DataFrame:
    if not oe_cols:
        return pd.DataFrame(index=filtered_df.index)
    all_metrics: dict[str, pd.Series] = {}
    for oe_col in oe_cols:
        if oe_col not in filtered_df.columns:
            continue
        task_name = oe_col.split("/")[-1]
        for idx, json_str in filtered_df[oe_col].items():
            if pd.isna(json_str):
                continue
            try:
                parsed = json.loads(json_str)
                if not isinstance(parsed, dict):
                    continue
                for metric_key, metric_value in parsed.items():
                    if metric_key == "task_config":
                        continue
                    col_name = f"oe_{task_name}_{metric_key}"
                    if isinstance(metric_value, dict):
                        continue
                    else:
                        if col_name not in all_metrics:
                            all_metrics[col_name] = pd.Series(
                                index=filtered_df.index, dtype="object"
                            )
                        all_metrics[col_name].loc[idx] = metric_value
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    if all_metrics:
        result_df = pd.DataFrame(all_metrics, index=filtered_df.index)
    else:
        result_df = pd.DataFrame(index=filtered_df.index)
    return result_df


def rebuild_run_df(filtered_df: pd.DataFrame, categorized_cols: dict) -> pd.DataFrame:
    result_df = pd.DataFrame(index=filtered_df.index)

    id_cols = categorized_cols.get("id_cols", [])
    for col in id_cols:
        if col in filtered_df.columns:
            result_df[col] = filtered_df[col]

    status_cols = categorized_cols.get("status_cols", [])
    for col in status_cols:
        if col in filtered_df.columns:
            result_df[col] = filtered_df[col]

    x_axis_cols = categorized_cols.get("x_axis_cols", [])
    for col in x_axis_cols:
        if col in filtered_df.columns:
            result_df[f"{col}_summary"] = filtered_df[col]

    summary_metrics_cols = categorized_cols.get("summary_metrics_cols", [])
    for col in summary_metrics_cols:
        if col in filtered_df.columns:
            result_df[f"{col}_summary"] = filtered_df[col]

    core_hpm_cols = categorized_cols.get("core_hpm_cols", [])
    for col in core_hpm_cols:
        if col in filtered_df.columns:
            if col == "learning_rate":
                result_df[f"{col}_hpm"] = filtered_df[col]
            else:
                result_df[col] = filtered_df[col]

    chat_cols = categorized_cols.get("chat_cols", [])
    for col in chat_cols:
        if col in filtered_df.columns:
            result_df[col] = filtered_df[col]

    eval_setting_cols = categorized_cols.get("eval_setting_cols", [])
    for col in eval_setting_cols:
        if col in filtered_df.columns:
            result_df[col] = filtered_df[col]

    dpo_hpm_cols = categorized_cols.get("dpo_hpm_cols", [])
    for col in dpo_hpm_cols:
        if col in filtered_df.columns:
            result_df[col] = filtered_df[col]

    dpo_eval_cols = categorized_cols.get("dpo_eval_cols", [])
    for col in dpo_eval_cols:
        if col in filtered_df.columns:
            result_df[f"{col}_summary"] = filtered_df[col]

    complex_cols = categorized_cols.get("complex_cols", [])
    if "wandb_tags" in complex_cols:
        tag_df = parse_wandb_tags(filtered_df)
        result_df = pd.concat([result_df, tag_df], axis=1)

    oe_cols = categorized_cols.get("oe_cols", [])
    if oe_cols:
        oe_metrics_df = parse_oe_eval_metrics(filtered_df, oe_cols)
        result_df = pd.concat([result_df, oe_metrics_df], axis=1)

    return result_df


def create_unified_df(runs_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    result = parse_and_clean_runs_df(runs_df)
    filtered_df = result["filtered_df"]
    assert isinstance(filtered_df, pd.DataFrame)
    parsed_runs_df = rebuild_run_df(filtered_df, result)
    history_clean = history_df.drop(columns=["project"])
    unified_df = history_clean.merge(parsed_runs_df, on="run_id", how="inner")
    unified_df = add_datadecide_columns(unified_df)
    return unified_df
