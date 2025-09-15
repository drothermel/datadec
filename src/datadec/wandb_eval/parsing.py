import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from datadec.constants import HARDCODED_SIZE_MAPPING
from datadec.data import DataDecide
from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_constants as wconsts


def extract_dataset_from_model_path(model_path: str) -> Optional[str]:
    if pd.isna(model_path) or not isinstance(model_path, str):
        return None

    match = re.search(r"DataDecide-([^/]+?)(?:-\d+[MB])?/snapshots", model_path)
    if match:
        dataset_part = match.group(1)
        return dataset_part
    return None


def map_wandb_dataset_to_datadecide(wandb_dataset: str) -> Optional[str]:
    if not wandb_dataset:
        return None

    mapping = {
        "dolma1_7": "Dolma1.7",
        "dclm-baseline": "DCLM-Baseline",
        "dclm-baseline-25p-dolma1.7-75p": "DCLM-Baseline 25% / Dolma 75%",
        "dclm-baseline-50p-dolma1.7-50p": "DCLM-Baseline 50% / Dolma 50%",
        "dclm-baseline-75p-dolma1.7-25p": "DCLM-Baseline 75% / Dolma 25%",
        "dclm-baseline-qc-10p": "DCLM-Baseline (QC 10%)",
        "dclm-baseline-qc-20p": "DCLM-Baseline (QC 20%)",
        "dclm-baseline-qc-7p-fw2": "DCLM-Baseline (QC 7%, FW2)",
        "dclm-baseline-qc-7p-fw3": "DCLM-Baseline (QC 7%, FW3)",
        "dclm-baseline-qc-fw-10p": "DCLM-Baseline (QC FW 10%)",
        "dclm-baseline-qc-fw-3p": "DCLM-Baseline (QC FW 3%)",
    }

    return mapping.get(wandb_dataset)


def map_wandb_model_size_to_datadecide(model_size: int) -> Optional[str]:
    if pd.isna(model_size):
        return None

    closest_param = None
    min_diff = float("inf")
    for param_str, numeric_val in HARDCODED_SIZE_MAPPING.items():
        diff = abs(numeric_val - model_size)
        if diff < min_diff:
            min_diff = diff
            closest_param = param_str

    return closest_param


def add_datadecide_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model_name_or_path" in df.columns:
        df["wandb_dataset"] = df["model_name_or_path"].apply(
            extract_dataset_from_model_path
        )
        df["data"] = df["wandb_dataset"].apply(map_wandb_dataset_to_datadecide)
    else:
        df["data"] = None
    if "model_size" in df.columns:
        df["params"] = df["model_size"].apply(map_wandb_model_size_to_datadecide)
    else:
        df["params"] = None

    return df


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


def get_created_time_key(df: pd.DataFrame) -> str:
    for tk in wconsts.TIME_KEYS:
        if tk in df.columns:
            return tk
    assert False, "No time key found"


def filter_broken_initial_testing_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[get_created_time_key(df)] >= wconsts.EARLIEST_GOOD_RUN_DATE]


def filter_dpo_test_runs(df: pd.DataFrame) -> pd.DataFrame:
    if "wandb_tags" not in df.columns:
        return df

    dpo_mask = df["wandb_tags"].fillna("").str.contains("dpo_tune_cache", na=False)
    return df[~dpo_mask]


def drop_wandb_constant_ignored_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in wconsts.ALL_DROP_COLS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} problematic columns: {cols_to_drop}")
    return df


def split_obj_vs_nonobj_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    object_columns = []
    nonobject_columns = []
    for col in df.columns:
        if df[col].dtype == "object":
            object_columns.append(col)
        else:
            nonobject_columns.append(col)
    return object_columns, nonobject_columns


def filter_constant_and_nanconstant_cols(df: pd.DataFrame) -> list[str]:
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
                all_nan_columns.append(col)  # Only NaN values
            else:
                all_constant_columns.append(col)  # Only one constant value
        elif nunique_with_nan == 2 and has_nan:
            constant_or_nan_columns.append(col)  # One constant + NaN
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


def convert_objects_and_normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            non_null_data = df[col].dropna()
            if len(non_null_data) == 0:
                continue

            first_val = non_null_data.iloc[0]
            if isinstance(first_val, (dict, list, tuple, set)):
                try:
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if x is not None else None
                    )
                except Exception:
                    df[col] = df[col].astype(str)
                continue
            try:
                all_same_type = all(
                    type(val) is type(first_val) for val in non_null_data
                )
                if all_same_type:
                    if isinstance(first_val, bool):
                        df[col] = df[col].astype("boolean")
                    elif isinstance(first_val, int):
                        df[col] = df[col].astype("Int64")
                    elif isinstance(first_val, float):
                        non_nan_vals = df[col].dropna()
                        if len(non_nan_vals) > 0 and all(
                            val == int(val) for val in non_nan_vals
                        ):
                            df[col] = df[col].astype("Int64")
                        else:
                            df[col] = pd.to_numeric(df[col], errors="ignore")
                    elif isinstance(first_val, str):
                        df[col] = df[col].astype("string")
                else:
                    numeric_df = pd.to_numeric(df[col], errors="coerce")
                    if not numeric_df.isna().all():
                        numeric_vals = numeric_df.dropna()
                        if len(numeric_vals) > 0 and all(
                            val == int(val) for val in numeric_vals
                        ):
                            df[col] = numeric_df.astype("Int64")
                        else:
                            df[col] = numeric_df
            except Exception:
                continue

        if pd.api.types.is_float_dtype(df[col]):
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                try:
                    is_whole = all(val == int(val) for val in non_null_data)
                    if is_whole:
                        df[col] = df[col].astype("Int64")
                except (ValueError, OverflowError, TypeError):
                    pass
    return df


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


def assert_no_remaining_noise_columns(col_categories: dict) -> None:
    assert len(col_categories["all_nan"]) == 0, (
        f"Found {len(col_categories['all_nan'])} unexpected all-NaN columns: {col_categories['all_nan']}"
    )
    assert len(col_categories["all_constant"]) == 0, (
        f"Found {len(col_categories['all_constant'])} unexpected all-constant columns: {col_categories['all_constant']}"
    )
    assert len(col_categories["constant_or_nan"]) == 0, (
        f"Found {len(col_categories['constant_or_nan'])} unexpected constant-or-NaN columns: {col_categories['constant_or_nan']}"
    )


def categorize_columns_by_key_sets(all_cols):
    categorized = {name: [] for name in wconsts.KEY_SETS.keys()}
    remaining_cols = list(all_cols)
    for category_name, key_list in wconsts.KEY_SETS.items():
        matched_cols = [col for col in remaining_cols if col in key_list]
        categorized[category_name] = matched_cols
        remaining_cols = [col for col in remaining_cols if col not in matched_cols]
    return categorized, remaining_cols


def parse_and_clean_runs_df(runs_df):
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
        f"Found {len(object_cols)} unexpected object columns after conversion: {object_cols}"  # noqa: E501
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


def preprocess_object_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, oe_eval_cols = extract_oe_eval_metrics_cols(df)
    df = convert_objects_and_normalize_dtypes(df)
    return df, {"oe_eval_cols": oe_eval_cols}


def parse_wandb_tags(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if "wandb_tags" not in filtered_df.columns:
        return pd.DataFrame(index=filtered_df.index)
    all_tags = set()
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


def parse_oe_eval_metrics(filtered_df: pd.DataFrame, oe_cols: list) -> pd.DataFrame:
    if not oe_cols:
        return pd.DataFrame(index=filtered_df.index)
    all_metrics = {}
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


def fix_oe_metrics_to_final_step_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    oe_cols = [col for col in df.columns if col.startswith("oe_")]
    if not oe_cols:
        return df
    for run_id in df["run_id"].unique():
        run_mask = df["run_id"] == run_id
        run_data = df[run_mask]
        if len(run_data) > 1:
            max_step = run_data["step"].max()
            non_final_mask = run_mask & (df["step"] != max_step)
            df.loc[non_final_mask, oe_cols] = pd.NA
    return df


def copy_final_oe_to_pretraining_metrics(
    df: pd.DataFrame, pretraining_df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    wandb_eval_cols = [col for col in df.columns if col.startswith("oe_")]
    pretrain_eval_cols = [
        col
        for col in pretraining_df.columns
        if any(col == wandb_col[3:] for wandb_col in wandb_eval_cols)
    ]
    metric_mapping = {}
    for wandb_col in wandb_eval_cols:
        pretrain_equivalent = wandb_col[3:]
        if pretrain_equivalent in pretrain_eval_cols:
            metric_mapping[wandb_col] = pretrain_equivalent
    print(
        f"Found {len(metric_mapping)} matching metric pairs for continuous scaling curves"
    )
    for pretrain_col in metric_mapping.values():
        if pretrain_col not in df.columns:
            df[pretrain_col] = pd.NA
    for run_id in df["run_id"].unique():
        run_mask = df["run_id"] == run_id
        run_data = df[run_mask]
        final_row = run_data[
            run_data[[col for col in wandb_eval_cols if col in run_data.columns]]
            .notna()
            .any(axis=1)
        ]
        if len(final_row) > 0:
            final_row = final_row.iloc[-1]
            for wandb_col, pretrain_col in metric_mapping.items():
                if wandb_col in final_row and pd.notna(final_row[wandb_col]):
                    final_step_mask = run_mask & (df.index == final_row.name)
                    df.loc[final_step_mask, pretrain_col] = final_row[wandb_col]
    return df


def integrate_pretraining_data(unified_df: pd.DataFrame) -> pd.DataFrame:
    dd = DataDecide()
    pretraining_df = dd.mean_eval
    pretrain_max = (
        pretraining_df.groupby(["params", "data"])
        .agg({"tokens": "max", "compute": "max"})
        .reset_index()
    )
    pretrain_max.columns = [
        "params",
        "data",
        "pretrain_tokens_max",
        "pretrain_compute_max",
    ]
    result_df = unified_df.merge(pretrain_max, on=["params", "data"], how="left")
    result_df["pretraining"] = result_df["pretrain_tokens_max"].notna()
    result_df["cumulative_tokens"] = result_df["pretrain_tokens_max"].fillna(
        0
    ) + result_df["total_tokens"].fillna(0)
    if "total_compute_est" in result_df.columns:
        result_df["cumulative_compute"] = result_df["pretrain_compute_max"].fillna(
            0
        ) + result_df["total_compute_est"].fillna(0)
    result_df = fix_oe_metrics_to_final_step_only(result_df)
    result_df = copy_final_oe_to_pretraining_metrics(result_df, pretraining_df)
    result_df = add_pretraining_progression_rows(result_df, pretraining_df)
    return result_df


def add_pretraining_progression_rows(
    finetuning_df: pd.DataFrame, pretraining_df: pd.DataFrame
) -> pd.DataFrame:
    pretrain_combinations = pretraining_df[["params", "data"]].drop_duplicates()
    pretraining_rows = []
    for _, combo in pretrain_combinations.iterrows():
        params = combo["params"]
        data = combo["data"]
        matching_runs = finetuning_df[
            (finetuning_df["params"] == params)
            & (finetuning_df["data"] == data)
            & (finetuning_df["pretraining"])
        ]
        if len(matching_runs) == 0:
            continue
        pretrain_progression = pretraining_df[
            (pretraining_df["params"] == params) & (pretraining_df["data"] == data)
        ].copy()
        unique_runs = matching_runs["run_id"].unique()
        for run_id in unique_runs:
            run_metadata = matching_runs[matching_runs["run_id"] == run_id].iloc[0]
            for _, pretrain_row in pretrain_progression.iterrows():
                new_row = run_metadata.copy()
                new_row["total_tokens"] = pretrain_row["tokens"]
                new_row["cumulative_tokens"] = pretrain_row["tokens"]
                new_row["step"] = pretrain_row["step"]
                if "compute" in pretrain_row:
                    new_row["total_compute_est"] = pretrain_row["compute"]
                    new_row["cumulative_compute"] = pretrain_row["compute"]
                for col in pretrain_row.index:
                    if col in ["params", "data", "step", "tokens", "compute"]:
                        continue
                    if col in new_row.index:
                        new_row[col] = pretrain_row[col]
                oe_cols = [col for col in new_row.index if col.startswith("oe_")]
                for col in oe_cols:
                    new_row[col] = pd.NA
                new_row["pretraining_phase"] = True
                pretraining_rows.append(new_row)

    if pretraining_rows:
        pretrain_df_new = pd.DataFrame(pretraining_rows)
        finetuning_df["pretraining_phase"] = False
        combined_df = pd.concat([pretrain_df_new, finetuning_df], ignore_index=True)
        print(f"Added {len(pretrain_df_new)} pretraining progression rows")
        return combined_df
    else:
        finetuning_df["pretraining_phase"] = False
        return finetuning_df


def create_unified_df(runs_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    result = parse_and_clean_runs_df(runs_df)
    parsed_runs_df = rebuild_run_df(result["filtered_df"], result)
    history_clean = history_df.drop(columns=["project"])
    unified_df = history_clean.merge(parsed_runs_df, on="run_id", how="inner")
    unified_df = add_datadecide_columns(unified_df)
    return unified_df


def create_unified_df_with_pretraining(
    runs_df: pd.DataFrame, history_df: pd.DataFrame
) -> pd.DataFrame:
    unified_df = create_unified_df(runs_df, history_df)
    return integrate_pretraining_data(unified_df)


def create_and_save_pretrain_posttrain_df(
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    if save_path is None:
        save_path = wconsts.PRETRAIN_POSTTRAIN_DF_PATH
    print("Creating unified pretraining + finetuning DataFrame...")
    runs_df, history_df = analysis_helpers.load_df()
    final_df = create_unified_df_with_pretraining(runs_df, history_df)
    large_int_cols = [
        "total_tokens",
        "cumulative_tokens",
        "pretrain_tokens_max",
        "cumulative_compute",
        "pretrain_compute_max",
        "total_compute_est",
    ]
    for col in large_int_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype("float64")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_path}...")
    if save_path.endswith(".parquet"):
        final_df.to_parquet(save_path, index=False)
    else:
        final_df.to_pickle(save_path)
    print(f"✓ Saved {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
    print("✓ Contains pretraining + finetuning with continuous scaling curves")
    return final_df
