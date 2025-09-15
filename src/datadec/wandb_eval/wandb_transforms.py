import json
import re
from typing import Dict, List, Optional, Set

import pandas as pd

from datadec import constants as consts
from datadec.wandb_eval import wandb_constants as wconsts


def extract_dataset_from_model_path(model_path: str) -> Optional[str]:
    if pd.isna(model_path) or not isinstance(model_path, str):
        return None
    match = re.search(r"DataDecide-([^/]+?)(?:-\d+[MB])?/snapshots", model_path)
    if match:
        return match.group(1)
    return None


def map_wandb_dataset_to_datadecide(wandb_dataset: str) -> Optional[str]:
    if not wandb_dataset:
        return None
    return wconsts.WANDB_DATASET_TO_DATADECIDE_MAPPING.get(wandb_dataset)


def map_wandb_model_size_to_datadecide(model_size: int) -> Optional[str]:
    if pd.isna(model_size):
        return None
    closest_param = None
    min_diff = float("inf")
    for param_str, numeric_val in consts.HARDCODED_SIZE_MAPPING.items():
        diff = abs(numeric_val - model_size)
        if diff < min_diff:
            min_diff = diff
            closest_param = param_str

    return closest_param


def add_datadecide_columns(df: pd.DataFrame) -> pd.DataFrame:
    assert "model_name_or_path" in df.columns, "model_name_or_path column not found"
    assert "model_size" in df.columns, "model_size column not found"
    df = df.copy()
    df["wandb_dataset"] = df["model_name_or_path"].apply(
        extract_dataset_from_model_path
    )
    df["data"] = df["wandb_dataset"].apply(map_wandb_dataset_to_datadecide)
    df["params"] = df["model_size"].apply(map_wandb_model_size_to_datadecide)
    return df


def extract_hyperparameters(run_name: str, ignore: List[str] = None) -> Dict[str, any]:
    if ignore is None:
        ignore = wconsts.DEFAULT_IGNORE_PARAMS
    params = {}

    # Try new format first: YYMMDD-HHMMSS
    date_time_match = re.search(r"^(\d{6})-(\d{6})_", run_name)
    if date_time_match:
        # Convert YYMMDD-HHMMSS to sortable ISO datetime format
        date_part = date_time_match.group(1)
        time_part = date_time_match.group(2)
        year = "20" + date_part[:2]  # Assume 20XX
        month = date_part[2:4]
        day = date_part[4:6]
        hour = time_part[:2]
        minute = time_part[2:4]
        second = time_part[4:6]
        params["run_datetime"] = f"{year}-{month}-{day} {hour}:{minute}:{second}"
    else:
        # Try old format: YYYY_MM_DD-HH_MM_SS
        date_time_match = re.search(
            r"^(\d{4}_\d{2}_\d{2})-(\d{2}_\d{2}_\d{2})_", run_name
        )
        if date_time_match:
            # Convert old format to sortable ISO datetime format
            date_str = date_time_match.group(1).replace("_", "-")
            time_str = date_time_match.group(2).replace("_", ":")
            params["run_datetime"] = f"{date_str} {time_str}"

    # Extract experiment name - try new format first, then old format, then fallback
    exp_name_match = re.search(r"^\d{6}-\d{6}_(.+?)_DD-", run_name)
    if exp_name_match:
        # New format: YYMMDD-HHMMSS_experiment_name_DD-
        full_exp_name = exp_name_match.group(1)
        # Clean up experiment name by removing model sizes and token configs, but keep method names that are part of experiment name
        clean_exp_name = re.sub(r"_(?:\d+[MB]|\d+Mtx\d+).*$", "", full_exp_name)
        params["exp_name"] = clean_exp_name if clean_exp_name else full_exp_name
    else:
        # Old format: YYYY_MM_DD-HH_MM_SS_experiment_name_DD-
        exp_name_match = re.search(
            r"^\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_(.+?)_DD-", run_name
        )
        if exp_name_match:
            full_exp_name = exp_name_match.group(1)
            # Clean up experiment name by removing model sizes and token configs, but keep method names that are part of experiment name
            clean_exp_name = re.sub(r"_(?:\d+[MB]|\d+Mtx\d+).*$", "", full_exp_name)
            params["exp_name"] = clean_exp_name if clean_exp_name else full_exp_name
        else:
            # Fallback: extract between datetime and first recognizable parameter
            fallback_match = re.search(
                r"^\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_(.+?)_(?:dd__|dolma|dclm)",
                run_name,
            )
            if fallback_match:
                full_exp_name = fallback_match.group(1)
                # Clean up experiment name
                clean_exp_name = re.sub(r"_(?:\d+[MB]|\d+Mtx\d+).*$", "", full_exp_name)
                params["exp_name"] = clean_exp_name if clean_exp_name else full_exp_name

    # Extract from the second DD part (after _Ft_) if present, which has the actual model info
    ft_dd_match = re.search(
        r"_Ft_DD-([^-]+(?:_[^-]+)*)[-_](\d+)([MB])[-_](\d+)[-_]\d+", run_name
    )
    if ft_dd_match:
        # Format: _Ft_DD-dclm_qc10p-60M-29042-0
        params["data"] = ft_dd_match.group(1)  # dclm_qc10p
        params["params"] = f"{ft_dd_match.group(2)}{ft_dd_match.group(3)}"  # 60M
        params["checkpoint"] = ft_dd_match.group(4)  # 29042
    else:
        # Fallback to original simple patterns
        dataset_match = re.search(r"DD-([^-]+)-", run_name)
        if dataset_match:
            params["data"] = dataset_match.group(1)

        model_match = re.search(r"-(\d+)([MB])_", run_name)
        if model_match:
            size = model_match.group(1)
            unit = model_match.group(2)
            params["params"] = f"{size}{unit}"

        checkpoint_match = re.search(r"-\d+M_(\w+)_\d+Mtx", run_name)
        if checkpoint_match:
            params["checkpoint"] = checkpoint_match.group(1)

    mtx_match = re.search(r"(\d+)Mtx(\d+)", run_name)
    if mtx_match:
        dataset_tokens = int(mtx_match.group(1))
        epochs_from_name = int(mtx_match.group(2))
        params["epochs"] = epochs_from_name
        params["total_tok"] = dataset_tokens
    else:
        # Check for __NNNMt format (e.g., __100Mt)
        mt_only_match = re.search(r"__(\d+)Mt(?:_|$)", run_name)
        if mt_only_match:
            dataset_tokens = int(mt_only_match.group(1))
            params["total_tok"] = dataset_tokens
        else:
            legacy_match = re.search(r"main_(\d+)Mtx(\d+)", run_name)
            if legacy_match:
                base = int(legacy_match.group(1))
                mult = int(legacy_match.group(2))
                params["token_base"] = base
                params["token_mult"] = mult
                params["total_tok"] = base * mult

    explicit_params = re.findall(r"--(\w+)=([^\s_]+)", run_name)
    for param_name, param_value in explicit_params:
        try:
            if "." in param_value or "e" in param_value.lower():
                float_val = float(param_value)
                if float_val.is_integer():
                    params[param_name] = int(float_val)
                else:
                    params[param_name] = float_val
            else:
                params[param_name] = int(param_value)
        except ValueError:
            params[param_name] = param_value

    if "learning_rate" in params:
        params["lr"] = params["learning_rate"]
        del params["learning_rate"]

    run_name_lower = run_name.lower()
    for method in wconsts.METHODS:
        if method in run_name_lower:
            params["method"] = method
            break

    params = {f"{k}_rnp": v for k, v in params.items() if k not in ignore}
    return params


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
                    continue
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


def infer_training_method(df: pd.DataFrame) -> pd.DataFrame:
    assert "method" not in df.columns, "Method column already exists"
    df = df.copy()
    dpo_hparams = wconsts.CORE_DPO_HPM_COLS
    has_dpo_params = df[dpo_hparams].notna().any(axis=1)
    df["method"] = "finetune"
    df.loc[has_dpo_params, "method"] = "dpo"
    return df


def get_columns_for_groups(df: pd.DataFrame, column_groups: List[str]) -> Set[str]:
    selected_columns = set()
    for group_name in column_groups:
        if group_name in wconsts.KEY_SETS:
            group_columns = wconsts.KEY_SETS[group_name]
            existing_columns = [col for col in group_columns if col in df.columns]
            selected_columns.update(existing_columns)
        else:
            print(f"Warning: Unknown column group '{group_name}'")
    if "method" in df.columns:
        selected_columns.add("method")
    return selected_columns


def select_column_groups(
    df: pd.DataFrame, column_groups: List[str], include_oe_evaluations: bool = False
) -> pd.DataFrame:
    if not column_groups:
        return df
    selected_columns = []
    for col in wconsts.ADDED_COLS:
        if col in df.columns:
            selected_columns.append(col)
    for group_name in column_groups:
        if group_name in wconsts.KEY_SETS:
            group_columns = wconsts.KEY_SETS[group_name]
            for col in group_columns:
                if col in df.columns and col not in selected_columns:
                    selected_columns.append(col)
    if include_oe_evaluations:
        oe_eval_cols = [
            col
            for col in df.columns
            if any(
                task in col and metric in col
                for task in wconsts.OE_EVAL_TASKS
                for metric in wconsts.OE_EVAL_METRICS
            )
            and col not in selected_columns
        ]
        oe_eval_cols.sort()
        selected_columns.extend(oe_eval_cols)
    return df[selected_columns]


def add_extracted_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    run_name_col = None
    for col in wconsts.RUN_NAME_CANDIDATES:
        if col in df.columns:
            run_name_col = col
            break
    if run_name_col is None:
        print("Warning: No run name column found - skipping hyperparameter extraction")
        return df
    print(f"Using '{run_name_col}' column for hyperparameter extraction")

    extracted_params_list = []
    for run_name in df[run_name_col]:
        if pd.notna(run_name):
            params = extract_hyperparameters(str(run_name))
            if "data_rnp" in params:
                mapped_data = map_wandb_dataset_to_datadecide(params["data_rnp"])
                if mapped_data:
                    params["data_rnp"] = mapped_data
            extracted_params_list.append(params)
        else:
            extracted_params_list.append({})
    extracted_df = pd.DataFrame(extracted_params_list)
    result_df = pd.concat([df, extracted_df], axis=1)
    if len(extracted_df.columns) > 0:
        print(f"Extracted parameters: {list(extracted_df.columns)}")
    return result_df


def drop_method_specific_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    assert method in ["finetune"], "Only finetune method supported for now"
    if method == "finetune":
        dpo_cols_present = [col for col in wconsts.DPO_ONLY_COLS if col in df.columns]

        if dpo_cols_present:
            for col in dpo_cols_present:
                non_nan_count = df[col].notna().sum()
                assert non_nan_count == 0, (
                    f"DPO column '{col}' has {non_nan_count} non-NaN values in finetune data"
                )
            df = df.drop(columns=dpo_cols_present)
            print(f"Dropped {len(dpo_cols_present)} DPO-specific columns")
    return df
