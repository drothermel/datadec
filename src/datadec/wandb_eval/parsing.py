import pandas as pd

from datadec.wandb_eval import wandb_constants as wconsts

HISTORY_DPO_ONLY = [
    "rewards/chosen",
    "rewards/margin",
    "rewards/average",
    "rewards/accuracy",
    "rewards/rejected",
]
HISTORY_EXTRA_DROP_COLS = [
    "run_name",  # same as run_id, keep one
    "total_tokens_including_padding",  # keep total tokens instead
    "per_device_tps",
    "per_device_tps_including_padding",
]
HISTORY_DROP = [
    *HISTORY_DPO_ONLY,
    *HISTORY_EXTRA_DROP_COLS,
]

RUNS_EXTRA_DROP_COLS = []


def get_created_time_key(df: pd.DataFrame) -> str:
    for tk in wconsts.TIME_KEYS:
        if tk in df.columns:
            return tk
    assert False, "No time key found"


def filter_early_test_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[get_created_time_key(df)] >= wconsts.EARLIEST_GOOD_RUN_DATE]


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
    constant_columns = []
    nanconstant_columns = []
    other_columns = []
    for col in df.columns:
        if df[col].dtype == "object":
            assert False, "Filter object columns before finding constants"
        nunique = df[col].nunique()
        if nunique == 1:
            constant_columns.append(col)
        elif nunique == 2 and any(df[col].isna()):
            nanconstant_columns.append(col)
        other_columns.append(col)
    return {
        "constant": constant_columns,
        "nanconstant": nanconstant_columns,
        "other": other_columns,
    }


def filter_pretrain_metric_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if "pretrain_eval" in col]
