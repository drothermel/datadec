# %%
%load_ext autoreload
%autoreload 2
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

# Configure pandas to show all columns in interactive output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 500)  # Limit column content width for readability

sys.path.append(str(Path(__file__).parent.parent / "src"))

from datadec.wandb_eval import wandb_constants as wconsts
from datadec.wandb_eval.wandb_store import WandBStore
from datadec.wandb_eval.parsing import filter_broken_initial_testing_runs, drop_wandb_constant_ignored_cols, get_created_time_key, convert_objects_and_normalize_dtypes, add_datadecide_columns, map_wandb_dataset_to_datadecide
from datadec.wandb_eval.analysis_helpers import extract_hyperparameters
# %%

ADDED_COLS = ["method", "params", "data"]

# Column group definitions for incremental loading
INITIAL_SFT_GROUPS = ["id_cols", "status_cols", "core_hpm_cols"]
EXTENDED_SFT_GROUPS = INITIAL_SFT_GROUPS + ["x_axis_cols", "summary_metrics_cols"]
FULL_SFT_GROUPS = EXTENDED_SFT_GROUPS  # Evaluation columns added separately


@dataclass
class FilterConfig:
    method: Optional[str] = "finetune"
    completed_only: bool = True
    recent_only: bool = True
    required_params: Optional[List[str]] = None
    column_groups: List[str] = None
    include_oe_evaluations: bool = False

    def __post_init__(self):
        if self.required_params is None:
            self.required_params = []
        if self.column_groups is None:
            self.column_groups = []
        if self.method is not None and self.method not in wconsts.METHODS:
            raise ValueError(
                f"method must be one of {wconsts.METHODS} or None, got {self.method}"
            )

        # Validate column groups
        valid_groups = set(wconsts.KEY_SETS.keys())
        for group in self.column_groups:
            if group not in valid_groups:
                raise ValueError(
                    f"Invalid column group '{group}'. Valid groups: {sorted(valid_groups)}"
                )


@dataclass
class ExperimentalSummary:
    total_runs: int
    unique_conditions: int
    replications_per_condition: float
    model_sizes: List[float]
    learning_rates: List[float]
    dataset_sizes: List[float]


def load_sft_data(
    runs_df: pd.DataFrame,
    config: FilterConfig,
) -> pd.DataFrame:
    print(f"Starting with {len(runs_df)} runs and {len(runs_df.columns)} columns")

    runs_df = filter_broken_initial_testing_runs(runs_df)
    runs_df = add_extracted_hyperparameters(runs_df)
    runs_df = convert_objects_and_normalize_dtypes(runs_df)
    runs_df = add_datadecide_columns(runs_df)
    runs_df = infer_training_method(runs_df)
    runs_df = select_column_groups(runs_df, config)
    runs_df = drop_wandb_constant_ignored_cols(runs_df)
    print(f"After dropping constant/problematic columns: {runs_df.shape}")

    if config.method is not None:
        runs_df = runs_df[runs_df["method"] == config.method]
        print(f"After method filtering ({config.method}): {len(runs_df)} runs")
        runs_df = drop_method_specific_columns(runs_df, config.method)

    if config.completed_only:
        runs_df = runs_df[runs_df["state"] == "finished"]
        print(f"After completion filtering: {len(runs_df)} runs")

    if config.recent_only:
        time_col = get_created_time_key(runs_df)
        runs_df = runs_df[runs_df[time_col] >= wconsts.EARLIEST_GOOD_RUN_DATE]
        print(f"After recency filtering: {len(runs_df)} runs")

    assert len(runs_df) > 0, "No runs remain after filtering."
    return runs_df


def infer_training_method(runs_df: pd.DataFrame) -> pd.DataFrame:
    assert "method" not in runs_df.columns, "Method column already exists"
    runs_df = runs_df.copy()
    dpo_hparams = wconsts.CORE_DPO_HPM_COLS
    has_dpo_params = runs_df[dpo_hparams].notna().any(axis=1)
    runs_df["method"] = "finetune"
    runs_df.loc[has_dpo_params, "method"] = "dpo"
    return runs_df


def get_columns_for_groups(runs_df: pd.DataFrame, column_groups: List[str]) -> Set[str]:
    selected_columns = set()

    for group_name in column_groups:
        if group_name in wconsts.KEY_SETS:
            group_columns = wconsts.KEY_SETS[group_name]
            existing_columns = [col for col in group_columns if col in runs_df.columns]
            selected_columns.update(existing_columns)
            print(f"Group '{group_name}': {len(existing_columns)}/{len(group_columns)} columns available")
        else:
            print(f"Warning: Unknown column group '{group_name}'")

    if "method" in runs_df.columns:
        selected_columns.add("method")

    return selected_columns




def get_ordered_columns(runs_df: pd.DataFrame, config: FilterConfig) -> List[str]:
    ordered_columns = []

    # Step 1: Add ADDED_COLS first (if they exist in the DataFrame)
    for col in ADDED_COLS:
        if col in runs_df.columns:
            ordered_columns.append(col)

    # Step 2: Add columns from each group in the specified order
    for group_name in config.column_groups:
        if group_name in wconsts.KEY_SETS:
            # Maintain original ordering within each group
            group_columns = wconsts.KEY_SETS[group_name]
            for col in group_columns:
                if col in runs_df.columns and col not in ordered_columns:
                    ordered_columns.append(col)

    # Step 3: Add evaluation columns if requested
    if config.include_oe_evaluations:
        oe_eval_cols = [col for col in runs_df.columns
                       if any(task in col and metric in col
                             for task in wconsts.OE_EVAL_TASKS
                             for metric in wconsts.OE_EVAL_METRICS)
                       and col not in ordered_columns]
        # Sort evaluation columns for consistency
        oe_eval_cols.sort()
        ordered_columns.extend(oe_eval_cols)
        if oe_eval_cols:
            print(f"Added {len(oe_eval_cols)} OE evaluation columns")

    return ordered_columns


def select_column_groups(runs_df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    if not config.column_groups:
        print("No column groups specified - returning all columns")
        return runs_df

    # Get columns in proper hierarchical order
    ordered_columns = get_ordered_columns(runs_df, config)

    print(f"Selected {len(ordered_columns)} columns from groups: {config.column_groups}")
    print(f"Column order: ADDED_COLS -> {' -> '.join(config.column_groups)} -> evaluations")

    return runs_df[ordered_columns]


def add_extracted_hyperparameters(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract hyperparameters from run names/IDs and add with 'name_' prefix for comparison"""
    runs_df = runs_df.copy()

    # Check for run name columns in order of preference (from EXACT_MATCH_COLS)
    run_name_candidates = ["run_name", "run_id", "exp_name"]
    run_name_col = None

    for col in run_name_candidates:
        if col in runs_df.columns:
            run_name_col = col
            break

    if run_name_col is None:
        print(f"Warning: No run name column found in {run_name_candidates} - skipping hyperparameter extraction")
        return runs_df

    print(f"Using '{run_name_col}' column for hyperparameter extraction")

    extracted_params_list = []
    for run_name in runs_df[run_name_col]:
        if pd.notna(run_name):
            params = extract_hyperparameters(str(run_name), ignore=["run_date", "run_time"])

            # Map extracted dataset name to DataDecide format for comparison
            if 'data_rnp' in params:
                mapped_data = map_wandb_dataset_to_datadecide(params['data_rnp'])
                if mapped_data:
                    params['data_rnp'] = mapped_data

            extracted_params_list.append(params)
        else:
            extracted_params_list.append({})

    # Convert to DataFrame and merge
    extracted_df = pd.DataFrame(extracted_params_list)
    result_df = pd.concat([runs_df, extracted_df], axis=1)

    # Report what was extracted
    if len(extracted_df.columns) > 0:
        print(f"Extracted parameters from {run_name_col}: {list(extracted_df.columns)}")
        print(f"Sample extraction: {dict(list(extracted_df.iloc[0].items())[:3]) if len(extracted_df) > 0 else 'None'}")
    else:
        print(f"No parameters extracted from {run_name_col} column")

    return result_df


def drop_method_specific_columns(runs_df: pd.DataFrame, method: str) -> pd.DataFrame:
    assert method in ["finetune"], "Only finetune method supported for now"
    if method == "finetune":
        dpo_cols_present = [
            col for col in wconsts.DPO_ONLY_COLS if col in runs_df.columns
        ]

        if dpo_cols_present:
            for col in dpo_cols_present:
                non_nan_count = runs_df[col].notna().sum()
                assert non_nan_count == 0, (
                    f"DPO column '{col}' has {non_nan_count} non-NaN values in finetune data"
                )
            runs_df = runs_df.drop(columns=dpo_cols_present)
            print(f"Dropped {len(dpo_cols_present)} DPO-specific columns")
    return runs_df


def main():
    try:
        wandb_store = WandBStore(wconsts.DEFAULT_DB_CONNECTION)
        runs_df = wandb_store.get_runs()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None

    config = FilterConfig(
        method="finetune",
        completed_only=False,
        recent_only=False,
        column_groups=INITIAL_SFT_GROUPS,
    )
    filtered_df = load_sft_data(runs_df, config)
    print(f"\nTotal runs after filtering: {len(filtered_df)}")
    print(f"Training methods: {filtered_df['method'].value_counts().to_dict()}")
    if "state" in filtered_df.columns:
        print(f"Run states: {filtered_df['state'].value_counts().to_dict()}")
    return filtered_df


# %%
filtered_df = main()

# %%
filtered_df.head()

# %%
filtered_df["num_train_epochs"].unique()

# %%
if __name__ == "__main__":
    filtered_df = main()


# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
