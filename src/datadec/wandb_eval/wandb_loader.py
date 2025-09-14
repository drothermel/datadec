from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from datadec.wandb_eval import wandb_constants as wconsts
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_store import WandBStore


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

        valid_groups = set(wconsts.KEY_SETS.keys())
        for group in self.column_groups:
            if group not in valid_groups:
                raise ValueError(
                    f"Invalid column group '{group}'. Valid groups: {sorted(valid_groups)}"
                )


class WandBDataLoader:
    def __init__(self, db_connection: str = None):
        self.db_connection = db_connection or wconsts.DEFAULT_DB_CONNECTION

    def load_runs_and_history(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        wandb_store = WandBStore(self.db_connection)
        return wandb_store.get_runs(), wandb_store.get_history()

    def load_data(self, config: FilterConfig) -> pd.DataFrame:
        runs_df, _ = self.load_runs_and_history()
        return self._apply_filters(runs_df, config)

    def _apply_filters(
        self, runs_df: pd.DataFrame, config: FilterConfig
    ) -> pd.DataFrame:
        print(f"Starting with {len(runs_df)} runs and {len(runs_df.columns)} columns")

        runs_df = transforms.filter_broken_initial_testing_runs(runs_df)
        runs_df = transforms.add_extracted_hyperparameters(runs_df)
        runs_df = transforms.convert_objects_and_normalize_dtypes(runs_df)
        runs_df = transforms.add_datadecide_columns(runs_df)
        runs_df = transforms.infer_training_method(runs_df)
        runs_df = transforms.select_column_groups(
            runs_df, config.column_groups, config.include_oe_evaluations
        )
        runs_df = transforms.drop_wandb_constant_ignored_cols(runs_df)
        print(f"After dropping constant/problematic columns: {runs_df.shape}")

        if config.method is not None:
            runs_df = runs_df[runs_df["method"] == config.method]
            print(f"After method filtering ({config.method}): {len(runs_df)} runs")
            runs_df = transforms.drop_method_specific_columns(runs_df, config.method)

        if config.completed_only:
            runs_df = runs_df[runs_df["state"] == "finished"]
            print(f"After completion filtering: {len(runs_df)} runs")

        if config.recent_only:
            time_col = transforms.get_created_time_key(runs_df)
            runs_df = runs_df[runs_df[time_col] >= wconsts.EARLIEST_GOOD_RUN_DATE]
            print(f"After recency filtering: {len(runs_df)} runs")

        assert len(runs_df) > 0, "No runs remain after filtering."
        return runs_df
