import itertools

import pandas as pd

from datadec import constants as consts
from datadec import data_utils, df_utils, model_utils
from datadec.loader import DataFrameLoader
from datadec.paths import DataDecidePaths
from datadec.pipeline import DataPipeline


class DataDecide:
    def __init__(
        self,
        data_dir: str = "./data",
        recompute_from: str = None,
        verbose: bool = True,
    ):
        self.paths = DataDecidePaths(data_dir)
        self.loader = DataFrameLoader()
        self.pipeline = DataPipeline(self.paths)

        self.pipeline.run(recompute_from=recompute_from, verbose=verbose)

        self._dataset_details = data_utils.load_ds_details_df(
            self.paths.ds_details_path
        )
        self._model_details = model_utils.get_model_details_df()
        self.loader.cache_dataframe(self._model_details, "model_details")

        self._cache_derived_dataframes()

        if verbose:
            print(">> Finished setting up DataDecide.")

    def _cache_derived_dataframes(self):
        core_dataframes = [
            ("full_eval", "full_eval"),
            ("mean_eval", "mean_eval"),
        ]

        for cache_key, df_name in core_dataframes:
            path = self.paths.get_path(df_name)
            if path.exists():
                self.loader.load(path, cache_key)

        derived_cache_names = [
            ("full_eval_with_details", "full_eval_with_details"),
            ("mean_eval_with_lr", "mean_eval_with_lr"),
            ("std_eval", "std_eval"),
        ]

        for cache_key, df_name in derived_cache_names:
            path = self.paths.get_path(df_name)
            if path.exists():
                self.loader.load(path, cache_key)

    @property
    def all_data_param_combos(self):
        return list(
            itertools.product(
                consts.ALL_DATA_NAMES,
                consts.ALL_PARAM_STRS,
            )
        )

    @property
    def full_eval(self) -> pd.DataFrame:
        return self.loader.load(self.paths.get_path("full_eval"), "full_eval")

    @property
    def mean_eval(self) -> pd.DataFrame:
        return self.loader.load(self.paths.get_path("mean_eval"), "mean_eval")

    @property
    def dataset_details(self) -> pd.DataFrame:
        return self._dataset_details

    @property
    def model_details(self) -> pd.DataFrame:
        return self._model_details

    def load_dataframe(self, name: str) -> pd.DataFrame:
        return self.loader.load(self.paths.get_path(name), name)

    def merge_in_ds_and_model_details(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return df_utils.merge_in_ds_and_model_details(
            input_df, self.dataset_details, self.model_details
        )

    def get_filtered_df(
        self,
        filter_by_max_step: bool = True,
        return_means: bool = True,
        min_params: str = "10M",
        verbose: bool = False,
    ) -> pd.DataFrame:
        base_df = self.full_eval.copy()
        if verbose:
            print(
                f">> Initial shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

        if min_params is not None:
            min_params_numeric = model_utils.param_to_numeric(min_params)
            base_df = base_df[
                base_df["params"].apply(
                    lambda x: model_utils.param_to_numeric(x) >= min_params_numeric
                )
            ]
            if verbose:
                print(
                    f">> Above min params {min_params} shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
                )

        if filter_by_max_step:
            base_df = self._filter_by_max_step_to_use(base_df)
            if verbose:
                print(
                    f">> Filter by max step shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
                )

        if return_means:
            base_df, _ = df_utils.create_mean_std_df(base_df)
            if verbose:
                print(
                    f">> Create mean std df shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
                )

        return base_df

    def _filter_by_max_step_to_use(self, df: pd.DataFrame) -> pd.DataFrame:
        max_steps = df.groupby(["params", "data"])["step"].max().reset_index()
        max_steps.columns = ["params", "data", "max_step"]

        df_with_max = df.merge(max_steps, on=["params", "data"], how="left")
        filtered_df = df_with_max[df_with_max["step"] == df_with_max["max_step"]]

        return filtered_df.drop(columns=["max_step"])

    def clear_cache(self, cache_key: str = None) -> None:
        self.loader.clear_cache(cache_key)
