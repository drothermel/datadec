import itertools
from typing import List, Optional, Tuple, Union

import pandas as pd

from datadec import constants as consts
from datadec import data_utils, df_utils, model_utils
from datadec.loader import DataFrameLoader
from datadec.paths import DataDecidePaths
from datadec.pipeline import DataPipeline, verbose_print


class DataDecide:
    def __init__(
        self,
        data_dir: str = "./data",
        recompute_from: str = None,
        verbose: bool = True,
    ):
        self.paths = DataDecidePaths(data_dir)

        self.pipeline = DataPipeline(self.paths)
        self.pipeline.run(recompute_from=recompute_from, verbose=verbose)

        self.loader = DataFrameLoader(self.paths)
        self.loader.set_name(
            consts.MODEL_DETAILS_DF_NAME,
            model_utils.get_model_details_df(),
        )
        self.loader.set_name(
            consts.DATASET_DETAILS_DF_NAME,
            data_utils.get_data_recipe_details_df(self.paths.ds_details_csv_path),
        )
        verbose_print("Finished setting up DataDecide.", verbose)

    @property
    def all_data_param_combos(self):
        return list(
            itertools.product(
                consts.ALL_DATA_NAMES,
                consts.ALL_MODEL_SIZE_STRS,
            )
        )

    @property
    def full_eval(self) -> pd.DataFrame:
        return self.loader.load_name("full_eval")

    @property
    def mean_eval(self) -> pd.DataFrame:
        return self.loader.load_name("mean_eval")

    def load_dataframe(self, name: str) -> pd.DataFrame:
        return self.loader.load_name(name)

    def get_filtered_df(
        self,
        filter_types: List[str] = ["max_steps"],
        return_means: bool = True,
        min_params: str = "10M",
        verbose: bool = False,
    ) -> pd.DataFrame:
        base_df = self.full_eval.copy()
        df_utils.print_shape(base_df, "Initial", verbose)

        if min_params is not None:
            if isinstance(min_params, str):
                min_params = model_utils.param_to_numeric(min_params)
            base_df = base_df[base_df[consts.PARAM_NUMERIC_COL] >= min_params]
            df_utils.print_shape(base_df, f"Above min params {min_params}", verbose)

        # Apply filters based on filter_types
        for filter_type in filter_types:
            if filter_type == "max_steps":
                base_df = df_utils.filter_by_max_step_to_use(base_df)
                df_utils.print_shape(base_df, "LEQ to max step", verbose)
            elif filter_type == "ppl":
                base_df = df_utils.filter_ppl_rows(base_df)
                df_utils.print_shape(base_df, "Non-NaN perplexity", verbose)
            elif filter_type == "olmes":
                base_df = df_utils.filter_olmes_rows(base_df)
                df_utils.print_shape(base_df, "Non-NaN OLMES", verbose)
            else:
                available_types = ["max_steps", "ppl", "olmes"]
                raise ValueError(
                    f"Unknown filter_type '{filter_type}'. Available: {available_types}"
                )

        if return_means:
            base_df, _ = df_utils.create_mean_std_df(base_df)
            df_utils.print_shape(base_df, "Mean df", verbose)

        return base_df

    def easy_index_df(
        self,
        df_name: str = "full_eval",
        data: Optional[Union[str, List[str]]] = None,
        params: Optional[Union[str, List[str]]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        step: Optional[Union[int, List[int]]] = None,
        data_param_combos: Optional[List[Tuple[str, str]]] = None,
        keep_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df = self.load_dataframe(df_name)
        df = df_utils.select_by_data_param_combos(df, data, params, data_param_combos)
        if seed is not None:
            df = df[df["seed"].isin(seed if isinstance(seed, list) else [seed])]
        if step is not None:
            df = df[df["step"].isin(step if isinstance(step, list) else [step])]
        if keep_cols is not None:
            df = df[keep_cols]
        return df
