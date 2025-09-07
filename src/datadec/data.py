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
        input_df: Optional[pd.DataFrame] = None,
        filter_types: List[str] = ["max_steps"],
        return_means: bool = True,
        min_params: str = "10M",
        verbose: bool = False,
    ) -> pd.DataFrame:
        # Use input_df if provided, otherwise default to full_eval
        base_df = input_df if input_df is not None else self.full_eval

        # Apply quality filters using compositional utility
        df = self.filter_data_quality(
            base_df, filter_types=filter_types, verbose=verbose
        )

        # Apply parameter threshold filtering using compositional utility
        df = self.select_subset(df, min_params=min_params, verbose=verbose)

        # Apply aggregation if requested using compositional utility
        if return_means:
            df = self.aggregate_results(df, by_seeds=True, verbose=verbose)

        return df

    def easy_index_df(
        self,
        input_df: Optional[pd.DataFrame] = None,
        df_name: str = "full_eval",
        data: Optional[Union[str, List[str]]] = None,
        params: Optional[Union[str, List[str]]] = None,
        seeds: Optional[Union[int, List[int]]] = None,
        step: Optional[Union[int, List[int]]] = None,
        data_param_combos: Optional[List[Tuple[str, str]]] = None,
        keep_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        # Use input_df if provided, otherwise load by name
        if input_df is not None:
            base_df = input_df
        else:
            base_df = self.load_dataframe(df_name)

        # Handle exact step matching (select_subset uses ranges, but easy_index_df uses exact values)
        if step is not None:
            step_list = step if isinstance(step, list) else [step]
            base_df = base_df[base_df["step"].isin(step_list)]

        # Use compositional utility for most filtering
        return self.select_subset(
            base_df,
            data=data,
            params=params,
            seeds=seeds,
            data_param_combos=data_param_combos,
            columns=keep_cols,
        )

    def aggregate_results(
        self,
        input_df: pd.DataFrame,
        by_seeds: bool = True,
        return_std: bool = False,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        if not by_seeds:
            df_utils.print_shape(input_df, "No aggregation", verbose)
            return input_df

        if len(input_df) == 0:
            df_utils.print_shape(input_df, "Empty DataFrame, no aggregation", verbose)
            if return_std:
                return input_df.copy(), input_df.copy()
            return input_df.copy()

        df_utils.print_shape(input_df, "Before aggregation", verbose)
        mean_df, std_df = df_utils.create_mean_std_df(input_df)
        df_utils.print_shape(mean_df, "After aggregation (means)", verbose)

        if return_std:
            df_utils.print_shape(std_df, "After aggregation (stds)", verbose)
            return mean_df, std_df

        return mean_df

    def filter_data_quality(
        self,
        input_df: pd.DataFrame,
        filter_types: List[str] = ["max_steps"],
        verbose: bool = False,
    ) -> pd.DataFrame:
        df = input_df.copy()
        df_utils.print_shape(df, "Initial", verbose)

        if len(df) == 0:
            df_utils.print_shape(df, "Empty DataFrame, no filtering", verbose)
            return df

        for filter_type in filter_types:
            if filter_type == "max_steps":
                df = df_utils.filter_by_max_step_to_use(df)
                df_utils.print_shape(df, "LEQ to max step", verbose)
            elif filter_type == "ppl":
                df = df_utils.filter_ppl_rows(df)
                df_utils.print_shape(df, "Non-NaN perplexity", verbose)
            elif filter_type == "olmes":
                df = df_utils.filter_olmes_rows(df)
                df_utils.print_shape(df, "Non-NaN OLMES", verbose)
            else:
                available_types = ["max_steps", "ppl", "olmes"]
                raise ValueError(
                    f"Unknown filter_type '{filter_type}'. Available: {available_types}"
                )

        return df

    def select_subset(
        self,
        input_df: pd.DataFrame,
        data: Optional[Union[str, List[str]]] = None,
        params: Optional[Union[str, List[str]]] = None,
        seeds: Optional[Union[int, List[int]]] = None,
        step_lims: Optional[Tuple[Optional[int], Optional[int]]] = None,
        token_lims: Optional[Tuple[Optional[int], Optional[int]]] = None,
        min_params: Optional[str] = None,
        max_params: Optional[str] = None,
        data_param_combos: Optional[List[Tuple[str, str]]] = None,
        columns: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        metric_type: Optional[str] = None,
        include_id_columns: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        df = input_df.copy()
        df_utils.print_shape(df, "Initial subset selection", verbose)

        if len(df) == 0:
            df_utils.print_shape(df, "Empty DataFrame, no selection", verbose)
            return df

        # Data/param combination selection (from easy_index_df)
        df = df_utils.select_by_data_param_combos(df, data, params, data_param_combos)
        df_utils.print_shape(df, "After data/param selection", verbose)

        # Parameter threshold filtering (from get_filtered_df)
        if min_params is not None:
            min_params_numeric = (
                model_utils.param_to_numeric(min_params)
                if isinstance(min_params, str)
                else min_params
            )
            df = df[df[consts.PARAM_NUMERIC_COL] >= min_params_numeric]
            df_utils.print_shape(df, f"Above min params {min_params}", verbose)

        if max_params is not None:
            max_params_numeric = (
                model_utils.param_to_numeric(max_params)
                if isinstance(max_params, str)
                else max_params
            )
            df = df[df[consts.PARAM_NUMERIC_COL] <= max_params_numeric]
            df_utils.print_shape(df, f"Below max params {max_params}", verbose)

        # Seed selection (from easy_index_df)
        if seeds is not None:
            if not isinstance(seeds, list):
                seeds = [seeds]
            df = df[df["seed"].isin(seeds)]
            df_utils.print_shape(df, f"Seeds {seeds}", verbose)

        # Step range selection (enhanced from easy_index_df)
        if step_lims is not None:
            min_step, max_step = step_lims
            if min_step is not None:
                df = df[df["step"] >= min_step]
            if max_step is not None:
                df = df[df["step"] <= max_step]
            df_utils.print_shape(df, f"Step range {step_lims}", verbose)

        # Token range selection (new functionality)
        if token_lims is not None:
            min_tokens, max_tokens = token_lims
            if min_tokens is not None:
                df = df[df["tokens"] >= min_tokens]
            if max_tokens is not None:
                df = df[df["tokens"] <= max_tokens]
            df_utils.print_shape(df, f"Token range {token_lims}", verbose)

        # Column selection (new functionality)
        if columns or metrics or metric_type:
            selected_columns = self._build_column_list(
                df, columns, metrics, metric_type, include_id_columns
            )
            df = df[selected_columns]
            df_utils.print_shape(
                df, f"Selected {len(selected_columns)} columns", verbose
            )

        return df

    def _build_column_list(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]],
        metrics: Optional[List[str]],
        metric_type: Optional[str],
        include_id_columns: bool,
    ) -> List[str]:
        selected_columns = []

        # Always include ID columns if requested
        if include_id_columns:
            id_columns = ["params", "data", "seed", "step", "tokens"]
            for col in id_columns:
                if col in df.columns:
                    selected_columns.append(col)

        # Add explicitly specified columns
        if columns:
            for col in columns:
                if col in df.columns and col not in selected_columns:
                    selected_columns.append(col)

        # Add metric-type specific columns
        if metric_type:
            if metric_type == "ppl":
                ppl_columns = [col for col in df.columns if col in consts.PPL_TYPES]
                selected_columns.extend(
                    [col for col in ppl_columns if col not in selected_columns]
                )
            elif metric_type == "olmes":
                olmes_columns = [
                    col
                    for col in df.columns
                    if any(task in col for task in consts.OLMES_TASKS)
                ]
                selected_columns.extend(
                    [col for col in olmes_columns if col not in selected_columns]
                )
            else:
                available_types = ["ppl", "olmes"]
                raise ValueError(
                    f"Unknown metric_type '{metric_type}'. Available: {available_types}"
                )

        # Add explicitly named metrics (with validation)
        if metrics:
            for metric in metrics:
                if metric not in consts.ALL_KNOWN_METRICS:
                    raise ValueError(
                        f"Unknown metric '{metric}'. "
                        f"Use metric_type='ppl' or metric_type='olmes' for bulk selection, "
                        f"or see consts.ALL_KNOWN_METRICS for valid metric names."
                    )
                if metric in df.columns and metric not in selected_columns:
                    selected_columns.append(metric)

        # If no columns were selected, return all columns
        if not selected_columns:
            selected_columns = df.columns.tolist()

        return selected_columns
