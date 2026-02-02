from __future__ import annotations

import itertools
from pathlib import Path
from typing import Union

import pandas as pd

from datadec import constants as consts
from datadec import df_utils, model_utils, validation
from datadec.paths import DataDecidePaths
from datadec.pipeline import DataPipeline, verbose_print


def get_data_recipe_family(
    data_name: str, data_recipe_families: dict[str, list[str]] | None = None
) -> str:
    if data_recipe_families is None:
        data_recipe_families = consts.DATA_RECIPE_FAMILIES

    for family, names in data_recipe_families.items():
        if data_name in names:
            return family
    return "unknown"


def get_data_recipe_details_df(ds_details_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ds_details_path).rename(columns={"dataset": "data"})

    df["data"] = (
        df["data"]
        .str.replace("Dolma1.7 (no math code)", "Dolma1.7 (no math, code)")
        .str.replace("DCLM-Baseline (QC 7%", "DCLM-Baseline (QC 7%,")
    )

    return df


class DataFrameLoader:
    def __init__(self, paths: DataDecidePaths | None = None) -> None:
        self._cache: dict[str, pd.DataFrame] = {}
        self.paths: DataDecidePaths = paths if paths else DataDecidePaths()

    @property
    def cached_dataframes(self) -> list[str]:
        return list(self._cache.keys())

    def possible_dataframes(self) -> list[str]:
        return list(self.paths.available_dataframes)

    def written_dataframes(self) -> list[str]:
        written_dataframes = []
        for df_name in self.possible_dataframes():
            maybe_path = self.paths.get_existing_path(df_name)
            if maybe_path is not None:
                written_dataframes.append(df_name)
        return written_dataframes

    def set_name(self, name: str, df: pd.DataFrame) -> None:
        self._cache[name] = df

    def load_path(self, path: Path, name: str | None = None) -> pd.DataFrame:
        key = name if name is not None else str(path)
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(path)
        return self._cache[key]

    def load_name(self, name: str) -> pd.DataFrame:
        if not self.paths.check_name_in_paths(name):
            if not self.is_cached(name):
                raise ValueError(f"Unknown dataframe '{name}'")
            return self._cache[name]
        path = self.paths.get_path(name)
        return self.load_path(path, name)

    def is_cached(self, cache_key: str) -> bool:
        return cache_key in self._cache

    def clear_cache(self, cache_key: str | None = None) -> None:
        if cache_key is None:
            self._cache.clear()
        else:
            self._cache.pop(cache_key, None)

    def get_cache_size(self) -> int:
        return len(self._cache)


class DataDecide:
    def __init__(
        self,
        data_dir: str = "./data",
        recompute_from: str | None = None,
        verbose: bool = True,
    ) -> None:
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
            get_data_recipe_details_df(self.paths.ds_details_csv_path),
        )
        verbose_print("Finished setting up DataDecide.", verbose)

    @property
    def all_data_param_combos(self) -> list[tuple[str, str]]:
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

    @property
    def full_eval_melted(self) -> pd.DataFrame:
        return self.loader.load_name("full_eval_melted")

    @property
    def mean_eval_melted(self) -> pd.DataFrame:
        return self.loader.load_name("mean_eval_melted")

    def load_dataframe(self, name: str) -> pd.DataFrame:
        return self.loader.load_name(name)

    def aggregate_results(
        self,
        input_df: pd.DataFrame,
        by_seeds: bool = True,
        return_std: bool = False,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
        if len(input_df) == 0 or not by_seeds:
            df_utils.print_shape(input_df, "Empty DataFrame or no aggregation", verbose)
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
        filter_types: list[str] | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        if filter_types is None:
            filter_types = ["max_steps"]
        validation.validate_filter_types(filter_types)
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
        return df

    def select_subset(
        self,
        input_df: pd.DataFrame,
        data: str | list[str] | None = None,
        params: str | list[str] | None = None,
        seeds: int | list[int] | None = None,
        step_lims: tuple[int | None, int | None] | None = None,
        token_lims: tuple[int | None, int | None] | None = None,
        min_params: str | None = None,
        max_params: str | None = None,
        data_param_combos: list[tuple[str, str]] | None = None,
        columns: list[str] | None = None,
        metrics: list[str] | None = None,
        metric_type: str | None = None,
        include_id_columns: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        df = input_df.copy()
        df_utils.print_shape(df, "Initial subset selection", verbose)
        if len(df) == 0:
            df_utils.print_shape(df, "Empty DataFrame, no selection", verbose)
            return df

        data = data if data is None else self.select_data(data)
        params = params if params is None else self.select_params(params)
        df = df_utils.select_by_data_param_combos(df, data, params, data_param_combos)
        df_utils.print_shape(df, "After data/param selection", verbose)

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

        if seeds is not None:
            seeds = seeds if isinstance(seeds, list) else [seeds]
            df = df[df["seed"].isin(seeds)]
            df_utils.print_shape(df, f"Seeds {seeds}", verbose)

        if step_lims is not None:
            min_step, max_step = step_lims
            if min_step is not None:
                df = df[df["step"] >= min_step]
            if max_step is not None:
                df = df[df["step"] <= max_step]
            df_utils.print_shape(df, f"Step range {step_lims}", verbose)

        if token_lims is not None:
            min_tokens, max_tokens = token_lims
            if min_tokens is not None:
                df = df[df["tokens"] >= min_tokens]
            if max_tokens is not None:
                df = df[df["tokens"] <= max_tokens]
            df_utils.print_shape(df, f"Token range {token_lims}", verbose)

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
        columns: list[str] | None,
        metrics: list[str] | None,
        metric_type: str | None,
        include_id_columns: bool,
    ) -> list[str]:
        validation.validate_metric_type(metric_type)
        selected_columns: set[str] = set()

        if include_id_columns:
            selected_columns.update(
                col for col in consts.FULL_ID_COLUMNS if col in df.columns
            )

        if columns:
            selected_columns.update(col for col in columns if col in df.columns)

        if metric_type and metric_type == "ppl":
            selected_columns.update(
                col for col in consts.PPL_TYPES if col in df.columns
            )
        elif metric_type and metric_type == "olmes":
            selected_columns.update(
                col for col in consts.OLMES_TASKS if col in df.columns
            )
        if metrics:
            validation.validate_metrics(metrics)
            selected_columns.update(
                metric for metric in metrics if metric in df.columns
            )
        if not selected_columns:
            return list(df.columns)
        return list(selected_columns)

    def select_params(
        self,
        params: str | list[str] = "all",
        exclude: list[str] | None = None,
    ) -> list[str]:
        return validation._validated_select(
            choices=params,
            valid_options=consts.ALL_MODEL_SIZE_STRS,
            name="parameter size",
            exclude=exclude,
        )

    def select_data(
        self,
        data: str | list[str] = "all",
        exclude: list[str] | None = None,
    ) -> list[str]:
        return validation._validated_select(
            choices=data,
            valid_options=consts.ALL_DATA_NAMES,
            name="data recipe",
            exclude=exclude,
        )

    def melt_for_plotting(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
        include_seeds: bool = True,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        return df_utils.melt_for_plotting(
            df=df,
            metrics=metrics,
            include_seeds=include_seeds,
            drop_na=drop_na,
            id_columns=consts.FULL_ID_COLUMNS,
        )

    def prepare_plot_data(
        self,
        params: str | list[str] | None = None,
        data: str | list[str] | None = None,
        metrics: list[str] | None = None,
        aggregate_seeds: bool = False,
        input_df: pd.DataFrame | None = None,
        auto_filter: bool = True,
        melt: bool = True,
        verbose: bool = False,
        **select_subset_kwargs,
    ) -> pd.DataFrame:
        base_df = input_df if input_df is not None else self.full_eval

        if auto_filter and metrics:
            filter_types = validation.determine_filter_types(metrics)
            df = self.filter_data_quality(
                base_df, filter_types=filter_types, verbose=verbose
            )
        else:
            df = base_df.copy()

        df = self.select_subset(
            df,
            params=params,
            data=data,
            metrics=metrics,
            verbose=verbose,
            **select_subset_kwargs,
        )

        if aggregate_seeds:
            df = self.aggregate_results(df, by_seeds=True, verbose=verbose)

        if melt:
            return self.melt_for_plotting(
                df, metrics=metrics, include_seeds=not aggregate_seeds
            )
        return df
