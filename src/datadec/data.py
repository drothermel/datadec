import itertools

import pandas as pd

from datadec import constants as consts
from datadec import data_utils, df_utils, model_utils
from datadec.loader import DataFrameLoader
from datadec.paths import DataDecidePaths
from datadec.pipeline import DataPipeline


class DataDecide:
    """Main interface for DataDecide datasets.

    This class provides a clean API for accessing DataDecide evaluation datasets
    using composition with specialized components for data pipeline management
    and lazy loading.
    """

    def __init__(
        self, data_dir: str = "./data", force_reload: bool = False, verbose: bool = True
    ):
        """Initialize DataDecide with data directory and processing options.

        Args:
            data_dir: Directory for storing DataDecide data
            force_reload: If True, recreate all processed files
            verbose: If True, print progress messages during setup
        """
        self.paths = DataDecidePaths(data_dir)
        self.loader = DataFrameLoader()
        self.pipeline = DataPipeline(self.paths)

        # Run the data pipeline to ensure all files are ready
        self.pipeline.run(force_reload=force_reload, verbose=verbose)

        # Cache static data that doesn't need lazy loading
        self._ds_details_df = data_utils.load_ds_details_df(self.paths.ds_details_path)
        self._model_details_df = model_utils.get_model_details_df()
        self.loader.cache_dataframe(self._model_details_df, "model_details_df")

        print(">> Finished setting up DataDecide dataframes.")

    @property
    def all_data_param_combos(self):
        """Get all combinations of data names and parameter strings."""
        return list(
            itertools.product(
                consts.ALL_DATA_NAMES,
                consts.ALL_PARAM_STRS,
            )
        )

    # ------------ Raw Data Properties ------------

    @property
    def ppl_raw_df(self) -> pd.DataFrame:
        """Raw perplexity evaluation data."""
        return self.loader.load(self.paths.ppl_eval_raw_path, "ppl_raw_df")

    @property
    def dwn_raw_df(self) -> pd.DataFrame:
        """Raw downstream evaluation data."""
        return self.loader.load(self.paths.downstream_eval_raw_path, "dwn_raw_df")

    # ------------ Parsed Data Properties ------------

    @property
    def ppl_parsed_df(self) -> pd.DataFrame:
        """Parsed perplexity evaluation data."""
        return self.loader.load(self.paths.ppl_eval_parsed_path, "ppl_parsed_df")

    @property
    def dwn_parsed_df(self) -> pd.DataFrame:
        """Parsed downstream evaluation data."""
        return self.loader.load(self.paths.downstream_eval_parsed_path, "dwn_parsed_df")

    @property
    def step_to_token_compute_df(self) -> pd.DataFrame:
        """Step-to-token and step-to-compute mapping data."""
        return self.loader.load(
            self.paths.step_to_token_compute_path, "step_to_token_compute_df"
        )

    # ------------ Derived Dataset Properties ------------

    @property
    def full_eval_ds(self) -> pd.DataFrame:
        """Full evaluation dataset (merged perplexity + downstream)."""
        return self.loader.load(self.paths.full_eval_ds_path, "full_eval_ds")

    @property
    def mean_eval_ds(self) -> pd.DataFrame:
        """Mean evaluation dataset (averaged across seeds)."""
        return self.loader.load(self.paths.mean_eval_ds_path, "mean_eval_ds")

    @property
    def std_eval_ds(self) -> pd.DataFrame:
        """Standard deviation evaluation dataset."""
        return self.loader.load(self.paths.std_eval_ds_path, "std_eval_ds")

    # ------------ Static Data Properties ------------

    @property
    def ds_details_df(self) -> pd.DataFrame:
        """Dataset details and metadata."""
        return self._ds_details_df

    @property
    def model_details_df(self) -> pd.DataFrame:
        """Model configuration details."""
        return self._model_details_df

    # ------------ DataFrame Manipulation Helpers ------------

    def filter_by_max_step_to_use(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by maximum step to use for each parameter size."""
        return df_utils.filter_by_max_step_to_use(df)

    def merge_in_ds_and_model_details(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Merge dataset and model details into the input DataFrame."""
        return df_utils.merge_in_ds_and_model_details(
            input_df, self.ds_details_df, self.model_details_df
        )

    def get_max_ppl_vals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get maximum perplexity values across all perplexity columns."""
        return df_utils.get_max_ppl_vals(df)

    def set_step_val_to_max_ppl_val(
        self, df: pd.DataFrame, step: int = 0
    ) -> pd.DataFrame:
        """Set perplexity values at specific step to maximum values for NaN entries."""
        return df_utils.set_step_val_to_max_ppl_val(df, step)

    def create_mean_std_df(
        self, merged_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create mean and standard deviation DataFrames by averaging across seeds."""
        return df_utils.create_mean_std_df(merged_df)

    # ------------ Utility Methods ------------

    def index_dfs(
        self, df_name: str, params: str, data: str, step: int
    ) -> pd.DataFrame:
        """Index into a specific DataFrame by params, data, and step.

        Args:
            df_name: Name of the DataFrame property to access
            params: Parameter size string (e.g., "1B")
            data: Data recipe name
            step: Training step

        Returns:
            Filtered DataFrame with matching rows
        """
        df = getattr(self, df_name)
        return df[
            (df["params"] == params) & (df["data"] == data) & (df["step"] == step)
        ]

    def clear_cache(self, cache_key: str = None) -> None:
        """Clear cached DataFrames.

        Args:
            cache_key: Specific DataFrame to clear. If None, clears all cached data.
        """
        self.loader.clear_cache(cache_key)


def prep_base_df(
    data_dir: str = "./data",
    force_reload: bool = False,
    filter_by_max_step: bool = True,
    add_lr_cols: bool = True,
    return_means: bool = True,
    min_params: str = "10M",
    verbose: bool = False,
):
    dd = DataDecide(data_dir=data_dir, force_reload=force_reload, verbose=verbose)
    base_df = dd.full_eval_ds.copy()
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
        base_df = dd.filter_by_max_step_to_use(base_df)
        if verbose:
            print(
                f">> Filter by max step shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

    # Must merge before calculating lr cols
    base_df = dd.merge_in_ds_and_model_details(base_df)
    if verbose:
        print(
            f">> Merge in ds and model details shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
        )

    if add_lr_cols:
        base_df = model_utils.add_lr_cols(base_df)
        if verbose:
            print(
                f">> Add lr cols shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

    # Calculate means as last step
    if return_means:
        base_df, _ = dd.create_mean_std_df(base_df)
        if verbose:
            print(
                f">> Create mean std df shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )
    return base_df
