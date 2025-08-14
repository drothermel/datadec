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
        self,
        data_dir: str = "./data",
        recompute_from: str = None,
        force_reload: bool = None,
        verbose: bool = True,
    ):
        """Initialize DataDecide with data directory and processing options.

        Args:
            data_dir: Directory for storing DataDecide data
            recompute_from: Stage to start recomputing from (None for cached files)
                Options: None, "download", "metrics_expand", "parse", "merge", "aggregate", "all"
            force_reload: DEPRECATED - use recompute_from="all" instead
            verbose: If True, print progress messages during setup
        """
        self.paths = DataDecidePaths(data_dir)
        self.loader = DataFrameLoader()
        self.pipeline = DataPipeline(self.paths)

        # Handle legacy force_reload parameter
        if force_reload is not None:
            if verbose:
                print(
                    "Warning: force_reload is deprecated, use recompute_from='all' instead"
                )
            recompute_from = "all" if force_reload else recompute_from

        # Run the data pipeline to ensure all files are ready
        self.pipeline.run(recompute_from=recompute_from, verbose=verbose)

        # Cache static data that doesn't need lazy loading
        self._dataset_details = data_utils.load_ds_details_df(
            self.paths.ds_details_path
        )
        self._model_details = model_utils.get_model_details_df()
        self.loader.cache_dataframe(self._model_details, "model_details")

        # Pre-cache common derived DataFrames
        self._cache_derived_dataframes()

        print(">> Finished setting up DataDecide.")

    def _cache_derived_dataframes(self):
        """Pre-cache commonly accessed derived DataFrames for better performance."""
        # Pre-cache full and mean eval DataFrames since they're most commonly accessed
        try:
            # Cache the main datasets
            self.loader.load(self.paths.full_eval_ds_path, "full_eval")
            self.loader.load(self.paths.mean_eval_ds_path, "mean_eval")

            # Cache additional derived DataFrames if they exist
            derived_cache_configs = [
                ("full_eval_with_details", self.paths.full_eval_with_details_path),
                ("mean_eval_with_lr", self.paths.mean_eval_with_lr_path),
                ("std_eval", self.paths.std_eval_ds_path),
            ]

            for cache_key, path in derived_cache_configs:
                if path.exists():
                    self.loader.load(path, cache_key)

        except Exception:
            # If caching fails, continue without pre-caching
            # This ensures the DataDecide object still initializes successfully
            pass

    @property
    def all_data_param_combos(self):
        """Get all combinations of data names and parameter strings."""
        return list(
            itertools.product(
                consts.ALL_DATA_NAMES,
                consts.ALL_PARAM_STRS,
            )
        )

    # ------------ Main Dataset Properties ------------

    @property
    def full_eval(self) -> pd.DataFrame:
        """Full evaluation dataset (merged perplexity + downstream)."""
        return self.loader.load(self.paths.full_eval_ds_path, "full_eval")

    @property
    def mean_eval(self) -> pd.DataFrame:
        """Mean evaluation dataset (averaged across seeds)."""
        return self.loader.load(self.paths.mean_eval_ds_path, "mean_eval")

    @property
    def dataset_details(self) -> pd.DataFrame:
        """Dataset details and metadata."""
        return self._dataset_details

    @property
    def model_details(self) -> pd.DataFrame:
        """Model configuration details."""
        return self._model_details

    # ------------ Utility Methods ------------

    def load_dataframe(self, name: str) -> pd.DataFrame:
        """Load any intermediate or derived DataFrame by name.

        Available DataFrames:
        - "ppl_raw": Raw perplexity evaluation data
        - "dwn_raw": Raw downstream evaluation data
        - "ppl_parsed": Parsed perplexity evaluation data
        - "dwn_parsed": Parsed downstream evaluation data
        - "dwn_metrics_expanded": Downstream data after metrics expansion (slow step)
        - "step_to_token_compute": Step-to-token/compute mapping
        - "std_eval": Standard deviation evaluation dataset
        - "full_eval_with_details": Full eval merged with model/dataset details
        - "mean_eval_with_lr": Mean eval with learning rate columns

        Args:
            name: Name of the DataFrame to load

        Returns:
            Requested DataFrame

        Raises:
            ValueError: If DataFrame name is not recognized
        """
        # Map friendly names to file paths and cache keys
        dataframe_map = {
            "ppl_raw": (self.paths.ppl_eval_raw_path, "ppl_raw"),
            "dwn_raw": (self.paths.downstream_eval_raw_path, "dwn_raw"),
            "ppl_parsed": (self.paths.ppl_eval_parsed_path, "ppl_parsed"),
            "dwn_parsed": (self.paths.downstream_eval_parsed_path, "dwn_parsed"),
            "dwn_metrics_expanded": (
                self.paths.dwn_metrics_expanded_path,
                "dwn_metrics_expanded",
            ),
            "step_to_token_compute": (
                self.paths.step_to_token_compute_path,
                "step_to_token_compute",
            ),
            "std_eval": (self.paths.std_eval_ds_path, "std_eval"),
            "full_eval_with_details": (
                self.paths.full_eval_with_details_path,
                "full_eval_with_details",
            ),
            "mean_eval_with_lr": (
                self.paths.mean_eval_with_lr_path,
                "mean_eval_with_lr",
            ),
        }

        if name not in dataframe_map:
            available = ", ".join(sorted(dataframe_map.keys()))
            raise ValueError(f"Unknown DataFrame '{name}'. Available: {available}")

        file_path, cache_key = dataframe_map[name]
        return self.loader.load(file_path, cache_key)

    def merge_in_ds_and_model_details(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Merge dataset and model details into the input DataFrame."""
        return df_utils.merge_in_ds_and_model_details(
            input_df, self.dataset_details, self.model_details
        )

    def get_analysis_df(
        self,
        filter_by_max_step: bool = True,
        add_lr_cols: bool = True,
        return_means: bool = True,
        min_params: str = "10M",
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Get analysis-ready DataFrame with common transformations applied.

        This is the main method for getting a DataFrame ready for ML experiment analysis.
        It applies common filtering, adds model/dataset details, and optionally calculates
        learning rate schedules and averages across seeds.

        Args:
            filter_by_max_step: If True, filter to maximum step for each model/data combo
            add_lr_cols: If True, add learning rate schedule columns
            return_means: If True, return means across seeds; if False, return full data
            min_params: Minimum model size to include (e.g., "10M", "300M")
            verbose: If True, print shape information at each step

        Returns:
            Analysis-ready DataFrame with requested transformations
        """
        # Start with full evaluation dataset
        base_df = self.full_eval.copy()
        if verbose:
            print(
                f">> Initial shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

        # Filter by minimum parameters
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

        # Filter by max step if requested
        if filter_by_max_step:
            base_df = self._filter_by_max_step_to_use(base_df)
            if verbose:
                print(
                    f">> Filter by max step shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
                )

        # Must merge before calculating lr cols
        base_df = self.merge_in_ds_and_model_details(base_df)
        if verbose:
            print(
                f">> Merge in details shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

        # Add learning rate columns if requested
        if add_lr_cols:
            base_df = model_utils.add_lr_cols(base_df)
            if verbose:
                print(
                    f">> Add lr cols shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
                )

        # Calculate means as last step if requested
        if return_means:
            base_df, _ = df_utils.create_mean_std_df(base_df)
            if verbose:
                print(
                    f">> Create mean std df shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
                )

        return base_df

    def _filter_by_max_step_to_use(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to maximum step for each model/data combination.

        Args:
            df: DataFrame with params, data, and step columns

        Returns:
            Filtered DataFrame with only max step entries
        """
        # Group by model and data, keep only the maximum step for each
        max_steps = df.groupby(["params", "data"])["step"].max().reset_index()
        max_steps.columns = ["params", "data", "max_step"]

        # Merge back and filter
        df_with_max = df.merge(max_steps, on=["params", "data"], how="left")
        filtered_df = df_with_max[df_with_max["step"] == df_with_max["max_step"]]

        return filtered_df.drop(columns=["max_step"])

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
    """DEPRECATED: Use DataDecide().get_analysis_df() instead.

    This function is kept for backward compatibility but will be removed in a future version.
    Please use: dd = DataDecide(); df = dd.get_analysis_df() instead.
    """
    if verbose:
        print(
            "Warning: prep_base_df is deprecated. Use DataDecide().get_analysis_df() instead."
        )

    # Convert to new parameter format
    recompute_from = "all" if force_reload else None
    dd = DataDecide(data_dir=data_dir, recompute_from=recompute_from, verbose=verbose)

    return dd.get_analysis_df(
        filter_by_max_step=filter_by_max_step,
        add_lr_cols=add_lr_cols,
        return_means=return_means,
        min_params=min_params,
        verbose=verbose,
    )
