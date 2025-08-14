"""Data processing pipeline for DataDecide datasets.

This module contains the DataPipeline class which handles the complete ETL process
for downloading, parsing, and creating derived datasets from DataDecide data.
"""

import pandas as pd
from datasets import load_dataset

from datadec import constants as consts
from datadec import parsing
from datadec import df_utils
from datadec.paths import DataDecidePaths


class DataPipeline:
    """Handles the ETL pipeline for DataDecide datasets.

    This class manages the complete data processing workflow from downloading
    raw datasets to creating analysis-ready derived datasets.
    """

    def __init__(self, paths: DataDecidePaths):
        """Initialize the pipeline with path management.

        Args:
            paths: DataDecidePaths instance for managing file locations
        """
        self.paths = paths

    def download_raw_data(
        self, force_reload: bool = False, verbose: bool = False
    ) -> None:
        """Download raw datasets from HuggingFace.

        Downloads perplexity and downstream evaluation datasets and saves them
        as parquet files for further processing.

        Args:
            force_reload: If True, re-download even if files exist
            verbose: If True, print progress messages
        """
        # Download perplexity evaluation dataset
        if not self.paths.ppl_eval_raw_path.exists() or force_reload:
            if verbose:
                print("Downloading perplexity evaluation dataset...")
            ppl_dataset = load_dataset(
                consts.HF_DATASET_NAMES["perplexity_eval_ds"], split="train"
            )
            ppl_dataset.to_parquet(self.paths.ppl_eval_raw_path)

        # Download downstream evaluation dataset
        if not self.paths.downstream_eval_raw_path.exists() or force_reload:
            if verbose:
                print("Downloading downstream evaluation dataset...")
            dwn_dataset = load_dataset(
                consts.HF_DATASET_NAMES["downstream_eval_ds"], split="train"
            )
            dwn_dataset.to_parquet(self.paths.downstream_eval_raw_path)

    def extract_step_mappings(
        self, force_reload: bool = False, verbose: bool = False
    ) -> None:
        """Extract step-to-token and step-to-compute mappings.

        Creates a mapping DataFrame that converts training steps to tokens
        and compute for each model parameter configuration.

        Args:
            force_reload: If True, recreate even if file exists
            verbose: If True, print progress messages
        """
        if not self.paths.step_to_token_compute_path.exists() or force_reload:
            if verbose:
                print("Extracting step-to-token and step-to-compute mapping...")

            # Load downstream data to extract step mappings
            dwn_df = pd.read_parquet(self.paths.downstream_eval_raw_path)
            step_to_token_compute_df = parsing.make_step_to_token_compute_df(dwn_df)
            step_to_token_compute_df.to_parquet(self.paths.step_to_token_compute_path)

    def parse_data(self, force_reload: bool = False, verbose: bool = False) -> None:
        """Parse raw evaluation datasets into structured format.

        Converts raw evaluation data into clean, analysis-ready DataFrames
        with standardized column names and data types.

        Args:
            force_reload: If True, reparse even if files exist
            verbose: If True, print progress messages
        """
        # Parse perplexity data
        if not self.paths.ppl_eval_parsed_path.exists() or force_reload:
            if verbose:
                print("Parsing perplexity evaluation data...")
            ppl_df = pd.read_parquet(self.paths.ppl_eval_raw_path)
            ppl_parsed_df = parsing.parse_perplexity_dataframe(ppl_df)
            ppl_parsed_df.to_parquet(self.paths.ppl_eval_parsed_path)

        # Parse downstream data
        if not self.paths.downstream_eval_parsed_path.exists() or force_reload:
            if verbose:
                print("Parsing downstream evaluation data (this may take a while)...")
            dwn_df = pd.read_parquet(self.paths.downstream_eval_raw_path)
            dwn_parsed_df = parsing.parse_downstream_dataframe(dwn_df)
            dwn_parsed_df.to_parquet(self.paths.downstream_eval_parsed_path)

    def create_derived_datasets(
        self, force_reload: bool = False, verbose: bool = False
    ) -> None:
        """Create derived datasets by merging and aggregating parsed data.

        Creates the full evaluation dataset by merging perplexity and downstream
        results, then generates mean and standard deviation datasets.

        Args:
            force_reload: If True, recreate even if files exist
            verbose: If True, print progress messages
        """
        # Create full evaluation dataset
        if not self.paths.full_eval_ds_path.exists() or force_reload:
            if verbose:
                print("Creating full evaluation dataset...")

            # Load parsed data
            dwn_parsed_df = pd.read_parquet(self.paths.downstream_eval_parsed_path)
            ppl_parsed_df = pd.read_parquet(self.paths.ppl_eval_parsed_path)
            step_to_token_compute_df = pd.read_parquet(
                self.paths.step_to_token_compute_path
            )

            # Create merged dataset
            full_eval_ds = self._create_full_eval_df(
                dwn_parsed_df,
                ppl_parsed_df,
                step_to_token_compute_df,
            )
            full_eval_ds.to_parquet(self.paths.full_eval_ds_path)

        # Create mean and standard deviation datasets
        if (
            not self.paths.mean_eval_ds_path.exists()
            or not self.paths.std_eval_ds_path.exists()
            or force_reload
        ):
            if verbose:
                print("Creating mean and standard deviation datasets...")

            full_eval_ds = pd.read_parquet(self.paths.full_eval_ds_path)
            mean_eval_ds, std_eval_ds = df_utils.create_mean_std_df(full_eval_ds)

            mean_eval_ds.to_parquet(self.paths.mean_eval_ds_path)
            std_eval_ds.to_parquet(self.paths.std_eval_ds_path)

    def run(self, force_reload: bool = False, verbose: bool = False) -> None:
        """Run the complete data processing pipeline.

        Executes all pipeline steps in order: download -> extract mappings ->
        parse -> create derived datasets.

        Args:
            force_reload: If True, recreate all files even if they exist
            verbose: If True, print progress messages for each step
        """
        if verbose:
            print("Starting DataDecide pipeline...")

        self.download_raw_data(force_reload=force_reload, verbose=verbose)
        self.extract_step_mappings(force_reload=force_reload, verbose=verbose)
        self.parse_data(force_reload=force_reload, verbose=verbose)
        self.create_derived_datasets(force_reload=force_reload, verbose=verbose)

        if verbose:
            print("Pipeline completed successfully!")

    def _create_full_eval_df(
        self,
        dwn_parsed_df: pd.DataFrame,
        ppl_parsed_df: pd.DataFrame,
        step_to_token_compute_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create the full evaluation dataset by merging all data sources.

        Args:
            dwn_parsed_df: Parsed downstream evaluation data
            ppl_parsed_df: Parsed perplexity evaluation data
            step_to_token_compute_df: Step-to-token/compute mapping data

        Returns:
            Merged DataFrame with all evaluation metrics and metadata
        """
        # Merge downstream and perplexity data
        merged_df = dwn_parsed_df.merge(
            ppl_parsed_df,
            on=["params", "data", "seed", "step"],
            how="outer",
            suffixes=["_dwn", "_ppl"],
        )

        # Add token and compute columns
        merged_df = (
            merged_df.merge(step_to_token_compute_df, on="params", how="left")
            .assign(
                tokens=lambda x: x["step"] * x["tokens_per_step"],
                compute=lambda x: x["step"] * x["compute_per_step"],
            )
            .drop(columns=["tokens_per_step", "compute_per_step"])
        )

        # Reorder columns for better organization
        merged_df = parsing.reorder_df_cols(
            merged_df, consts.KEY_COLS + ["tokens", "compute"]
        )

        return merged_df
