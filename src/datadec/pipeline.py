import pandas as pd
from datasets import load_dataset

from datadec import constants as consts
from datadec import parsing
from datadec import df_utils
from datadec.paths import DataDecidePaths


class DataPipeline:
    def __init__(self, paths: DataDecidePaths):
        self.paths = paths

        self.pipeline_stages = [
            "download",
            "metrics_expand",
            "parse",
            "merge",
            "aggregate",
        ]

    def download_raw_data(
        self, force_reload: bool = False, verbose: bool = False
    ) -> None:
        ppl_path = self.paths.get_path("ppl_raw")
        dwn_path = self.paths.get_path("dwn_raw")

        if force_reload or not ppl_path.exists():
            if verbose:
                print("Downloading perplexity evaluation dataset...")
            ppl_ds = load_dataset(consts.HF_DATASET_NAMES["perplexity_eval_ds"])
            ppl_df = ppl_ds["train"].to_pandas()
            ppl_df.to_parquet(ppl_path)

        if force_reload or not dwn_path.exists():
            if verbose:
                print("Downloading downstream evaluation dataset...")
            dwn_ds = load_dataset(consts.HF_DATASET_NAMES["downstream_eval_ds"])
            dwn_df = dwn_ds["train"].to_pandas()
            dwn_df.to_parquet(dwn_path)

    def extract_step_token_compute_mapping(self, verbose: bool = False) -> None:
        dwn_df = pd.read_parquet(self.paths.get_path("dwn_raw"))
        step_data = parsing.make_step_to_token_compute_df(dwn_df)
        step_data.to_parquet(self.paths.get_path("step_to_token_compute"))
        if verbose:
            print("Extracting step-to-token and step-to-compute mapping...")

    def expand_metrics_column(self, verbose: bool = False) -> None:
        dwn_df = pd.read_parquet(self.paths.get_path("dwn_raw"))
        expanded_df = parsing.expand_downstream_metrics(dwn_df)
        expanded_df.to_parquet(self.paths.get_path("dwn_metrics_expanded"))

    def parse_datasets(self, verbose: bool = False) -> None:
        if verbose:
            print("Parsing perplexity evaluation data...")
        ppl_df = pd.read_parquet(self.paths.get_path("ppl_raw"))
        ppl_parsed = parsing.parse_perplexity_dataframe(ppl_df)
        ppl_parsed.to_parquet(self.paths.get_path("ppl_parsed"))

        if verbose:
            print("Completing downstream parsing...")
        dwn_expanded = pd.read_parquet(self.paths.get_path("dwn_metrics_expanded"))
        dwn_parsed = parsing.complete_downstream_parsing(dwn_expanded)
        dwn_parsed.to_parquet(self.paths.get_path("dwn_parsed"))

    def merge_datasets(self, verbose: bool = False) -> None:
        if verbose:
            print("Creating full evaluation dataset...")
        ppl_df = pd.read_parquet(self.paths.get_path("ppl_parsed"))
        dwn_df = pd.read_parquet(self.paths.get_path("dwn_parsed"))

        full_eval_df = df_utils.merge_ppl_and_dwn_dfs(ppl_df, dwn_df)
        full_eval_df.to_parquet(self.paths.get_path("full_eval"))

    def create_aggregated_datasets(self, verbose: bool = False) -> None:
        if verbose:
            print("Creating mean and standard deviation datasets...")
        full_eval_df = pd.read_parquet(self.paths.get_path("full_eval"))

        mean_df, std_df = df_utils.create_mean_std_df(full_eval_df)
        mean_df.to_parquet(self.paths.get_path("mean_eval"))
        std_df.to_parquet(self.paths.get_path("std_eval"))

    def run(self, recompute_from: str = None, verbose: bool = False) -> None:
        if recompute_from == "all":
            start_stage = 0
            if verbose:
                print("Starting DataDecide pipeline from 'all'...")
        elif recompute_from in self.pipeline_stages:
            start_stage = self.pipeline_stages.index(recompute_from)
            if verbose:
                print(f"Starting DataDecide pipeline from '{recompute_from}'...")
        else:
            if verbose:
                print("Starting DataDecide pipeline (using cached files)...")
            start_stage = len(self.pipeline_stages)

        if start_stage <= 0:
            self.download_raw_data(force_reload=True, verbose=verbose)

        if start_stage <= 1:
            self.extract_step_token_compute_mapping(verbose=verbose)
            self.expand_metrics_column(verbose=verbose)

        if start_stage <= 2:
            self.parse_datasets(verbose=verbose)

        if start_stage <= 3:
            self.merge_datasets(verbose=verbose)

        if start_stage <= 4:
            self.create_aggregated_datasets(verbose=verbose)

        if verbose:
            print("Pipeline completed successfully!")
