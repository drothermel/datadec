from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset

from datadec import constants as consts
from datadec import data_utils, df_utils, model_utils, parsing
from datadec.paths import DataDecidePaths


def download_dataset(
    path: Path,
    repo_id: str,
    split: str,
) -> None:
    raw_df = load_dataset(repo_id, split=split)
    raw_df.to_parquet(path)


def verbose_print(msg: str, verbose: bool = False) -> None:
    if verbose:
        print(f">> {msg}")


class DataPipeline:
    def __init__(self, paths: DataDecidePaths):
        self.paths = paths
        self.pipeline_stage_fxns = {
            "download": self.download_raw_data,
            "metrics_expand": self.expand_dwn_metrics_column,
            "parse": self.parse_datasets,
            "merge": self.merge_datasets,
            "enrich": self.enrich_dataset,
            "aggregate": self.create_aggregated_datasets,
        }

        self.stage_outputs = {
            "download": ["ppl_raw", "dwn_raw"],
            "metrics_expand": ["dwn_metrics_expanded"],
            "parse": ["ppl_parsed", "dwn_parsed"],
            "merge": ["full_eval_raw"],
            "enrich": ["full_eval"],
            "aggregate": ["mean_eval", "std_eval"],
        }

    def download_raw_data(self, verbose: bool = False) -> None:
        for raw_metric_type in ["ppl", "dwn"]:
            verbose_print(f"Downloading {raw_metric_type} raw data", verbose)
            download_dataset(
                path=self.paths.get_path(f"{raw_metric_type}_raw"),
                repo_id=consts.HF_DATASET_NAMES[f"{raw_metric_type}_eval_ds"],
                split=consts.HF_DATASET_SPLIT,
            )

    def expand_dwn_metrics_column(self, verbose: bool = False) -> None:
        verbose_print("Expanding downstream metrics column", verbose)
        dwn_df = pd.read_parquet(self.paths.get_path("dwn_raw"))
        expanded_df = parsing.list_col_to_columns(dwn_df, "metrics")
        expanded_df.to_parquet(self.paths.get_path("dwn_metrics_expanded"))

    def parse_datasets(self, verbose: bool = False) -> None:
        verbose_print("Downstream DF Parsing", verbose)
        dwn_expanded = pd.read_parquet(self.paths.get_path("dwn_metrics_expanded"))
        dwn_parsed = parsing.parse_downstream_expanded_df(dwn_expanded)
        dwn_parsed.to_parquet(self.paths.get_path("dwn_parsed"))

        verbose_print("Perplexity DF Parsing", verbose)
        ppl_df = pd.read_parquet(self.paths.get_path("ppl_raw"))
        ppl_parsed = parsing.parse_perplexity_df(ppl_df)
        ppl_parsed.to_parquet(self.paths.get_path("ppl_parsed"))

    def merge_datasets(self, verbose: bool = False) -> None:
        verbose_print("Merging ppl, dwn, dataset and model detail dfs", verbose)
        ppl_df = pd.read_parquet(self.paths.get_path("ppl_parsed"))
        dwn_df = pd.read_parquet(self.paths.get_path("dwn_parsed"))
        dataset_details_df = data_utils.get_data_recipe_details_df(
            self.paths.ds_details_csv_path
        )
        model_details_df = model_utils.get_model_details_df()

        # Merge all dfs and save
        full_eval_raw = parsing.merge_all_dfs(
            ppl_df, dwn_df, dataset_details_df, model_details_df
        )
        full_eval_raw.to_parquet(self.paths.get_path("full_eval_raw"))

    def enrich_dataset(self, verbose: bool = False) -> None:
        verbose_print("Begin dataset enrichment", verbose)
        df = pd.read_parquet(self.paths.get_path("full_eval_raw"))
        df = model_utils.add_lr_cols(df)
        df.to_parquet(self.paths.get_path("full_eval"))

    def create_aggregated_datasets(self, verbose: bool = False) -> None:
        if verbose:
            print("Creating mean and standard deviation datasets...")
        full_eval_df = pd.read_parquet(self.paths.get_path("full_eval"))

        mean_df, std_df = df_utils.create_mean_std_df(full_eval_df)
        mean_df.to_parquet(self.paths.get_path("mean_eval"))
        std_df.to_parquet(self.paths.get_path("std_eval"))

    def run(self, recompute_from: Optional[str] = None, verbose: bool = False) -> None:
        recompute_from = "download" if recompute_from == "all" else recompute_from
        for stage_name, stage_fxn in self.pipeline_stage_fxns.items():
            expected_output_exists = [
                self.paths.get_path(out).exists()
                for out in self.stage_outputs[stage_name]
            ]
            if not all(expected_output_exists) or recompute_from in [stage_name, None]:
                stage_fxn(verbose=verbose)
