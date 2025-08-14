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
            "enrich",
            "aggregate",
        ]

        self.stage_outputs = {
            "download": ["ppl_raw", "dwn_raw"],
            "metrics_expand": ["step_to_token_compute", "dwn_metrics_expanded"],
            "parse": ["ppl_parsed", "dwn_parsed"],
            "merge": ["full_eval_raw"],
            "enrich": ["full_eval"],
            "aggregate": ["mean_eval", "std_eval"],
        }

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
            print("Merging perplexity and downstream datasets...")
        ppl_df = pd.read_parquet(self.paths.get_path("ppl_parsed"))
        dwn_df = pd.read_parquet(self.paths.get_path("dwn_parsed"))

        full_eval_raw = df_utils.merge_ppl_and_dwn_dfs(ppl_df, dwn_df)
        full_eval_raw = parsing.reorder_df_cols(full_eval_raw, consts.KEY_COLS)
        full_eval_raw.to_parquet(self.paths.get_path("full_eval_raw"))

    def enrich_dataset(self, verbose: bool = False) -> None:
        if verbose:
            print("Creating full evaluation dataset with enrichments...")
        
        # Load raw merged data
        df = pd.read_parquet(self.paths.get_path("full_eval_raw"))
        
        # Add MMLU average
        df = df_utils.add_mmlu_average(df)
        
        # Load and merge dataset details
        from datadec import data_utils
        dataset_details = data_utils.load_ds_details_df(self.paths.ds_details_path)
        df = df.merge(dataset_details, on="data", how="left")
        
        # Load and merge model details  
        from datadec import model_utils
        model_details = model_utils.get_model_details_df()
        df = df.merge(model_details, on="params", how="left")
        
        # Add step-specific learning rate columns
        df = model_utils.add_lr_cols(df)
        
        # Reorder columns (KEY_COLS first)
        df = parsing.reorder_df_cols(df, consts.KEY_COLS)
        
        # Save as final full_eval
        df.to_parquet(self.paths.get_path("full_eval"))

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
            requested_stage = self.pipeline_stages.index(recompute_from)
            # Check if earlier stages have missing files
            earliest_missing = self._find_earliest_missing_stage(0)
            if earliest_missing < requested_stage:
                start_stage = earliest_missing
                missing_stage = self.pipeline_stages[earliest_missing]
                if verbose:
                    print(
                        f"Files from stage '{missing_stage}' are missing, adjusting to start from '{missing_stage}' instead of '{recompute_from}'"
                    )
            else:
                start_stage = requested_stage
                if verbose:
                    print(f"Starting DataDecide pipeline from '{recompute_from}'...")
        elif recompute_from is None:
            # Auto-detect earliest missing stage
            start_stage = self._find_earliest_missing_stage(0)
            if start_stage == len(self.pipeline_stages):
                if verbose:
                    print("All pipeline files exist, using cached data...")
            else:
                missing_stage = self.pipeline_stages[start_stage]
                if verbose:
                    print(
                        f"Auto-detected missing files from stage '{missing_stage}', starting pipeline from there..."
                    )
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
            self.enrich_dataset(verbose=verbose)

        if start_stage <= 5:
            self.create_aggregated_datasets(verbose=verbose)

        if verbose:
            print("Pipeline completed successfully!")

    def _verify_stage_files(self, stage_name: str) -> bool:
        """Check if all output files for a given stage exist."""
        if stage_name not in self.stage_outputs:
            return True

        for file_name in self.stage_outputs[stage_name]:
            file_path = self.paths.get_path(file_name)
            if not file_path.exists():
                return False
        return True

    def _find_earliest_missing_stage(self, from_stage: int = 0) -> int:
        """Find the index of the first stage with missing output files."""
        for i in range(from_stage, len(self.pipeline_stages)):
            stage_name = self.pipeline_stages[i]
            if not self._verify_stage_files(stage_name):
                return i
        return len(self.pipeline_stages)
