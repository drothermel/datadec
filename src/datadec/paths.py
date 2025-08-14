"""Path management for DataDecide datasets and processed files.

This module contains the DataDecidePaths class which handles all file path
construction and directory management for the DataDecide library.
"""

from pathlib import Path


class DataDecidePaths:
    """Manages file paths for DataDecide datasets and processed outputs.

    This class centralizes all path construction logic and ensures that
    necessary directories are created automatically.
    """

    def __init__(self, data_dir: str = "./data"):
        """Initialize paths with base data directory.

        Args:
            data_dir: Base directory for all DataDecide data storage
        """
        self.data_dir = Path(data_dir) / "datadecide"
        self.dataset_dir = self.data_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.ds_details_path = self.data_dir / "dataset_features.csv"

    def parquet_path(self, name: str) -> Path:
        """Get path for a parquet file in the data directory.

        Args:
            name: Base name for the parquet file (without extension)

        Returns:
            Path to the parquet file
        """
        return self.data_dir / f"{name}.parquet"

    def dataset_path(self, max_params_str: str) -> Path:
        """Get path for a dataset pickle file.

        Args:
            max_params_str: Parameter size string (e.g., "1B", "300M")

        Returns:
            Path to the dataset pickle file
        """
        return self.dataset_dir / f"dataset_{max_params_str}.pkl"

    # ------------ DataDecide Raw Paths ------------
    @property
    def ppl_eval_raw_path(self) -> Path:
        """Path to raw perplexity evaluation data."""
        return self.parquet_path("ppl_eval")

    @property
    def downstream_eval_raw_path(self) -> Path:
        """Path to raw downstream evaluation data."""
        return self.parquet_path("downstream_eval")

    # ------------ DataDecide Parsed Paths ------------
    @property
    def step_to_token_compute_path(self) -> Path:
        """Path to step-to-token-compute mapping data."""
        return self.parquet_path("step_to_token_compute")

    @property
    def ppl_eval_parsed_path(self) -> Path:
        """Path to parsed perplexity evaluation data."""
        return self.parquet_path("ppl_eval_parsed")

    @property
    def downstream_eval_parsed_path(self) -> Path:
        """Path to parsed downstream evaluation data."""
        return self.parquet_path("downstream_eval_parsed")

    @property
    def full_eval_ds_path(self) -> Path:
        """Path to full evaluation dataset (merged ppl + downstream)."""
        return self.parquet_path("full_eval")

    @property
    def mean_eval_ds_path(self) -> Path:
        """Path to mean evaluation dataset (averaged across seeds)."""
        return self.parquet_path("mean_eval")

    @property
    def std_eval_ds_path(self) -> Path:
        """Path to standard deviation evaluation dataset."""
        return self.parquet_path("std_eval")

    # ------------ Intermediate Processing Paths ------------
    @property
    def dwn_metrics_expanded_path(self) -> Path:
        """Path to downstream data after metrics column expansion (slow step)."""
        return self.parquet_path("dwn_metrics_expanded")

    @property
    def dwn_mmlu_averaged_path(self) -> Path:
        """Path to downstream data after MMLU averaging."""
        return self.parquet_path("dwn_mmlu_averaged")

    @property
    def dwn_pivoted_path(self) -> Path:
        """Path to downstream data after task metrics pivoting."""
        return self.parquet_path("dwn_pivoted")

    @property
    def ppl_dwn_merged_path(self) -> Path:
        """Path to merged perplexity and downstream data (before tokens/compute)."""
        return self.parquet_path("ppl_dwn_merged")

    # ------------ Derived Analysis Paths ------------
    @property
    def full_eval_with_details_path(self) -> Path:
        """Path to full eval merged with model/dataset details."""
        return self.parquet_path("full_eval_with_details")

    @property
    def full_eval_with_lr_path(self) -> Path:
        """Path to full eval with learning rate columns."""
        return self.parquet_path("full_eval_with_lr")

    @property
    def mean_eval_with_details_path(self) -> Path:
        """Path to mean eval merged with model/dataset details."""
        return self.parquet_path("mean_eval_with_details")

    @property
    def mean_eval_with_lr_path(self) -> Path:
        """Path to mean eval with learning rate columns."""
        return self.parquet_path("mean_eval_with_lr")
