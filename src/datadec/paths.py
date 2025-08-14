"""Path management for DataDecide datasets and processed files.

This module contains the DataDecidePaths class which handles all file path
construction and directory management for the DataDecide library.
"""

from pathlib import Path


class DataDecidePaths:
    """Manages file paths for DataDecide datasets and processed outputs.

    This class centralizes all path construction logic using a single dictionary
    that serves as the source of truth for all DataFrame file locations.
    """

    def __init__(self, data_dir: str = "./data"):
        """Initialize paths with base data directory.

        Args:
            data_dir: Base directory for all DataDecide data storage
        """
        self.data_dir = Path(data_dir) / "datadecide"
        self.dataset_dir = self.data_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Single source of truth - all DataFrame paths in one place
        self.dataframes = {
            # Raw data
            "ppl_raw": "ppl_eval",
            "dwn_raw": "downstream_eval",
            
            # Intermediate processing (in order of pipeline execution)
            "dwn_metrics_expanded": "dwn_metrics_expanded",  # The slow step (2-5 mins)
            "dwn_mmlu_averaged": "dwn_mmlu_averaged",
            "dwn_pivoted": "dwn_pivoted", 
            "ppl_dwn_merged": "ppl_dwn_merged",
            
            # Parsed data
            "ppl_parsed": "ppl_eval_parsed",
            "dwn_parsed": "downstream_eval_parsed",
            "step_to_token_compute": "step_to_token_compute",
            
            # Final datasets
            "full_eval": "full_eval",
            "mean_eval": "mean_eval",
            "std_eval": "std_eval",
            
            # Derived analysis DataFrames
            "full_eval_with_details": "full_eval_with_details",
            "full_eval_with_lr": "full_eval_with_lr",
            "mean_eval_with_details": "mean_eval_with_details",
            "mean_eval_with_lr": "mean_eval_with_lr",
        }

        # Special non-parquet paths (static files in package)
        package_root = Path(__file__).parent
        self.ds_details_path = package_root / "data" / "dataset_features.csv"

    def get_path(self, name: str) -> Path:
        """Get path for a DataFrame by name.

        Args:
            name: Name of the DataFrame (key from self.dataframes)

        Returns:
            Path to the parquet file for the DataFrame

        Raises:
            ValueError: If DataFrame name is not recognized
        """
        if name not in self.dataframes:
            available = ", ".join(sorted(self.dataframes.keys()))
            raise ValueError(f"Unknown dataframe '{name}'. Available: {available}")
        return self.data_dir / f"{self.dataframes[name]}.parquet"

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

    @property
    def available_dataframes(self) -> list[str]:
        """Get list of all available DataFrame names."""
        return list(self.dataframes.keys())