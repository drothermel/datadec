from pathlib import Path


class DataDecidePaths:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir) / "datadecide"
        self.dataset_dir = self.data_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.dataframes = {
            "ppl_raw": "ppl_eval",
            "dwn_raw": "downstream_eval",
            "dwn_metrics_expanded": "dwn_metrics_expanded",
            "dwn_mmlu_averaged": "dwn_mmlu_averaged",
            "dwn_pivoted": "dwn_pivoted",
            "ppl_dwn_merged": "ppl_dwn_merged",
            "ppl_parsed": "ppl_eval_parsed",
            "dwn_parsed": "downstream_eval_parsed",
            "step_to_token_compute": "step_to_token_compute",
            "full_eval_raw": "full_eval_raw",
            "full_eval": "full_eval",
            "mean_eval": "mean_eval",
            "std_eval": "std_eval",
            "full_eval_with_details": "full_eval_with_details",
            "full_eval_with_lr": "full_eval_with_lr",
            "mean_eval_with_details": "mean_eval_with_details",
            "mean_eval_with_lr": "mean_eval_with_lr",
        }

        package_root = Path(__file__).parent
        self.ds_details_path = package_root / "data" / "dataset_features.csv"

    def get_path(self, name: str) -> Path:
        if name not in self.dataframes:
            available = ", ".join(sorted(self.dataframes.keys()))
            raise ValueError(f"Unknown dataframe '{name}'. Available: {available}")
        return self.data_dir / f"{self.dataframes[name]}.parquet"

    def parquet_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.parquet"

    def dataset_path(self, max_params_str: str) -> Path:
        return self.dataset_dir / f"dataset_{max_params_str}.pkl"

    @property
    def available_dataframes(self) -> list[str]:
        return list(self.dataframes.keys())
