from __future__ import annotations

from pathlib import Path

from datadec.config import load_source_manifest

DEFAULT_DATA_DIR = "./data"


class DataDecidePaths:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        manifest = load_source_manifest()
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.dataframes = {
            "ppl_raw": manifest.ppl.output,
            "dwn_raw": manifest.olmes.output,
            "dwn_metrics_expanded": "dwn_metrics_expanded.parquet",
            "ppl_dwn_merged": "ppl_dwn_merged.parquet",
            "ppl_processed": "processed/ppl.parquet",
            "olmes_processed": "processed/olmes.parquet",
            "ppl_parsed": "ppl_eval_parsed.parquet",
            "dwn_parsed": "downstream_eval_parsed.parquet",
            "full_eval_raw": "full_eval_raw.parquet",
            "full_eval": "full_eval.parquet",
            "mean_eval": "mean_eval.parquet",
            "std_eval": "std_eval.parquet",
            "full_eval_melted": "full_eval_melted.parquet",
            "mean_eval_melted": "mean_eval_melted.parquet",
        }

    @property
    def available_dataframes(self) -> list[str]:
        return list(self.dataframes.keys())

    def check_name_in_paths(self, name: str) -> bool:
        return name in self.dataframes

    def get_path(self, name: str) -> Path:
        if name not in self.dataframes:
            available = ", ".join(sorted(self.dataframes.keys()))
            raise ValueError(f"Unknown dataframe '{name}'. Available: {available}")
        return self.data_dir / self.dataframes[name]

    def get_existing_path(self, name: str) -> Path | None:
        path = self.get_path(name)
        if not path.exists():
            return None
        return path

    def parquet_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.parquet"

    def dataset_path(self, max_params_str: str) -> Path:
        return self.dataset_dir / f"dataset_{max_params_str}.pkl"
