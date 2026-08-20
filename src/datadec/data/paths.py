from __future__ import annotations

from pathlib import Path

from datadec.config import (
    load_olmes_contract,
    load_scaling_law_contract,
    load_source_manifest,
)

DEFAULT_DATA_DIR = "./data"


class DataDecidePaths:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        manifest = load_source_manifest()
        self.data_dir = Path(data_dir)

        self.dataframes = {
            "ppl_raw": manifest.ppl.output,
            "dwn_raw": manifest.olmes.output,
            "ppl_processed": "processed/ppl.parquet",
            "olmes_processed": "processed/olmes.parquet",
        }

    def get_path(self, name: str) -> Path:
        if name not in self.dataframes:
            available = ", ".join(sorted(self.dataframes.keys()))
            raise ValueError(f"Unknown dataframe '{name}'. Available: {available}")
        return self.data_dir / self.dataframes[name]

    def scaling_law_raw_paths(self) -> tuple[Path, ...]:
        contract = load_scaling_law_contract()
        raw_directory = self.data_dir / contract.raw_directory
        return tuple(
            raw_directory / filename for filename in contract.source_precedence
        )

    def scaling_law_evaluations_path(self) -> Path:
        contract = load_scaling_law_contract()
        return self.data_dir / contract.tables.evaluations.path

    def scaling_law_checkpoint_losses_path(self) -> Path:
        contract = load_scaling_law_contract()
        return self.data_dir / contract.tables.checkpoint_losses.path

    def olmes_details_tasks_path(self, recipe: str) -> Path:
        return self._olmes_details_table_path("detailed_tasks", recipe)

    def olmes_details_instances_path(self, recipe: str) -> Path:
        return self._olmes_details_table_path("detailed_instances", recipe)

    def olmes_details_choices_path(self, recipe: str) -> Path:
        return self._olmes_details_table_path("detailed_choices", recipe)

    def _olmes_details_table_path(self, table_name: str, recipe: str) -> Path:
        contract = load_olmes_contract()
        table = getattr(contract.tables, table_name)
        template = table.path_template
        if template is None:
            raise ValueError(f"OLMES {table_name} table has no path_template")
        return self.data_dir / template.format(recipe=recipe)
