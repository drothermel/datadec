import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from datadec.optional import add_marimo_display

DATADEC_DATA_DIR_ENV = "DATADEC_DATA_DIR"

__all__ = ["DATADEC_DATA_DIR_ENV", "Paths"]


@add_marimo_display()
class Paths(BaseModel):
    """Default paths for metrics-all ingestion.

    Set ``DATADEC_DATA_DIR`` to override the default base directory (``~/data``).
    """
    @staticmethod
    def _default_data_dir() -> Path:
        env_value = os.getenv(DATADEC_DATA_DIR_ENV)
        if env_value:
            return Path(env_value).expanduser()
        return Path.home() / "data"

    data_dir: Path = Field(default_factory=_default_data_dir)
    data_cache_dir: Path = Field(default_factory=lambda data: data["data_dir"] / "cache")
    metrics_all_dir: Path = Field(
        default_factory=lambda data: data["data_dir"] / "datadec"
    )

    @model_validator(mode="before")
    def validate_path_types(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, Path | str):
                    raise ValueError(f"Invalid path type for {key}: {type(value)}")
                data[key] = Path(value)
        return data
