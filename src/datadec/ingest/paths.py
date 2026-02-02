from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from marimo_utils.display import add_marimo_display

__all__ = ["Paths"]


@add_marimo_display()
class Paths(BaseModel):
    username: str = "drotherm"

    data_dir: Path = Field(
        default_factory=lambda data: Path.home() / data["username"] / "data"
    )
    data_cache_dir: Path = Field(
        default_factory=lambda data: data["data_dir"] / "cache"
    )
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
