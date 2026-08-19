from __future__ import annotations

import tomllib
from datetime import date
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

_CONFIG_PACKAGE = "datadec"
_SOURCE_CONFIGS_DIR = Path(__file__).parents[2] / "configs"


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingConstants(ConfigModel):
    max_sequence_length: int
    token_length_multiplier: int
    model_size_normalization: int
    learning_rate_exponent: float
    learning_rate_base: float
    final_learning_rate_ratio: float
    batch_size_coefficient: int
    batch_size_exponent: float
    gpus_per_node: int
    microbatch_size: int


class ModelDefaults(ConfigModel):
    default_seed: int
    length_str: str
    lr_warmup_start: float
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    weight_tying: bool
    alibi: bool
    rope: bool
    flash_attention: bool
    attention_dropout: float
    attention_layer_norm: bool
    include_bias: bool
    layer_norm_type: str
    layer_norm_with_affine: bool
    layer_norm_eps: float
    bias_for_layer_norm: bool
    attention_layer_norm_with_affine: bool
    activation_type: str
    residual_dropout: float
    embedding_dropout: float
    max_sequence_length: int
    vocab_size: int
    embedding_size: int
    eos_token_id: int
    pad_token_id: int
    init_device: str
    init_fn: str
    init_std: float
    init_cutoff_factor: int


class ModelDefinition(ConfigModel):
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    true_size: int
    max_step: int


class RecipeGroups(ConfigModel):
    base_recipes: list[str]
    base_and_quality_control: list[str]
    without_ablations: list[str]
    custom_families: dict[str, list[str]]
    perplexity_performance: dict[str, list[str]]
    olmes_performance: dict[str, list[str]]


class DataDecideCatalog(ConfigModel):
    mmlu_tasks: list[str]
    olmes_tasks: list[str]
    metric_names: list[str]
    drop_metrics: list[str]
    training: TrainingConstants
    model_defaults: ModelDefaults
    models: list[ModelDefinition]
    data_recipe_families: dict[str, list[str]]
    seed_map: dict[str, int]
    perplexity_name_map: dict[str, str]
    recipe_groups: RecipeGroups

    @model_validator(mode="after")
    def validate_unique_models(self) -> Self:
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        return self


class DatasetSource(ConfigModel):
    id: str
    provider: Literal["datasets"]
    repo_id: str
    revision: str
    split: str
    output: str


class DetailSource(ConfigModel):
    id: str
    provider: Literal["huggingface_hub"]
    repo_type: Literal["dataset"]
    repo_id: str
    revision: str
    filename_template: str
    output_root: str
    recipes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_unique_recipes(self) -> Self:
        if len(self.recipes) != len(set(self.recipes)):
            raise ValueError("OLMES detail recipes must be unique")
        return self


class ArchiveSource(ConfigModel):
    id: str
    provider: Literal["google_drive_folder"]
    url: str
    downloaded_on: date


class SourceManifest(ConfigModel):
    ppl: DatasetSource
    olmes: DatasetSource
    olmes_details: DetailSource
    archives: tuple[ArchiveSource, ...]


def config_file(filename: str) -> Traversable:
    packaged = files(_CONFIG_PACKAGE).joinpath("configs", filename)
    if packaged.is_file():
        return packaged

    source = _SOURCE_CONFIGS_DIR / filename
    if source.is_file():
        return source
    raise FileNotFoundError(f"DataDecide config file not found: {filename}")


def _load_toml(filename: str) -> dict[str, object]:
    with config_file(filename).open("rb") as file:
        return tomllib.load(file)


@cache
def load_catalog() -> DataDecideCatalog:
    return DataDecideCatalog.model_validate(_load_toml("catalog.toml"))


@cache
def load_source_manifest() -> SourceManifest:
    return SourceManifest.model_validate(_load_toml("sources.toml"))


DATASET_FEATURES_CSV: Traversable = config_file("dataset_features.csv")

__all__ = [
    "DATASET_FEATURES_CSV",
    "ArchiveSource",
    "DataDecideCatalog",
    "DatasetSource",
    "DetailSource",
    "SourceManifest",
    "config_file",
    "load_catalog",
    "load_source_manifest",
]
