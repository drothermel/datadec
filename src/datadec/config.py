from __future__ import annotations

import tomllib
from datetime import date
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class PublishedResultFile(ConfigModel):
    id: str
    path: str
    expected_size: int = Field(gt=0)
    category: Literal["scaling_law", "published_results"]

    @model_validator(mode="after")
    def validate_path_and_category(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            not path.parts
            or path.is_absolute()
            or path.as_posix() != self.path
            or ".." in path.parts
        ):
            raise ValueError("published result paths must be normalized relative paths")
        is_raw = path.parts[0] == "raw_data"
        if is_raw != (self.category == "scaling_law"):
            raise ValueError("only raw_data files may use the scaling_law category")
        return self


class PublishedResultsManifest(ConfigModel):
    folder_url: Literal[
        "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
    ]
    files: tuple[PublishedResultFile, ...]

    @model_validator(mode="after")
    def validate_unique_files(self) -> Self:
        ids = [file.id for file in self.files]
        if len(ids) != len(set(ids)):
            raise ValueError("published result Google Drive file IDs must be unique")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("published result paths must be unique")
        return self


class OLMESColumnContract(ConfigModel):
    name: str
    logical_type: Literal["string", "int64", "float64", "bool"]
    nullable: bool


class OLMESTableContract(ConfigModel):
    path: str | None = None
    path_template: str | None = None
    primary_key: tuple[str, ...]
    sort_key: tuple[str, ...]
    columns: tuple[OLMESColumnContract, ...]

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        if (self.path is None) == (self.path_template is None):
            raise ValueError("OLMES tables must define exactly one path")

        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("OLMES table column names must be unique")

        for key_name, key in (
            ("primary key", self.primary_key),
            ("sort key", self.sort_key),
        ):
            if not key:
                raise ValueError(f"OLMES table {key_name} must not be empty")
            if len(key) != len(set(key)):
                raise ValueError(f"OLMES table {key_name} columns must be unique")
            missing = set(key).difference(names)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"OLMES table {key_name} columns are missing: {missing_names}"
                )
        return self


class OLMESAggregatePrimaryMetricPolicy(ConfigModel):
    mmlu: str
    arc_challenge: str
    arc_easy: str
    boolq: str
    csqa: str
    hellaswag: str
    openbookqa: str
    piqa: str
    socialiqa: str
    winogrande: str


class OLMESMetricContract(ConfigModel):
    aggregate: tuple[str, ...]
    detailed_tasks: tuple[str, ...]
    detailed_instances: tuple[str, ...]
    detailed_choices: tuple[str, ...]
    not_reproducible_from_details: tuple[str, ...]
    aggregate_primary_metric: OLMESAggregatePrimaryMetricPolicy
    detailed_primary_metric_source: Literal["task_config.primary_metric"]
    detailed_primary_metric_column: Literal["primary_score"]

    @model_validator(mode="after")
    def validate_unique_metric_classifications(self) -> Self:
        for name, metrics in (
            ("aggregate", self.aggregate),
            ("detailed tasks", self.detailed_tasks),
            ("detailed instances", self.detailed_instances),
            ("detailed choices", self.detailed_choices),
            ("not reproducible", self.not_reproducible_from_details),
        ):
            if len(metrics) != len(set(metrics)):
                raise ValueError(f"OLMES {name} metrics must be unique")
        return self


class OLMESTables(ConfigModel):
    aggregate: OLMESTableContract
    detailed_tasks: OLMESTableContract
    detailed_instances: OLMESTableContract
    detailed_choices: OLMESTableContract


class OLMESIdentityContract(ConfigModel):
    native_id_kinds: tuple[Literal["integer", "string", "null"], ...]

    @model_validator(mode="after")
    def validate_native_id_kinds(self) -> Self:
        if (
            set(self.native_id_kinds) != {"integer", "string", "null"}
            or len(self.native_id_kinds) != 3
        ):
            raise ValueError("OLMES native ID kinds must cover integer, string, null")
        return self


class OLMESContract(ConfigModel):
    recipe_map: dict[str, str]
    seed_map: dict[int, str]
    identity: OLMESIdentityContract
    metrics: OLMESMetricContract
    tables: OLMESTables

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        for name, mapping in (
            ("recipe", self.recipe_map),
            ("seed", self.seed_map),
        ):
            if len(mapping.values()) != len(set(mapping.values())):
                raise ValueError(f"OLMES {name} mappings must be unique")

        for table in (
            self.tables.detailed_tasks,
            self.tables.detailed_instances,
            self.tables.detailed_choices,
        ):
            columns = {column.name: column for column in table.columns}
            seed_value = columns.get("seed_value")
            seed = columns.get("seed")
            if (
                seed_value is None
                or seed_value.logical_type != "int64"
                or seed_value.nullable
                or seed is None
                or seed.logical_type != "string"
                or seed.nullable
                or "seed_value" not in table.primary_key
                or "seed_value" not in table.sort_key
                or "seed" in table.primary_key
                or "seed" in table.sort_key
            ):
                raise ValueError(
                    "OLMES detailed tables use seed_value keys and seed attributes"
                )

        metric_tables = (
            (self.metrics.aggregate, self.tables.aggregate),
            (self.metrics.detailed_tasks, self.tables.detailed_tasks),
            (self.metrics.detailed_instances, self.tables.detailed_instances),
            (self.metrics.detailed_choices, self.tables.detailed_choices),
        )
        for metric_names, table in metric_tables:
            column_names = tuple(column.name for column in table.columns)
            if column_names[-len(metric_names) :] != metric_names:
                raise ValueError(
                    "OLMES metric classifications must match table column order"
                )

        aggregate_columns = {
            column.name: column for column in self.tables.aggregate.columns
        }
        for name in self.metrics.aggregate:
            column = aggregate_columns[name]
            if column.logical_type != "float64" or not column.nullable:
                raise ValueError(
                    "OLMES aggregate metrics must be nullable float64 columns"
                )

        policy_metrics = self.metrics.aggregate_primary_metric.model_dump().values()
        if not set(policy_metrics).issubset(self.metrics.aggregate):
            raise ValueError(
                "OLMES aggregate primary metric policy references unknown metrics"
            )
        if self.metrics.detailed_primary_metric_column not in (
            self.metrics.detailed_tasks
        ):
            raise ValueError(
                "OLMES detailed primary metric must be a detailed task metric"
            )
        if not set(self.metrics.not_reproducible_from_details).issubset(
            self.metrics.aggregate
        ):
            raise ValueError("OLMES non-reproducible metrics must be aggregate metrics")
        return self

    def validate_references(
        self,
        *,
        catalog: DataDecideCatalog,
        source_manifest: SourceManifest,
    ) -> Self:
        source_recipes = set(source_manifest.olmes_details.recipes)
        if set(self.recipe_map) != source_recipes:
            raise ValueError(
                "OLMES recipe mapping must exactly cover source detail recipes"
            )

        catalog_recipes = {
            recipe
            for family in catalog.data_recipe_families.values()
            for recipe in family
        }
        if set(self.recipe_map.values()) != catalog_recipes:
            raise ValueError(
                "OLMES recipe mapping must be a bijection with catalog recipes"
            )
        if set(self.seed_map.values()) != set(catalog.seed_map):
            raise ValueError("OLMES seed mapping must exactly cover catalog seeds")
        return self


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


@cache
def load_published_results_manifest() -> PublishedResultsManifest:
    return PublishedResultsManifest.model_validate(_load_toml("published_results.toml"))


@cache
def load_olmes_contract() -> OLMESContract:
    contract = OLMESContract.model_validate(_load_toml("olmes.toml"))
    return contract.validate_references(
        catalog=load_catalog(), source_manifest=load_source_manifest()
    )


__all__ = [
    "ArchiveSource",
    "DataDecideCatalog",
    "DatasetSource",
    "DetailSource",
    "OLMESAggregatePrimaryMetricPolicy",
    "OLMESColumnContract",
    "OLMESContract",
    "OLMESIdentityContract",
    "OLMESMetricContract",
    "OLMESTableContract",
    "OLMESTables",
    "PublishedResultFile",
    "PublishedResultsManifest",
    "SourceManifest",
    "config_file",
    "load_catalog",
    "load_olmes_contract",
    "load_published_results_manifest",
    "load_source_manifest",
]
