from __future__ import annotations

import tomllib
from datetime import date
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from string import Formatter
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
    flops_per_token_per_parameter: int = Field(gt=0)


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
    nominal_parameter_count: int = Field(gt=0)
    training_parameter_count: int = Field(gt=0)
    exact_parameter_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_nominal_parameter_count(self) -> Self:
        suffix_multipliers = {"M": 1_000_000, "B": 1_000_000_000}
        suffix = self.name[-1:]
        try:
            expected = int(self.name[:-1]) * suffix_multipliers[suffix]
        except (KeyError, ValueError):
            raise ValueError(
                "model names must encode nominal parameter counts with M or B"
            ) from None
        if self.nominal_parameter_count != expected:
            raise ValueError("model nominal_parameter_count must match its model name")
        return self


class RecipeGroups(ConfigModel):
    base_recipes: list[str]
    base_and_quality_control: list[str]
    without_ablations: list[str]
    custom_families: dict[str, list[str]]
    perplexity_performance: dict[str, list[str]]
    olmes_performance: dict[str, list[str]]


class DataDecideCatalog(ConfigModel):
    metric_names: list[str]
    drop_metrics: list[str]
    training: TrainingConstants
    model_defaults: ModelDefaults
    models: list[ModelDefinition]
    data_recipe_families: dict[str, list[str]]
    seed_map: dict[str, int]
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


type PublishedResultCategory = Literal[
    "scaling_law", "published_results", "published_figures"
]
type PublishedResultSchema = Literal[
    "transformed",
    "prediction_model_scale",
    "processed_ladder",
    "cheap_decisions",
    "new_eval_decision_accuracy",
    "new_eval_means",
    "target_pairs",
]
type PublishedResultUnit = Literal[
    "cheap-decisions",
    "new-eval-intermediates",
    "outputs2",
    "per-task-arc-challenge",
    "per-task-arc-easy",
    "per-task-boolq",
    "per-task-csqa",
    "per-task-hellaswag",
    "per-task-mmlu",
    "per-task-openbookqa",
    "per-task-piqa",
    "per-task-socialiqa",
    "per-task-winogrande",
    "processed-data-current",
    "processed-data-pre-extra-real",
]


_SCHEMA_FILENAMES: dict[PublishedResultSchema, frozenset[str]] = {
    "transformed": frozenset({"1_metric_transformed.csv", "1_primary_transformed.csv"}),
    "prediction_model_scale": frozenset({"2_prediction_model_scale.csv"}),
    "processed_ladder": frozenset(
        {
            "results_ladder_5xC_seeds_cleaned_correct_params.csv",
            "results_ladder_5xC_seeds_cleaned_correct_params_pre_extra_real.csv",
            "results_ladder_5xC_seeds_dirty_correct_params.csv",
            "results_ladder_5xC_seeds_dirty_correct_params_pre_extra_real.csv",
        }
    ),
    "cheap_decisions": frozenset({"cheap_decisions_stacked_rc_pred_all.csv"}),
    "new_eval_decision_accuracy": frozenset({"davidh_new_evals_decision_accuracy.csv"}),
    "new_eval_means": frozenset({"davidh_new_evals_means_df.csv"}),
    "target_pairs": frozenset({"0_target_pairs.json"}),
}


def _published_result_unit_for_path(path: PurePosixPath) -> PublishedResultUnit:
    if path.as_posix() == "cheap_decisions_stacked_rc_pred_all.csv":
        return "cheap-decisions"
    if path.parts[0] == "new_eval_intermediates":
        return "new-eval-intermediates"
    if path.parts[0] == "outputs2":
        return "outputs2"
    if len(path.parts) >= 2 and path.parts[0] == "per_task_out":
        task_units: dict[str, PublishedResultUnit] = {
            "arc_challenge_out": "per-task-arc-challenge",
            "arc_easy_out": "per-task-arc-easy",
            "boolq_out": "per-task-boolq",
            "csqa_out": "per-task-csqa",
            "hellaswag_out": "per-task-hellaswag",
            "mmlu_out": "per-task-mmlu",
            "openbookqa_out": "per-task-openbookqa",
            "piqa_out": "per-task-piqa",
            "socialiqa_out": "per-task-socialiqa",
            "winogrande_out": "per-task-winogrande",
        }
        try:
            return task_units[path.parts[1]]
        except KeyError:
            pass
    if path.parts[0] == "processed_data":
        if path.name.endswith("_pre_extra_real.csv"):
            return "processed-data-pre-extra-real"
        return "processed-data-current"
    raise ValueError("structured published result path has no publication unit")


def _published_result_schema_for_path(path: PurePosixPath) -> PublishedResultSchema:
    if path.as_posix() == "cheap_decisions_stacked_rc_pred_all.csv":
        return "cheap_decisions"
    if path.parts[0] == "new_eval_intermediates":
        schemas: dict[str, PublishedResultSchema] = {
            "davidh_new_evals_decision_accuracy.csv": "new_eval_decision_accuracy",
            "davidh_new_evals_means_df.csv": "new_eval_means",
        }
        try:
            return schemas[path.name]
        except KeyError:
            pass
    if (
        path.parts[0] == "processed_data"
        and path.name in _SCHEMA_FILENAMES["processed_ladder"]
    ):
        return "processed_ladder"
    if path.parts[0] == "outputs2" or (
        len(path.parts) >= 2 and path.parts[0] == "per_task_out"
    ):
        for schema in ("target_pairs", "transformed", "prediction_model_scale"):
            if path.name in _SCHEMA_FILENAMES[schema]:
                return schema
    raise ValueError("structured published result path has no schema family")


class PublishedResultFile(ConfigModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    id: str
    path: str
    expected_size: int = Field(gt=0)
    category: PublishedResultCategory
    publication_unit: PublishedResultUnit | None = None
    schema_: PublishedResultSchema | None = Field(default=None, alias="schema")

    @property
    def schema(self) -> PublishedResultSchema | None:
        return self.schema_

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
        suffix = path.suffix.lower()
        is_raw = path.parts[0] == "raw_data"
        if is_raw != (self.category == "scaling_law"):
            raise ValueError("only raw_data files may use the scaling_law category")
        expected_extensions = {
            "scaling_law": {".csv"},
            "published_results": {".csv", ".json"},
            "published_figures": {".pdf", ".png"},
        }
        if suffix not in expected_extensions[self.category]:
            raise ValueError(
                f"invalid extension for published result category {self.category}"
            )
        is_structured = self.category == "published_results"
        if is_structured != (
            self.publication_unit is not None and self.schema is not None
        ):
            raise ValueError(
                "structured published results require publication_unit and schema; "
                "other categories must omit both"
            )
        if is_structured:
            try:
                expected_schema = _published_result_schema_for_path(path)
            except ValueError:
                raise ValueError(
                    "published result schema does not match its source path"
                ) from None
            if self.schema != expected_schema:
                raise ValueError(
                    "published result schema does not match its source path"
                )
            try:
                expected_unit = _published_result_unit_for_path(path)
            except ValueError:
                raise ValueError(
                    "published result publication_unit does not match its source path"
                ) from None
            if self.publication_unit != expected_unit:
                raise ValueError(
                    "published result publication_unit does not match its source path"
                )
        return self

    def parquet_relative_path(self) -> PurePosixPath:
        if self.category != "published_results":
            raise ValueError("only structured published results have Parquet outputs")
        return PurePosixPath(self.path).with_suffix(".parquet")


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
        outputs = [
            file.parquet_relative_path().as_posix()
            for file in self.files
            if file.category == "published_results"
        ]
        if len(outputs) != len(set(outputs)):
            raise ValueError("published result Parquet output paths must be unique")
        return self


def _validate_remote_path(path: str, *, description: str) -> None:
    parsed = PurePosixPath(path)
    if (
        not path
        or not parsed.parts
        or parsed.is_absolute()
        or parsed.as_posix() != path
        or ".." in parsed.parts
        or "\\" in path
    ):
        raise ValueError(f"{description} must be a normalized relative POSIX path")


def _template_fields(template: str, *, description: str) -> tuple[str, ...]:
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"{description} must be a valid format template") from error

    if any(format_spec or conversion for _, _, format_spec, conversion in parsed):
        raise ValueError(f"{description} must not use conversions or format specs")
    return tuple(field for _, field, _, _ in parsed if field is not None)


def _validate_template(
    template: str,
    *,
    description: str,
    expected_fields: tuple[str, ...],
) -> None:
    fields = _template_fields(template, description=description)
    if fields != expected_fields:
        expected = ", ".join(expected_fields)
        raise ValueError(f"{description} must contain exactly {{{expected}}}")


def _remote_path_for_local(local_path: str) -> str:
    path = PurePosixPath(local_path)
    if not path.parts or path.parts[0] != "processed":
        raise ValueError("published local table paths must be under processed/")
    return PurePosixPath(*path.parts[1:]).as_posix()


class PublishingTarget(ConfigModel):
    repo_id: Literal["drotherm/dd_parsed"]
    revision: Literal["main"]


class SingleFilePublishingContract(ConfigModel):
    remote_path: str
    commit_message: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        _validate_remote_path(self.remote_path, description="publication remote path")
        if not self.commit_message.strip():
            raise ValueError("publication commit message must not be empty")
        return self


class ScalingLawPublishingContract(ConfigModel):
    evaluations_remote_path: str
    checkpoint_losses_remote_path: str
    commit_message: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        paths = (self.evaluations_remote_path, self.checkpoint_losses_remote_path)
        for path in paths:
            _validate_remote_path(
                path, description="scaling-law publication remote path"
            )
        if len(paths) != len(set(paths)):
            raise ValueError("scaling-law publication remote paths must be unique")
        if not self.commit_message.strip():
            raise ValueError("scaling-law publication commit message must not be empty")
        return self


class OLMESDetailsPublishingContract(ConfigModel):
    tasks_remote_path_template: str
    instances_remote_path_template: str
    choices_remote_path_template: str
    commit_message_template: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        templates = self.remote_path_templates()
        if len(templates) != len(set(templates)):
            raise ValueError("OLMES detail remote path templates must be unique")
        for template in templates:
            _validate_template(
                template,
                description="OLMES detail remote path template",
                expected_fields=("recipe",),
            )
            _validate_remote_path(
                template.format(recipe="representative-recipe"),
                description="OLMES detail remote path template",
            )
        _validate_template(
            self.commit_message_template,
            description="OLMES detail commit message template",
            expected_fields=("recipe",),
        )
        return self

    def remote_path_templates(self) -> tuple[str, str, str]:
        return (
            self.tasks_remote_path_template,
            self.instances_remote_path_template,
            self.choices_remote_path_template,
        )


class PublishedResultsPublishingContract(ConfigModel):
    remote_root: str
    commit_message_template: str

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        _validate_remote_path(
            self.remote_root, description="published-results remote root"
        )
        _validate_template(
            self.commit_message_template,
            description="published-results commit message template",
            expected_fields=("unit",),
        )
        return self


class PublishingContract(ConfigModel):
    target: PublishingTarget
    ppl: SingleFilePublishingContract
    olmes: SingleFilePublishingContract
    scaling_law: ScalingLawPublishingContract
    olmes_details: OLMESDetailsPublishingContract
    published_results: PublishedResultsPublishingContract

    def validate_references(
        self,
        *,
        olmes_contract: OLMESContract,
        scaling_law_contract: ScalingLawContract,
        source_manifest: SourceManifest,
    ) -> Self:
        expected_paths = {
            "olmes": _remote_path_for_local(olmes_contract.tables.aggregate.path or ""),
            "scaling-law evaluations": _remote_path_for_local(
                scaling_law_contract.tables.evaluations.path
            ),
            "scaling-law checkpoint losses": _remote_path_for_local(
                scaling_law_contract.tables.checkpoint_losses.path
            ),
        }
        configured_paths = {
            "olmes": self.olmes.remote_path,
            "scaling-law evaluations": self.scaling_law.evaluations_remote_path,
            "scaling-law checkpoint losses": (
                self.scaling_law.checkpoint_losses_remote_path
            ),
        }
        for name, expected in expected_paths.items():
            if configured_paths[name] != expected:
                raise ValueError(
                    f"{name} remote path must correspond to its local table path"
                )

        detail_tables = (
            olmes_contract.tables.detailed_tasks,
            olmes_contract.tables.detailed_instances,
            olmes_contract.tables.detailed_choices,
        )
        expected_templates = tuple(
            _remote_path_for_local(table.path_template or "") for table in detail_tables
        )
        if self.olmes_details.remote_path_templates() != expected_templates:
            raise ValueError(
                "OLMES detail remote path templates must correspond to local tables"
            )

        recipes = source_manifest.olmes_details.recipes
        if set(recipes) != set(olmes_contract.recipe_map):
            raise ValueError(
                "OLMES detail publication recipes must match configured recipes"
            )
        expanded_detail_paths = tuple(
            template.format(recipe=recipe)
            for recipe in recipes
            for template in self.olmes_details.remote_path_templates()
        )
        remote_paths = (
            self.ppl.remote_path,
            self.olmes.remote_path,
            self.scaling_law.evaluations_remote_path,
            self.scaling_law.checkpoint_losses_remote_path,
            *expanded_detail_paths,
            self.published_results.remote_root,
        )
        if len(remote_paths) != len(set(remote_paths)):
            raise ValueError("publication remote paths must be unique")
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


class ScalingLawSeedPolicy(ConfigModel):
    excluded_legacy_values: tuple[int, ...]
    missing: Literal["exclude_legacy_input"]
    unknown_non_null: Literal["error"]

    @model_validator(mode="after")
    def validate_unique_excluded_values(self) -> Self:
        if len(self.excluded_legacy_values) != len(set(self.excluded_legacy_values)):
            raise ValueError("scaling-law excluded legacy seeds must be unique")
        if self.excluded_legacy_values != (6198,):
            raise ValueError(
                "scaling-law excluded legacy seeds must contain only seed 6198"
            )
        return self


class ScalingLawColumnContract(ConfigModel):
    name: str
    logical_type: Literal["string", "int64", "float64", "bool"]
    nullable: bool


class ScalingLawTableContract(ConfigModel):
    path: str
    primary_key: tuple[str, ...]
    sort_key: tuple[str, ...]
    columns: tuple[ScalingLawColumnContract, ...]

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            not path.parts
            or path.is_absolute()
            or path.as_posix() != self.path
            or ".." in path.parts
        ):
            raise ValueError(
                "scaling-law table paths must be normalized relative paths"
            )

        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("scaling-law table column names must be unique")

        for key_name, key in (
            ("primary key", self.primary_key),
            ("sort key", self.sort_key),
        ):
            if not key:
                raise ValueError(f"scaling-law table {key_name} must not be empty")
            if len(key) != len(set(key)):
                raise ValueError(f"scaling-law table {key_name} columns must be unique")
            missing = set(key).difference(names)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"scaling-law table {key_name} columns are missing: {missing_names}"
                )
        return self


class ScalingLawTables(ConfigModel):
    evaluations: ScalingLawTableContract
    checkpoint_losses: ScalingLawTableContract


class ScalingLawContract(ConfigModel):
    raw_directory: str
    source_precedence: tuple[str, ...]
    models: tuple[str, ...]
    excluded_source_groups: tuple[str, ...]
    source_group_aliases: dict[str, str]
    source_group_map: dict[str, str]
    seed_map: dict[int, str]
    seed_policy: ScalingLawSeedPolicy
    tables: ScalingLawTables

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        raw_directory = PurePosixPath(self.raw_directory)
        if (
            not raw_directory.parts
            or raw_directory.is_absolute()
            or raw_directory.as_posix() != self.raw_directory
            or ".." in raw_directory.parts
        ):
            raise ValueError(
                "scaling-law raw directory must be a normalized relative path"
            )

        if not self.source_precedence:
            raise ValueError("scaling-law source precedence must not be empty")
        if len(self.source_precedence) != len(set(self.source_precedence)):
            raise ValueError("scaling-law source precedence must be unique")
        for filename in self.source_precedence:
            path = PurePosixPath(filename)
            if len(path.parts) != 1 or path.name != filename:
                raise ValueError("scaling-law sources must be bare filenames")

        for name, mapping in (
            ("source group alias", self.source_group_aliases),
            ("source group", self.source_group_map),
            ("seed", self.seed_map),
        ):
            if not mapping:
                raise ValueError(f"scaling-law {name} mapping must not be empty")
            if len(mapping.values()) != len(set(mapping.values())):
                raise ValueError(f"scaling-law {name} mappings must be unique")

        if len(self.excluded_source_groups) != len(set(self.excluded_source_groups)):
            raise ValueError("scaling-law excluded source groups must be unique")
        configured_source_groups = (
            set(self.source_group_map)
            | set(self.source_group_aliases)
            | set(self.excluded_source_groups)
        )
        if len(configured_source_groups) != (
            len(self.source_group_map)
            + len(self.source_group_aliases)
            + len(self.excluded_source_groups)
        ):
            raise ValueError(
                "scaling-law canonical, aliased, and excluded source groups "
                "must be disjoint"
            )
        unknown_alias_targets = set(self.source_group_aliases.values()).difference(
            self.source_group_map
        )
        if unknown_alias_targets:
            raise ValueError(
                "scaling-law source group aliases must reference canonical groups"
            )

        if set(self.seed_map).intersection(self.seed_policy.excluded_legacy_values):
            raise ValueError("clean and excluded scaling-law seeds must be disjoint")

        if len(self.models) != len(set(self.models)):
            raise ValueError("scaling-law models must be unique")

        expected_keys = {
            "evaluations": ("recipe", "params", "seed_value", "step", "task"),
            "checkpoint_losses": ("recipe", "params", "seed_value", "step"),
        }
        expected_prefixes = {
            "evaluations": (
                ("source_file", "string", False),
                ("recipe", "string", False),
                ("data", "string", False),
                ("params", "string", False),
                ("seed_value", "int64", False),
                ("seed", "string", False),
                ("step", "int64", False),
                ("task", "string", False),
                ("chinchilla", "string", False),
                ("tokens", "int64", False),
                ("compute", "float64", False),
            ),
            "checkpoint_losses": (
                ("source_file", "string", False),
                ("recipe", "string", False),
                ("data", "string", False),
                ("params", "string", False),
                ("seed_value", "int64", False),
                ("seed", "string", False),
                ("step", "int64", False),
                ("chinchilla", "string", False),
                ("tokens", "int64", False),
                ("compute", "float64", False),
            ),
        }
        checkpoint_metrics = (
            "c4_en_validation_cross_entropy",
            "dolma_common_crawl_validation_cross_entropy",
            "pile_validation_cross_entropy",
            "wikitext_103_validation_cross_entropy",
            "train_cross_entropy",
            "throughput_total_tokens",
        )
        for name, table in (
            ("evaluations", self.tables.evaluations),
            ("checkpoint_losses", self.tables.checkpoint_losses),
        ):
            if (
                table.primary_key != expected_keys[name]
                or table.sort_key != expected_keys[name]
            ):
                raise ValueError(
                    f"scaling-law {name} primary and sort keys must match identity"
                )
            actual_prefix = tuple(
                (column.name, column.logical_type, column.nullable)
                for column in table.columns[: len(expected_prefixes[name])]
            )
            if actual_prefix != expected_prefixes[name]:
                raise ValueError(
                    f"scaling-law {name} identity and provenance columns are invalid"
                )

        loss_columns = self.tables.checkpoint_losses.columns[-len(checkpoint_metrics) :]
        if tuple(column.name for column in loss_columns) != checkpoint_metrics or any(
            column.logical_type != "float64" or not column.nullable
            for column in loss_columns
        ):
            raise ValueError(
                "scaling-law checkpoint loss metrics must be nullable float64 "
                "columns in canonical order"
            )

        if self.tables.evaluations.path == self.tables.checkpoint_losses.path:
            raise ValueError("scaling-law output table paths must be unique")
        return self

    def validate_references(
        self,
        *,
        catalog: DataDecideCatalog,
        olmes_contract: OLMESContract,
        published_results_manifest: PublishedResultsManifest,
    ) -> Self:
        manifest_sources = tuple(
            PurePosixPath(file.path).name
            for file in published_results_manifest.files
            if file.category == "scaling_law"
        )
        if self.source_precedence != manifest_sources:
            raise ValueError(
                "scaling-law source precedence must exactly cover the published "
                "results manifest in precedence order"
            )

        catalog_models = tuple(model.name for model in catalog.models)
        if self.models != catalog_models:
            raise ValueError("scaling-law models must exactly match catalog models")

        canonical_recipes = set(self.source_group_map.values())
        if canonical_recipes != set(olmes_contract.recipe_map):
            raise ValueError(
                "scaling-law source groups must map bijectively to OLMES recipes"
            )
        catalog_recipes = {
            recipe
            for family in catalog.data_recipe_families.values()
            for recipe in family
        }
        display_labels = {
            olmes_contract.recipe_map[recipe] for recipe in canonical_recipes
        }
        if display_labels != catalog_recipes:
            raise ValueError(
                "scaling-law recipe display labels must derive from OLMES recipes"
            )

        if self.seed_map != olmes_contract.seed_map:
            raise ValueError("scaling-law seed map must exactly match OLMES seeds")

        evaluation_metrics = self.tables.evaluations.columns[
            -len(olmes_contract.metrics.aggregate) :
        ]
        aggregate_columns = {
            column.name: column for column in olmes_contract.tables.aggregate.columns
        }
        if tuple(column.name for column in evaluation_metrics) != (
            olmes_contract.metrics.aggregate
        ):
            raise ValueError(
                "scaling-law evaluation metrics must exactly match OLMES aggregate "
                "metrics"
            )
        if any(
            column.logical_type != aggregate_columns[column.name].logical_type
            or column.nullable != aggregate_columns[column.name].nullable
            for column in evaluation_metrics
        ):
            raise ValueError(
                "scaling-law evaluation metric types must match OLMES aggregate columns"
            )
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


@cache
def load_scaling_law_contract() -> ScalingLawContract:
    contract = ScalingLawContract.model_validate(_load_toml("scaling_law.toml"))
    return contract.validate_references(
        catalog=load_catalog(),
        olmes_contract=load_olmes_contract(),
        published_results_manifest=load_published_results_manifest(),
    )


@cache
def load_publishing_contract() -> PublishingContract:
    contract = PublishingContract.model_validate(_load_toml("publishing.toml"))
    return contract.validate_references(
        olmes_contract=load_olmes_contract(),
        scaling_law_contract=load_scaling_law_contract(),
        source_manifest=load_source_manifest(),
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
    "OLMESDetailsPublishingContract",
    "PublishedResultFile",
    "PublishedResultCategory",
    "PublishedResultSchema",
    "PublishedResultUnit",
    "PublishedResultsManifest",
    "PublishedResultsPublishingContract",
    "PublishingContract",
    "PublishingTarget",
    "ScalingLawColumnContract",
    "ScalingLawContract",
    "ScalingLawPublishingContract",
    "ScalingLawSeedPolicy",
    "ScalingLawTableContract",
    "ScalingLawTables",
    "SingleFilePublishingContract",
    "SourceManifest",
    "config_file",
    "load_catalog",
    "load_olmes_contract",
    "load_published_results_manifest",
    "load_publishing_contract",
    "load_scaling_law_contract",
    "load_source_manifest",
]
