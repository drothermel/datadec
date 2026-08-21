from __future__ import annotations

from datetime import datetime
from enum import UNIQUE, StrEnum, verify
from math import isfinite
from pathlib import PurePosixPath
from typing import Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


JsonScalar: TypeAlias = str | int | float | bool | None


class PaperModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@verify(UNIQUE)
class ClaimKind(StrEnum):
    EMPIRICAL_NUMERIC = "empirical_numeric"
    EMPIRICAL_COMPARISON = "empirical_comparison"
    EMPIRICAL_TREND = "empirical_trend"
    EMPIRICAL_PLOT = "empirical_plot"
    METHOD_DEFINITION = "method_definition"
    DESCRIPTIVE_METADATA = "descriptive_metadata"
    EXTERNAL_BACKGROUND = "external_background"
    NORMATIVE_OR_FUTURE = "normative_or_future"


PRIMARY_CLAIM_KINDS = frozenset(
    {
        ClaimKind.EMPIRICAL_NUMERIC,
        ClaimKind.EMPIRICAL_COMPARISON,
        ClaimKind.EMPIRICAL_TREND,
        ClaimKind.EMPIRICAL_PLOT,
    }
)


@verify(UNIQUE)
class ValidationOutcome(StrEnum):
    REPRODUCED = "reproduced"
    APPROXIMATELY_REPRODUCED = "approximately_reproduced"
    DIRECTIONALLY_CONSISTENT = "directionally_consistent"
    NOT_REPRODUCED = "not_reproduced"
    NOT_ASSESSABLE_FROM_DD_PARSED = "not_assessable_from_dd_parsed"
    METADATA_DISCREPANCY = "metadata_discrepancy"
    DESCRIPTIVE_ONLY = "descriptive_only"
    EXTERNAL_OR_BACKGROUND = "external_or_background"


@verify(UNIQUE)
class AnalysisId(StrEnum):
    SINGLE_SCALE = "single_scale"
    PER_TASK = "per_task"
    PROXY_METRICS = "proxy_metrics"
    NOISE_SPREAD = "noise_spread"
    SCALING_LAW = "scaling_law"


@verify(UNIQUE)
class AttemptRole(StrEnum):
    DEFAULT = "default"
    SENSITIVITY = "sensitivity"


@verify(UNIQUE)
class PredicateOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"


@verify(UNIQUE)
class ComparisonPredicate(StrEnum):
    EXACT = "exact"
    ROUNDED = "rounded"
    ABSOLUTE_TOLERANCE = "absolute_tolerance"
    BOOLEAN_TRUE = "boolean_true"
    DIRECTIONAL = "directional"
    NONEMPTY_PLOT = "nonempty_plot"


@verify(UNIQUE)
class CheckpointRule(StrEnum):
    EXACT = "exact"
    LATEST_COMMON_COMPLETE = "latest_common_complete"
    PRECEDING_COMMON_COMPLETE = "preceding_common_complete"


@verify(UNIQUE)
class AxisScale(StrEnum):
    LINEAR = "linear"
    LOG = "log"


def _validate_repository_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError("paths must be normalized repository-relative POSIX paths")
    return value


def _validate_finite_json(value: JsonValue | JsonScalar, path: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} must contain only finite JSON numbers")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")


def _require_unique(values: tuple[str, ...], description: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{description} must be unique")


class PaperClaim(PaperModel):
    id: str = Field(min_length=1)
    source_file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    paper_elements: tuple[str, ...] = ()
    kind: ClaimKind
    family: str = Field(min_length=1)
    paper_target: JsonScalar = None
    attempt_ids: tuple[str, ...] = ()
    method_dependency_claim_ids: tuple[str, ...] = ()
    citation_keys: tuple[str, ...] = ()
    supporting_outcome: ValidationOutcome | None = None
    non_assessable_reason: str | None = Field(default=None, min_length=1)

    _validate_source_file = field_validator("source_file")(_validate_repository_path)

    @field_validator("paper_target")
    @classmethod
    def validate_paper_target(cls, value: JsonScalar) -> JsonScalar:
        _validate_finite_json(value, "paper_target")
        return value

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError(
                "claim line_end must be greater than or equal to line_start"
            )
        for description, values in (
            ("claim attempt IDs", self.attempt_ids),
            ("claim method dependency IDs", self.method_dependency_claim_ids),
            ("claim citation keys", self.citation_keys),
        ):
            _require_unique(values, description)
        if self.id in self.method_dependency_claim_ids:
            raise ValueError("claims cannot depend on themselves")

        is_primary = self.kind in PRIMARY_CLAIM_KINDS
        is_non_assessable = (
            self.supporting_outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
        )
        if is_primary:
            if bool(self.attempt_ids) == is_non_assessable:
                raise ValueError(
                    "primary claims require attempts or a not-assessable outcome"
                )
            if is_non_assessable != (self.non_assessable_reason is not None):
                raise ValueError(
                    "not-assessable claims require exactly one non-assessable reason"
                )
            if self.attempt_ids and self.supporting_outcome is not None:
                raise ValueError(
                    "assessable primary claims cannot have supporting outcomes"
                )
        else:
            if self.attempt_ids:
                raise ValueError("nonempirical claims cannot have executable attempts")
            if self.non_assessable_reason is not None:
                raise ValueError(
                    "nonempirical claims cannot have empirical non-assessable reasons"
                )
            allowed_outcomes = {
                ClaimKind.METHOD_DEFINITION: {ValidationOutcome.DESCRIPTIVE_ONLY},
                ClaimKind.DESCRIPTIVE_METADATA: {
                    ValidationOutcome.DESCRIPTIVE_ONLY,
                    ValidationOutcome.METADATA_DISCREPANCY,
                },
                ClaimKind.EXTERNAL_BACKGROUND: {
                    ValidationOutcome.EXTERNAL_OR_BACKGROUND
                },
                ClaimKind.NORMATIVE_OR_FUTURE: {ValidationOutcome.DESCRIPTIVE_ONLY},
            }
            if self.supporting_outcome not in allowed_outcomes[self.kind]:
                raise ValueError(
                    "supporting claim outcome does not match its claim kind"
                )
        return self


class SourceRegion(PaperModel):
    id: str = Field(min_length=1)
    source_file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    kind: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_ids: tuple[str, ...] = ()
    non_claim_reason: str | None = Field(default=None, min_length=1)

    _validate_source_file = field_validator("source_file")(_validate_repository_path)

    @model_validator(mode="after")
    def validate_region(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError(
                "region line_end must be greater than or equal to line_start"
            )
        if bool(self.claim_ids) == bool(self.non_claim_reason):
            raise ValueError(
                "source regions require exactly one of claim_ids or non_claim_reason"
            )
        _require_unique(self.claim_ids, "source region claim IDs")
        return self


class ClaimRegistry(PaperModel):
    format_version: Literal[2]
    claims: tuple[PaperClaim, ...]
    source_regions: tuple[SourceRegion, ...] = ()

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        claim_ids = tuple(claim.id for claim in self.claims)
        region_ids = tuple(region.id for region in self.source_regions)
        _require_unique(claim_ids, "paper claim IDs")
        _require_unique(region_ids, "paper source region IDs")
        known_claim_ids = set(claim_ids)
        unknown_references = {
            dependency
            for claim in self.claims
            for dependency in claim.method_dependency_claim_ids
            if dependency not in known_claim_ids
        }
        unknown_references.update(
            claim_id
            for region in self.source_regions
            for claim_id in region.claim_ids
            if claim_id not in known_claim_ids
        )
        if unknown_references:
            unknown = ", ".join(sorted(unknown_references))
            raise ValueError(f"paper registry references unknown claims: {unknown}")
        return self


class PaperIdentity(PaperModel):
    arxiv_id: str = Field(pattern=r"^[0-9]{4}\.[0-9]{5}v[1-9][0-9]*$")
    source_url: str = Field(min_length=1)
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_root: str
    entrypoint: str

    _validate_source_root = field_validator("source_root")(_validate_repository_path)
    _validate_entrypoint = field_validator("entrypoint")(_validate_repository_path)

    @model_validator(mode="after")
    def validate_entrypoint(self) -> Self:
        if len(PurePosixPath(self.entrypoint).parts) != 1:
            raise ValueError("paper entrypoint must be relative to source_root")
        return self


class InputTableSpec(PaperModel):
    id: str = Field(min_length=1)
    path: str
    columns: tuple[str, ...]
    remote_path: str | None = None

    _validate_path = field_validator("path")(_validate_repository_path)

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        paths = (
            (self.path,) if self.remote_path is None else (self.path, self.remote_path)
        )
        if any("published-results" in PurePosixPath(path).parts for path in paths):
            raise ValueError("validation inputs cannot reference published-results")
        if self.remote_path is not None:
            _validate_repository_path(self.remote_path)
        if not self.columns:
            raise ValueError("input tables require declared columns")
        _require_unique(self.columns, "input table columns")
        return self


class AttemptInput(PaperModel):
    table_id: str = Field(min_length=1)
    columns: tuple[str, ...]

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        if not self.columns:
            raise ValueError("attempt inputs require declared columns")
        _require_unique(self.columns, "attempt input columns")
        return self


class AttemptSpec(PaperModel):
    id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    default: bool
    parent_attempt_id: str | None = Field(default=None, min_length=1)
    analysis_id: AnalysisId
    inputs: tuple[AttemptInput, ...]
    recipe_ids: tuple[str, ...] = ()
    seed_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    metric_ids: tuple[str, ...] = ()
    model_sizes: tuple[str, ...] = ()
    checkpoints: tuple[str, ...] = ()
    transformation_ids: tuple[str, ...]
    comparison_rule_id: str = Field(min_length=1)
    sensitivity_ids: tuple[str, ...] = ()
    plot_series_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if not self.inputs:
            raise ValueError("attempts require at least one declared input")
        if not self.transformation_ids:
            raise ValueError("attempts require ordered transformations")
        for description, values in (
            ("attempt input table IDs", tuple(item.table_id for item in self.inputs)),
            ("attempt recipe IDs", self.recipe_ids),
            ("attempt seed IDs", self.seed_ids),
            ("attempt task IDs", self.task_ids),
            ("attempt metric IDs", self.metric_ids),
            ("attempt model sizes", self.model_sizes),
            ("attempt checkpoints", self.checkpoints),
            ("attempt transformation IDs", self.transformation_ids),
            ("attempt sensitivity IDs", self.sensitivity_ids),
            ("attempt plot-series IDs", self.plot_series_ids),
        ):
            _require_unique(values, description)
        if self.default:
            expected_id = f"{self.claim_id.lower()}-default"
            if self.id != expected_id:
                raise ValueError(f"default attempt ID must be {expected_id}")
            if self.parent_attempt_id is not None:
                raise ValueError("default attempts cannot have parent attempts")
        elif self.parent_attempt_id is None:
            raise ValueError("sensitivity attempts require a parent attempt")
        return self


class ComparisonRule(PaperModel):
    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    predicate: ComparisonPredicate
    displayed_decimal_places: int | None = Field(default=None, ge=0)
    absolute_tolerance: float | None = Field(default=None, gt=0)
    threshold_grid: tuple[float, ...] = ()

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if self.predicate is ComparisonPredicate.ROUNDED:
            if (
                self.displayed_decimal_places is None
                or self.absolute_tolerance is not None
            ):
                raise ValueError("rounded comparison requires only displayed precision")
        elif self.predicate is ComparisonPredicate.ABSOLUTE_TOLERANCE:
            if (
                self.absolute_tolerance is None
                or self.displayed_decimal_places is not None
            ):
                raise ValueError(
                    "tolerance comparison requires only absolute tolerance"
                )
            if not self.threshold_grid:
                raise ValueError("tolerance comparison requires a threshold grid")
            if self.absolute_tolerance not in self.threshold_grid:
                raise ValueError("comparison threshold grid must include the default")
        elif (
            self.displayed_decimal_places is not None
            or self.absolute_tolerance is not None
        ):
            raise ValueError("comparison predicate does not accept numeric parameters")
        if any(not isfinite(value) or value <= 0 for value in self.threshold_grid):
            raise ValueError("comparison thresholds must be finite and positive")
        if tuple(sorted(set(self.threshold_grid))) != self.threshold_grid:
            raise ValueError("comparison thresholds must be sorted and unique")
        return self


class CheckpointPolicy(PaperModel):
    final_rule: Literal[CheckpointRule.LATEST_COMMON_COMPLETE]
    completeness_dimensions: tuple[str, ...]
    one_step_across_universe: Literal[True]

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if not self.completeness_dimensions:
            raise ValueError("checkpoint completeness dimensions must not be empty")
        _require_unique(self.completeness_dimensions, "checkpoint dimensions")
        return self


class SensitivityPolicy(PaperModel):
    preceding_common_complete_steps: Literal[2]
    include_paper_step_when_present: Literal[True]
    fixed_before_computation: Literal[True]


class AnalysisPolicy(PaperModel):
    id: AnalysisId
    transformation_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_transformations(self) -> Self:
        if not self.transformation_ids:
            raise ValueError("analysis policies require transformations")
        _require_unique(self.transformation_ids, "analysis policy transformation IDs")
        return self


class ValidationOutputs(PaperModel):
    runs_root: str
    report: str
    figures_root: str
    manifest_filename: Literal["manifest.json"] = "manifest.json"
    targets_filename: Literal["targets.json"] = "targets.json"
    attempts_filename: Literal["attempts.json"] = "attempts.json"
    plot_series_filename: Literal["plot-series.json"] = "plot-series.json"

    _validate_paths = field_validator("runs_root", "report", "figures_root")(
        _validate_repository_path
    )


class PaperValidationContract(PaperModel):
    paper: PaperIdentity
    inputs: tuple[InputTableSpec, ...]
    attempts: tuple[AttemptSpec, ...]
    comparison_rules: tuple[ComparisonRule, ...]
    checkpoint_policy: CheckpointPolicy
    sensitivity_policy: SensitivityPolicy
    analysis_policies: tuple[AnalysisPolicy, ...]
    outputs: ValidationOutputs

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        input_ids = tuple(item.id for item in self.inputs)
        attempt_ids = tuple(item.id for item in self.attempts)
        rule_ids = tuple(item.id for item in self.comparison_rules)
        policy_ids = tuple(item.id.value for item in self.analysis_policies)
        for description, values in (
            ("validation input IDs", input_ids),
            ("validation attempt IDs", attempt_ids),
            ("comparison rule IDs", rule_ids),
            ("analysis policy IDs", policy_ids),
        ):
            _require_unique(values, description)
        known_inputs = {item.id: set(item.columns) for item in self.inputs}
        known_attempts = set(attempt_ids)
        known_rules = set(rule_ids)
        known_policies = set(policy_ids)
        sensitivity_ids = tuple(
            sensitivity_id
            for attempt in self.attempts
            for sensitivity_id in attempt.sensitivity_ids
        )
        _require_unique(sensitivity_ids, "validation sensitivity IDs")
        if set(sensitivity_ids) & known_attempts:
            raise ValueError("sensitivity IDs must differ from default attempt IDs")
        for attempt in self.attempts:
            for attempt_input in attempt.inputs:
                if attempt_input.table_id not in known_inputs:
                    raise ValueError(
                        f"attempt {attempt.id} references unknown input {attempt_input.table_id}"
                    )
                unknown_columns = (
                    set(attempt_input.columns) - known_inputs[attempt_input.table_id]
                )
                if unknown_columns:
                    unknown = ", ".join(sorted(unknown_columns))
                    raise ValueError(
                        f"attempt {attempt.id} references unknown columns: {unknown}"
                    )
            if attempt.comparison_rule_id not in known_rules:
                raise ValueError(
                    f"attempt {attempt.id} references unknown comparison rule"
                )
            if attempt.analysis_id.value not in known_policies:
                raise ValueError(f"attempt {attempt.id} has no analysis policy")
            if attempt.parent_attempt_id not in known_attempts | {None}:
                raise ValueError(f"attempt {attempt.id} references unknown parent")
        return self


class ContentIdentity(PaperModel):
    id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CodeTrace(PaperModel):
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    diff_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class RuntimeTrace(PaperModel):
    python_version: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    dependency_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PaperTarget(PaperModel):
    claim_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    kind: ClaimKind
    source_file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    source_text: str = Field(min_length=1)
    value: JsonScalar = None

    _validate_source_file = field_validator("source_file")(_validate_repository_path)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.kind not in PRIMARY_CLAIM_KINDS:
            raise ValueError("paper targets must represent empirical claims")
        if self.line_end < self.line_start:
            raise ValueError(
                "target line_end must be greater than or equal to line_start"
            )
        _validate_finite_json(self.value, "target value")
        return self


class RowPredicate(PaperModel):
    column: str = Field(min_length=1)
    operator: PredicateOperator
    value: JsonScalar | tuple[JsonScalar, ...]

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        is_set_operator = self.operator in {
            PredicateOperator.IN,
            PredicateOperator.NOT_IN,
        }
        if is_set_operator != isinstance(self.value, tuple):
            raise ValueError(
                "set predicates require tuple values and scalar predicates do not"
            )
        if isinstance(self.value, tuple) and not self.value:
            raise ValueError("set predicate values must not be empty")
        _validate_finite_json(self.value, "predicate value")
        return self


class RowSelection(PaperModel):
    logical_table_id: str = Field(min_length=1)
    columns: tuple[str, ...]
    predicates: tuple[RowPredicate, ...]
    local_parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    remote_dataset_revision: str | None = Field(default=None, min_length=1)
    selected_row_count: int = Field(ge=0)
    selected_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.columns:
            raise ValueError("row selections require columns")
        _require_unique(self.columns, "row selection columns")
        return self


class CheckpointSelection(PaperModel):
    requested_meaning: str = Field(min_length=1)
    rule: CheckpointRule
    actual_step: int = Field(ge=0)
    completeness_dimensions: tuple[str, ...]
    expected_group_count: int = Field(ge=0)
    selected_group_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.completeness_dimensions:
            raise ValueError("checkpoint selection requires completeness dimensions")
        _require_unique(self.completeness_dimensions, "checkpoint dimensions")
        if self.selected_group_count > self.expected_group_count:
            raise ValueError("selected checkpoint groups cannot exceed expected groups")
        if (
            self.rule
            in {
                CheckpointRule.LATEST_COMMON_COMPLETE,
                CheckpointRule.PRECEDING_COMMON_COMPLETE,
            }
            and self.selected_group_count != self.expected_group_count
        ):
            raise ValueError("common-complete checkpoints must include every group")
        return self


class NamedCount(PaperModel):
    name: str = Field(min_length=1)
    value: int = Field(ge=0)


class AttemptResult(PaperModel):
    attempt_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    role: AttemptRole
    parent_attempt_id: str | None = Field(default=None, min_length=1)
    comparison_rule_id: str = Field(min_length=1)
    comparison_rule_version: int = Field(ge=1)
    transformation_ids: tuple[str, ...]
    row_selections: tuple[RowSelection, ...]
    checkpoint_selections: tuple[CheckpointSelection, ...] = ()
    target_value: JsonScalar = None
    computed_value: JsonValue | None = None
    unrounded_difference: float | None = None
    seeds: tuple[str, ...] = ()
    denominator: int | None = Field(default=None, ge=0)
    exclusions: tuple[NamedCount, ...] = ()
    missing_groups: tuple[str, ...] = ()
    target_ties: int = Field(default=0, ge=0)
    predicted_ties: int = Field(default=0, ge=0)
    standard_deviation: float | None = Field(default=None, ge=0)
    ddof: int | None = Field(default=None, ge=0)
    outcome: ValidationOutcome
    diagnostics: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    plot_series_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if not self.transformation_ids or not self.row_selections:
            raise ValueError(
                "attempt results require transformations and row selections"
            )
        for description, values in (
            ("result transformation IDs", self.transformation_ids),
            ("result seeds", self.seeds),
            ("result missing groups", self.missing_groups),
            ("result plot-series IDs", self.plot_series_ids),
            ("result exclusion names", tuple(item.name for item in self.exclusions)),
        ):
            _require_unique(values, description)
        if self.role is AttemptRole.DEFAULT:
            if self.parent_attempt_id is not None:
                raise ValueError("default results cannot have parent attempts")
        elif self.parent_attempt_id is None:
            raise ValueError("sensitivity results require parent attempts")
        if (self.standard_deviation is None) != (self.ddof is None):
            raise ValueError("standard deviation and DDOF must be recorded together")
        _validate_finite_json(self.target_value, "attempt target value")
        _validate_finite_json(self.computed_value, "attempt computed value")
        if self.unrounded_difference is not None and not isfinite(
            self.unrounded_difference
        ):
            raise ValueError("unrounded difference must be finite")
        return self


class MetadataDiscrepancy(PaperModel):
    claim_id: str = Field(min_length=1)
    paper_locator: str = Field(min_length=1)
    paper_value: JsonScalar
    metadata_source: str = Field(min_length=1)
    metadata_value: JsonValue
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        _validate_finite_json(self.paper_value, "metadata paper value")
        _validate_finite_json(self.metadata_value, "metadata available value")
        return self


class AxisSpec(PaperModel):
    measure: str = Field(min_length=1)
    scale: AxisScale
    unit: str = Field(min_length=1)


class DimensionValue(PaperModel):
    name: str = Field(min_length=1)
    value: JsonScalar

    @field_validator("value")
    @classmethod
    def validate_finite(cls, value: JsonScalar) -> JsonScalar:
        _validate_finite_json(value, "plot dimension")
        return value


class MeasureValue(PaperModel):
    name: str = Field(min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("plot measures must be finite")
        return value


class PlotPoint(PaperModel):
    dimensions: tuple[DimensionValue, ...] = ()
    measures: tuple[MeasureValue, ...]

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.measures:
            raise ValueError("plot points require measures")
        _require_unique(
            tuple(item.name for item in self.dimensions), "point dimensions"
        )
        _require_unique(tuple(item.name for item in self.measures), "point measures")
        return self


class PlotSeries(PaperModel):
    id: str = Field(min_length=1)
    figure: str = Field(min_length=1)
    panel: str = Field(min_length=1)
    semantic_kind: str = Field(min_length=1)
    x_axis: AxisSpec
    y_axis: AxisSpec
    dimensions: tuple[str, ...] = ()
    measures: tuple[str, ...]
    attempt_id: str = Field(min_length=1)
    actual_checkpoint: int | None = Field(default=None, ge=0)
    counts: tuple[NamedCount, ...] = ()
    points: tuple[PlotPoint, ...]
    paper_analog: bool = True

    @model_validator(mode="after")
    def validate_series(self) -> Self:
        if self.paper_analog and not self.points:
            raise ValueError("paper-analog plot series must not be empty")
        for description, values in (
            ("plot dimensions", self.dimensions),
            ("plot measures", self.measures),
            ("plot count names", tuple(item.name for item in self.counts)),
        ):
            _require_unique(values, description)
        if (
            self.x_axis.measure not in self.measures
            or self.y_axis.measure not in self.measures
        ):
            raise ValueError("plot axes must reference declared measures")
        expected_dimensions = set(self.dimensions)
        expected_measures = set(self.measures)
        for point in self.points:
            if {item.name for item in point.dimensions} != expected_dimensions:
                raise ValueError("plot point dimensions must match the series")
            if {item.name for item in point.measures} != expected_measures:
                raise ValueError("plot point measures must match the series")
        return self


class AnalysisManifest(PaperModel):
    run_format: Literal[2] = 2
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    started_at: datetime
    completed_at: datetime
    code_trace: CodeTrace | None = None
    runtime_trace: RuntimeTrace | None = None
    input_identities: tuple[ContentIdentity, ...]
    targets_identity: ContentIdentity
    attempts_identity: ContentIdentity
    plot_series_identity: ContentIdentity

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("run timestamps must include timezone offsets")
        if self.completed_at < self.started_at:
            raise ValueError("run completion timestamp cannot precede start timestamp")
        input_ids = tuple(identity.id for identity in self.input_identities)
        _require_unique(input_ids, "manifest input identity IDs")
        bundle_ids = {
            self.targets_identity.id,
            self.attempts_identity.id,
            self.plot_series_identity.id,
        }
        if bundle_ids != {"targets.json", "attempts.json", "plot-series.json"}:
            raise ValueError("manifest bundle identities must name the three run files")
        return self


class AnalysisBundle(PaperModel):
    manifest: AnalysisManifest
    targets: tuple[PaperTarget, ...]
    metadata_discrepancies: tuple[MetadataDiscrepancy, ...] = ()
    attempts: tuple[AttemptResult, ...]
    plot_series: tuple[PlotSeries, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        target_ids = tuple(target.claim_id for target in self.targets)
        attempt_ids = tuple(attempt.attempt_id for attempt in self.attempts)
        series_ids = tuple(series.id for series in self.plot_series)
        discrepancy_ids = tuple(item.claim_id for item in self.metadata_discrepancies)
        for description, values in (
            ("bundle target claim IDs", target_ids),
            ("bundle attempt IDs", attempt_ids),
            ("bundle plot-series IDs", series_ids),
            ("bundle metadata discrepancy claim IDs", discrepancy_ids),
        ):
            _require_unique(values, description)
        known_targets = set(target_ids)
        known_attempts = set(attempt_ids)
        known_series = set(series_ids)
        for attempt in self.attempts:
            if attempt.claim_id not in known_targets:
                raise ValueError("attempt result references an unknown paper target")
            if set(attempt.plot_series_ids) - known_series:
                raise ValueError("attempt result references unknown plot series")
        for series in self.plot_series:
            if series.attempt_id not in known_attempts:
                raise ValueError("plot series references an unknown attempt")
            attempt = next(
                item for item in self.attempts if item.attempt_id == series.attempt_id
            )
            if series.id not in attempt.plot_series_ids:
                raise ValueError("plot series is not declared by its attempt")
        return self


__all__ = [
    "AnalysisBundle",
    "AnalysisId",
    "AnalysisManifest",
    "AnalysisPolicy",
    "AttemptInput",
    "AttemptResult",
    "AttemptRole",
    "AttemptSpec",
    "AxisScale",
    "AxisSpec",
    "CheckpointPolicy",
    "CheckpointRule",
    "CheckpointSelection",
    "ClaimKind",
    "ClaimRegistry",
    "CodeTrace",
    "ComparisonPredicate",
    "ComparisonRule",
    "ContentIdentity",
    "DimensionValue",
    "InputTableSpec",
    "MeasureValue",
    "MetadataDiscrepancy",
    "NamedCount",
    "PaperClaim",
    "PaperIdentity",
    "PaperTarget",
    "PaperValidationContract",
    "PlotPoint",
    "PlotSeries",
    "PredicateOperator",
    "RowPredicate",
    "RowSelection",
    "RuntimeTrace",
    "SensitivityPolicy",
    "SourceRegion",
    "ValidationOutcome",
    "ValidationOutputs",
]
