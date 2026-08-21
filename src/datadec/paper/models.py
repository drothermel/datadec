from __future__ import annotations

from datetime import datetime
from enum import UNIQUE, StrEnum, verify
from math import isfinite
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class PaperModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@verify(UNIQUE)
class ClaimOwnership(StrEnum):
    DATADEC_EMPIRICAL = "datadec_empirical"
    METHOD_DESIGN = "method_design"
    ARTIFACT_RELEASE = "artifact_release"
    QUALITATIVE_INTERPRETATION = "qualitative_interpretation"
    EXTERNAL_CITATION = "external_citation"


@verify(UNIQUE)
class EvidenceBoundary(StrEnum):
    PAPER_OR_FINAL_ARTIFACT = "paper_or_final_artifact"
    AUTHOR_DOWNSTREAM_TABLE = "author_downstream_table"
    AGGREGATE_EVALUATION = "aggregate_evaluation"
    INSTANCE_AND_CHOICE = "instance_and_choice"
    EVALUATION_RERUN = "evaluation_rerun"
    TRAINING_RERUN = "training_rerun"
    CORPUS_CONSTRUCTION = "corpus_construction"


@verify(UNIQUE)
class MethodProvenance(StrEnum):
    PAPER_DERIVED = "paper_derived"
    UPSTREAM_INFORMED = "upstream_informed"
    ARTIFACT_DERIVED = "artifact_derived"


@verify(UNIQUE)
class ExpectationKind(StrEnum):
    LITERAL = "literal"
    NUMERIC = "numeric"
    PREDICATE = "predicate"
    CITATION_TRACE = "citation_trace"


@verify(UNIQUE)
class PolicyStatus(StrEnum):
    SETTLED = "settled"
    UNRESOLVED = "unresolved"


@verify(UNIQUE)
class Verdict(StrEnum):
    REPRODUCED = "reproduced"
    CONTRADICTED = "contradicted"
    INTERNALLY_INCONSISTENT = "internally_inconsistent"
    SOURCE_ONLY_MATCH = "source_only_match"
    BLOCKED_MISSING_INPUT = "blocked_missing_input"
    BLOCKED_UNSPECIFIED_METHOD = "blocked_unspecified_method"
    EXTERNAL_OR_CITATION_DEPENDENT = "external_or_citation_dependent"
    NOT_ATTEMPTED = "not_attempted"
    NOT_APPLICABLE = "not_applicable"


@verify(UNIQUE)
class CodeTreeState(StrEnum):
    CLEAN = "clean"
    DIRTY = "dirty"


@verify(UNIQUE)
class BlockerKind(StrEnum):
    MISSING_INPUT = "missing_input"
    UNSPECIFIED_METHOD = "unspecified_method"
    EXTERNAL_OR_CITATION_DEPENDENT = "external_or_citation_dependent"
    NOT_ATTEMPTED = "not_attempted"
    NOT_APPLICABLE = "not_applicable"


def _validate_repository_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
    ):
        raise ValueError("paths must be normalized repository-relative POSIX paths")
    return value


class PaperClaim(PaperModel):
    id: str = Field(min_length=1)
    source_file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    owner: ClaimOwnership
    expectation_kind: ExpectationKind
    expectation: str | int | float | bool
    required_evidence_boundary: EvidenceBoundary
    verifier_id: str | None = None
    method_id: str | None = None
    policy_id: str | None = None
    unresolved_method_id: str | None = None
    input_refs: tuple[str, ...] = ()
    prerequisite_claim_ids: tuple[str, ...] = ()
    paper_elements: tuple[str, ...] = ()
    citation_keys: tuple[str, ...] = ()

    _validate_source_file = field_validator("source_file")(_validate_repository_path)

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError(
                "claim line_end must be greater than or equal to line_start"
            )
        if self.owner is ClaimOwnership.EXTERNAL_CITATION and not self.citation_keys:
            raise ValueError("external-citation claims must include citation keys")
        if self.verifier_id is not None and self.unresolved_method_id is not None:
            raise ValueError(
                "claim verifier_id and unresolved_method_id are mutually exclusive"
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
        has_claims = bool(self.claim_ids)
        has_non_claim_reason = bool(self.non_claim_reason)
        if has_claims == has_non_claim_reason:
            raise ValueError(
                "source regions require exactly one of nonempty claim_ids or "
                "non_claim_reason"
            )
        return self


class ClaimRegistry(PaperModel):
    claims: tuple[PaperClaim, ...]
    source_regions: tuple[SourceRegion, ...] = ()

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        claim_ids = [claim.id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("paper claim IDs must be unique")

        region_ids = [region.id for region in self.source_regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("paper source region IDs must be unique")

        unknown_claim_ids = {
            claim_id
            for region in self.source_regions
            for claim_id in region.claim_ids
            if claim_id not in set(claim_ids)
        }
        if unknown_claim_ids:
            unknown = ", ".join(sorted(unknown_claim_ids))
            raise ValueError(
                f"paper source regions reference unknown claims: {unknown}"
            )
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


class PaperContractReferences(PaperModel):
    catalog: str
    sources: str
    olmes: str
    scaling_law: str
    published_results: str
    claims_contract: str

    _validate_paths = field_validator(
        "catalog",
        "sources",
        "olmes",
        "scaling_law",
        "published_results",
        "claims_contract",
    )(_validate_repository_path)


class MethodProvenanceEntry(PaperModel):
    id: str = Field(min_length=1)
    provenance: MethodProvenance
    description: str = Field(min_length=1)
    paper_elements: tuple[str, ...] = ()


class NamedPolicy(PaperModel):
    id: str = Field(min_length=1)
    status: PolicyStatus
    statement: str = Field(min_length=1)


class PaperOutputs(PaperModel):
    runs_root: str
    generated_results_root: str
    report: str
    reproduced_figures_root: str
    observations_filename: str
    run_manifest_filename: str

    _validate_paths = field_validator(
        "runs_root",
        "generated_results_root",
        "report",
        "reproduced_figures_root",
        "observations_filename",
        "run_manifest_filename",
    )(_validate_repository_path)

    @model_validator(mode="after")
    def validate_filenames(self) -> Self:
        for description, filename in (
            ("observations", self.observations_filename),
            ("run manifest", self.run_manifest_filename),
        ):
            if len(PurePosixPath(filename).parts) != 1:
                raise ValueError(
                    f"paper {description} filename must be a bare filename"
                )
        return self


@verify(UNIQUE)
class OperationalizationBasis(StrEnum):
    PAPER = "paper"
    REPOSITORY_OPERATIONALIZED = "repository_operationalized"


def _validate_analysis_aliases(
    values: tuple[str, ...], *, label: str, expected_count: int | None = None
) -> None:
    if not values:
        raise ValueError(f"OLMES {label} aliases must not be empty")
    if any(not value for value in values):
        raise ValueError(f"OLMES {label} aliases must not contain empty values")
    if len(values) != len(set(values)):
        raise ValueError(f"OLMES {label} aliases must be unique")
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(f"OLMES {label} must contain exactly {expected_count} aliases")


class OlmesRecipeUniverse(PaperModel):
    aliases: tuple[str, ...]
    basis: OperationalizationBasis
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aliases(self) -> Self:
        _validate_analysis_aliases(self.aliases, label="recipe", expected_count=25)
        return self


class OlmesSeedPolicy(PaperModel):
    target_aliases: tuple[str, ...]
    prediction_aliases: tuple[str, ...]
    basis: OperationalizationBasis
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aliases(self) -> Self:
        _validate_analysis_aliases(
            self.target_aliases, label="target seed", expected_count=3
        )
        _validate_analysis_aliases(
            self.prediction_aliases, label="prediction seed", expected_count=3
        )
        if len(set(self.target_aliases) | set(self.prediction_aliases)) != 5:
            raise ValueError(
                "OLMES target and prediction seeds must cover exactly five aliases"
            )
        return self


class OlmesMetricPolicy(PaperModel):
    target_column: str = Field(min_length=1)
    proxy_columns: tuple[str, ...]
    basis: OperationalizationBasis
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        _validate_analysis_aliases(
            self.proxy_columns, label="proxy metric", expected_count=15
        )
        if self.target_column in self.proxy_columns:
            raise ValueError("OLMES target metric must not also be a proxy metric")
        return self


class OlmesTaskGroupingPolicy(PaperModel):
    non_mmlu_tasks: tuple[str, ...]
    mmlu_subjects: tuple[str, ...]
    mmlu_task_name: str = Field(min_length=1)
    mmlu_subject_weighting: Literal["equal"]
    olmes_task_weighting: Literal["equal"]
    basis: OperationalizationBasis
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        _validate_analysis_aliases(
            self.non_mmlu_tasks, label="non-MMLU task", expected_count=9
        )
        _validate_analysis_aliases(
            self.mmlu_subjects, label="MMLU subject", expected_count=57
        )
        source_tasks = set(self.non_mmlu_tasks) | set(self.mmlu_subjects)
        if len(source_tasks) != 66:
            raise ValueError("OLMES non-MMLU and MMLU task aliases must be disjoint")
        if self.mmlu_task_name in source_tasks:
            raise ValueError(
                "OLMES aggregate MMLU task name must differ from source task aliases"
            )
        return self


class OlmesFinalCheckpoint(PaperModel):
    model_size: str = Field(min_length=1)
    step: int = Field(ge=0)


class OlmesFinalCheckpointPolicy(PaperModel):
    checkpoints: tuple[OlmesFinalCheckpoint, ...]
    basis: OperationalizationBasis
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_checkpoints(self) -> Self:
        sizes = tuple(checkpoint.model_size for checkpoint in self.checkpoints)
        _validate_analysis_aliases(sizes, label="final-checkpoint model size")
        return self


class OlmesComparisonPolicy(PaperModel):
    target_ties: Literal["exclude"]
    predicted_ties: Literal["count_as_incorrect"]
    basis: OperationalizationBasis
    description: str = Field(min_length=1)


class OlmesStandardDeviationPolicy(PaperModel):
    attempt_ddof: int = Field(ge=0)
    within_recipe_noise_ddof: int = Field(ge=0)
    spread_ddof: int = Field(ge=0)
    basis: OperationalizationBasis
    description: str = Field(min_length=1)


class OlmesMissingDataPolicy(PaperModel):
    behavior: Literal["record"]
    allow_complete_case: Literal[False]
    basis: OperationalizationBasis
    description: str = Field(min_length=1)


class OlmesComputePolicy(PaperModel):
    flops_per_token_per_parameter: int = Field(gt=0)
    parameter_count_column: Literal["exact_parameter_count"]
    token_count_column: Literal["tokens"]
    target_training_tokens: int = Field(gt=0)
    denominator_scope: Literal["single_target_run"]
    target_run_count: Literal[1]
    target_compute_denominator: int = Field(gt=0)
    basis: OperationalizationBasis
    description: str = Field(min_length=1)


class OlmesAnalysisContract(PaperModel):
    recipes: OlmesRecipeUniverse
    target_model_size: str = Field(min_length=1)
    seeds: OlmesSeedPolicy
    metrics: OlmesMetricPolicy
    task_grouping: OlmesTaskGroupingPolicy
    final_checkpoints: OlmesFinalCheckpointPolicy
    noise_model_size: str = Field(min_length=1)
    comparison: OlmesComparisonPolicy
    standard_deviation: OlmesStandardDeviationPolicy
    missing_data: OlmesMissingDataPolicy
    compute: OlmesComputePolicy


class PaperReproductionContract(PaperModel):
    paper: PaperIdentity
    contracts: PaperContractReferences
    methods: tuple[MethodProvenanceEntry, ...]
    policies: tuple[NamedPolicy, ...]
    olmes_analysis: OlmesAnalysisContract
    outputs: PaperOutputs

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        for description, values in (
            ("method", self.methods),
            ("policy", self.policies),
        ):
            ids = [value.id for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"paper reproduction {description} IDs must be unique")
        return self


class ContentIdentity(PaperModel):
    id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CodeIdentity(PaperModel):
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    tree_state: CodeTreeState
    dirty_diff_artifact_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_tree_state(self) -> Self:
        if self.tree_state is CodeTreeState.DIRTY:
            if self.dirty_diff_artifact_id is None:
                raise ValueError(
                    "dirty code identity requires a canonical diff artifact ID"
                )
        elif self.dirty_diff_artifact_id is not None:
            raise ValueError(
                "clean code identity cannot reference a dirty diff artifact"
            )
        return self


class RuntimeIdentity(PaperModel):
    python_version: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ObservationCount(PaperModel):
    name: str = Field(min_length=1)
    value: int = Field(ge=0)


class ObservationBlocker(PaperModel):
    kind: BlockerKind
    reason: str = Field(min_length=1)
    missing_input_ids: tuple[str, ...] = ()
    unresolved_method_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_blocker(self) -> Self:
        if len(self.missing_input_ids) != len(set(self.missing_input_ids)):
            raise ValueError("blocker missing input IDs must be unique")
        if tuple(sorted(self.missing_input_ids)) != self.missing_input_ids:
            raise ValueError("blocker missing input IDs must be sorted")
        if self.kind is BlockerKind.MISSING_INPUT:
            if not self.missing_input_ids:
                raise ValueError("missing-input blocker requires missing input IDs")
            if self.unresolved_method_id is not None:
                raise ValueError(
                    "missing-input blocker cannot include an unresolved method ID"
                )
        elif self.kind is BlockerKind.UNSPECIFIED_METHOD:
            if self.unresolved_method_id is None:
                raise ValueError(
                    "unspecified-method blocker requires an unresolved method ID"
                )
            if self.missing_input_ids:
                raise ValueError(
                    "unspecified-method blocker cannot include missing input IDs"
                )
        elif self.missing_input_ids or self.unresolved_method_id is not None:
            raise ValueError(
                "this blocker kind cannot include missing inputs or an unresolved method"
            )
        return self


def _validate_finite_json(value: JsonValue, path: str = "observed_value") -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} must contain only finite JSON numbers")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")


class Observation(PaperModel):
    claim_id: str = Field(min_length=1)
    verifier_id: str | None = Field(default=None, min_length=1)
    method_id: str | None = Field(default=None, min_length=1)
    method_provenance: MethodProvenance | None = None
    method_reference_artifact_id: str | None = Field(default=None, min_length=1)
    policy_id: str | None = Field(default=None, min_length=1)
    actual_evidence_boundary: EvidenceBoundary | None = None
    verdict: Verdict
    observed_value: JsonValue | None = None
    diagnostics: tuple[str, ...] = ()
    denominator: int | None = Field(default=None, ge=0)
    counts: tuple[ObservationCount, ...] = ()
    input_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    blocker: ObservationBlocker | None = None

    @field_validator("observed_value")
    @classmethod
    def validate_observed_value(cls, value: JsonValue | None) -> JsonValue | None:
        if value is not None:
            _validate_finite_json(value)
        return value

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        for description, values in (
            ("input IDs", self.input_ids),
            ("artifact IDs", self.artifact_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"observation {description} must be unique")
            if tuple(sorted(values)) != values:
                raise ValueError(f"observation {description} must be sorted")

        count_names = tuple(count.name for count in self.counts)
        if len(count_names) != len(set(count_names)):
            raise ValueError("observation count names must be unique")
        if tuple(sorted(count_names)) != count_names:
            raise ValueError("observation counts must be sorted by name")

        if (self.method_id is None) != (self.method_provenance is None):
            raise ValueError(
                "observation method ID and method provenance must be provided together"
            )
        if self.method_provenance is MethodProvenance.UPSTREAM_INFORMED:
            if self.method_reference_artifact_id is None:
                raise ValueError(
                    "upstream-informed method requires a reference artifact ID"
                )
        elif self.method_reference_artifact_id is not None:
            raise ValueError(
                "method reference artifact is only valid for upstream-informed methods"
            )
        if (
            self.method_reference_artifact_id is not None
            and self.method_reference_artifact_id not in self.artifact_ids
        ):
            raise ValueError(
                "method reference artifact must appear in observation artifact IDs"
            )

        blocker_kind_by_verdict = {
            Verdict.BLOCKED_MISSING_INPUT: BlockerKind.MISSING_INPUT,
            Verdict.BLOCKED_UNSPECIFIED_METHOD: BlockerKind.UNSPECIFIED_METHOD,
            Verdict.EXTERNAL_OR_CITATION_DEPENDENT: (
                BlockerKind.EXTERNAL_OR_CITATION_DEPENDENT
            ),
            Verdict.NOT_ATTEMPTED: BlockerKind.NOT_ATTEMPTED,
            Verdict.NOT_APPLICABLE: BlockerKind.NOT_APPLICABLE,
        }
        required_blocker_kind = blocker_kind_by_verdict.get(self.verdict)
        if required_blocker_kind is None:
            if self.blocker is not None:
                raise ValueError("evidence verdicts cannot include a blocker")
            if self.actual_evidence_boundary is None:
                raise ValueError(
                    "evidence verdicts require an actual evidence boundary"
                )
            if self.observed_value is None and not self.diagnostics:
                raise ValueError(
                    "evidence verdicts require an observed value or diagnostics"
                )
        elif self.blocker is None or self.blocker.kind is not required_blocker_kind:
            raise ValueError(
                "terminal non-evidence verdict requires a matching blocker"
            )

        if (
            self.verdict is Verdict.SOURCE_ONLY_MATCH
            and self.actual_evidence_boundary
            not in {
                EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT,
                EvidenceBoundary.AUTHOR_DOWNSTREAM_TABLE,
            }
        ):
            raise ValueError(
                "source-only match requires paper, final-artifact, or author-table evidence"
            )
        if self.verdict is Verdict.REPRODUCED and self.actual_evidence_boundary in {
            EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT,
            EvidenceBoundary.AUTHOR_DOWNSTREAM_TABLE,
        }:
            raise ValueError(
                "reproduced verdict requires independently recomputed evidence"
            )
        return self


class ObservationFileIdentity(PaperModel):
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)
    observation_count: int = Field(ge=0)

    _validate_filename = field_validator("filename")(_validate_repository_path)

    @model_validator(mode="after")
    def validate_filename(self) -> Self:
        if len(PurePosixPath(self.filename).parts) != 1:
            raise ValueError("observation filename must be a bare filename")
        return self


class RunManifest(PaperModel):
    run_format: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    started_at: datetime
    completed_at: datetime
    paper_identity: ContentIdentity
    config_identity: ContentIdentity
    claims_identity: ContentIdentity
    code_identity: CodeIdentity
    runtime_identity: RuntimeIdentity
    input_identities: tuple[ContentIdentity, ...] = ()
    artifact_identities: tuple[ContentIdentity, ...] = ()
    observations_identity: ObservationFileIdentity
    complete: Literal[True] = True

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("run timestamps must include timezone offsets")
        if self.completed_at < self.started_at:
            raise ValueError("run completion timestamp cannot precede start timestamp")
        for description, identities in (
            ("input", self.input_identities),
            ("artifact", self.artifact_identities),
        ):
            ids = tuple(identity.id for identity in identities)
            if len(ids) != len(set(ids)):
                raise ValueError(f"run {description} identity IDs must be unique")
            if tuple(sorted(ids)) != ids:
                raise ValueError(f"run {description} identities must be sorted by ID")
        dirty_diff_id = self.code_identity.dirty_diff_artifact_id
        if dirty_diff_id is not None and dirty_diff_id not in {
            identity.id for identity in self.artifact_identities
        }:
            raise ValueError(
                "dirty code diff artifact must appear in run artifact identities"
            )
        return self


class RunBundle(PaperModel):
    manifest: RunManifest
    observations: tuple[Observation, ...]


__all__ = [
    "BlockerKind",
    "ClaimOwnership",
    "ClaimRegistry",
    "CodeIdentity",
    "CodeTreeState",
    "ContentIdentity",
    "EvidenceBoundary",
    "ExpectationKind",
    "MethodProvenance",
    "MethodProvenanceEntry",
    "NamedPolicy",
    "Observation",
    "ObservationBlocker",
    "ObservationCount",
    "ObservationFileIdentity",
    "OlmesAnalysisContract",
    "OlmesComparisonPolicy",
    "OlmesComputePolicy",
    "OlmesFinalCheckpoint",
    "OlmesFinalCheckpointPolicy",
    "OlmesMetricPolicy",
    "OlmesMissingDataPolicy",
    "OlmesRecipeUniverse",
    "OlmesSeedPolicy",
    "OlmesStandardDeviationPolicy",
    "OlmesTaskGroupingPolicy",
    "OperationalizationBasis",
    "PaperClaim",
    "PaperContractReferences",
    "PaperIdentity",
    "PaperOutputs",
    "PaperReproductionContract",
    "PolicyStatus",
    "RunBundle",
    "RunManifest",
    "RuntimeIdentity",
    "SourceRegion",
    "Verdict",
]
