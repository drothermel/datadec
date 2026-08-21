from __future__ import annotations

from enum import UNIQUE, StrEnum, verify
from pathlib import PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class PaperReproductionContract(PaperModel):
    paper: PaperIdentity
    contracts: PaperContractReferences
    methods: tuple[MethodProvenanceEntry, ...]
    policies: tuple[NamedPolicy, ...]
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


__all__ = [
    "ClaimOwnership",
    "ClaimRegistry",
    "EvidenceBoundary",
    "ExpectationKind",
    "MethodProvenance",
    "MethodProvenanceEntry",
    "NamedPolicy",
    "PaperClaim",
    "PaperContractReferences",
    "PaperIdentity",
    "PaperOutputs",
    "PaperReproductionContract",
    "PolicyStatus",
    "SourceRegion",
]
