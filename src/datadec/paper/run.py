from __future__ import annotations

import hashlib
import platform
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TypeVar

from pydantic import BaseModel

from datadec.config import (
    DataDecideCatalog,
    OLMESContract,
    PublishedResultsManifest,
    ScalingLawContract,
    SourceManifest,
)
from datadec.paper.contracts import load_claim_registry
from datadec.paper.models import (
    BlockerKind,
    ClaimOwnership,
    ClaimRegistry,
    CodeIdentity,
    CodeTreeState,
    ContentIdentity,
    EvidenceBoundary,
    Observation,
    ObservationBlocker,
    PaperClaim,
    PaperReproductionContract,
    RunBundle,
    RuntimeIdentity,
    Verdict,
)
from datadec.paper.policies import resolve_olmes_policy
from datadec.paper.report import render_report_file
from datadec.paper.runs import (
    create_run_bundle,
    load_run_bundle,
    validate_run_qualification,
)
from datadec.paper.source import (
    CitationReport,
    CoverageReport,
    DependencyReport,
    raw_line_slice_sha256,
    scan_tex_dependencies,
    validate_citations,
    validate_source_coverage,
)
from datadec.paper.verifiers.olmes import (
    NormalizedOlmesPolicy,
    NormalizedOlmesVerification,
    verify_normalized_olmes_parquet,
)
from datadec.paper.verifiers.suite import (
    CheckStatus,
    SuiteFact,
    SuiteRowVerification,
    SuiteVerification,
    parse_suite_table,
    verify_suite,
)

_REPRODUCTION_CONFIG_PATH = "configs/paper_reproduction.toml"
_DEPENDENCY_LOCK_PATH = "uv.lock"
_SUITE_TABLE_PATH = "docs/paper/tables/suite_stats.tex"
_EXPECTED_CITATION_KEY_COUNT = 43
_NORMALIZED_OLMES_INPUT_ID = "normalized-olmes-input"
_ARTIFACT_RELEASE_MANIFEST_ID = "artifact-release-manifest"
_EVALUATION_RERUN_RESULTS_ID = "evaluation-rerun-results"
_TRAINING_RUN_MANIFEST_ID = "training-run-manifest"
_CORPUS_CONSTRUCTION_MANIFEST_ID = "corpus-construction-manifest"
_ABSTRACT_SINGLE_SCALE_CLAIM_ID = "DD-0011"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class RepositoryValidation:
    repository_root: Path
    contract: PaperReproductionContract
    registry: ClaimRegistry
    coverage: CoverageReport
    dependencies: DependencyReport
    citations: CitationReport
    olmes_policy: NormalizedOlmesPolicy
    suite: SuiteVerification


def _repository_file(root: Path, relative_path: str) -> Path:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or path.as_posix() != relative_path or ".." in path.parts:
        raise ValueError(
            "repository paths must be normalized repository-relative POSIX paths"
        )
    candidate = root.joinpath(*path.parts)
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes repository root: {relative_path}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"required repository file not found: {relative_path}")
    return candidate


def _load_toml_model(root: Path, relative_path: str, model: type[_ModelT]) -> _ModelT:
    with _repository_file(root, relative_path).open("rb") as file:
        return model.model_validate(tomllib.load(file))


def _paper_entrypoint(contract: PaperReproductionContract) -> str:
    return (
        PurePosixPath(contract.paper.source_root) / contract.paper.entrypoint
    ).as_posix()


def validate_repository(root: str | Path) -> RepositoryValidation:
    """Validate the complete repository-owned first-run paper surface."""
    repository_root = Path(root).resolve(strict=True)
    contract = _load_toml_model(
        repository_root,
        _REPRODUCTION_CONFIG_PATH,
        PaperReproductionContract,
    )
    registry = load_claim_registry(
        _repository_file(repository_root, contract.contracts.claims_contract)
    )

    catalog = _load_toml_model(
        repository_root, contract.contracts.catalog, DataDecideCatalog
    )
    sources = _load_toml_model(
        repository_root, contract.contracts.sources, SourceManifest
    )
    published_results = _load_toml_model(
        repository_root,
        contract.contracts.published_results,
        PublishedResultsManifest,
    )
    olmes = _load_toml_model(
        repository_root, contract.contracts.olmes, OLMESContract
    ).validate_references(catalog=catalog, source_manifest=sources)
    _load_toml_model(
        repository_root,
        contract.contracts.scaling_law,
        ScalingLawContract,
    ).validate_references(
        catalog=catalog,
        olmes_contract=olmes,
        published_results_manifest=published_results,
    )

    coverage = validate_source_coverage(repository_root, registry)
    entrypoint = _paper_entrypoint(contract)
    dependencies = scan_tex_dependencies(repository_root, entrypoint)
    active_dependency_files = {
        *dependencies.tex_files,
        *dependencies.graphics_files,
        *dependencies.bibliography_files,
        *dependencies.bibliography_style_files,
        *dependencies.bbl_files,
    }
    disconnected_source_files = tuple(
        sorted(set(coverage.source_files) - active_dependency_files)
    )
    if disconnected_source_files:
        raise ValueError(
            "covered source files are disconnected from the active paper entrypoint: "
            f"{disconnected_source_files!r}"
        )
    citations = validate_citations(repository_root, entrypoint)
    if len(citations.citation_keys) != _EXPECTED_CITATION_KEY_COUNT:
        raise ValueError(
            "active paper citation-key count drifted: expected "
            f"{_EXPECTED_CITATION_KEY_COUNT}, found {len(citations.citation_keys)}"
        )
    claim_citation_keys = {
        key for claim in registry.claims for key in claim.citation_keys
    }
    if claim_citation_keys != set(citations.citation_keys):
        missing = tuple(sorted(set(citations.citation_keys) - claim_citation_keys))
        unexpected = tuple(sorted(claim_citation_keys - set(citations.citation_keys)))
        raise ValueError(
            "claim citation traces differ from active paper citations: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    olmes_policy = resolve_olmes_policy(
        contract,
        catalog=catalog,
        olmes_contract=olmes,
    )
    suite = verify_suite(
        parse_suite_table(_repository_file(repository_root, _SUITE_TABLE_PATH)),
        catalog,
    )
    return RepositoryValidation(
        repository_root=repository_root,
        contract=contract,
        registry=registry,
        coverage=coverage,
        dependencies=dependencies,
        citations=citations,
        olmes_policy=olmes_policy,
        suite=suite,
    )


def _external_observation(claim: PaperClaim) -> Observation:
    citation_keys = tuple(sorted(claim.citation_keys))
    return Observation(
        claim_id=claim.id,
        verdict=Verdict.EXTERNAL_OR_CITATION_DEPENDENT,
        observed_value=list(citation_keys),
        diagnostics=(f"citation keys: {', '.join(citation_keys)}",),
        blocker=ObservationBlocker(
            kind=BlockerKind.EXTERNAL_OR_CITATION_DEPENDENT,
            reason="the claim is attributed to external literature and was traced only to its active citation keys",
        ),
    )


def _missing_observation(
    claim: PaperClaim,
    missing_input_ids: tuple[str, ...],
    reason: str,
    *,
    input_ids: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        claim_id=claim.id,
        verdict=Verdict.BLOCKED_MISSING_INPUT,
        input_ids=tuple(sorted(input_ids)),
        blocker=ObservationBlocker(
            kind=BlockerKind.MISSING_INPUT,
            reason=reason,
            missing_input_ids=tuple(sorted(missing_input_ids)),
        ),
    )


def _unspecified_observation(
    claim: PaperClaim,
    unresolved_method_id: str,
    reason: str,
    *,
    input_ids: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        claim_id=claim.id,
        verdict=Verdict.BLOCKED_UNSPECIFIED_METHOD,
        input_ids=tuple(sorted(input_ids)),
        blocker=ObservationBlocker(
            kind=BlockerKind.UNSPECIFIED_METHOD,
            reason=reason,
            unresolved_method_id=unresolved_method_id,
        ),
    )


def _not_attempted_observation(claim: PaperClaim) -> Observation:
    return Observation(
        claim_id=claim.id,
        verdict=Verdict.NOT_ATTEMPTED,
        blocker=ObservationBlocker(
            kind=BlockerKind.NOT_ATTEMPTED,
            reason="no claim-specific mapping to the available normalized evaluation facts is implemented",
        ),
    )


def _source_observation(root: Path, claim: PaperClaim) -> Observation:
    digest = raw_line_slice_sha256(
        root,
        claim.source_file,
        claim.line_start,
        claim.line_end,
    )
    return Observation(
        claim_id=claim.id,
        actual_evidence_boundary=claim.required_evidence_boundary,
        verdict=Verdict.SOURCE_ONLY_MATCH,
        diagnostics=(
            f"source locator: {claim.source_file}:{claim.line_start}-{claim.line_end}",
            f"source line-slice SHA256: {digest}",
            "source presence is not an independent reproduction",
        ),
    )


def _suite_fact_observation(claim: PaperClaim, fact: SuiteFact) -> Observation:
    if fact.status is CheckStatus.UNSUPPORTED:
        return _missing_observation(
            claim,
            (_TRAINING_RUN_MANIFEST_ID,),
            fact.reason or "the suite verifier lacks completed training-run evidence",
        )
    verdict = (
        Verdict.SOURCE_ONLY_MATCH
        if fact.status is CheckStatus.MATCH
        else Verdict.CONTRADICTED
    )
    return Observation(
        claim_id=claim.id,
        actual_evidence_boundary=fact.available_evidence_boundary,
        verdict=verdict,
        observed_value=fact.observed,
        diagnostics=(
            f"suite fact {fact.id}: expected {fact.expected!r}, observed {fact.observed!r}",
            "catalog-derived evidence is below the required training-rerun boundary",
        ),
    )


def _suite_row_observation(claim: PaperClaim, row: SuiteRowVerification) -> Observation:
    mismatches = tuple(
        f"{match.field.value}: expected {match.expected_display}, observed {match.observed_display}"
        for match in row.field_matches
        if not match.matches
    )
    observed = {
        match.field.value: match.observed_display for match in row.field_matches
    }
    return Observation(
        claim_id=claim.id,
        actual_evidence_boundary=row.available_evidence_boundary,
        verdict=(Verdict.SOURCE_ONLY_MATCH if row.matches else Verdict.CONTRADICTED),
        observed_value=observed,
        diagnostics=(
            *(mismatches or ("all displayed suite fields match",)),
            "catalog-derived evidence is below the required training-rerun boundary",
        ),
    )


def _abstract_single_scale_observation(
    claim: PaperClaim,
    policy: NormalizedOlmesPolicy,
    verification: NormalizedOlmesVerification | None,
    olmes_input_id: str | None,
) -> Observation:
    model_size = policy.noise_size
    step = policy.final_step_by_size[model_size]
    metric = policy.target_metric_column
    summary_id = f"olmes-summary:params={model_size}:step={step}:metric={metric}"
    input_ids = (olmes_input_id,) if olmes_input_id is not None else ()
    summary = None
    if verification is not None:
        summary = next(
            (
                value
                for value in verification.checkpoint_summaries
                if value.model_size == model_size
                and value.step == step
                and value.metric == metric
            ),
            None,
        )
    if summary is None:
        return _missing_observation(
            claim,
            (summary_id,),
            "the exact paper-final 150M primary-metric decision summary is absent",
            input_ids=input_ids,
        )
    return _unspecified_observation(
        claim,
        claim.unresolved_method_id or "approximate_numeric_tolerance",
        "the exact summary is present, but the reproduction contract specifies no numeric tolerance for approximately 0.80",
        input_ids=input_ids,
    )


def build_observations(
    validation: RepositoryValidation,
    *,
    olmes_verification: NormalizedOlmesVerification | None = None,
    olmes_input_id: str | None = None,
) -> tuple[Observation, ...]:
    """Build one truthful terminal observation for every active claim."""
    suite_facts = {
        fact.claim_id: fact
        for fact in validation.suite.facts
        if fact.claim_id is not None
    }
    suite_rows = {row.claim_id: row for row in validation.suite.rows}
    observations: list[Observation] = []
    for claim in validation.registry.claims:
        if claim.owner is ClaimOwnership.EXTERNAL_CITATION:
            observation = _external_observation(claim)
        elif claim.id == _ABSTRACT_SINGLE_SCALE_CLAIM_ID:
            observation = _abstract_single_scale_observation(
                claim,
                validation.olmes_policy,
                olmes_verification,
                olmes_input_id,
            )
        elif claim.id in suite_facts:
            observation = _suite_fact_observation(claim, suite_facts[claim.id])
        elif claim.id in suite_rows:
            observation = _suite_row_observation(claim, suite_rows[claim.id])
        elif claim.unresolved_method_id is not None:
            observation = _unspecified_observation(
                claim,
                claim.unresolved_method_id,
                "the claim registry records an unresolved claim-specific method",
            )
        elif claim.owner is ClaimOwnership.ARTIFACT_RELEASE:
            observation = _missing_observation(
                claim,
                (_ARTIFACT_RELEASE_MANIFEST_ID,),
                "no pinned artifact-release manifest is available in this run",
            )
        elif claim.required_evidence_boundary in {
            EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT,
            EvidenceBoundary.AUTHOR_DOWNSTREAM_TABLE,
        }:
            observation = _source_observation(validation.repository_root, claim)
        elif claim.required_evidence_boundary is EvidenceBoundary.EVALUATION_RERUN:
            observation = _missing_observation(
                claim,
                (_EVALUATION_RERUN_RESULTS_ID,),
                "no evaluation-rerun results are available in this run",
            )
        elif claim.required_evidence_boundary is EvidenceBoundary.TRAINING_RERUN:
            observation = _missing_observation(
                claim,
                (_TRAINING_RUN_MANIFEST_ID,),
                "no completed training-run manifest is available in this run",
            )
        elif claim.required_evidence_boundary is EvidenceBoundary.CORPUS_CONSTRUCTION:
            observation = _missing_observation(
                claim,
                (_CORPUS_CONSTRUCTION_MANIFEST_ID,),
                "no corpus-construction manifest is available in this run",
            )
        elif claim.required_evidence_boundary in {
            EvidenceBoundary.AGGREGATE_EVALUATION,
            EvidenceBoundary.INSTANCE_AND_CHOICE,
        }:
            observation = _not_attempted_observation(claim)
        else:  # pragma: no cover - the closed enum is exhausted above
            raise AssertionError(
                f"unclassified evidence boundary for claim {claim.id}: "
                f"{claim.required_evidence_boundary}"
            )
        observations.append(observation)

    ordered = tuple(sorted(observations, key=lambda value: value.claim_id))
    if len(ordered) != len(validation.registry.claims):
        raise AssertionError("observation construction did not preserve claim count")
    return ordered


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(root: Path, relative_path: str) -> ContentIdentity:
    path = _repository_file(root, relative_path)
    return ContentIdentity(id=relative_path, sha256=_sha256_file(path))


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_code_identity(root: Path) -> CodeIdentity:
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ValueError(f"repository root is not the Git top level: {root}")
    dirty = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ValueError("paper verification runs require a clean Git tree")
    return CodeIdentity(
        commit_sha=_git_output(root, "rev-parse", "HEAD"),
        tree_state=CodeTreeState.CLEAN,
    )


def run_repository(
    root: str | Path,
    run_id: str,
    olmes_path: str | Path | None = None,
) -> RunBundle:
    """Run available repository verifiers and create one immutable bundle."""
    repository_root = Path(root).resolve(strict=True)
    code_identity = _clean_code_identity(repository_root)
    started_at = datetime.now(timezone.utc)
    validation = validate_repository(repository_root)

    input_identities = [
        _file_identity(repository_root, _paper_entrypoint(validation.contract)),
        _file_identity(repository_root, _DEPENDENCY_LOCK_PATH),
    ]
    olmes_verification: NormalizedOlmesVerification | None = None
    olmes_input_id: str | None = None
    if olmes_path is not None:
        normalized_path = Path(olmes_path)
        if not normalized_path.is_absolute():
            normalized_path = repository_root / normalized_path
        normalized_path = normalized_path.resolve(strict=True)
        before_digest = _sha256_file(normalized_path)
        olmes_verification = verify_normalized_olmes_parquet(
            normalized_path,
            validation.olmes_policy,
        )
        after_digest = _sha256_file(normalized_path)
        if after_digest != before_digest:
            raise ValueError(
                "normalized OLMES input changed while it was being verified"
            )
        olmes_input_id = _NORMALIZED_OLMES_INPUT_ID
        input_identities.append(
            ContentIdentity(id=olmes_input_id, sha256=before_digest)
        )

    observations = build_observations(
        validation,
        olmes_verification=olmes_verification,
        olmes_input_id=olmes_input_id,
    )
    if _clean_code_identity(repository_root) != code_identity:
        raise ValueError("Git commit or clean-tree identity changed during the run")
    completed_at = datetime.now(timezone.utc)
    lock_identity = _file_identity(repository_root, _DEPENDENCY_LOCK_PATH)
    return create_run_bundle(
        repository_root / validation.contract.outputs.runs_root,
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        paper_identity=ContentIdentity(
            id=f"arxiv:{validation.contract.paper.arxiv_id}",
            sha256=validation.contract.paper.archive_sha256,
        ),
        config_identity=_file_identity(repository_root, _REPRODUCTION_CONFIG_PATH),
        claims_identity=_file_identity(
            repository_root,
            validation.contract.contracts.claims_contract,
        ),
        code_identity=code_identity,
        runtime_identity=RuntimeIdentity(
            python_version=platform.python_version(),
            implementation=platform.python_implementation(),
            platform=platform.platform(),
            dependency_lock_sha256=lock_identity.sha256,
        ),
        active_claim_ids=(claim.id for claim in validation.registry.claims),
        observations=observations,
        input_identities=input_identities,
        observations_filename=validation.contract.outputs.observations_filename,
        manifest_filename=validation.contract.outputs.run_manifest_filename,
    )


def render_repository(root: str | Path, run_id: str) -> Path:
    """Validate and atomically render one selected immutable run bundle."""
    validation = validate_repository(root)
    bundle = load_run_bundle(
        validation.repository_root / validation.contract.outputs.runs_root,
        run_id,
        active_claim_ids=(claim.id for claim in validation.registry.claims),
        manifest_filename=validation.contract.outputs.run_manifest_filename,
    )
    validate_run_qualification(bundle.manifest)

    expected_paper = ContentIdentity(
        id=f"arxiv:{validation.contract.paper.arxiv_id}",
        sha256=validation.contract.paper.archive_sha256,
    )
    expected_config = _file_identity(
        validation.repository_root, _REPRODUCTION_CONFIG_PATH
    )
    expected_claims = _file_identity(
        validation.repository_root,
        validation.contract.contracts.claims_contract,
    )
    for description, expected, actual in (
        ("paper", expected_paper, bundle.manifest.paper_identity),
        ("config", expected_config, bundle.manifest.config_identity),
        ("claims", expected_claims, bundle.manifest.claims_identity),
    ):
        if actual != expected:
            raise ValueError(
                f"run {description} identity does not match the current repository"
            )

    output_path = validation.repository_root / validation.contract.outputs.report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_report_file(
        validation.registry,
        bundle.manifest,
        bundle.observations,
        output_path,
    )
    return output_path


__all__ = [
    "RepositoryValidation",
    "build_observations",
    "render_repository",
    "run_repository",
    "validate_repository",
]
