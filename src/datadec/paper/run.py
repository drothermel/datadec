from __future__ import annotations

import hashlib
import importlib
import platform
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import orjson
import pyarrow.parquet as pq

from datadec.paper.contracts import load_claim_registry, load_validation_contract
from datadec.paper.models import (
    PRIMARY_CLAIM_KINDS,
    AnalysisBundle,
    AnalysisId,
    AttemptResult,
    AttemptRole,
    ClaimKind,
    ClaimRegistry,
    ContentIdentity,
    MetadataDiscrepancy,
    PaperTarget,
    PaperValidationContract,
    PlotSeries,
    RuntimeTrace,
    ValidationOutcome,
)
from datadec.paper.registry import validate_cross_contracts
from datadec.paper.runs import create_analysis_bundle, load_analysis_bundle
from datadec.paper.source import (
    CitationReport,
    CoverageReport,
    DependencyReport,
    scan_tex_dependencies,
    validate_citations,
    validate_source_coverage,
)

_VALIDATION_CONFIG_PATH = "configs/paper_validation.toml"
_CLAIMS_PATH = "docs/paper/claims.toml"
_DEPENDENCY_LOCK_PATH = "uv.lock"
_PATH_PARAMETER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_ADAPTER_IMPORTS: Mapping[AnalysisId, tuple[str, str]] = {
    AnalysisId.SINGLE_SCALE: (
        "datadec.paper.verifiers.single_scale",
        "run_single_scale_attempts",
    ),
    AnalysisId.PER_TASK: (
        "datadec.paper.verifiers.single_scale",
        "run_per_task_attempts",
    ),
    AnalysisId.PROXY_METRICS: (
        "datadec.paper.verifiers.proxy_metrics",
        "run_proxy_metrics_attempts",
    ),
    AnalysisId.NOISE_SPREAD: (
        "datadec.paper.verifiers.proxy_metrics",
        "run_noise_spread_attempts",
    ),
    AnalysisId.SCALING_LAW: (
        "datadec.paper.verifiers.scaling",
        "run_scaling_law_attempts",
    ),
    AnalysisId.MATH_CODE: (
        "datadec.paper.verifiers.math_code",
        "run_math_code_attempts",
    ),
}

AnalysisAdapter = Callable[
    ...,
    tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]],
]
MetadataComparator = Callable[
    [str | Path, ClaimRegistry], tuple[MetadataDiscrepancy, ...]
]


@dataclass(frozen=True, slots=True)
class ValidationInput:
    table_id: str
    paths: tuple[Path, ...]
    identity: ContentIdentity


@dataclass(frozen=True, slots=True)
class SupportingDisposition:
    claim_id: str
    kind: ClaimKind
    outcome: ValidationOutcome


@dataclass(frozen=True, slots=True)
class ValidationSurface:
    repository_root: Path
    data_root: Path
    registry: ClaimRegistry
    contract: PaperValidationContract
    coverage: CoverageReport
    dependencies: DependencyReport
    citations: CitationReport
    inputs: tuple[ValidationInput, ...]
    supporting_dispositions: tuple[SupportingDisposition, ...]

    @property
    def input_identities(self) -> Mapping[str, ContentIdentity]:
        return {item.table_id: item.identity for item in self.inputs}


@dataclass(frozen=True, slots=True)
class RenderedOutputs:
    report: Path
    figures: tuple[Path, ...]


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


def _paper_entrypoint(contract: PaperValidationContract) -> str:
    return (
        PurePosixPath(contract.paper.source_root) / contract.paper.entrypoint
    ).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_data_root(repository_root: Path, data_dir: str | Path) -> Path:
    candidate = Path(data_dir)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return candidate.resolve(strict=True)


def _input_paths(data_root: Path, configured_path: str) -> tuple[Path, ...]:
    path = PurePosixPath(configured_path)
    if path.is_absolute() or path.as_posix() != configured_path or ".." in path.parts:
        raise ValueError(
            "validation input paths must be normalized data-relative POSIX paths"
        )
    pattern = _PATH_PARAMETER.sub("*", configured_path)
    paths = tuple(sorted(data_root.glob(pattern)))
    if not paths:
        raise FileNotFoundError(
            f"validation input did not match any files: {configured_path}"
        )
    for candidate in paths:
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(data_root)
        except ValueError as error:
            raise ValueError(
                f"validation input escapes data directory: {configured_path}"
            ) from error
        if candidate.is_symlink() or not resolved.is_file():
            raise ValueError("validation inputs must be non-symlink regular files")
    return paths


def _input_identity(
    table_id: str, paths: tuple[Path, ...], data_root: Path
) -> ContentIdentity:
    file_values = tuple(
        {
            "path": path.resolve(strict=True).relative_to(data_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    )
    if len(file_values) == 1:
        digest = file_values[0]["sha256"]
    else:
        digest = hashlib.sha256(
            orjson.dumps(file_values, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()
    return ContentIdentity(id=table_id, sha256=digest)


def _validate_input_schema(
    table_id: str, paths: tuple[Path, ...], required_columns: tuple[str, ...]
) -> None:
    expected = set(required_columns)
    for path in paths:
        actual = set(pq.read_schema(path).names)
        missing = tuple(sorted(expected - actual))
        if missing:
            raise ValueError(
                f"validation input {table_id} is missing columns {missing} in {path}"
            )


def _supporting_dispositions(
    registry: ClaimRegistry,
) -> tuple[SupportingDisposition, ...]:
    dispositions: list[SupportingDisposition] = []
    for claim in registry.claims:
        if claim.kind in PRIMARY_CLAIM_KINDS:
            continue
        if claim.supporting_outcome is None:  # guarded by PaperClaim validation
            raise AssertionError(f"supporting claim {claim.id} has no disposition")
        dispositions.append(
            SupportingDisposition(
                claim_id=claim.id,
                kind=claim.kind,
                outcome=claim.supporting_outcome,
            )
        )
    return tuple(sorted(dispositions, key=lambda value: value.claim_id))


def validate_repository(
    root: str | Path,
    data_dir: str | Path,
) -> ValidationSurface:
    """Validate the static finding contracts, manuscript, and dd_parsed inputs."""
    repository_root = Path(root).resolve(strict=True)
    data_root = _resolve_data_root(repository_root, data_dir)
    contract = load_validation_contract(
        _repository_file(repository_root, _VALIDATION_CONFIG_PATH)
    )
    registry = load_claim_registry(_repository_file(repository_root, _CLAIMS_PATH))
    validate_cross_contracts(registry, contract)

    entrypoint = _paper_entrypoint(contract)
    coverage = validate_source_coverage(repository_root, registry, entrypoint)
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

    inputs: list[ValidationInput] = []
    for spec in contract.inputs:
        paths = _input_paths(data_root, spec.path)
        _validate_input_schema(spec.id, paths, spec.columns)
        inputs.append(
            ValidationInput(
                table_id=spec.id,
                paths=paths,
                identity=_input_identity(spec.id, paths, data_root),
            )
        )
    return ValidationSurface(
        repository_root=repository_root,
        data_root=data_root,
        registry=registry,
        contract=contract,
        coverage=coverage,
        dependencies=dependencies,
        citations=citations,
        inputs=tuple(sorted(inputs, key=lambda value: value.table_id)),
        supporting_dispositions=_supporting_dispositions(registry),
    )


def _paper_targets(registry: ClaimRegistry) -> tuple[PaperTarget, ...]:
    return tuple(
        PaperTarget(
            claim_id=claim.id,
            family=claim.family,
            kind=claim.kind,
            source_file=claim.source_file,
            line_start=claim.line_start,
            line_end=claim.line_end,
            source_text=claim.text,
            value=claim.paper_target,
        )
        for claim in sorted(registry.claims, key=lambda value: value.id)
        if claim.kind in PRIMARY_CLAIM_KINDS
    )


def _load_analysis_adapters() -> Mapping[AnalysisId, AnalysisAdapter]:
    adapters: dict[AnalysisId, AnalysisAdapter] = {}
    for analysis_id, (module_name, function_name) in _ADAPTER_IMPORTS.items():
        module = importlib.import_module(module_name)
        adapter = getattr(module, function_name)
        if not callable(adapter):
            raise TypeError(f"analysis adapter is not callable: {function_name}")
        adapters[analysis_id] = adapter
    return adapters


def _load_metadata_comparator() -> MetadataComparator:
    module = importlib.import_module("datadec.paper.verifiers.metadata")
    comparator = getattr(module, "compare_descriptive_metadata")
    if not callable(comparator):
        raise TypeError("metadata comparator is not callable")
    return comparator


def _validate_adapter_results(
    *,
    analysis_id: AnalysisId,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    results: tuple[AttemptResult, ...],
    plot_series: tuple[PlotSeries, ...],
) -> None:
    claims = {claim.id: claim for claim in registry.claims}
    specs = tuple(spec for spec in contract.attempts if spec.analysis_id is analysis_id)
    specs_by_id = {spec.id: spec for spec in specs}
    sensitivity_parents = {
        sensitivity_id: spec
        for spec in specs
        for sensitivity_id in spec.sensitivity_ids
    }
    result_specs = {}
    supplied_ids = tuple(result.attempt_id for result in results)
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError(f"{analysis_id.value} adapter returned duplicate attempts")
    attempts_by_id = {result.attempt_id: result for result in results}
    supplied_defaults = {
        result.attempt_id for result in results if result.role is AttemptRole.DEFAULT
    }
    expected_defaults = set(specs_by_id)
    if supplied_defaults != expected_defaults:
        raise ValueError(
            f"{analysis_id.value} adapter did not complete its configured attempts: "
            f"missing={sorted(expected_defaults - supplied_defaults)}, "
            f"unexpected={sorted(supplied_defaults - expected_defaults)}"
        )
    allowed_ids = expected_defaults | set(sensitivity_parents)
    unexpected_results = sorted(set(supplied_ids) - allowed_ids)
    if unexpected_results:
        raise ValueError(
            f"{analysis_id.value} adapter returned undeclared results: "
            f"{unexpected_results}"
        )

    rules = {rule.id: rule for rule in contract.comparison_rules}
    for result in results:
        spec = specs_by_id.get(result.attempt_id)
        if spec is None:
            spec = sensitivity_parents[result.attempt_id]
            if result.role is not AttemptRole.SENSITIVITY:
                raise ValueError(
                    "declared sensitivities require sensitivity-role results"
                )
            if result.parent_attempt_id != spec.id:
                raise ValueError("sensitivity result has the wrong parent attempt")
        elif result.role is not AttemptRole.DEFAULT:
            raise ValueError("configured default attempts require default-role results")
        result_specs[result.attempt_id] = spec
        rule = rules[spec.comparison_rule_id]
        if (
            result.claim_id != spec.claim_id
            or result.evidence_level is not spec.evidence_level
            or result.comparison_rule_id != rule.id
            or result.comparison_rule_version != rule.version
            or result.transformation_ids != spec.transformation_ids
            or result.target_value != claims[result.claim_id].paper_target
        ):
            raise ValueError(
                f"adapter result {result.attempt_id} differs from its static contract"
            )
        declared_inputs = {item.table_id for item in spec.inputs}
        if any(
            selection.logical_table_id not in declared_inputs
            for selection in result.row_selections
        ):
            raise ValueError(
                f"adapter result {result.attempt_id} uses an undeclared input"
            )

    supplied_series_ids = tuple(series.id for series in plot_series)
    if len(supplied_series_ids) != len(set(supplied_series_ids)):
        raise ValueError(f"{analysis_id.value} adapter returned duplicate plot series")
    expected_series = {
        series_id: spec for spec in specs for series_id in spec.plot_series_ids
    }
    required_series = {
        series_id
        for series_id, spec in expected_series.items()
        if attempts_by_id[spec.id].outcome
        is not ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    }
    missing_series = required_series - set(supplied_series_ids)
    unexpected_series = set(supplied_series_ids) - set(expected_series)
    if missing_series or unexpected_series:
        raise ValueError(
            f"{analysis_id.value} adapter plot series differ from config: "
            f"missing={sorted(missing_series)}, "
            f"unexpected={sorted(unexpected_series)}"
        )
    series_by_attempt: dict[str, set[str]] = {}
    for series in plot_series:
        if series.attempt_id != expected_series[series.id].id:
            raise ValueError(
                f"plot series {series.id} is attached to the wrong attempt"
            )
        series_by_attempt.setdefault(series.attempt_id, set()).add(series.id)
    for result in results:
        declared = set(result_specs[result.attempt_id].plot_series_ids)
        referenced = set(result.plot_series_ids)
        if referenced - declared:
            raise ValueError(
                f"adapter result {result.attempt_id} references undeclared plot series"
            )
        if referenced != series_by_attempt.get(result.attempt_id, set()):
            raise ValueError(
                f"adapter result {result.attempt_id} plot-series references differ "
                "from returned series"
            )


def _runtime_trace(repository_root: Path) -> RuntimeTrace:
    lock_path = repository_root / _DEPENDENCY_LOCK_PATH
    return RuntimeTrace(
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        dependency_lock_sha256=(
            _sha256_file(lock_path) if lock_path.is_file() else None
        ),
    )


def run_validation(
    root: str | Path,
    run_id: str,
    data_dir: str | Path,
    *,
    adapter_registry: Mapping[AnalysisId, AnalysisAdapter] | None = None,
    metadata_comparator: MetadataComparator | None = None,
) -> AnalysisBundle:
    """Run every configured analysis and persist one complete format-3 bundle."""
    started_at = datetime.now(timezone.utc)
    surface = validate_repository(root, data_dir)
    adapters = (
        _load_analysis_adapters() if adapter_registry is None else adapter_registry
    )
    if set(adapters) != set(AnalysisId):
        raise ValueError(
            "analysis adapter registry must contain exactly the closed analysis IDs"
        )
    compare_metadata = (
        _load_metadata_comparator()
        if metadata_comparator is None
        else metadata_comparator
    )
    metadata_discrepancies = tuple(
        compare_metadata(surface.repository_root, surface.registry)
    )

    identities = surface.input_identities
    attempts: list[AttemptResult] = []
    plot_series: list[PlotSeries] = []
    for analysis_id in AnalysisId:
        analysis_attempts, analysis_series = adapters[analysis_id](
            repository_root=surface.repository_root,
            data_root=surface.data_root,
            registry=surface.registry,
            contract=surface.contract,
            input_identities=identities,
        )
        analysis_attempts = tuple(analysis_attempts)
        analysis_series = tuple(analysis_series)
        _validate_adapter_results(
            analysis_id=analysis_id,
            registry=surface.registry,
            contract=surface.contract,
            results=analysis_attempts,
            plot_series=analysis_series,
        )
        attempts.extend(analysis_attempts)
        plot_series.extend(analysis_series)

    current_identities = {
        item.table_id: _input_identity(item.table_id, item.paths, surface.data_root)
        for item in surface.inputs
    }
    if current_identities != dict(identities):
        raise ValueError("validation inputs changed while analyses were running")

    completed_at = datetime.now(timezone.utc)
    return create_analysis_bundle(
        surface.repository_root / surface.contract.outputs.runs_root,
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        runtime_trace=_runtime_trace(surface.repository_root),
        input_identities=identities.values(),
        targets=_paper_targets(surface.registry),
        metadata_discrepancies=metadata_discrepancies,
        attempts=attempts,
        plot_series=plot_series,
    )


def render_validation(root: str | Path, run_id: str) -> RenderedOutputs:
    """Render one completed bundle without reopening scientific inputs."""
    repository_root = Path(root).resolve(strict=True)
    contract = load_validation_contract(
        _repository_file(repository_root, _VALIDATION_CONFIG_PATH)
    )
    bundle = load_analysis_bundle(
        repository_root / contract.outputs.runs_root,
        run_id,
    )
    report_module = importlib.import_module("datadec.paper.report")
    transaction_module = importlib.import_module("datadec.paper.output_transaction")
    renderer = getattr(report_module, "render_bundle_outputs")
    rendered = renderer(bundle)
    report_path = repository_root / contract.outputs.report
    figures_root = repository_root / contract.outputs.figures_root
    figure_paths = tuple(
        figures_root / filename for filename, _content in rendered.figures
    )
    replace_outputs = getattr(transaction_module, "replace_output_set")
    replace_outputs(
        (
            (report_path, rendered.report),
            *(
                (figures_root / filename, content)
                for filename, content in rendered.figures
            ),
        ),
        exact_directories={figures_root: figure_paths},
    )
    return RenderedOutputs(report=report_path, figures=figure_paths)


__all__ = [
    "AnalysisAdapter",
    "MetadataComparator",
    "RenderedOutputs",
    "SupportingDisposition",
    "ValidationInput",
    "ValidationSurface",
    "render_validation",
    "run_validation",
    "validate_repository",
]
