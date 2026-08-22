from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import datadec.paper.run as run_module
from datadec.paper.contracts import load_validation_contract
from datadec.paper.models import (
    PRIMARY_CLAIM_KINDS,
    AnalysisId,
    AttemptResult,
    AttemptRole,
    AxisScale,
    AxisSpec,
    DimensionValue,
    MeasureValue,
    MetadataDiscrepancy,
    PlotPoint,
    PlotSeries,
    RowSelection,
    ValidationOutcome,
)
from datadec.paper.run import run_validation, validate_repository

_REPOSITORY_ROOT = Path(__file__).parents[2]
_KEY_SHA256 = "c" * 64


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dd-parsed")
    contract = load_validation_contract(
        _REPOSITORY_ROOT / "configs/paper_validation.toml"
    )
    for spec in contract.inputs:
        relative = spec.path.replace("{recipe}", "fixture-recipe")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {column: pa.array([], type=pa.string()) for column in spec.columns}
        )
        pq.write_table(table, path)
    return root


@pytest.fixture(scope="module")
def surface(data_dir: Path) -> run_module.ValidationSurface:
    return validate_repository(_REPOSITORY_ROOT, data_dir)


def _result_for_spec(
    spec,  # type: ignore[no-untyped-def]
    surface: run_module.ValidationSurface,
) -> AttemptResult:
    claim = next(
        claim for claim in surface.registry.claims if claim.id == spec.claim_id
    )
    rule = next(
        rule
        for rule in surface.contract.comparison_rules
        if rule.id == spec.comparison_rule_id
    )
    return AttemptResult(
        attempt_id=spec.id,
        claim_id=spec.claim_id,
        role=AttemptRole.DEFAULT,
        evidence_level=spec.evidence_level,
        comparison_rule_id=rule.id,
        comparison_rule_version=rule.version,
        transformation_ids=spec.transformation_ids,
        row_selections=tuple(
            RowSelection(
                logical_table_id=item.table_id,
                columns=item.columns,
                predicates=(),
                local_parquet_sha256=surface.input_identities[item.table_id].sha256,
                selected_row_count=1,
                selected_key_sha256=_KEY_SHA256,
            )
            for item in spec.inputs
        ),
        target_value=claim.paper_target,
        computed_value=True,
        outcome=ValidationOutcome.REPRODUCED,
        plot_series_ids=spec.plot_series_ids,
    )


def _series_for_spec(spec) -> tuple[PlotSeries, ...]:  # type: ignore[no-untyped-def]
    return tuple(
        PlotSeries(
            id=series_id,
            figure="paper-analog",
            panel=spec.claim_id,
            semantic_kind="decision_accuracy",
            x_axis=AxisSpec(measure="compute", scale=AxisScale.LOG, unit="FLOPs"),
            y_axis=AxisSpec(
                measure="decision_accuracy", scale=AxisScale.LINEAR, unit="ratio"
            ),
            dimensions=("model_size",),
            measures=("compute", "decision_accuracy"),
            attempt_id=spec.id,
            points=(
                PlotPoint(
                    dimensions=(DimensionValue(name="model_size", value="150M"),),
                    measures=(
                        MeasureValue(name="compute", value=1.0),
                        MeasureValue(name="decision_accuracy", value=0.8),
                    ),
                ),
            ),
        )
        for series_id in spec.plot_series_ids
    )


def _adapters(
    surface: run_module.ValidationSurface,
) -> dict[AnalysisId, run_module.AnalysisAdapter]:
    adapters: dict[AnalysisId, run_module.AnalysisAdapter] = {}
    for analysis_id in AnalysisId:
        specs = tuple(
            spec
            for spec in surface.contract.attempts
            if spec.analysis_id is analysis_id
        )

        def adapter(*, _specs=specs, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["repository_root"] == surface.repository_root
            assert kwargs["data_root"] == surface.data_root
            assert kwargs["registry"] is surface.registry
            assert kwargs["contract"] is surface.contract
            assert kwargs["input_identities"] == surface.input_identities
            return (
                tuple(_result_for_spec(spec, surface) for spec in _specs),
                tuple(series for spec in _specs for series in _series_for_spec(spec)),
            )

        adapters[analysis_id] = adapter
    return adapters


def test_repository_validation_covers_current_static_and_input_surface(
    surface: run_module.ValidationSurface,
    data_dir: Path,
) -> None:
    assert len(surface.registry.claims) == 455
    assert (
        sum(claim.kind in PRIMARY_CLAIM_KINDS for claim in surface.registry.claims)
        == 79
    )
    assert len(surface.supporting_dispositions) == 376
    assert len(surface.inputs) == 5
    assert set(surface.input_identities) == {
        "cheap_decisions",
        "new_eval_decision_accuracy",
        "new_eval_means",
        "olmes_aggregate",
        "scaling_evaluations",
    }
    assert not (data_dir / "processed/olmes-details").exists()
    assert len(surface.coverage.claim_ids) == 455
    assert len(surface.citations.citation_keys) == 43


def test_validation_rejects_input_schema_drift(data_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "data"
    for path in data_dir.rglob("*.parquet"):
        destination = copied / path.relative_to(data_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    aggregate = copied / "processed/olmes.parquet"
    table = pq.read_table(aggregate).drop(["primary_metric"])
    pq.write_table(table, aggregate)

    with pytest.raises(ValueError, match="missing columns.*primary_metric"):
        validate_repository(_REPOSITORY_ROOT, copied)


def test_validation_rejects_missing_used_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="processed/olmes.parquet"):
        validate_repository(_REPOSITORY_ROOT, tmp_path)


def test_validation_code_has_structural_author_result_and_parameter_update_denial() -> (
    None
):
    source = inspect.getsource(run_module)
    compact = source.lower().replace("_", "")
    assert "published-results" not in source
    assert "published_results" not in source
    assert "optimizer" not in compact
    assert "backward(" not in compact
    assert "fit(" not in compact


def test_run_creates_79_executable_empirical_results(
    surface: run_module.ValidationSurface,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_surface = replace(surface, repository_root=tmp_path)
    monkeypatch.setattr(
        run_module, "validate_repository", lambda root, data: temporary_surface
    )
    discrepancy = MetadataDiscrepancy(
        claim_id="DD-0269",
        paper_locator="docs/paper/tables/suite_stats.tex:1",
        paper_value=2048,
        metadata_source="configs/catalog.toml",
        metadata_value=4096,
        note="Current metadata differs from the paper description.",
    )

    bundle = run_validation(
        tmp_path,
        "complete-run",
        surface.data_root,
        adapter_registry=_adapters(temporary_surface),
        metadata_comparator=lambda root, registry: (discrepancy,),
    )

    assert bundle.manifest.run_format == 3
    assert len(bundle.targets) == 79
    assert len(bundle.attempts) == 79
    assert bundle.metadata_discrepancies == (discrepancy,)
    assert (
        sum(
            attempt.outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
            for attempt in bundle.attempts
        )
        == 0
    )
    assert all(
        attempt.missing_groups
        and all(
            selection.selected_row_count == 0 for selection in attempt.row_selections
        )
        for attempt in bundle.attempts
        if attempt.outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    )
    assert {
        path.name
        for path in (tmp_path / "data/paper-validation/runs/complete-run").iterdir()
    } == {"manifest.json", "targets.json", "attempts.json", "plot-series.json"}


def test_dirty_or_non_git_repository_is_not_a_run_gate(
    surface: run_module.ValidationSurface,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_surface = replace(surface, repository_root=tmp_path)
    monkeypatch.setattr(
        run_module, "validate_repository", lambda root, data: temporary_surface
    )

    bundle = run_validation(
        tmp_path,
        "dirty-allowed",
        surface.data_root,
        adapter_registry=_adapters(temporary_surface),
        metadata_comparator=lambda root, registry: (),
    )

    assert bundle.manifest.code_trace is None


def test_render_reads_only_the_completed_bundle_and_replaces_rendered_set(
    surface: run_module.ValidationSurface,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(manifest=SimpleNamespace(run_id="selected"))
    rendered = SimpleNamespace(
        report=b"report\n",
        figures=(("outcome-audit.svg", b"<svg/>"),),
    )
    replaced: dict[str, object] = {}
    monkeypatch.setattr(run_module, "load_analysis_bundle", lambda root, run_id: bundle)
    monkeypatch.setattr(
        run_module,
        "validate_repository",
        lambda *args: pytest.fail("render must not reopen validation inputs"),
    )

    def load_module(name: str):  # type: ignore[no-untyped-def]
        if name == "datadec.paper.report":
            return SimpleNamespace(render_bundle_outputs=lambda value: rendered)
        if name == "datadec.paper.output_transaction":
            return SimpleNamespace(
                replace_output_set=lambda outputs,
                *,
                exact_directories: replaced.update(
                    outputs=tuple(outputs), exact_directories=exact_directories
                )
            )
        raise AssertionError(name)

    monkeypatch.setattr(run_module.importlib, "import_module", load_module)

    result = run_module.render_validation(_REPOSITORY_ROOT, "selected")

    assert result.report == _REPOSITORY_ROOT / surface.contract.outputs.report
    assert result.figures == (
        _REPOSITORY_ROOT / surface.contract.outputs.figures_root / "outcome-audit.svg",
    )
    assert replaced["outputs"] == (
        (result.report, b"report\n"),
        (result.figures[0], b"<svg/>"),
    )
    assert replaced["exact_directories"] == {
        _REPOSITORY_ROOT / surface.contract.outputs.figures_root: result.figures
    }


def test_adapter_may_not_serialize_unfinished_analysis_as_missing_data(
    surface: run_module.ValidationSurface,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_surface = replace(surface, repository_root=tmp_path)
    adapters = _adapters(temporary_surface)
    adapters[AnalysisId.SINGLE_SCALE] = lambda **kwargs: ((), ())
    monkeypatch.setattr(
        run_module, "validate_repository", lambda root, data: temporary_surface
    )

    with pytest.raises(ValueError, match="did not complete.*missing="):
        run_validation(
            tmp_path,
            "unfinished",
            surface.data_root,
            adapter_registry=adapters,
            metadata_comparator=lambda root, registry: (),
        )
    assert not (tmp_path / "data/paper-validation/runs/unfinished").exists()


def test_configured_plot_series_are_required(
    surface: run_module.ValidationSurface,
) -> None:
    spec = next(
        spec
        for spec in surface.contract.attempts
        if spec.analysis_id is AnalysisId.SINGLE_SCALE
    )
    configured = spec.model_copy(update={"plot_series_ids": ("dd-test-paper-analog",)})
    contract = surface.contract.model_copy(
        update={
            "attempts": tuple(
                configured if item.id == spec.id else item
                for item in surface.contract.attempts
            )
        }
    )
    results = tuple(
        _result_for_spec(item, surface)
        for item in contract.attempts
        if item.analysis_id is AnalysisId.SINGLE_SCALE
    )

    with pytest.raises(ValueError, match="plot series differ from config"):
        run_module._validate_adapter_results(
            analysis_id=AnalysisId.SINGLE_SCALE,
            registry=surface.registry,
            contract=contract,
            results=results,
            plot_series=(),
        )


def test_adapter_result_evidence_must_match_attempt_spec(
    surface: run_module.ValidationSurface,
) -> None:
    specs = tuple(
        spec
        for spec in surface.contract.attempts
        if spec.analysis_id is AnalysisId.SCALING_LAW
    )
    results = tuple(_result_for_spec(spec, surface) for spec in specs)
    mismatched = results[0].model_copy(update={"evidence_level": "lower_level_rows"})

    with pytest.raises(ValueError, match="differs from its static contract"):
        run_module._validate_adapter_results(
            analysis_id=AnalysisId.SCALING_LAW,
            registry=surface.registry,
            contract=surface.contract,
            results=(mismatched, *results[1:]),
            plot_series=tuple(
                series for spec in specs for series in _series_for_spec(spec)
            ),
        )


def test_not_assessable_plot_attempt_may_omit_configured_series(
    surface: run_module.ValidationSurface,
) -> None:
    specs = tuple(
        spec
        for spec in surface.contract.attempts
        if spec.analysis_id is AnalysisId.SINGLE_SCALE
    )
    missing_spec = next(spec for spec in specs if spec.plot_series_ids)
    results = tuple(
        _result_for_spec(spec, surface).model_copy(
            update={
                "computed_value": None,
                "diagnostics": ("configured plot surface is incomplete",),
                "missing_groups": ("task=missing",),
                "outcome": ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
                "plot_series_ids": (),
            }
        )
        if spec.id == missing_spec.id
        else _result_for_spec(spec, surface)
        for spec in specs
    )
    series = tuple(
        value
        for spec in specs
        if spec.id != missing_spec.id
        for value in _series_for_spec(spec)
    )

    run_module._validate_adapter_results(
        analysis_id=AnalysisId.SINGLE_SCALE,
        registry=surface.registry,
        contract=surface.contract,
        results=results,
        plot_series=series,
    )


def test_unexpected_plot_series_is_rejected(
    surface: run_module.ValidationSurface,
) -> None:
    specs = tuple(
        spec
        for spec in surface.contract.attempts
        if spec.analysis_id is AnalysisId.SINGLE_SCALE
    )
    spec = next(item for item in specs if item.plot_series_ids)
    unexpected = _series_for_spec(spec)[0].model_copy(update={"id": "unexpected"})

    with pytest.raises(ValueError, match="unexpected=.*unexpected"):
        run_module._validate_adapter_results(
            analysis_id=AnalysisId.SINGLE_SCALE,
            registry=surface.registry,
            contract=surface.contract,
            results=tuple(_result_for_spec(item, surface) for item in specs),
            plot_series=(
                unexpected,
                *(
                    value
                    for item in specs
                    if item.id != spec.id
                    for value in _series_for_spec(item)
                ),
            ),
        )


def test_closed_format_strings_are_exact() -> None:
    assert {analysis.value for analysis in AnalysisId} == {
        "single_scale",
        "per_task",
        "proxy_metrics",
        "noise_spread",
        "scaling_law",
        "math_code",
    }
    assert {outcome.value for outcome in ValidationOutcome} == {
        "reproduced",
        "approximately_reproduced",
        "directionally_consistent",
        "not_reproduced",
        "not_assessable_from_dd_parsed",
        "metadata_discrepancy",
        "descriptive_only",
        "external_or_background",
    }
