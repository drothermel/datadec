from __future__ import annotations

import builtins
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from datadec.paper.models import (
    AnalysisBundle,
    AnalysisManifest,
    AttemptResult,
    AttemptRole,
    CheckpointRule,
    CheckpointSelection,
    ClaimKind,
    ContentIdentity,
    EvidenceLevel,
    MetadataDiscrepancy,
    NamedCount,
    PaperTarget,
    PredicateOperator,
    RowPredicate,
    RowSelection,
    ValidationOutcome,
)
from datadec.paper.output_transaction import replace_output_set
from datadec.paper.report import (
    _compact_json_cell,
    render_bundle_outputs,
    render_report,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)


def _manifest() -> AnalysisManifest:
    return AnalysisManifest(
        run_id="validation-run",
        started_at=_NOW,
        completed_at=_NOW + timedelta(minutes=1),
        input_identities=(ContentIdentity(id="olmes", sha256=_SHA_A),),
        targets_identity=ContentIdentity(id="targets.json", sha256=_SHA_A),
        attempts_identity=ContentIdentity(id="attempts.json", sha256=_SHA_B),
        plot_series_identity=ContentIdentity(id="plot-series.json", sha256=_SHA_C),
    )


def _target(claim_id: str, family: str, value: object) -> PaperTarget:
    return PaperTarget(
        claim_id=claim_id,
        family=family,
        kind=ClaimKind.EMPIRICAL_NUMERIC,
        source_file="docs/paper/example_paper.tex",
        line_start=10,
        line_end=11,
        source_text=f"Paper text for {claim_id}",
        value=value,
    )


def _selection(*, rows: int = 12) -> RowSelection:
    return RowSelection(
        logical_table_id="olmes",
        columns=("size", "step", "metric"),
        predicates=(
            RowPredicate(column="size", operator=PredicateOperator.EQ, value="150M"),
        ),
        local_parquet_sha256=_SHA_A,
        remote_dataset_revision="dd-parsed-rev",
        selected_row_count=rows,
        selected_key_sha256=_SHA_B,
    )


def _attempt(
    attempt_id: str,
    claim_id: str,
    *,
    role: AttemptRole = AttemptRole.DEFAULT,
    parent: str | None = None,
    outcome: ValidationOutcome = ValidationOutcome.REPRODUCED,
    computed: object = 0.8033333333333333,
    difference: float | None = 0.0033333333333332993,
    diagnostics: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    missing_groups: tuple[str, ...] = (),
    evidence_level: EvidenceLevel = EvidenceLevel.LOWER_LEVEL_ROWS,
) -> AttemptResult:
    return AttemptResult(
        attempt_id=attempt_id,
        claim_id=claim_id,
        role=role,
        parent_attempt_id=parent,
        evidence_level=evidence_level,
        comparison_rule_id="approximately-80-percent",
        comparison_rule_version=2,
        transformation_ids=("macro-average-mmlu", "pairwise-decisions"),
        row_selections=(_selection(),),
        checkpoint_selections=(
            CheckpointSelection(
                requested_meaning="final",
                rule=CheckpointRule.LATEST_COMMON_COMPLETE,
                actual_step=37_500,
                completeness_dimensions=("recipe", "seed", "task"),
                expected_group_count=4_950,
                selected_group_count=4_950,
            ),
        ),
        target_value=0.8,
        computed_value=computed,
        unrounded_difference=difference,
        seeds=("2", "3", "4"),
        denominator=900,
        exclusions=(NamedCount(name="excluded_target_ties", value=2),),
        missing_groups=missing_groups,
        target_ties=2,
        predicted_ties=1,
        standard_deviation=0.029627314724385286,
        ddof=1,
        outcome=outcome,
        diagnostics=diagnostics,
        limitations=limitations,
    )


def _bundle() -> AnalysisBundle:
    default = _attempt("dd-0001-default", "DD-0001")
    sensitivity = _attempt(
        "dd-0001-preceding-1",
        "DD-0001",
        role=AttemptRole.SENSITIVITY,
        parent=default.attempt_id,
        outcome=ValidationOutcome.DIRECTIONALLY_CONSISTENT,
        computed=0.77,
        difference=-0.03,
    )
    unavailable = _attempt(
        "dd-0002-default",
        "DD-0002",
        outcome=ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
        computed=None,
        difference=None,
        diagnostics=("required task observations are absent",),
        limitations=("training is outside validation scope",),
        missing_groups=("task=missing",),
        evidence_level=EvidenceLevel.AUTHOR_DERIVED_AGGREGATE,
    )
    return AnalysisBundle(
        manifest=_manifest(),
        targets=(
            _target("DD-0003", "z-family", "trend & target"),
            _target("DD-0002", "b-family", 0.9),
            _target("DD-0001", "a_family", 0.8),
        ),
        metadata_discrepancies=(
            MetadataDiscrepancy(
                claim_id="DD-META",
                paper_locator="example_paper.tex:20",
                paper_value="2024",
                metadata_source="dd_parsed metadata",
                metadata_value="2048",
                note="description differs; not an empirical outcome",
            ),
        ),
        attempts=(unavailable, sensitivity, default),
        plot_series=(),
    )


def test_report_is_deterministic_and_covers_finding_contract() -> None:
    bundle = _bundle()

    report = render_report(bundle)
    reordered = bundle.model_copy(
        update={
            "targets": tuple(reversed(bundle.targets)),
            "attempts": tuple(reversed(bundle.attempts)),
        }
    )

    assert render_report(reordered) == report
    assert r"## Family: a\_family" in report
    assert report.index(r"## Family: a\_family") < report.index(
        "## Unassessable from dd_parsed"
    )
    assert (
        "| DD-0001 | dd-0001-default | default | lower_level_rows | 0.8 | "
        "0.8033333333333333 | 0.00333333333333 | reproduced |"
    ) in report
    assert "dd-0001-preceding-1" in report
    assert "Sensitivity attempts remain separate comparison rows" in report
    assert "author_derived_aggregate" in report
    assert "not independent fit or evaluation reruns" in report
    assert "approximately-80-percent v2" in report
    assert "macro-average-mmlu, pairwise-decisions" in report
    assert "rows=12; denominator=900; ties=2/1" in report
    assert "| 37500 |" in report
    assert "DD-0002" in report and "task=missing" in report
    assert "| DD-0002 | b-family | author_derived_aggregate |" in report
    assert "DD-0003" in report and "No attempt result persisted" in report
    assert "## Metadata discrepancies" in report
    assert "description differs; not an empirical outcome" in report
    assert "Metadata comparisons are descriptive and are excluded" in report
    assert "## Traceability appendix" in report
    assert "| Input | olmes |" in report
    assert "| Bundle | attempts.json |" in report
    assert len(report.encode()) < 20_000


def test_report_uses_persisted_results_without_input_reads_or_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    attempt = bundle.attempts[-1].model_copy(
        update={"computed_value": 999, "unrounded_difference": 123.456}
    )
    bundle = bundle.model_copy(update={"attempts": (*bundle.attempts[:-1], attempt)})

    def fail_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("rendering must not open scientific or repository inputs")

    monkeypatch.setattr(builtins, "open", fail_read)

    report = render_report(bundle)

    assert "| lower_level_rows | 0.8 | 999 | 123.456 | reproduced |" in report


def test_compact_json_cell_keeps_denominator_discrepancy_visible() -> None:
    value = {
        "released_relative_error_percent": 230.80877955632104,
        "paper_denominator_relative_error_percent": 64.3872145727109,
        "displayed_released_relative_error_percent": 230.8,
        "displayed_paper_denominator_relative_error_percent": 64.4,
        "absolute_error_percent": 65.36898720839734,
        "released_relative_denominator": "absolute_prediction",
        "paper_stated_relative_denominator": "actual_or_target",
        "relative_error_denominator_discrepancy": True,
    }

    rendered = _compact_json_cell(value)

    assert "released\\_relative\\_error\\_percent" in rendered
    assert "paper\\_denominator\\_relative\\_error\\_percent" in rendered
    assert "230.80877955632104" in rendered
    assert "64.3872145727109" in rendered


def test_render_bundle_outputs_returns_report_and_named_valid_svg_bytes() -> None:
    outputs = render_bundle_outputs(_bundle())

    assert outputs.report == render_report(_bundle()).encode()
    assert tuple(name for name, _ in outputs.figures) == ("outcome-audit.svg",)
    assert all(isinstance(content, bytes) for _, content in outputs.figures)
    root = ET.fromstring(outputs.figures[0][1])
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    audit = outputs.figures[0][1].decode()
    assert "reproduced: 1" in audit
    assert "not_assessable_from_dd_parsed: 1" in audit
    assert "metadata_discrepancy" not in audit


def test_rendered_bytes_are_accepted_by_output_transaction(tmp_path: Path) -> None:
    outputs = render_bundle_outputs(_bundle())
    report_path = tmp_path / "report.md"
    audit_path = tmp_path / outputs.figures[0][0]

    replace_output_set(
        ((report_path, outputs.report), (audit_path, outputs.figures[0][1]))
    )

    assert report_path.read_bytes() == outputs.report
    assert audit_path.read_bytes() == outputs.figures[0][1]
