from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from datadec.config import config_file, load_paper_validation_contract
from datadec.paper import (
    AnalysisId,
    AnalysisManifest,
    AttemptInput,
    AttemptSpec,
    AxisScale,
    AxisSpec,
    CheckpointRule,
    ClaimKind,
    ComparisonPredicate,
    ComparisonRule,
    ContentIdentity,
    EvidenceLevel,
    InputTableSpec,
    MeasureValue,
    PaperClaim,
    PlotPoint,
    PlotSeries,
    PredicateOperator,
    RowPredicate,
    SourceRegion,
    ValidationOutcome,
)
from datadec.paper.models import ComparisonParameter, ComparisonParameterName

_REPOSITORY_ROOT = Path(__file__).parents[2]
_SHA256 = "a" * 64


def _claim(**updates: Any) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "id": "DD-0001",
        "source_file": "docs/paper/example_paper.tex",
        "line_start": 10,
        "line_end": 11,
        "text": "The paper makes a testable claim.",
        "kind": ClaimKind.EMPIRICAL_NUMERIC,
        "family": "headline",
        "paper_target": 0.8,
        "attempt_ids": ("dd-0001-default",),
    }
    claim.update(updates)
    return claim


def _attempt(**updates: Any) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "id": "dd-0001-default",
        "claim_id": "DD-0001",
        "default": True,
        "analysis_id": AnalysisId.SINGLE_SCALE,
        "evidence_level": EvidenceLevel.LOWER_LEVEL_ROWS,
        "inputs": (AttemptInput(table_id="olmes", columns=("recipe", "score")),),
        "transformation_ids": ("rank", "compare"),
        "comparison_rule_id": "approx-v1",
    }
    attempt.update(updates)
    return attempt


def _region(**updates: Any) -> dict[str, Any]:
    region: dict[str, Any] = {
        "id": "region-1",
        "source_file": "docs/paper/example_paper.tex",
        "line_start": 10,
        "line_end": 12,
        "kind": "prose",
        "content_sha256": _SHA256,
        "claim_ids": ("DD-0001",),
    }
    region.update(updates)
    return region


def test_config_imports_in_a_clean_interpreter() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import datadec.config"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_validation_vocabulary_is_exact() -> None:
    assert {kind.value for kind in ClaimKind} == {
        "empirical_numeric",
        "empirical_comparison",
        "empirical_trend",
        "empirical_plot",
        "method_definition",
        "descriptive_metadata",
        "external_background",
        "normative_or_future",
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
    assert {analysis.value for analysis in AnalysisId} == {
        "single_scale",
        "per_task",
        "proxy_metrics",
        "noise_spread",
        "scaling_law",
        "math_code",
    }
    assert {level.value for level in EvidenceLevel} == {
        "lower_level_rows",
        "author_derived_aggregate",
    }
    assert (
        ComparisonParameterName.COMPUTE_LOG10_BIN_WIDTH.value
        == "compute_log10_bin_width"
    )
    assert (
        ComparisonParameterName.FRONTIER_DIFFERENCE_MAXIMUM.value
        == "frontier_difference_maximum"
    )
    assert ComparisonParameterName.PREDICTED_TIE_CREDIT.value == "predicted_tie_credit"


def test_current_validation_config_declares_all_assessable_defaults() -> None:
    contract = load_paper_validation_contract()

    assert config_file("paper_validation.toml").is_file()
    assert len(contract.attempts) == 79
    assert all(attempt.default for attempt in contract.attempts)
    assert all(
        attempt.id == f"{attempt.claim_id.lower()}-default"
        for attempt in contract.attempts
    )
    assert {attempt.analysis_id: 0 for attempt in contract.attempts}.keys() == set(
        AnalysisId
    )
    assert contract.checkpoint_policy.final_rule is (
        CheckpointRule.LATEST_COMMON_COMPLETE
    )
    assert contract.sensitivity_policy.preceding_common_complete_steps == 2
    headline = next(
        attempt for attempt in contract.attempts if attempt.id == "dd-0011-default"
    )
    assert headline.sensitivity_ids == (
        "dd-0011-preceding-common-complete-1",
        "dd-0011-preceding-common-complete-2",
        "dd-0011-paper-step",
        "dd-0011-comparison-absolute-tolerance-grid-1",
        "dd-0011-comparison-absolute-tolerance-grid-3",
    )
    assert contract.outputs.runs_root == "data/paper-validation/runs"
    assert contract.outputs.report == "docs/paper-validation-report.md"
    assert contract.outputs.figures_root == "docs/paper/validation-figures"
    inputs = {item.id: item for item in contract.inputs}
    assert {
        input_id: (item.path, item.remote_path, item.columns)
        for input_id, item in inputs.items()
        if item.evidence_level is EvidenceLevel.AUTHOR_DERIVED_AGGREGATE
    } == {
        "cheap_decisions": (
            "processed/published-results/cheap_decisions_stacked_rc_pred_all.parquet",
            "published-results/cheap_decisions_stacked_rc_pred_all.parquet",
            (
                "task",
                "mix",
                "metric",
                "setup",
                "step_1_y",
                "step_2_y",
                "stacked_y",
                "step_1_pred",
                "step_2_pred",
                "stacked_pred",
                "abs_error_step_1",
                "abs_error_step_2",
                "abs_error_stacked",
                "rel_error_stacked",
            ),
        ),
        "new_eval_decision_accuracy": (
            "processed/published-results/new_eval_intermediates/davidh_new_evals_decision_accuracy.parquet",
            "published-results/new_eval_intermediates/davidh_new_evals_decision_accuracy.parquet",
            (
                "size",
                "task",
                "target_ranking",
                "logits_per_byte_corr",
                "logits_per_char_corr",
                "primary_score",
            ),
        ),
        "new_eval_means": (
            "processed/published-results/new_eval_intermediates/davidh_new_evals_means_df.parquet",
            "published-results/new_eval_intermediates/davidh_new_evals_means_df.parquet",
            (
                "size",
                "task",
                "primary_score",
                "logits_per_byte_corr",
                "logits_per_char_corr",
            ),
        ),
    }
    assert all(
        item.evidence_level is EvidenceLevel.LOWER_LEVEL_ROWS
        for input_id, item in inputs.items()
        if input_id
        not in {
            "cheap_decisions",
            "new_eval_decision_accuracy",
            "new_eval_means",
        }
    )

    with pytest.raises(ValidationError, match="frozen"):
        contract.outputs.report = "other.md"


def test_qualitative_attempts_use_frozen_typed_rules() -> None:
    contract = load_paper_validation_contract()
    attempt_ids = {
        "dd-0010-default",
        "dd-0051-default",
        "dd-0052-default",
        "dd-0053-default",
        "dd-0055-default",
        "dd-0056-default",
        "dd-0057-default",
        "dd-0098-default",
        "dd-0142-default",
        "dd-0149-default",
        "dd-0150-default",
        "dd-0164-default",
        "dd-0165-default",
        "dd-0166-default",
        "dd-0167-default",
        "dd-0168-default",
        *(f"dd-{claim:04d}-default" for claim in range(174, 180)),
        "dd-0194-default",
        *(f"dd-{claim:04d}-default" for claim in range(196, 200)),
        *(f"dd-{claim:04d}-default" for claim in range(202, 208)),
        "dd-0211-default",
        "dd-0212-default",
        "dd-0356-default",
    }
    attempts = {attempt.id: attempt for attempt in contract.attempts}
    rules = {rule.id: rule for rule in contract.comparison_rules}

    assert attempt_ids <= attempts.keys()
    assert all(
        rules[attempts[attempt_id].comparison_rule_id].parameters
        for attempt_id in attempt_ids
    )
    assert all(
        parameter.default in parameter.sensitivity_grid
        for attempt_id in attempt_ids
        for parameter in rules[attempts[attempt_id].comparison_rule_id].parameters
    )
    predictable = rules[attempts["dd-0356-default"].comparison_rule_id]
    assert predictable.parameter(
        ComparisonParameterName.ACCURACY_THRESHOLD
    ).sensitivity_grid == (0.75, 0.8, 0.85)
    clustering = rules[attempts["dd-0202-default"].comparison_rule_id]
    assert clustering.parameter(
        ComparisonParameterName.SILHOUETTE_MINIMUM
    ).sensitivity_grid == (0.15, 0.25, 0.35)
    compute_equivalence = rules[attempts["dd-0165-default"].comparison_rule_id]
    assert compute_equivalence.parameter(
        ComparisonParameterName.COMPUTE_LOG10_BIN_WIDTH
    ).sensitivity_grid == (0.05, 0.1, 0.2)
    noise_improvement = attempts["dd-0057-default"]
    assert set(noise_improvement.sensitivity_ids) >= {
        "dd-0057-preceding-common-complete-1",
        "dd-0057-preceding-common-complete-2",
        "dd-0057-comparison-fraction-threshold-grid-2",
        "dd-0057-comparison-fraction-threshold-grid-3",
    }
    assert len(noise_improvement.sensitivity_ids) == len(
        set(noise_improvement.sensitivity_ids)
    )
    for attempt_id in attempt_ids:
        attempt = attempts[attempt_id]
        rule = rules[attempt.comparison_rule_id]
        expected_ids = {
            f"{attempt.claim_id.lower()}-comparison-"
            f"{parameter.name.value.replace('_', '-')}-grid-{grid_index}"
            for parameter in rule.parameters
            for grid_index, value in enumerate(parameter.sensitivity_grid, start=1)
            if value != parameter.default
        }
        assert expected_ids <= set(attempt.sensitivity_ids)


def test_newly_assessable_attempts_have_exact_evidence_and_inputs() -> None:
    contract = load_paper_validation_contract()
    attempts = {attempt.claim_id: attempt for attempt in contract.attempts}
    scaling_claim_ids = {
        "DD-0013",
        "DD-0054",
        "DD-0119",
        "DD-0180",
        "DD-0181",
        "DD-0189",
        "DD-0192",
        *(f"DD-{claim:04d}" for claim in range(301, 309)),
        "DD-0311",
        "DD-0312",
        "DD-0330",
        "DD-0368",
        "DD-0369",
    }
    math_code_claim_ids = {
        "DD-0017",
        "DD-0018",
        "DD-0213",
        "DD-0221",
        "DD-0222",
        "DD-0224",
        "DD-0225",
        "DD-0226",
        "DD-0227",
        "DD-0413",
        "DD-0414",
    }
    newly_assessable = scaling_claim_ids | {"DD-0165"} | math_code_claim_ids

    assert len(newly_assessable) == 32
    assert newly_assessable <= attempts.keys()
    assert all(
        attempts[claim_id].evidence_level is EvidenceLevel.AUTHOR_DERIVED_AGGREGATE
        for claim_id in scaling_claim_ids | math_code_claim_ids
    )
    assert attempts["DD-0165"].evidence_level is EvidenceLevel.LOWER_LEVEL_ROWS
    assert all(
        {item.table_id for item in attempts[claim_id].inputs} == {"cheap_decisions"}
        for claim_id in scaling_claim_ids
    )


def test_new_scaling_and_math_code_rules_freeze_audited_thresholds() -> None:
    contract = load_paper_validation_contract()
    attempts = {attempt.claim_id: attempt for attempt in contract.attempts}
    rules = {rule.id: rule for rule in contract.comparison_rules}

    def parameter(claim_id: str, name: ComparisonParameterName) -> ComparisonParameter:
        return rules[attempts[claim_id].comparison_rule_id].parameter(name)

    for claim_id in {"DD-0013", "DD-0054", "DD-0180", "DD-0181", "DD-0368"}:
        assert (
            parameter(
                claim_id, ComparisonParameterName.FRONTIER_DIFFERENCE_MAXIMUM
            ).default
            == 0.0
        )
    assert parameter(
        "DD-0119", ComparisonParameterName.MARKED_GAP_MINIMUM
    ).sensitivity_grid == (0.02, 0.05)
    for claim_id in {"DD-0311", "DD-0330"}:
        assert parameter(
            claim_id, ComparisonParameterName.OVERLAP_RANGE_MAXIMUM
        ).sensitivity_grid == (0.005, 0.01)
    assert parameter(
        "DD-0369", ComparisonParameterName.PREDICTED_TIE_CREDIT
    ).sensitivity_grid == (0.0, 0.5)
    for claim_id in {"DD-0017", "DD-0018"}:
        assert parameter(
            claim_id, ComparisonParameterName.ACCURACY_THRESHOLD
        ).sensitivity_grid == (0.75, 0.8, 0.85)
        assert (
            parameter(claim_id, ComparisonParameterName.MAXIMUM_SCALE_PERCENT).default
            == 0.01
        )
    for claim_id in {"DD-0213", "DD-0226"}:
        assert (
            parameter(claim_id, ComparisonParameterName.MARKED_GAP_MINIMUM).default
            == 0.0
        )
    assert (
        parameter("DD-0221", ComparisonParameterName.STRONG_BASELINE_THRESHOLD).default
        == 0.75
    )
    assert (
        parameter("DD-0222", ComparisonParameterName.TRIVIAL_TOLERANCE).default == 0.05
    )
    assert rules[attempts["DD-0224"].comparison_rule_id].threshold_grid == (
        0.025,
        0.05,
        0.1,
    )
    assert (
        parameter("DD-0225", ComparisonParameterName.STRONG_BASELINE_THRESHOLD).default
        == 0.75
    )
    assert parameter(
        "DD-0227", ComparisonParameterName.ACCURACY_THRESHOLD
    ).sensitivity_grid == (0.75, 0.8, 0.85)
    assert (
        parameter(
            "DD-0413", ComparisonParameterName.NONTRIVIAL_ACCURACY_THRESHOLD
        ).default
        == 0.6
    )
    assert (
        parameter("DD-0413", ComparisonParameterName.STRONG_BASELINE_THRESHOLD).default
        == 0.75
    )
    assert parameter(
        "DD-0414", ComparisonParameterName.TRIVIAL_TOLERANCE
    ).sensitivity_grid == (0.05, 0.1)


def test_contract_rejects_attempt_evidence_that_differs_from_its_inputs() -> None:
    contract = load_paper_validation_contract()
    payload = contract.model_dump(mode="python")
    payload["attempts"][0]["evidence_level"] = EvidenceLevel.AUTHOR_DERIVED_AGGREGATE

    with pytest.raises(ValidationError, match="evidence level must be"):
        type(contract).model_validate(payload)

    mixed_payload = contract.model_dump(mode="python")
    mixed_payload["attempts"][0]["inputs"] += (
        {"table_id": "cheap_decisions", "columns": ("task",)},
    )
    with pytest.raises(
        ValidationError, match="evidence level must be author_derived_aggregate"
    ):
        type(contract).model_validate(mixed_payload)


def test_published_result_input_paths_are_explicitly_allowed_and_still_safe() -> None:
    table = InputTableSpec(
        id="published",
        path="processed/published-results/table.parquet",
        remote_path="published-results/table.parquet",
        columns=("value",),
        evidence_level=EvidenceLevel.AUTHOR_DERIVED_AGGREGATE,
    )

    assert table.path == "processed/published-results/table.parquet"
    with pytest.raises(ValidationError, match="repository-relative"):
        InputTableSpec(
            id="unsafe",
            path="../published-results/table.parquet",
            columns=("value",),
            evidence_level=EvidenceLevel.AUTHOR_DERIVED_AGGREGATE,
        )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"line_start": 12, "line_end": 11}, "line_end"),
        ({"attempt_ids": ()}, "require attempts"),
        (
            {
                "attempt_ids": (),
                "supporting_outcome": ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
            },
            "non-assessable reason",
        ),
        ({"unknown": "value"}, "Extra inputs are not permitted"),
    ],
)
def test_empirical_claim_contract_rejects_invalid_claims(
    updates: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        PaperClaim.model_validate(_claim(**updates))


def test_nonassessable_and_supporting_claims_have_no_attempts() -> None:
    nonassessable = PaperClaim.model_validate(
        _claim(
            attempt_ids=(),
            supporting_outcome=ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
            non_assessable_reason="Required observations are absent.",
        )
    )
    supporting = PaperClaim.model_validate(
        _claim(
            kind=ClaimKind.METHOD_DEFINITION,
            paper_target=None,
            attempt_ids=(),
            supporting_outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
        )
    )

    assert not nonassessable.attempt_ids
    assert not supporting.attempt_ids

    with pytest.raises(ValidationError, match="nonempirical"):
        PaperClaim.model_validate(
            _claim(
                kind=ClaimKind.METHOD_DEFINITION,
                supporting_outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
            )
        )


def test_attempt_specs_pin_default_ids_and_ordered_transformations() -> None:
    attempt = AttemptSpec.model_validate(_attempt())

    assert attempt.transformation_ids == ("rank", "compare")

    with pytest.raises(ValidationError, match="default attempt ID"):
        AttemptSpec.model_validate(_attempt(id="claim-default"))
    with pytest.raises(ValidationError, match="parent attempt"):
        AttemptSpec.model_validate(_attempt(default=False, id="dd-0001-sensitivity"))


def test_comparison_rules_freeze_default_and_sensitivity_thresholds() -> None:
    rule = ComparisonRule(
        id="approximately-one-point-v1",
        version=1,
        predicate=ComparisonPredicate.ABSOLUTE_TOLERANCE,
        absolute_tolerance=0.01,
        threshold_grid=(0.005, 0.01, 0.02),
    )

    assert rule.threshold_grid == (0.005, 0.01, 0.02)

    with pytest.raises(ValidationError, match="must include the default"):
        ComparisonRule(
            id="bad",
            version=1,
            predicate=ComparisonPredicate.ABSOLUTE_TOLERANCE,
            absolute_tolerance=0.01,
            threshold_grid=(0.005, 0.02),
        )


def test_typed_comparison_parameters_freeze_defaults_and_sensitivities() -> None:
    rule = ComparisonRule(
        id="predictable-v1",
        version=1,
        predicate=ComparisonPredicate.BOOLEAN_TRUE,
        parameters=(
            ComparisonParameter(
                name=ComparisonParameterName.ACCURACY_THRESHOLD,
                default=0.8,
                sensitivity_grid=(0.75, 0.8, 0.85),
            ),
        ),
    )

    assert rule.parameter(ComparisonParameterName.ACCURACY_THRESHOLD).default == 0.8

    with pytest.raises(ValidationError, match="must include the default"):
        ComparisonParameter(
            name=ComparisonParameterName.ACCURACY_THRESHOLD,
            default=0.8,
            sensitivity_grid=(0.75, 0.85),
        )
    with pytest.raises(ValidationError, match="parameter names"):
        ComparisonRule(
            id="duplicate",
            version=1,
            predicate=ComparisonPredicate.BOOLEAN_TRUE,
            parameters=rule.parameters * 2,
        )


def test_row_predicates_are_typed() -> None:
    scalar = RowPredicate(column="step", operator=PredicateOperator.EQ, value=37500)
    choices = RowPredicate(
        column="seed", operator=PredicateOperator.IN, value=(1, 2, 3)
    )

    assert scalar.value == 37500
    assert choices.value == (1, 2, 3)

    with pytest.raises(ValidationError, match="set predicates require tuple"):
        RowPredicate(column="step", operator=PredicateOperator.EQ, value=(37500,))


def test_run_format_3_manifest_has_only_trace_and_bundle_identities() -> None:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    manifest = AnalysisManifest(
        run_id="run-1",
        started_at=now,
        completed_at=now,
        input_identities=(ContentIdentity(id="olmes", sha256=_SHA256),),
        targets_identity=ContentIdentity(id="targets.json", sha256=_SHA256),
        attempts_identity=ContentIdentity(id="attempts.json", sha256=_SHA256),
        plot_series_identity=ContentIdentity(id="plot-series.json", sha256=_SHA256),
    )

    assert manifest.run_format == 3
    assert "qualification" not in manifest.model_dump()
    assert "tree_state" not in manifest.model_dump()


def test_paper_analog_plot_series_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        PlotSeries(
            id="series-1",
            figure="figure-1",
            panel="a",
            semantic_kind="paper_analog",
            x_axis=AxisSpec(measure="compute", scale=AxisScale.LOG, unit="FLOPs"),
            y_axis=AxisSpec(measure="accuracy", scale=AxisScale.LINEAR, unit="ratio"),
            measures=("compute", "accuracy"),
            attempt_id="dd-0001-default",
            points=(),
        )

    with pytest.raises(ValidationError, match="finite"):
        PlotPoint(
            measures=(
                MeasureValue(name="compute", value=float("inf")),
                MeasureValue(name="accuracy", value=0.8),
            )
        )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"claim_ids": (), "non_claim_reason": None}, "exactly one"),
        (
            {"claim_ids": ("DD-0001",), "non_claim_reason": "heading"},
            "exactly one",
        ),
        ({"line_start": 13, "line_end": 12}, "line_end"),
    ],
)
def test_source_region_requires_claims_xor_non_claim_reason(
    updates: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        SourceRegion.model_validate(_region(**updates))
