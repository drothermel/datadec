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
    MeasureValue,
    PaperClaim,
    PlotPoint,
    PlotSeries,
    PredicateOperator,
    RowPredicate,
    SourceRegion,
    ValidationOutcome,
)

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
    }


def test_current_validation_config_declares_all_assessable_defaults() -> None:
    config_path = _REPOSITORY_ROOT / "configs/paper_validation.toml"
    contract = load_paper_validation_contract()

    assert config_file("paper_validation.toml").is_file()
    assert len(contract.attempts) == 68
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
    )
    assert contract.outputs.runs_root == "data/paper-validation/runs"
    assert contract.outputs.report == "docs/paper-validation-report.md"
    assert contract.outputs.figures_root == "docs/paper/validation-figures"
    assert "published-results" not in config_path.read_text()

    with pytest.raises(ValidationError, match="frozen"):
        contract.outputs.report = "other.md"


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


def test_row_predicates_are_typed() -> None:
    scalar = RowPredicate(column="step", operator=PredicateOperator.EQ, value=37500)
    choices = RowPredicate(
        column="seed", operator=PredicateOperator.IN, value=(1, 2, 3)
    )

    assert scalar.value == 37500
    assert choices.value == (1, 2, 3)

    with pytest.raises(ValidationError, match="set predicates require tuple"):
        RowPredicate(column="step", operator=PredicateOperator.EQ, value=(37500,))


def test_run_format_2_manifest_has_only_trace_and_bundle_identities() -> None:
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

    assert manifest.run_format == 2
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
