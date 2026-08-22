from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from datadec.paper.contracts import load_claim_registry, load_validation_contract
from datadec.paper.models import (
    AttemptRole,
    CheckpointRule,
    ContentIdentity,
    ValidationOutcome,
)
from datadec.paper.verifiers.single_scale import (
    run_per_task_attempts,
    run_single_scale_attempts,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]
_FIXTURE = Path(__file__).parent / "fixtures" / "olmes_single_scale_regression.parquet"


@pytest.fixture
def adapter_inputs(tmp_path: Path) -> tuple[Path, ContentIdentity]:
    destination = tmp_path / "processed" / "olmes.parquet"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(_FIXTURE, destination)
    with destination.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return tmp_path, ContentIdentity(id="olmes_aggregate", sha256=digest)


def _contracts():
    return (
        load_claim_registry(_REPOSITORY_ROOT / "docs/paper/claims.toml"),
        load_validation_contract(_REPOSITORY_ROOT / "configs/paper_validation.toml"),
    )


def test_single_scale_adapter_persists_headline_and_aggregate_series(
    adapter_inputs: tuple[Path, ContentIdentity],
) -> None:
    data_root, identity = adapter_inputs
    registry, contract = _contracts()

    attempts, series = run_single_scale_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=data_root,
        registry=registry,
        contract=contract,
        input_identities={identity.id: identity},
    )

    assert {
        item.attempt_id for item in attempts if item.role is AttemptRole.DEFAULT
    } == {
        "dd-0010-default",
        "dd-0011-default",
        "dd-0164-default",
        "dd-0165-default",
        "dd-0169-default",
        "dd-0356-default",
    }
    default = next(item for item in attempts if item.attempt_id == "dd-0011-default")
    assert default.role is AttemptRole.DEFAULT
    assert default.target_value == "decision_accuracy approximately 0.80"
    assert default.computed_value == 0.8033333333333333
    assert default.outcome is ValidationOutcome.APPROXIMATELY_REPRODUCED
    assert default.denominator == 900
    assert default.standard_deviation == 0.029627314724385286
    assert default.ddof == 1
    assert default.seeds == ("default", "small aux 2", "small aux 3")
    assert default.target_ties == 0
    assert default.predicted_ties == 0
    assert tuple(item.actual_step for item in default.checkpoint_selections) == (
        69_369,
        37_500,
    )
    assert any("38157" in diagnostic for diagnostic in default.diagnostics)
    assert all(
        selection.local_parquet_sha256 == identity.sha256
        for selection in default.row_selections
    )
    headline_sensitivities = tuple(
        item
        for item in attempts
        if item.parent_attempt_id == "dd-0011-default"
        and "preceding-common-complete" in item.attempt_id
    )
    assert tuple(item.computed_value for item in headline_sensitivities) == (
        0.7977777777777778,
        0.7999999999999999,
    )
    assert all(item.role is AttemptRole.SENSITIVITY for item in headline_sensitivities)
    assert tuple(
        item.checkpoint_selections[1].rule for item in headline_sensitivities
    ) == (
        CheckpointRule.PRECEDING_COMMON_COMPLETE,
        CheckpointRule.PRECEDING_COMMON_COMPLETE,
    )
    assert tuple(
        item.checkpoint_selections[1].actual_step for item in headline_sensitivities
    ) == (36_250, 35_000)

    assert {item.id for item in series} == {
        "dd-0169-paper-analog",
        "dd-0356-paper-analog",
    }
    aggregate = next(item for item in series if item.id == "dd-0169-paper-analog")
    assert aggregate.id == "dd-0169-paper-analog"
    assert aggregate.attempt_id == "dd-0169-default"
    assert len(aggregate.points) == 6
    assert {point.dimensions[0].value for point in aggregate.points} == {
        "150M",
        "1B",
    }
    aggregate_result = next(
        item for item in attempts if item.attempt_id == "dd-0169-default"
    )
    assert aggregate.points
    assert aggregate_result.outcome is ValidationOutcome.DESCRIPTIVE_ONLY
    assert aggregate_result.plot_series_ids == (aggregate.id,)

    equivalence = next(
        item for item in attempts if item.attempt_id == "dd-0165-default"
    )
    assert equivalence.outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    assert equivalence.computed_value == {
        "matched_pairs": [],
        "matched_pair_count": 0,
        "minimum_accuracy_difference": None,
        "minimum_allowed_difference": -0.0033333333333333335,
        "satisfied": False,
    }
    assert equivalence.missing_groups == (
        "checkpoint_pair=exact_compute_intermediate_to_final",
    )
    assert equivalence.diagnostics == (
        "No exact-compute intermediate/final checkpoint pairs exist in the selected "
        "dd_parsed surface.",
    )

    tolerance_sensitivities = {
        item.attempt_id: item
        for item in attempts
        if item.parent_attempt_id == "dd-0011-default"
        and "absolute-tolerance" in item.attempt_id
    }
    assert {
        attempt_id: result.outcome
        for attempt_id, result in tolerance_sensitivities.items()
    } == {
        "dd-0011-comparison-absolute-tolerance-grid-1": (
            ValidationOutcome.APPROXIMATELY_REPRODUCED
        ),
        "dd-0011-comparison-absolute-tolerance-grid-3": (
            ValidationOutcome.APPROXIMATELY_REPRODUCED
        ),
    }
    assert {
        result.computed_value["absolute_tolerance"]
        for result in tolerance_sensitivities.values()
    } == {0.005, 0.02}

    annotation_results = {
        item.attempt_id: item
        for item in attempts
        if item.parent_attempt_id == "dd-0356-default"
        and "-comparison-" in item.attempt_id
    }
    assert {
        attempt_id: result.outcome for attempt_id, result in annotation_results.items()
    } == {
        "dd-0356-comparison-accuracy-threshold-grid-1": (
            ValidationOutcome.APPROXIMATELY_REPRODUCED
        ),
        "dd-0356-comparison-accuracy-threshold-grid-3": (
            ValidationOutcome.NOT_REPRODUCED
        ),
    }
    assert {
        result.computed_value["threshold"] for result in annotation_results.values()
    } == {0.75, 0.85}


def test_per_task_adapter_persists_all_ten_logical_task_curves(
    adapter_inputs: tuple[Path, ContentIdentity],
) -> None:
    data_root, identity = adapter_inputs
    registry, contract = _contracts()

    attempts, series = run_per_task_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=data_root,
        registry=registry,
        contract=contract,
        input_identities={identity.id: identity},
    )

    assert {
        item.attempt_id for item in attempts if item.role is AttemptRole.DEFAULT
    } == {
        "dd-0051-default",
        "dd-0052-default",
        "dd-0053-default",
        "dd-0142-default",
        "dd-0148-default",
        "dd-0149-default",
        "dd-0150-default",
        "dd-0166-default",
        "dd-0167-default",
        "dd-0168-default",
        *(f"dd-{claim:04d}-default" for claim in range(174, 180)),
    }
    plot_result = next(
        item for item in attempts if item.attempt_id == "dd-0148-default"
    )
    assert plot_result.outcome is ValidationOutcome.DESCRIPTIVE_ONLY
    assert plot_result.plot_series_ids == ("dd-0148-paper-analog",)
    assert {item.id for item in series} == {
        "dd-0148-paper-analog",
        "dd-0149-paper-analog",
        "dd-0150-paper-analog",
        *(f"dd-{claim:04d}-paper-analog" for claim in range(174, 180)),
    }
    per_task = next(item for item in series if item.id == "dd-0148-paper-analog")
    assert per_task.id == "dd-0148-paper-analog"
    assert len(per_task.points) == 60
    assert {point.dimensions[0].value for point in per_task.points} == {
        "arc_challenge",
        "arc_easy",
        "boolq",
        "csqa",
        "hellaswag",
        "mmlu",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
    }
    assert all(point.measures[0].value > 0 for point in per_task.points)
    comparison_sensitivities = tuple(
        item for item in attempts if "-comparison-" in item.attempt_id
    )
    assert comparison_sensitivities
    assert all(
        item.role is AttemptRole.SENSITIVITY
        and item.parent_attempt_id is not None
        and not item.plot_series_ids
        for item in comparison_sensitivities
    )


def test_adapter_rejects_an_unverified_input_identity(
    adapter_inputs: tuple[Path, ContentIdentity],
) -> None:
    data_root, _identity = adapter_inputs
    registry, contract = _contracts()

    with pytest.raises(ValueError, match="identity differs"):
        run_single_scale_attempts(
            repository_root=_REPOSITORY_ROOT,
            data_root=data_root,
            registry=registry,
            contract=contract,
            input_identities={
                "olmes_aggregate": ContentIdentity(
                    id="olmes_aggregate",
                    sha256="0" * 64,
                )
            },
        )
