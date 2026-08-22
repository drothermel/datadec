from __future__ import annotations

import hashlib
import math
import os
import shutil
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from datadec.paper.contracts import load_claim_registry, load_validation_contract
from datadec.paper.models import (
    AttemptRole,
    CheckpointRule,
    ContentIdentity,
    EvidenceLevel,
    ValidationOutcome,
)
from datadec.paper.verifiers.single_scale import (
    _ComputeEquivalencePoint,
    _compute_equivalence,
    _log_compute_bucket_index,
    run_per_task_attempts,
    run_single_scale_attempts,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]
_FIXTURE = Path(__file__).parent / "fixtures" / "olmes_single_scale_regression.parquet"
_FULL_DATA_ROOT = Path(
    os.environ.get("DATADEC_DD_PARSED_ROOT", _REPOSITORY_ROOT / "data")
)
_MINIMUM_DIFFERENCE = -1 / 300


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
    assert equivalence.evidence_level is EvidenceLevel.LOWER_LEVEL_ROWS
    assert equivalence.outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    assert equivalence.computed_value == {
        "compute_log10_bin_width": 0.1,
        "target_model_compute": 7.06209840434774e20,
        "matched_groups": [],
        "matched_bin_count": 0,
        "matched_group_count": 0,
        "passing_group_count": 0,
        "minimum_accuracy_difference": None,
        "mean_accuracy_difference": None,
        "median_accuracy_difference": None,
        "minimum_allowed_difference": -0.0033333333333333335,
        "zero_compute_checkpoint_count": 0,
        "same_size_pair_count": 1,
        "satisfied": False,
    }
    assert equivalence.missing_groups == (
        "compute_bucket=cross_size_intermediate_to_final",
    )
    assert equivalence.diagnostics == (
        "No cross-size intermediate/final matches exist in fixed log10 compute "
        "buckets of width 0.1.",
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


def test_adapter_rejects_missing_accuracy(
    adapter_inputs: tuple[Path, ContentIdentity],
) -> None:
    data_root, _ = adapter_inputs
    path = data_root / "processed/olmes.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "primary_metric"] = None
    frame.to_parquet(path, index=False)
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    registry, contract = _contracts()

    with pytest.raises(ValueError, match="missing primary-metric accuracy"):
        run_single_scale_attempts(
            repository_root=_REPOSITORY_ROOT,
            data_root=data_root,
            registry=registry,
            contract=contract,
            input_identities={
                "olmes_aggregate": ContentIdentity(
                    id="olmes_aggregate",
                    sha256=digest,
                )
            },
        )


def _point(
    model_size: str,
    step: int,
    compute: float,
    accuracy: float,
) -> _ComputeEquivalencePoint:
    return _ComputeEquivalencePoint(
        model_size=model_size,
        step=step,
        compute=compute,
        decision_accuracy=accuracy,
    )


def _equivalence(
    points: tuple[_ComputeEquivalencePoint, ...],
    *,
    width: float = 0.1,
    minimum_difference: float = _MINIMUM_DIFFERENCE,
) -> dict[str, object]:
    result, _ = _compute_equivalence(
        points,
        target_model_compute=10.0,
        bin_width=width,
        minimum_difference=minimum_difference,
    )
    return result


def test_compute_bucket_boundaries_are_half_open() -> None:
    upper = 10.0**0.1
    lower = 10.0**-0.1

    assert (
        _log_compute_bucket_index(
            math.nextafter(lower, 0.0), target_model_compute=1.0, bin_width=0.1
        )
        == -2
    )
    assert (
        _log_compute_bucket_index(lower, target_model_compute=1.0, bin_width=0.1) == -1
    )
    assert (
        _log_compute_bucket_index(
            math.nextafter(1.0, 0.0), target_model_compute=1.0, bin_width=0.1
        )
        == -1
    )
    assert _log_compute_bucket_index(1.0, target_model_compute=1.0, bin_width=0.1) == 0
    assert (
        _log_compute_bucket_index(
            math.nextafter(upper, 1.0), target_model_compute=1.0, bin_width=0.1
        )
        == 0
    )
    assert (
        _log_compute_bucket_index(upper, target_model_compute=1.0, bin_width=0.1) == 1
    )


def test_compute_equivalence_averages_intermediate_checkpoints() -> None:
    result = _equivalence(
        (
            _point("intermediate", 1, 1.0, 0.60),
            _point("intermediate", 2, 1.02, 0.80),
            _point("intermediate", 3, 10.0, 0.90),
            _point("final", 1, 1.01, 0.68),
        )
    )

    assert result["matched_group_count"] == 1
    groups = cast(list[dict[str, object]], result["matched_groups"])
    group = groups[0]
    assert group["intermediate_steps"] == [1, 2]
    assert group["intermediate_checkpoint_count"] == 2
    assert group["intermediate_accuracy"] == pytest.approx(0.70)
    assert group["final_accuracy"] == 0.68
    assert group["accuracy_difference"] == pytest.approx(0.02)


def test_compute_equivalence_excludes_same_size_pairs() -> None:
    result = _equivalence(
        (
            _point("one-size", 1, 1.0, 0.60),
            _point("one-size", 2, 1.01, 0.61),
        )
    )

    assert result["matched_group_count"] == 0
    assert result["same_size_pair_count"] == 1
    assert not result["satisfied"]


def test_compute_equivalence_threshold_is_inclusive() -> None:
    exact = _equivalence(
        (
            _point("intermediate", 1, 1.0, 0.0),
            _point("intermediate", 2, 10.0, 0.5),
            _point("final", 1, 1.01, 1 / 300),
        )
    )
    below = _equivalence(
        (
            _point("intermediate", 1, 1.0, 0.0),
            _point("intermediate", 2, 10.0, 0.5),
            _point("final", 1, 1.01, math.nextafter(1 / 300, math.inf)),
        )
    )

    assert exact["minimum_accuracy_difference"] == _MINIMUM_DIFFERENCE
    assert exact["satisfied"]
    assert cast(float, below["minimum_accuracy_difference"]) < _MINIMUM_DIFFERENCE
    assert not below["satisfied"]


def test_compute_equivalence_counts_zero_compute() -> None:
    result, exclusions = _compute_equivalence(
        (
            _point("intermediate", 1, 0.0, 0.5),
            _point("intermediate", 2, 10.0, 0.6),
            _point("final", 1, 1.0, 0.6),
        ),
        target_model_compute=10.0,
        bin_width=0.1,
        minimum_difference=_MINIMUM_DIFFERENCE,
    )

    assert result["zero_compute_checkpoint_count"] == 1
    assert {item.name: item.value for item in exclusions}[
        "zero_compute_checkpoints"
    ] == 1


@pytest.mark.parametrize("compute", [-1.0, math.inf, -math.inf, math.nan])
def test_compute_equivalence_rejects_invalid_compute(compute: float) -> None:
    with pytest.raises(ValueError, match="compute"):
        _equivalence((_point("model", 1, compute, 0.5),))


def test_compute_equivalence_rejects_duplicate_checkpoint_points() -> None:
    with pytest.raises(ValueError, match="duplicate.*checkpoint"):
        _equivalence(
            (
                _point("model", 1, 1.0, 0.5),
                _point("model", 1, 1.1, 0.6),
            )
        )


def test_compute_equivalence_without_a_common_bin_is_not_satisfied() -> None:
    result = _equivalence(
        (
            _point("intermediate", 1, 1.0, 0.7),
            _point("intermediate", 2, 10.0, 0.8),
            _point("final", 1, 3.0, 0.7),
        )
    )

    assert result["matched_group_count"] == 0
    assert result["minimum_accuracy_difference"] is None
    assert not result["satisfied"]


def test_compute_widths_rerun_bucket_assignment_independently() -> None:
    points = (
        _point("intermediate", 1, 1.15, 0.70),
        _point("intermediate", 2, 10.0, 0.80),
        _point("final", 1, 1.10, 0.68),
    )

    assert _equivalence(points, width=0.05)["matched_group_count"] == 0
    assert _equivalence(points, width=0.10)["matched_group_count"] == 1
    assert _equivalence(points, width=0.20)["matched_group_count"] == 1


def test_distilled_750m_90m_compute_equivalence_regression() -> None:
    result, _ = _compute_equivalence(
        (
            _point("750M", 1_250, 6.02768343564288e18, 0.6677777777777778),
            _point("750M", 26_250, 1.2658135214850048e20, 0.8155555555555556),
            _point("90M", 29_901, 5.758063377068851e18, 0.7855555555555555),
            _point("1B", 69_369, 7.06209840434774e20, 0.9355555555555556),
        ),
        target_model_compute=7.06209840434774e20,
        bin_width=0.1,
        minimum_difference=_MINIMUM_DIFFERENCE,
    )

    groups = cast(list[dict[str, object]], result["matched_groups"])
    group = next(
        item
        for item in groups
        if item["intermediate_model_size"] == "750M"
        and item["final_model_size"] == "90M"
    )
    assert group["intermediate_steps"] == [1_250]
    assert group["final_step"] == 29_901
    assert group["intermediate_compute_minimum"] == 6.02768343564288e18
    assert group["final_compute"] == 5.758063377068851e18
    assert group["compute_ratio"] == pytest.approx(1.046824781340)
    assert group["accuracy_difference"] == pytest.approx(-0.11777777777777776)


@pytest.mark.skipif(
    not (_FULL_DATA_ROOT / "processed/olmes.parquet").is_file(),
    reason="full local dd_parsed OLMES mirror is unavailable",
)
def test_full_local_olmes_compute_equivalence_smoke() -> None:
    registry, contract = _contracts()
    path = _FULL_DATA_ROOT / "processed/olmes.parquet"
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    attempts, _ = run_single_scale_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=_FULL_DATA_ROOT,
        registry=registry,
        contract=contract,
        input_identities={
            "olmes_aggregate": ContentIdentity(
                id="olmes_aggregate",
                sha256=digest,
            )
        },
    )
    equivalence = {
        item.attempt_id: item
        for item in attempts
        if item.attempt_id == "dd-0165-default"
        or item.parent_attempt_id == "dd-0165-default"
    }

    default = equivalence["dd-0165-default"]
    assert default.evidence_level is EvidenceLevel.LOWER_LEVEL_ROWS
    assert default.outcome is ValidationOutcome.NOT_REPRODUCED
    assert default.computed_value["matched_bin_count"] == 12
    assert default.computed_value["matched_group_count"] == 29
    assert default.computed_value["passing_group_count"] == 15
    assert default.computed_value["minimum_accuracy_difference"] == pytest.approx(
        -0.117777778
    )
    assert default.computed_value["mean_accuracy_difference"] == pytest.approx(
        0.014880268
    )
    assert default.computed_value["median_accuracy_difference"] == pytest.approx(
        -0.002222222
    )

    narrow = equivalence["dd-0165-comparison-compute-log10-bin-width-grid-1"]
    assert narrow.computed_value["compute_log10_bin_width"] == 0.05
    assert narrow.computed_value["matched_bin_count"] == 12
    assert narrow.computed_value["matched_group_count"] == 21
    assert narrow.computed_value["passing_group_count"] == 9
    assert narrow.computed_value["minimum_accuracy_difference"] == pytest.approx(
        -0.117777778
    )
    assert narrow.outcome is ValidationOutcome.NOT_REPRODUCED

    wide = equivalence["dd-0165-comparison-compute-log10-bin-width-grid-3"]
    assert wide.computed_value["compute_log10_bin_width"] == 0.2
    assert wide.computed_value["matched_bin_count"] == 11
    assert wide.computed_value["matched_group_count"] == 38
    assert wide.computed_value["passing_group_count"] == 19
    assert wide.computed_value["minimum_accuracy_difference"] == pytest.approx(
        -0.117777778
    )
    assert wide.outcome is ValidationOutcome.NOT_REPRODUCED
