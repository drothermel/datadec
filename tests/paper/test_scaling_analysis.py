from __future__ import annotations

import math
from dataclasses import replace

import pytest

from datadec.paper.scaling import (
    AccuracyAtCompute,
    CheckpointLoss,
    EvaluationPoint,
    FinalLoss,
    FitFailureReason,
    PredictionCell,
    RankingAtSize,
    ScaleCompute,
    ScalingCoordinates,
    ScalingFitError,
    ScalingPolicy,
    ScalingTarget,
    ScalingVariant,
    accuracy_compute_single_3p,
    accuracy_from_loss,
    accuracy_tokens_parameters_single_5p,
    adjacent_size_crossover_counts,
    aggregate_final_losses,
    compare_stepwise_single_scale,
    construct_13_size_subsets,
    fit_scaling_law,
    held_out_prediction,
    loss_compute_2p,
    loss_compute_3p,
    loss_tokens_parameters_5p,
    pair_decision_accuracy,
    prediction_errors,
    predict_scaling_fit,
    select_stage_two_points,
    stepwise_frontier,
    summed_compute,
)


def test_paper_equations_match_golden_values() -> None:
    assert loss_compute_3p(16.0, 8.0, 0.5, 1.0) == 3.0
    assert loss_compute_2p(16.0, 8.0, 0.5) == 2.0
    assert loss_tokens_parameters_5p(16.0, 9.0, 8.0, 0.5, 6.0, 0.5, 1.0) == 5.0
    assert accuracy_from_loss(2.0, 0.8, -2.0, 2.0, 0.1) == 0.5
    assert accuracy_compute_single_3p(
        16.0, 8.0, 0.5, 1.0, 0.8, -2.0, 2.0, 0.1
    ) == pytest.approx(0.19536233761769406)
    assert accuracy_tokens_parameters_single_5p(
        16.0, 9.0, -8.0, 0.5, -6.0, 0.5, 1.0, 0.8, 0.1
    ) == pytest.approx(0.13794069854205343)


def test_final_loss_aggregation_uses_configured_tail_and_final_coordinates() -> None:
    checkpoints = tuple(
        CheckpointLoss(
            "small",
            ScalingCoordinates(10.0, progress * 100.0, progress * 6_000.0),
            progress,
            loss,
        )
        for progress, loss in (
            (0.85, 2.0),
            (0.90, 1.8),
            (0.95, 1.6),
            (1.00, 1.4),
        )
    )

    result = aggregate_final_losses(checkpoints)
    wider = aggregate_final_losses(
        checkpoints, policy=ScalingPolicy(final_window_fraction=0.20)
    )

    assert result[0].size_id == "small"
    assert result[0].coordinates == ScalingCoordinates(10.0, 100.0, 6_000.0)
    assert result[0].loss == pytest.approx(1.6)
    assert result[0].averaged_checkpoint_count == 3
    assert wider[0].loss == pytest.approx(1.7)
    assert wider[0].averaged_checkpoint_count == 4


def test_stage_two_helper_and_strict_late_selection_are_predeclared() -> None:
    points = tuple(
        EvaluationPoint(
            "small",
            ScalingCoordinates(10.0, 100.0 * progress, 6_000.0 * progress),
            progress,
            2.0 - progress,
            progress,
        )
        for progress in (0.25, 0.50, 0.75, 1.0)
    )

    late = select_stage_two_points(points, variant=ScalingVariant.LATE)
    helper_late = select_stage_two_points(
        points,
        variant=ScalingVariant.HELPER_LATE,
        policy=ScalingPolicy(late_progress_threshold=0.75),
    )

    assert tuple(point.progress for point in late.points) == (0.75, 1.0)
    assert late.selected_observed_count == 2
    assert late.helper_count == 0
    assert helper_late.selected_observed_count == 1
    assert helper_late.helper_count == 1
    assert (helper_late.points[-1].loss, helper_late.points[-1].score) == (0.0, 1.0)


def _synthetic_fit_data() -> tuple[tuple[FinalLoss, ...], tuple[EvaluationPoint, ...]]:
    final_losses: list[FinalLoss] = []
    evaluations: list[EvaluationPoint] = []
    for index in range(16):
        parameter_count = 2.0 ** (index % 4 + 1)
        final_tokens = 3.0 ** (index // 4 + 1) * (1 + index * 0.03)
        final_compute = 6 * parameter_count * final_tokens
        final_coordinates = ScalingCoordinates(
            parameter_count, final_tokens, final_compute
        )
        final_loss = loss_compute_3p(final_compute, 0.8, 0.35, 1.2)
        size_id = f"size-{index:02d}"
        final_losses.append(FinalLoss(size_id, final_coordinates, final_loss, 2))
        for progress in (0.25, 0.50, 0.75, 1.0):
            coordinates = ScalingCoordinates(
                parameter_count,
                final_tokens * progress,
                final_compute * progress,
            )
            loss = loss_compute_3p(coordinates.compute, 0.8, 0.35, 1.2)
            score = accuracy_from_loss(loss, 0.75, -4.0, 1.3, 0.1)
            evaluations.append(
                EvaluationPoint(size_id, coordinates, progress, loss, score)
            )
    return tuple(final_losses), tuple(evaluations)


@pytest.mark.parametrize("variant", tuple(ScalingVariant))
def test_all_eight_variants_fit_and_predict_finite(variant: ScalingVariant) -> None:
    final_losses, evaluations = _synthetic_fit_data()

    fit = fit_scaling_law(final_losses, evaluations, variant=variant)
    prediction = predict_scaling_fit(fit, ScalingCoordinates(32.0, 500.0, 96_000.0))

    assert math.isfinite(prediction)
    if variant in {
        ScalingVariant.THREE_PARAMETER_SINGLE_STEP,
        ScalingVariant.FIVE_PARAMETER_ND_SINGLE_STEP,
    }:
        assert fit.combined is not None
        assert fit.stage_one is None
        assert fit.stage_two is None
    else:
        assert fit.combined is None
        assert fit.stage_one is not None
        assert fit.stage_two is not None


def test_multistart_fit_is_exactly_deterministic() -> None:
    final_losses, evaluations = _synthetic_fit_data()

    first = fit_scaling_law(
        final_losses,
        reversed(evaluations),
        variant=ScalingVariant.THREE_PARAMETER_TWO_STAGE,
    )
    second = fit_scaling_law(
        reversed(final_losses),
        evaluations,
        variant=ScalingVariant.THREE_PARAMETER_TWO_STAGE,
    )

    assert second == first


def test_held_out_prediction_excludes_target_and_reports_both_errors() -> None:
    final_losses, evaluations = _synthetic_fit_data()
    target_coordinates = ScalingCoordinates(32.0, 500.0, 96_000.0)
    target_loss = loss_compute_3p(target_coordinates.compute, 0.8, 0.35, 1.2)
    target = ScalingTarget(
        "1B", target_coordinates, accuracy_from_loss(target_loss, 0.75, -4, 1.3, 0.1)
    )

    result = held_out_prediction(
        (*final_losses, FinalLoss("1B", target_coordinates, target_loss, 2)),
        (
            *evaluations,
            EvaluationPoint(
                "1B", target_coordinates, 1.0, target_loss, target.actual_score
            ),
        ),
        target=target,
        variant=ScalingVariant.THREE_PARAMETER_TWO_STAGE,
    )

    assert result.predicted_score == pytest.approx(target.actual_score, abs=1e-8)
    assert result.absolute_error_percent < 1e-6
    assert result.relative_error_percent < 1e-6
    assert result.fit.stage_one is not None
    assert result.fit.stage_one.diagnostics.observation_count == len(final_losses)


def test_prediction_errors_are_computed_per_cell_before_averaging() -> None:
    result = prediction_errors(
        (
            PredictionCell("task-b", predicted=0.20, actual=0.25),
            PredictionCell("task-a", predicted=0.60, actual=0.50),
        )
    )

    assert tuple(cell.cell_id for cell in result.cells) == ("task-a", "task-b")
    assert result.mean_absolute_error_percent == pytest.approx(7.5)
    assert result.mean_relative_error_percent == pytest.approx(20.0)


def test_constructs_all_21_paper_size_subsets() -> None:
    sizes = tuple(f"size-{index:02d}" for index in range(13))

    result = construct_13_size_subsets(sizes)

    assert len(result) == 21
    assert result[0].sizes == sizes[:3]
    assert result[10].sizes == sizes
    assert result[11].sizes == sizes[1:]
    assert result[-1].sizes == sizes[10:]
    assert (
        summed_compute(
            tuple(ScaleCompute(size, index + 1.0) for index, size in enumerate(sizes)),
            sizes=result[0].sizes,
        )
        == 6.0
    )


def test_pair_decision_accuracy_uses_supplied_target_ranks() -> None:
    result = pair_decision_accuracy(
        (("a", 1), ("b", 2), ("c", 3)),
        (("a", 0.9), ("b", 0.1), ("c", 0.2)),
    )

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.correct == 2
    assert result.denominator == 3
    assert result.total_pairs == 3


def test_stepwise_frontier_and_multi_scale_comparison() -> None:
    single = (
        AccuracyAtCompute("single-10", 10, 0.5),
        AccuracyAtCompute("single-20", 20, 0.4),
        AccuracyAtCompute("single-30", 30, 0.7),
    )
    multi = (
        AccuracyAtCompute("multi-15", 15, 0.55),
        AccuracyAtCompute("multi-25", 25, 0.60),
        AccuracyAtCompute("multi-35", 35, 0.65),
    )

    frontier = stepwise_frontier(single)
    comparisons = compare_stepwise_single_scale(multi, single)

    assert tuple(point.point_id for point in frontier) == ("single-10", "single-30")
    assert tuple(item.single_scale.point_id for item in comparisons) == (
        "single-10",
        "single-10",
        "single-30",
    )
    assert tuple(item.accuracy_difference for item in comparisons) == pytest.approx(
        (0.05, 0.10, -0.05)
    )


def test_adjacent_size_crossover_counts_pairwise_sign_flips() -> None:
    result = adjacent_size_crossover_counts(
        (
            RankingAtSize("small", (("a", 3), ("b", 2), ("c", 1))),
            RankingAtSize("medium", (("a", 2), ("b", 3), ("c", 1))),
            RankingAtSize("large", (("a", 2), ("b", 2), ("c", 3))),
        )
    )

    assert result[0].crossover_count == 1
    assert result[0].comparable_pairs == 3
    assert result[0].tied_pairs == 0
    assert result[1].crossover_count == 2
    assert result[1].comparable_pairs == 2
    assert result[1].tied_pairs == 1


def test_fit_failures_are_typed_for_insufficient_and_degenerate_inputs() -> None:
    final_losses, evaluations = _synthetic_fit_data()
    with pytest.raises(ScalingFitError) as insufficient:
        fit_scaling_law(
            final_losses[:2],
            evaluations,
            variant=ScalingVariant.THREE_PARAMETER_TWO_STAGE,
        )
    assert (
        insufficient.value.failure.reason is FitFailureReason.INSUFFICIENT_OBSERVATIONS
    )
    assert insufficient.value.failure.stage == "stage_one"

    same_compute_losses = tuple(
        replace(
            point,
            coordinates=ScalingCoordinates(
                point.coordinates.parameter_count,
                point.coordinates.token_count,
                10.0,
            ),
        )
        for point in final_losses
    )
    with pytest.raises(ScalingFitError) as degenerate:
        fit_scaling_law(
            same_compute_losses,
            evaluations,
            variant=ScalingVariant.THREE_PARAMETER_TWO_STAGE,
        )
    assert degenerate.value.failure.reason is FitFailureReason.DEGENERATE_PREDICTOR


def test_ill_conditioned_fit_can_fail_or_remain_visible_by_policy() -> None:
    final_losses, evaluations = _synthetic_fit_data()
    collinear = tuple(
        replace(
            point,
            coordinates=ScalingCoordinates(
                point.coordinates.parameter_count,
                point.coordinates.parameter_count * 10,
                point.coordinates.compute,
            ),
        )
        for point in final_losses
    )
    strict = ScalingPolicy(
        optimizer=replace(ScalingPolicy().optimizer, max_condition_number=1.0)
    )
    with pytest.raises(ScalingFitError) as error:
        fit_scaling_law(
            collinear,
            evaluations,
            variant=ScalingVariant.FIVE_PARAMETER_ND_TWO_STAGE,
            policy=strict,
        )
    assert error.value.failure.reason is FitFailureReason.ILL_CONDITIONED

    permissive = ScalingPolicy(
        optimizer=replace(
            ScalingPolicy().optimizer,
            max_condition_number=1.0,
            fail_on_ill_conditioned=False,
        )
    )
    result = fit_scaling_law(
        collinear,
        evaluations,
        variant=ScalingVariant.FIVE_PARAMETER_ND_TWO_STAGE,
        policy=permissive,
    )
    assert result.stage_one is not None
    assert result.stage_one.diagnostics.condition_number > 1.0
