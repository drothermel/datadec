from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from datadec.paper.proxy_metrics import (
    CheckpointTaskScore,
    ChoiceEvidence,
    CrossoverTiePolicy,
    InstanceEvidence,
    InstanceExclusionReason,
    LikelihoodNormalization,
    LogicalTaskScore,
    PredictedTiePolicy,
    ScaleRecipeScore,
    ThresholdComparison,
    calculate_all_proxy_metrics,
    calculate_proxy_metrics,
    compare_logical_task_scores,
    evaluate_threshold_sensitivity,
    latest_common_complete_noise_spread,
    metric_at_least_as_good,
    noise_spread_change,
    summarize_crossovers,
)


def _choice(
    instance_id: int,
    choice_index: int,
    probability: float,
    *,
    tokens: int = 1,
    characters: int = 1,
) -> ChoiceEvidence:
    return ChoiceEvidence(
        instance_id=instance_id,
        choice_index=choice_index,
        log_probability=math.log(probability),
        token_count=tokens,
        character_count=characters,
    )


def test_proxy_formulas_are_item_first_then_macro_averaged() -> None:
    instances = (InstanceEvidence(0, 0), InstanceEvidence(1, 0))
    choices = (
        _choice(0, 0, 0.6),
        _choice(0, 1, 0.3),
        _choice(0, 2, 0.1),
        _choice(1, 0, 0.2),
        _choice(1, 1, 0.8),
    )
    result = calculate_proxy_metrics(
        instances,
        choices,
        normalization=LikelihoodNormalization.RAW,
    )
    permuted = calculate_proxy_metrics(
        reversed(instances),
        reversed(choices),
        normalization=LikelihoodNormalization.RAW,
    )

    assert result.correct_probability == pytest.approx(0.4)
    assert result.margin == pytest.approx(-0.15)
    assert result.normalized_correct_probability == pytest.approx(0.4)
    assert result.total_probability == pytest.approx(1.0)
    assert result.accuracy == 0.5
    assert result.total_instances == 2
    assert result.denominator == 2
    assert result.exclusion_count == 0
    assert permuted == result


def test_calculation_evidence_is_frozen_and_slotted() -> None:
    evidence = InstanceEvidence(0, 0)

    assert not hasattr(evidence, "__dict__")
    with pytest.raises(FrozenInstanceError):
        evidence.correct_choice_index = 1


def test_token_and_character_variants_normalize_log_likelihood_before_exp() -> None:
    results = calculate_all_proxy_metrics(
        (InstanceEvidence(0, 0),),
        (
            _choice(0, 0, 0.25, tokens=2, characters=4),
            _choice(0, 1, 0.125, tokens=1, characters=1),
        ),
    )
    by_normalization = {result.normalization: result for result in results}

    assert tuple(by_normalization) == tuple(LikelihoodNormalization)
    assert by_normalization[
        LikelihoodNormalization.RAW
    ].correct_probability == pytest.approx(0.25)
    assert by_normalization[
        LikelihoodNormalization.PER_TOKEN
    ].correct_probability == pytest.approx(0.5)
    assert by_normalization[
        LikelihoodNormalization.PER_CHARACTER
    ].correct_probability == pytest.approx(math.sqrt(0.5))
    assert by_normalization[LikelihoodNormalization.RAW].total_probability == (
        pytest.approx(0.375)
    )
    assert by_normalization[LikelihoodNormalization.PER_TOKEN].accuracy == 1.0


def test_missing_choice_groups_are_excluded_with_exact_reasons() -> None:
    result = calculate_proxy_metrics(
        tuple(
            InstanceEvidence(index, index if index == 2 else 0) for index in range(4)
        ),
        (
            _choice(1, 0, 0.5),
            _choice(2, 0, 0.5),
            _choice(2, 1, 0.5),
            _choice(3, 0, 0.5),
            _choice(3, 1, 0.5),
        ),
        normalization=LikelihoodNormalization.RAW,
    )

    assert result.total_instances == 4
    assert result.denominator == 1
    assert result.exclusion_count == 3
    assert tuple(exclusion.reason for exclusion in result.excluded_instances) == (
        InstanceExclusionReason.MISSING_CHOICES,
        InstanceExclusionReason.NO_INCORRECT_CHOICE,
        InstanceExclusionReason.MISSING_CORRECT_CHOICE,
    )


@pytest.mark.parametrize(
    "instances, choices, match",
    [
        (
            (InstanceEvidence(0, 0), InstanceEvidence(0, 0)),
            (),
            "duplicate instance evidence",
        ),
        (
            (InstanceEvidence(0, 0),),
            (_choice(0, 0, 0.5), _choice(0, 0, 0.4)),
            "duplicate choice evidence",
        ),
        (
            (InstanceEvidence(0, 0),),
            (_choice(1, 0, 0.5),),
            "unknown instance",
        ),
        (
            (InstanceEvidence(0, 0),),
            (_choice(0, 0, 0.5), _choice(0, 2, 0.4)),
            "contiguous from zero",
        ),
    ],
)
def test_proxy_calculation_rejects_malformed_evidence(
    instances: tuple[InstanceEvidence, ...],
    choices: tuple[ChoiceEvidence, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        calculate_proxy_metrics(
            instances, choices, normalization=LikelihoodNormalization.RAW
        )


def test_choice_ties_count_as_incorrect_and_remain_visible() -> None:
    result = calculate_proxy_metrics(
        (InstanceEvidence(0, 0),),
        (_choice(0, 0, 0.5), _choice(0, 1, 0.5)),
        normalization=LikelihoodNormalization.RAW,
    )

    assert result.accuracy == 0.0
    assert result.margin == pytest.approx(0.0)
    assert result.top_ties == 1
    assert result.instances[0].top_tie is True


def test_all_excluded_instances_return_visible_empty_aggregate() -> None:
    result = calculate_proxy_metrics(
        (InstanceEvidence(0, 0),),
        (),
        normalization=LikelihoodNormalization.RAW,
    )

    assert result.denominator == 0
    assert result.correct_probability is None
    assert result.accuracy is None
    assert (
        result.excluded_instances[0].reason is InstanceExclusionReason.MISSING_CHOICES
    )


def _task_scores(task: str, values: dict[str, float]) -> tuple[LogicalTaskScore, ...]:
    return tuple(
        LogicalTaskScore(task=task, recipe=recipe, score=score)
        for recipe, score in values.items()
    )


def test_logical_task_comparison_accounts_for_all_ties_and_macro_averages() -> None:
    primary = (
        *_task_scores("mmlu", {"a": 3.0, "b": 2.0, "c": 1.0}),
        *_task_scores("arc_easy", {"a": 1.0, "b": 1.0, "c": 2.0}),
    )
    proxy = (
        *_task_scores("mmlu", {"a": 3.0, "b": 1.0, "c": 1.0}),
        *_task_scores("arc_easy", {"a": 1.0, "b": 2.0, "c": 2.0}),
    )

    result = compare_logical_task_scores(
        primary,
        proxy,
        logical_tasks=("mmlu", "arc_easy"),
        predicted_tie_policy=PredictedTiePolicy.COUNT_AS_INCORRECT,
    )

    by_task = {summary.task: summary.result for summary in result.tasks}
    assert by_task["mmlu"].accuracy == pytest.approx(2 / 3)
    assert by_task["mmlu"].predicted_ties == 1
    assert by_task["arc_easy"].accuracy == 0.5
    assert by_task["arc_easy"].denominator == 2
    assert by_task["arc_easy"].excluded_pairs == 1
    assert by_task["arc_easy"].target_ties == 1
    assert by_task["arc_easy"].predicted_ties == 1
    assert result.macro_accuracy == pytest.approx(7 / 12)
    assert result.task_denominator == 2


def test_predicted_tie_exclusion_is_an_explicit_sensitivity() -> None:
    primary = _task_scores("mmlu", {"a": 3.0, "b": 2.0, "c": 1.0})
    proxy = _task_scores("mmlu", {"a": 3.0, "b": 1.0, "c": 1.0})

    result = compare_logical_task_scores(
        primary,
        proxy,
        logical_tasks=("mmlu",),
        predicted_tie_policy=PredictedTiePolicy.EXCLUDE,
    )

    assert result.macro_accuracy == 1.0
    assert result.tasks[0].result.denominator == 2
    assert result.tasks[0].result.excluded_pairs == 1


def test_mmlu_subjects_cannot_be_weighted_as_independent_logical_tasks() -> None:
    scores = _task_scores("mmlu_history", {"a": 1.0, "b": 2.0})

    with pytest.raises(ValueError, match="macro-averaged"):
        compare_logical_task_scores(
            scores,
            scores,
            logical_tasks=("mmlu_history",),
            predicted_tie_policy=PredictedTiePolicy.COUNT_AS_INCORRECT,
        )


def _checkpoint_grid(
    step: int, values: dict[tuple[str, str], float]
) -> list[CheckpointTaskScore]:
    return [
        CheckpointTaskScore(
            model_size="150M",
            step=step,
            task="mmlu",
            metric="correct_prob_per_char",
            recipe=recipe,
            seed=seed,
            score=score,
        )
        for (recipe, seed), score in values.items()
    ]


def test_noise_spread_uses_latest_common_complete_step_and_sample_sd() -> None:
    grid = {("a", "s1"): 1.0, ("a", "s2"): 3.0, ("b", "s1"): 5.0, ("b", "s2"): 7.0}
    scores = [
        *_checkpoint_grid(10, grid),
        *_checkpoint_grid(20, dict(tuple(grid.items())[:-1])),
        *_checkpoint_grid(30, grid),
    ]

    result = latest_common_complete_noise_spread(
        reversed(scores),
        model_size="150M",
        expected_recipes=("a", "b"),
        expected_seeds=("s1", "s2"),
        expected_tasks=("mmlu",),
        expected_metrics=("correct_prob_per_char",),
    )

    assert result.step == 30
    assert result.complete_steps == (10, 30)
    assert result.preceding_complete_steps == (10,)
    assert result.expected_group_count == 4
    assert result.selected_group_count == 4
    noise_spread = result.results[0]
    assert noise_spread.noise == pytest.approx(math.sqrt(2))
    assert noise_spread.spread == pytest.approx(2 * math.sqrt(2))
    assert noise_spread.within_recipe_ddof == 1
    assert noise_spread.spread_ddof == 1
    assert noise_spread.within_recipe_denominator == 1
    assert noise_spread.spread_denominator == 1


def test_noise_spread_rejects_absent_common_complete_step() -> None:
    scores = _checkpoint_grid(
        10, {("a", "s1"): 1.0, ("a", "s2"): 2.0, ("b", "s1"): 3.0}
    )

    with pytest.raises(ValueError, match="no common complete checkpoint"):
        latest_common_complete_noise_spread(
            scores,
            model_size="150M",
            expected_recipes=("a", "b"),
            expected_seeds=("s1", "s2"),
            expected_tasks=("mmlu",),
            expected_metrics=("correct_prob_per_char",),
        )


def _crossover_scores() -> tuple[ScaleRecipeScore, ...]:
    values = {
        1.0: {"a": 3.0, "b": 2.0, "c": 1.0},
        2.0: {"a": 2.0, "b": 2.0, "c": 1.5},
        3.0: {"a": 1.0, "b": 3.0, "c": 2.0},
    }
    return tuple(
        ScaleRecipeScore(scale=scale, recipe=recipe, score=score)
        for scale, recipe_scores in values.items()
        for recipe, score in recipe_scores.items()
    )


def test_crossover_summary_exposes_tie_policy_and_endpoint_reversals() -> None:
    adjacent = summarize_crossovers(
        _crossover_scores(),
        expected_recipes=("a", "b", "c"),
        tie_policy=CrossoverTiePolicy.ADJACENT_NON_TIED,
    )
    bridged = summarize_crossovers(
        reversed(_crossover_scores()),
        expected_recipes=("c", "b", "a"),
        tie_policy=CrossoverTiePolicy.BRIDGE_TIED_POINTS,
    )

    assert adjacent.crossover_count == 1
    assert adjacent.pairs_with_crossover == 1
    assert adjacent.endpoint_reversals == 2
    assert adjacent.excluded_tied_transitions == 2
    assert bridged.crossover_count == 2
    assert bridged.pairs_with_crossover == 2
    assert bridged.endpoint_reversals == 2
    assert any(
        event.recipe_a == "a"
        and event.recipe_b == "b"
        and event.scale_before == 1.0
        and event.scale_after == 3.0
        for event in bridged.events
    )


def test_fixed_threshold_sensitivity_uses_only_supplied_grid_and_order() -> None:
    result = evaluate_threshold_sensitivity(
        0.8,
        thresholds=(0.75, 0.8, 0.85),
        comparison=ThresholdComparison.AT_LEAST,
    )

    assert tuple(evaluation.threshold for evaluation in result.evaluations) == (
        0.75,
        0.8,
        0.85,
    )
    assert tuple(evaluation.satisfied for evaluation in result.evaluations) == (
        True,
        True,
        False,
    )


def test_proxy_and_noise_predicates_require_explicit_tolerances() -> None:
    assert metric_at_least_as_good(0.79, 0.8, tolerance=0.01)
    assert not metric_at_least_as_good(0.79, 0.8, tolerance=0.0)

    change = noise_spread_change(
        primary_noise=0.3,
        primary_spread=0.5,
        proxy_noise=0.2,
        proxy_spread=0.51,
        minimum_change=0.05,
    )

    assert change.noise_reduction == pytest.approx(0.1)
    assert change.spread_increase == pytest.approx(0.01)
    assert change.noise_improved is True
    assert change.spread_improved is False
    assert change.either_improved is True
