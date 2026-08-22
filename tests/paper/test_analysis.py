from __future__ import annotations

import math

import pytest

from datadec.paper.analysis import (
    OLMES_NON_MMLU_TASKS,
    TiePolicy,
    construct_recipe_pairs,
    decision_accuracy,
    macro_average_olmes,
    noise_and_spread,
    percent_target_compute,
    summarize_prediction_attempts,
    theoretical_training_flops,
)


def _olmes_scores() -> list[tuple[str, float]]:
    non_mmlu = [
        (task, index / 10) for index, task in enumerate(OLMES_NON_MMLU_TASKS, start=1)
    ]
    return [*non_mmlu, ("mmlu_alpha", 0.0), ("mmlu_beta", 1.0)]


def test_macro_average_aggregates_mmlu_before_ten_task_average() -> None:
    result = macro_average_olmes(
        _olmes_scores(), expected_mmlu_subjects=("mmlu_alpha", "mmlu_beta")
    )

    assert result.score == pytest.approx(0.5)
    assert result.mmlu_score == 0.5
    assert result.task_denominator == 10
    assert result.mmlu_subject_denominator == 2


def test_macro_average_is_invariant_to_input_permutation() -> None:
    scores = _olmes_scores()

    forward = macro_average_olmes(
        scores, expected_mmlu_subjects=("mmlu_alpha", "mmlu_beta")
    )
    reversed_input = macro_average_olmes(
        reversed(scores), expected_mmlu_subjects=("mmlu_beta", "mmlu_alpha")
    )

    assert reversed_input == forward


@pytest.mark.parametrize(
    "scores, match",
    [
        ([*_olmes_scores(), ("arc_easy", 0.2)], "duplicate OLMES task"),
        (_olmes_scores()[1:], "incomplete OLMES task scores"),
    ],
)
def test_macro_average_rejects_duplicate_or_missing_rows(
    scores: list[tuple[str, float]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        macro_average_olmes(scores, expected_mmlu_subjects=("mmlu_alpha", "mmlu_beta"))


def test_construct_recipe_pairs_is_complete_and_deterministic() -> None:
    result = construct_recipe_pairs((("c", 3.0), ("a", 1.0), ("b", 2.0)))

    assert [
        (pair.recipe_a, pair.recipe_b, pair.score_a, pair.score_b) for pair in result
    ] == [
        ("a", "b", 1.0, 2.0),
        ("a", "c", 1.0, 3.0),
        ("b", "c", 2.0, 3.0),
    ]


def test_construct_recipe_pairs_rejects_duplicate_recipe_scores() -> None:
    with pytest.raises(ValueError, match="duplicate recipe"):
        construct_recipe_pairs((("a", 1.0), ("a", 2.0)))


def test_decision_accuracy_uses_pairwise_sign_agreement() -> None:
    result = decision_accuracy(
        (("a", 3.0), ("b", 2.0), ("c", 1.0)),
        (("c", 2.0), ("a", 3.0), ("b", 1.0)),
        tie_policy=TiePolicy.EXCLUDE,
    )

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.correct == 2
    assert result.denominator == 3
    assert result.total_pairs == 3
    assert result.excluded_pairs == 0
    assert result.target_ties == 0
    assert result.predicted_ties == 0


def test_decision_accuracy_excludes_target_and_predicted_ties() -> None:
    result = decision_accuracy(
        (("a", 1.0), ("b", 1.0), ("c", 2.0)),
        (("a", 1.0), ("b", 2.0), ("c", 2.0)),
        tie_policy=TiePolicy.EXCLUDE,
    )

    assert result.accuracy == 1.0
    assert result.denominator == 1
    assert result.excluded_pairs == 2
    assert result.target_ties == 1
    assert result.predicted_ties == 1


def test_decision_accuracy_can_count_predicted_tie_as_incorrect() -> None:
    result = decision_accuracy(
        (("a", 1.0), ("b", 1.0), ("c", 2.0)),
        (("a", 1.0), ("b", 2.0), ("c", 2.0)),
        tie_policy=TiePolicy.COUNT_AS_INCORRECT,
    )

    assert result.accuracy == 0.5
    assert result.correct == 1
    assert result.denominator == 2
    assert result.excluded_pairs == 1


def test_decision_accuracy_can_fail_on_ties() -> None:
    with pytest.raises(ValueError, match="tie encountered"):
        decision_accuracy(
            (("a", 1.0), ("b", 2.0)),
            (("a", 1.0), ("b", 1.0)),
            tie_policy=TiePolicy.ERROR,
        )


def test_decision_accuracy_rejects_different_recipe_universes() -> None:
    with pytest.raises(ValueError, match="recipe universes differ"):
        decision_accuracy(
            (("a", 1.0), ("b", 2.0)),
            (("a", 1.0), ("c", 2.0)),
            tie_policy=TiePolicy.EXCLUDE,
        )


def test_prediction_attempt_summary_distinguishes_population_and_sample_sd() -> None:
    population = summarize_prediction_attempts((3.0, 1.0, 2.0), ddof=0)
    sample = summarize_prediction_attempts((1.0, 2.0, 3.0), ddof=1)

    assert population.mean == 2.0
    assert population.sd == pytest.approx(math.sqrt(2 / 3))
    assert population.count == 3
    assert population.ddof == 0
    assert population.denominator == 3
    assert sample.mean == 2.0
    assert sample.sd == 1.0
    assert sample.ddof == 1
    assert sample.denominator == 2


def test_prediction_attempt_summary_rejects_invalid_ddof() -> None:
    with pytest.raises(ValueError, match="less than observation count"):
        summarize_prediction_attempts((1.0,), ddof=1)


def test_compute_equations_use_explicit_costs_and_denominator() -> None:
    first_cost = theoretical_training_flops(2, 3)
    zero_cost = theoretical_training_flops(2, 0)
    result = percent_target_compute((12.0, first_cost), target_compute=96.0)

    assert first_cost == 36.0
    assert zero_cost == 0.0
    assert result.percent == 50.0
    assert result.included_compute == 48.0
    assert result.target_compute == 96.0
    assert result.included_cost_count == 2


@pytest.mark.parametrize(
    "parameters, tokens",
    [(0, 1), (1, -1), (-1, 1), (math.inf, 1)],
)
def test_theoretical_training_flops_rejects_invalid_compute_inputs(
    parameters: float, tokens: float
) -> None:
    with pytest.raises(ValueError):
        theoretical_training_flops(parameters, tokens)


@pytest.mark.parametrize(
    "costs, target",
    [([], 1.0), ([1.0], 0.0), ([-1.0], 1.0), ([math.nan], 1.0)],
)
def test_percent_target_compute_rejects_invalid_budget(
    costs: list[float], target: float
) -> None:
    with pytest.raises(ValueError):
        percent_target_compute(costs, target_compute=target)


def _noise_grid() -> list[tuple[str, str, float]]:
    return [
        ("a", "seed-1", 1.0),
        ("a", "seed-2", 3.0),
        ("b", "seed-1", 5.0),
        ("b", "seed-2", 7.0),
    ]


def test_noise_and_spread_population_closed_form_and_permutation() -> None:
    result = noise_and_spread(
        _noise_grid(),
        expected_recipes=("a", "b"),
        expected_seeds=("seed-1", "seed-2"),
        within_recipe_ddof=0,
        spread_ddof=0,
    )
    permuted = noise_and_spread(
        reversed(_noise_grid()),
        expected_recipes=("b", "a"),
        expected_seeds=("seed-2", "seed-1"),
        within_recipe_ddof=0,
        spread_ddof=0,
    )

    assert result.noise == 1.0
    assert result.spread == 2.0
    assert result.recipe_count == 2
    assert result.seed_count == 2
    assert result.within_recipe_denominator == 2
    assert result.spread_denominator == 2
    assert permuted == result


def test_noise_and_spread_sample_closed_form() -> None:
    result = noise_and_spread(
        _noise_grid(),
        expected_recipes=("a", "b"),
        expected_seeds=("seed-1", "seed-2"),
        within_recipe_ddof=1,
        spread_ddof=1,
    )

    assert result.noise == pytest.approx(math.sqrt(2))
    assert result.spread == pytest.approx(2 * math.sqrt(2))
    assert result.within_recipe_ddof == 1
    assert result.spread_ddof == 1
    assert result.within_recipe_denominator == 1
    assert result.spread_denominator == 1


@pytest.mark.parametrize(
    "observations, match",
    [
        (_noise_grid()[:-1], "incomplete recipe/seed grid"),
        ([*_noise_grid(), ("a", "seed-1", 9.0)], "duplicate recipe/seed"),
    ],
)
def test_noise_and_spread_rejects_incomplete_or_duplicate_grid(
    observations: list[tuple[str, str, float]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        noise_and_spread(
            observations,
            expected_recipes=("a", "b"),
            expected_seeds=("seed-1", "seed-2"),
            within_recipe_ddof=0,
            spread_ddof=0,
        )
