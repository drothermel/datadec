from __future__ import annotations

import math

import pandas as pd
import pytest

from datadec.paper.analysis import MMLU_SUBJECTS, OLMES_NON_MMLU_TASKS, TiePolicy
from datadec.paper.models import EvidenceBoundary
from datadec.paper.verifiers.olmes import (
    FactStatus,
    FinalCheckpoint,
    MissingDataBehavior,
    NormalizedOlmesPolicy,
    OlmesAggregateScore,
    OlmesTaskGrouping,
    RecipeMean,
    TargetRanking,
    aggregate_olmes_scores,
    compute_single_scale_decisions,
    compute_task_metric_noise_spread,
    select_canonical_final_checkpoints,
    validate_normalized_olmes_frame,
    verify_normalized_olmes,
)

TASK_GROUPING = OlmesTaskGrouping(
    non_mmlu_tasks=OLMES_NON_MMLU_TASKS,
    mmlu_subjects=MMLU_SUBJECTS,
    mmlu_task_name="mmlu",
)


def _policy(
    *,
    recipes: tuple[str, ...] = ("a", "b"),
    target_seeds: tuple[str, ...] = ("target-1", "target-2", "target-3"),
    prediction_seeds: tuple[str, ...] = ("prediction-1", "prediction-2"),
    tie_policy: TiePolicy = TiePolicy.EXCLUDE,
    attempt_ddof: int = 0,
    within_recipe_ddof: int = 0,
    spread_ddof: int = 0,
    missing_behavior: MissingDataBehavior = MissingDataBehavior.ERROR,
    target_compute: float = 6_000.0,
) -> NormalizedOlmesPolicy:
    return NormalizedOlmesPolicy(
        recipes=recipes,
        target_size="1B",
        target_seeds=target_seeds,
        prediction_seeds=prediction_seeds,
        target_metric_column="primary_metric",
        proxy_metric_columns=("proxy_metric",),
        task_grouping=TASK_GROUPING,
        final_checkpoints=(
            FinalCheckpoint(model_size="1B", step=100),
            FinalCheckpoint(model_size="150M", step=50),
        ),
        noise_size="150M",
        tie_policy=tie_policy,
        attempt_ddof=attempt_ddof,
        within_recipe_ddof=within_recipe_ddof,
        spread_ddof=spread_ddof,
        missing_data_behavior=missing_behavior,
        parameter_count_column="exact_parameter_count",
        token_count_column="tokens",
        target_compute_denominator=target_compute,
    )


def _task_rows(
    *,
    model_size: str = "150M",
    recipe: str = "a",
    seed: str = "prediction-1",
    step: int = 50,
    non_mmlu_value: float = 0.0,
    mmlu_value: float = 1.0,
    proxy_offset: float = 0.0,
    parameter_count: int = 10,
    tokens: int = 20,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in (*OLMES_NON_MMLU_TASKS, *MMLU_SUBJECTS):
        value = mmlu_value if task in MMLU_SUBJECTS else non_mmlu_value
        rows.append(
            {
                "params": model_size,
                "data": recipe,
                "seed": seed,
                "step": step,
                "task": task,
                "exact_parameter_count": parameter_count,
                "tokens": tokens,
                "primary_metric": value,
                "proxy_metric": value + proxy_offset,
            }
        )
    return rows


def _aggregate_score(
    recipe: str,
    seed: str,
    score: float,
    *,
    metric: str = "proxy_metric",
    model_size: str = "150M",
    step: int = 25,
    parameters: float = 10.0,
    tokens: float = 20.0,
) -> OlmesAggregateScore:
    return OlmesAggregateScore(
        model_size=model_size,
        recipe=recipe,
        seed=seed,
        step=step,
        metric=metric,
        score=score,
        mmlu_score=score,
        parameter_count=parameters,
        token_count=tokens,
    )


def test_aggregate_is_permutation_invariant_and_two_stage_weights_mmlu() -> None:
    frame = pd.DataFrame(_task_rows())

    scores, missing = aggregate_olmes_scores(frame, _policy())
    permuted, permuted_missing = aggregate_olmes_scores(
        frame.sample(frac=1, random_state=7).reset_index(drop=True), _policy()
    )

    assert missing == ()
    assert permuted_missing == ()
    assert permuted == scores
    assert len(scores) == 2
    assert scores[0].mmlu_score == 1.0
    assert scores[0].score == pytest.approx(0.1)


def test_normalized_key_validation_rejects_duplicates() -> None:
    rows = _task_rows()
    frame = pd.DataFrame([*rows, rows[0]])

    with pytest.raises(ValueError, match="duplicate normalized OLMES keys"):
        validate_normalized_olmes_frame(frame, _policy())


def test_missing_task_is_error_or_structured_result_by_policy() -> None:
    frame = pd.DataFrame(_task_rows()[:-1])

    with pytest.raises(ValueError, match="task_aggregation"):
        aggregate_olmes_scores(frame, _policy())

    scores, missing = aggregate_olmes_scores(
        frame, _policy(missing_behavior=MissingDataBehavior.RECORD)
    )

    assert scores == ()
    assert len(missing) == 2
    assert {item.metric for item in missing} == {"primary_metric", "proxy_metric"}
    assert all(item.missing_tasks == (MMLU_SUBJECTS[-1],) for item in missing)


def test_final_checkpoint_selection_uses_explicit_map_and_exposes_missing() -> None:
    policy = _policy(missing_behavior=MissingDataBehavior.RECORD)
    scores, _ = aggregate_olmes_scores(pd.DataFrame(_task_rows()), policy)

    result = select_canonical_final_checkpoints(scores, policy)

    assert len(result.scores) == 2
    assert len(result.missing) == 18
    assert {item.stage for item in result.missing} == {"canonical_final"}
    assert any(
        item.model_size == "1B"
        and item.recipe == "a"
        and item.seed == "target-1"
        and item.step == 100
        for item in result.missing
    )


def test_final_selection_does_not_substitute_a_later_checkpoint() -> None:
    policy = _policy(missing_behavior=MissingDataBehavior.RECORD)
    scores, _ = aggregate_olmes_scores(pd.DataFrame(_task_rows(step=51)), policy)

    result = select_canonical_final_checkpoints(scores, policy)

    assert result.scores == ()
    assert all(item.step in {50, 100} for item in result.missing)


def test_25_recipe_single_scale_attempts_have_complete_300_pair_denominator() -> None:
    recipes = tuple(f"recipe-{index:02d}" for index in range(25))
    seeds = ("prediction-1", "prediction-2", "prediction-3")
    policy = _policy(
        recipes=recipes,
        prediction_seeds=seeds,
        attempt_ddof=1,
        target_compute=1_200.0,
    )
    target = TargetRanking(
        metric="primary_metric",
        model_size="1B",
        step=100,
        seed_count=3,
        scores=tuple(
            RecipeMean(recipe=recipe, score=float(index))
            for index, recipe in enumerate(recipes)
        ),
    )
    scores = tuple(
        _aggregate_score(recipe, seed, float(index))
        for seed in seeds
        for index, recipe in enumerate(recipes)
    )

    attempts, summaries, missing = compute_single_scale_decisions(
        scores, target, policy
    )

    assert missing == ()
    assert len(attempts) == 3
    assert {attempt.accuracy.total_pairs for attempt in attempts} == {300}
    assert {attempt.accuracy.denominator for attempt in attempts} == {300}
    assert all(len(attempt.pairs) == 300 for attempt in attempts)
    assert len(summaries) == 1
    assert summaries[0].seed_count == 3
    assert summaries[0].ddof == 1
    assert summaries[0].sd_denominator == 2
    assert summaries[0].percent_target_compute == 100.0


def test_target_and_prediction_ties_are_explicit_in_pair_results() -> None:
    recipes = ("a", "b", "c")
    policy = _policy(recipes=recipes, prediction_seeds=("prediction-1",))
    target = TargetRanking(
        metric="primary_metric",
        model_size="1B",
        step=100,
        seed_count=3,
        scores=(
            RecipeMean("a", 1.0),
            RecipeMean("b", 1.0),
            RecipeMean("c", 2.0),
        ),
    )
    scores = (
        _aggregate_score("a", "prediction-1", 1.0),
        _aggregate_score("b", "prediction-1", 2.0),
        _aggregate_score("c", "prediction-1", 2.0),
    )

    attempts, summaries, missing = compute_single_scale_decisions(
        scores, target, policy
    )

    assert missing == ()
    assert len(summaries) == 1
    assert attempts[0].accuracy.total_pairs == 3
    assert attempts[0].accuracy.denominator == 1
    assert attempts[0].accuracy.excluded_pairs == 2
    assert attempts[0].accuracy.target_ties == 1
    assert attempts[0].accuracy.predicted_ties == 1
    assert sum(pair.excluded for pair in attempts[0].pairs) == 2


def test_prediction_seed_alignment_never_complete_cases() -> None:
    policy = _policy(missing_behavior=MissingDataBehavior.RECORD)
    target = TargetRanking(
        metric="primary_metric",
        model_size="1B",
        step=100,
        seed_count=3,
        scores=(RecipeMean("a", 1.0), RecipeMean("b", 2.0)),
    )
    scores = (
        _aggregate_score("a", "prediction-1", 1.0),
        _aggregate_score("b", "prediction-1", 2.0),
        _aggregate_score("a", "prediction-2", 1.0),
    )

    attempts, summaries, missing = compute_single_scale_decisions(
        scores, target, policy
    )

    assert len(attempts) == 1
    assert summaries == ()
    assert len(missing) == 1
    assert missing[0].seed == "prediction-2"
    assert missing[0].missing_recipes == ("b",)


def test_noise_and_spread_use_explicit_ddof_and_two_stage_mmlu() -> None:
    rows: list[dict[str, object]] = []
    values = {
        ("a", "prediction-1"): 1.0,
        ("a", "prediction-2"): 3.0,
        ("b", "prediction-1"): 5.0,
        ("b", "prediction-2"): 7.0,
    }
    for (recipe, seed), value in values.items():
        rows.extend(
            _task_rows(
                recipe=recipe,
                seed=seed,
                non_mmlu_value=value,
                mmlu_value=value,
            )
        )
    policy = _policy(within_recipe_ddof=1, spread_ddof=1)

    results, missing = compute_task_metric_noise_spread(pd.DataFrame(rows), policy)

    assert missing == ()
    assert len(results) == 20
    primary_arc = next(
        result
        for result in results
        if result.task == "arc_easy" and result.metric == "primary_metric"
    )
    primary_mmlu = next(
        result
        for result in results
        if result.task == "mmlu" and result.metric == "primary_metric"
    )
    assert primary_arc.result.noise == pytest.approx(math.sqrt(2))
    assert primary_arc.result.spread == pytest.approx(2 * math.sqrt(2))
    assert primary_arc.result.within_recipe_denominator == 1
    assert primary_arc.result.spread_denominator == 1
    assert primary_mmlu.result == primary_arc.result


def test_verification_emits_long_form_facts_with_evidence_boundary() -> None:
    rows: list[dict[str, object]] = []
    for recipe_index, recipe in enumerate(("a", "b")):
        for seed in ("target-1", "target-2", "target-3"):
            rows.extend(
                _task_rows(
                    model_size="1B",
                    recipe=recipe,
                    seed=seed,
                    step=100,
                    non_mmlu_value=float(recipe_index),
                    mmlu_value=float(recipe_index),
                    parameter_count=100,
                    tokens=10,
                )
            )
        for seed in ("prediction-1", "prediction-2"):
            rows.extend(
                _task_rows(
                    recipe=recipe,
                    seed=seed,
                    non_mmlu_value=float(recipe_index),
                    mmlu_value=float(recipe_index),
                )
            )
    policy = _policy(target_compute=6_000.0)

    result = verify_normalized_olmes(pd.DataFrame(rows), policy)

    assert result.missing == ()
    assert result.target_ranking is not None
    assert len(result.seed_decisions) == 2
    assert len(result.checkpoint_summaries) == 1
    assert len(result.noise_spread) == 20
    assert all(
        fact.input_evidence_boundary is EvidenceBoundary.AGGREGATE_EVALUATION
        for fact in result.facts
    )
    assert all(fact.status is FactStatus.COMPLETE for fact in result.facts)
    seed_fact = next(
        fact
        for fact in result.facts
        if fact.fact == "single_scale_seed_decision_accuracy"
    )
    assert seed_fact.denominator == 1
    assert seed_fact.seed_count == 1


def test_record_policy_emits_machine_readable_missing_fact() -> None:
    policy = _policy(missing_behavior=MissingDataBehavior.RECORD)

    result = verify_normalized_olmes(pd.DataFrame(_task_rows()[:-1]), policy)

    missing_facts = [fact for fact in result.facts if fact.status is FactStatus.MISSING]
    assert missing_facts
    assert all(fact.value is None and fact.denominator == 0 for fact in missing_facts)
    assert all(
        fact.input_evidence_boundary is EvidenceBoundary.AGGREGATE_EVALUATION
        for fact in missing_facts
    )
