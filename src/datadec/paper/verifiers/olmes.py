from __future__ import annotations

import math
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import Iterable

import pandas as pd

from datadec.paper.analysis import (
    DecisionAccuracy,
    NoiseSpread,
    TiePolicy,
    decision_accuracy,
    macro_average_olmes,
    noise_and_spread,
    percent_target_compute,
    summarize_prediction_attempts,
    theoretical_training_flops,
)
from datadec.paper.models import EvidenceBoundary

NORMALIZED_KEY_COLUMNS: tuple[str, ...] = (
    "params",
    "data",
    "seed",
    "step",
    "task",
)


@verify(UNIQUE)
class MissingDataBehavior(StrEnum):
    ERROR = "error"
    RECORD = "record"


@verify(UNIQUE)
class FactStatus(StrEnum):
    COMPLETE = "complete"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class OlmesTaskGrouping:
    non_mmlu_tasks: tuple[str, ...]
    mmlu_subjects: tuple[str, ...]
    mmlu_task_name: str

    def __post_init__(self) -> None:
        _validate_identifiers(self.non_mmlu_tasks, label="non-MMLU task")
        _validate_identifiers(self.mmlu_subjects, label="MMLU subject")
        _validate_identifier(self.mmlu_task_name, label="MMLU task name")
        if len(self.non_mmlu_tasks) != 9:
            raise ValueError("OLMES grouping must contain exactly 9 non-MMLU tasks")
        if len(self.mmlu_subjects) != 57:
            raise ValueError("OLMES grouping must contain exactly 57 MMLU subjects")
        overlap = set(self.non_mmlu_tasks).intersection(self.mmlu_subjects)
        if overlap:
            raise ValueError(f"OLMES task groups overlap: {tuple(sorted(overlap))!r}")
        if self.mmlu_task_name in set(self.non_mmlu_tasks) | set(self.mmlu_subjects):
            raise ValueError("MMLU task name must be distinct from source task names")


@dataclass(frozen=True, slots=True)
class FinalCheckpoint:
    model_size: str
    step: int

    def __post_init__(self) -> None:
        _validate_identifier(self.model_size, label="model size")
        _validate_non_negative_int(self.step, label="final checkpoint step")


@dataclass(frozen=True, slots=True)
class NormalizedOlmesPolicy:
    recipes: tuple[str, ...]
    target_size: str
    target_seeds: tuple[str, ...]
    prediction_seeds: tuple[str, ...]
    target_metric_column: str
    proxy_metric_columns: tuple[str, ...]
    task_grouping: OlmesTaskGrouping
    final_checkpoints: tuple[FinalCheckpoint, ...]
    noise_size: str
    tie_policy: TiePolicy
    attempt_ddof: int
    within_recipe_ddof: int
    spread_ddof: int
    missing_data_behavior: MissingDataBehavior
    parameter_count_column: str
    token_count_column: str
    target_compute_denominator: float

    def __post_init__(self) -> None:
        _validate_identifiers(self.recipes, label="recipe")
        _validate_identifier(self.target_size, label="target size")
        _validate_identifiers(self.target_seeds, label="target seed")
        _validate_identifiers(self.prediction_seeds, label="prediction seed")
        _validate_identifier(self.target_metric_column, label="target metric column")
        _validate_identifiers(self.proxy_metric_columns, label="proxy metric column")
        _validate_identifier(self.noise_size, label="noise model size")
        _validate_identifier(
            self.parameter_count_column, label="parameter-count column"
        )
        _validate_identifier(self.token_count_column, label="token-count column")
        if not isinstance(self.task_grouping, OlmesTaskGrouping):
            raise TypeError("task_grouping must be an OlmesTaskGrouping")
        if not isinstance(self.tie_policy, TiePolicy):
            raise TypeError("tie_policy must be a TiePolicy")
        if not isinstance(self.missing_data_behavior, MissingDataBehavior):
            raise TypeError("missing_data_behavior must be a MissingDataBehavior")
        _validate_non_negative_int(self.attempt_ddof, label="attempt ddof")
        _validate_non_negative_int(self.within_recipe_ddof, label="within-recipe ddof")
        _validate_non_negative_int(self.spread_ddof, label="spread ddof")
        if self.attempt_ddof >= len(self.prediction_seeds):
            raise ValueError("attempt ddof must be less than prediction seed count")
        if self.within_recipe_ddof >= len(self.prediction_seeds):
            raise ValueError(
                "within-recipe ddof must be less than prediction seed count"
            )
        if self.spread_ddof >= len(self.recipes):
            raise ValueError("spread ddof must be less than recipe count")

        if not isinstance(self.final_checkpoints, tuple):
            raise TypeError("final_checkpoints must be supplied as a tuple")
        for checkpoint in self.final_checkpoints:
            if not isinstance(checkpoint, FinalCheckpoint):
                raise TypeError("final_checkpoints must contain FinalCheckpoint values")
        final_sizes = tuple(
            checkpoint.model_size for checkpoint in self.final_checkpoints
        )
        if not final_sizes:
            raise ValueError("final checkpoint mapping must not be empty")
        if len(final_sizes) != len(set(final_sizes)):
            raise ValueError("final checkpoint model sizes must be unique")
        missing_sizes = {self.target_size, self.noise_size}.difference(final_sizes)
        if missing_sizes:
            raise ValueError(
                "final checkpoint mapping is missing required model sizes: "
                f"{tuple(sorted(missing_sizes))!r}"
            )

        target_compute = _finite_number(
            self.target_compute_denominator, label="target compute denominator"
        )
        if target_compute <= 0:
            raise ValueError("target compute denominator must be positive")

    @property
    def metric_columns(self) -> tuple[str, ...]:
        return _ordered_unique((self.target_metric_column, *self.proxy_metric_columns))

    @property
    def final_step_by_size(self) -> dict[str, int]:
        return {
            checkpoint.model_size: checkpoint.step
            for checkpoint in self.final_checkpoints
        }


@dataclass(frozen=True, slots=True)
class MissingInput:
    stage: str
    model_size: str
    step: int
    metric: str
    recipe: str | None
    seed: str | None
    missing_tasks: tuple[str, ...] = ()
    missing_recipes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OlmesAggregateScore:
    model_size: str
    recipe: str
    seed: str
    step: int
    metric: str
    score: float
    mmlu_score: float
    parameter_count: float
    token_count: float


@dataclass(frozen=True, slots=True)
class CanonicalFinalSelection:
    scores: tuple[OlmesAggregateScore, ...]
    missing: tuple[MissingInput, ...]


@dataclass(frozen=True, slots=True)
class RecipeMean:
    recipe: str
    score: float


@dataclass(frozen=True, slots=True)
class TargetRanking:
    metric: str
    model_size: str
    step: int
    seed_count: int
    scores: tuple[RecipeMean, ...]


@dataclass(frozen=True, slots=True)
class PairDecision:
    recipe_a: str
    recipe_b: str
    target_sign: int
    predicted_sign: int
    correct: bool
    excluded: bool
    target_tie: bool
    predicted_tie: bool


@dataclass(frozen=True, slots=True)
class SeedDecisionAccuracy:
    model_size: str
    step: int
    metric: str
    seed: str
    accuracy: DecisionAccuracy
    pairs: tuple[PairDecision, ...]
    percent_target_compute: float
    included_compute: float


@dataclass(frozen=True, slots=True)
class CheckpointDecisionSummary:
    model_size: str
    step: int
    metric: str
    mean_accuracy: float
    sd_accuracy: float
    seed_count: int
    ddof: int
    sd_denominator: int
    percent_target_compute: float
    attempts: tuple[SeedDecisionAccuracy, ...]


@dataclass(frozen=True, slots=True)
class TaskMetricNoiseSpread:
    model_size: str
    step: int
    task: str
    metric: str
    result: NoiseSpread


@dataclass(frozen=True, slots=True)
class FactRow:
    fact: str
    status: FactStatus
    dimensions: tuple[tuple[str, str], ...]
    value: float | None
    denominator: int
    exclusions: int
    target_ties: int
    predicted_ties: int
    seed_count: int
    input_evidence_boundary: EvidenceBoundary


@dataclass(frozen=True, slots=True)
class NormalizedOlmesVerification:
    canonical_finals: CanonicalFinalSelection
    target_ranking: TargetRanking | None
    seed_decisions: tuple[SeedDecisionAccuracy, ...]
    checkpoint_summaries: tuple[CheckpointDecisionSummary, ...]
    noise_spread: tuple[TaskMetricNoiseSpread, ...]
    missing: tuple[MissingInput, ...]
    facts: tuple[FactRow, ...]


def _validate_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string: {value!r}")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _validate_identifiers(values: tuple[str, ...], *, label: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} values must be supplied as a tuple")
    if not values:
        raise ValueError(f"{label} values must not be empty")
    for value in values:
        _validate_identifier(value, label=label)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique")


def _validate_non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer: {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative: {value}")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a real number: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite: {value!r}")
    return number


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _missing_or_raise(
    missing: MissingInput,
    *,
    behavior: MissingDataBehavior,
) -> MissingInput:
    if behavior is MissingDataBehavior.ERROR:
        raise ValueError(
            "incomplete normalized OLMES input: "
            f"stage={missing.stage!r}, model_size={missing.model_size!r}, "
            f"step={missing.step}, metric={missing.metric!r}, "
            f"recipe={missing.recipe!r}, seed={missing.seed!r}, "
            f"missing_tasks={missing.missing_tasks!r}, "
            f"missing_recipes={missing.missing_recipes!r}"
        )
    return missing


def validate_normalized_olmes_frame(
    frame: pd.DataFrame, policy: NormalizedOlmesPolicy
) -> None:
    """Validate the normalized aggregate-evaluation boundary and unique keys."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = (
        set(NORMALIZED_KEY_COLUMNS)
        | set(policy.metric_columns)
        | {
            policy.parameter_count_column,
            policy.token_count_column,
        }
    )
    missing_columns = tuple(sorted(required.difference(frame.columns)))
    if missing_columns:
        raise ValueError(
            f"normalized OLMES frame is missing columns: {missing_columns!r}"
        )
    null_key_columns = tuple(
        column for column in NORMALIZED_KEY_COLUMNS if frame[column].isna().any()
    )
    if null_key_columns:
        raise ValueError(f"normalized OLMES keys contain nulls: {null_key_columns!r}")
    duplicate_mask = frame.duplicated(list(NORMALIZED_KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicate_keys = tuple(
            tuple(row)
            for row in frame.loc[duplicate_mask, list(NORMALIZED_KEY_COLUMNS)]
            .drop_duplicates()
            .sort_values(list(NORMALIZED_KEY_COLUMNS), kind="stable")
            .itertuples(index=False, name=None)
        )
        raise ValueError(f"duplicate normalized OLMES keys: {duplicate_keys!r}")


def aggregate_olmes_scores(
    frame: pd.DataFrame, policy: NormalizedOlmesPolicy
) -> tuple[tuple[OlmesAggregateScore, ...], tuple[MissingInput, ...]]:
    """Aggregate 57 MMLU subjects, then equal-weight the ten OLMES tasks."""
    validate_normalized_olmes_frame(frame, policy)
    columns = (
        *NORMALIZED_KEY_COLUMNS,
        policy.parameter_count_column,
        policy.token_count_column,
        *policy.metric_columns,
    )
    grouped: dict[tuple[str, str, str, int], list[tuple[object, ...]]] = {}
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        model_size = _validate_identifier(row[0], label="params")
        recipe = _validate_identifier(row[1], label="data")
        seed = _validate_identifier(row[2], label="seed")
        step = _validate_non_negative_int(row[3], label="step")
        grouped.setdefault((model_size, recipe, seed, step), []).append(row)

    expected_tasks = set(policy.task_grouping.non_mmlu_tasks) | set(
        policy.task_grouping.mmlu_subjects
    )
    scores: list[OlmesAggregateScore] = []
    missing_inputs: list[MissingInput] = []
    for group_key in sorted(grouped):
        model_size, recipe, seed, step = group_key
        rows = grouped[group_key]
        tasks = {_validate_identifier(row[4], label="task") for row in rows}
        unexpected_tasks = tuple(sorted(tasks.difference(expected_tasks)))
        if unexpected_tasks:
            raise ValueError(
                f"unexpected normalized OLMES tasks for {group_key!r}: "
                f"{unexpected_tasks!r}"
            )
        missing_tasks = tuple(sorted(expected_tasks.difference(tasks)))
        parameters = {
            _finite_number(row[5], label=f"parameter count for {group_key!r}")
            for row in rows
        }
        tokens = {
            _finite_number(row[6], label=f"token count for {group_key!r}")
            for row in rows
        }
        if len(parameters) != 1 or len(tokens) != 1:
            raise ValueError(
                "parameter and token columns must be invariant within checkpoint: "
                f"{group_key!r}"
            )
        parameter_count = next(iter(parameters))
        token_count = next(iter(tokens))
        if parameter_count <= 0:
            raise ValueError(f"parameter count must be positive for {group_key!r}")
        if token_count < 0 or (step > 0 and token_count == 0):
            raise ValueError(
                "token count must be non-negative and positive after step 0 for "
                f"{group_key!r}"
            )

        for metric_index, metric in enumerate(policy.metric_columns, start=7):
            null_metric_tasks = tuple(
                sorted(
                    _validate_identifier(row[4], label="task")
                    for row in rows
                    if pd.isna(row[metric_index])
                )
            )
            metric_missing_tasks = tuple(
                sorted(set(missing_tasks) | set(null_metric_tasks))
            )
            if metric_missing_tasks:
                missing_inputs.append(
                    _missing_or_raise(
                        MissingInput(
                            stage="task_aggregation",
                            model_size=model_size,
                            step=step,
                            metric=metric,
                            recipe=recipe,
                            seed=seed,
                            missing_tasks=metric_missing_tasks,
                        ),
                        behavior=policy.missing_data_behavior,
                    )
                )
                continue
            task_scores = tuple(
                (
                    _validate_identifier(row[4], label="task"),
                    _finite_number(
                        row[metric_index],
                        label=f"{metric!r} for {group_key!r}",
                    ),
                )
                for row in rows
            )
            aggregate = macro_average_olmes(
                task_scores,
                expected_mmlu_subjects=policy.task_grouping.mmlu_subjects,
            )
            scores.append(
                OlmesAggregateScore(
                    model_size=model_size,
                    recipe=recipe,
                    seed=seed,
                    step=step,
                    metric=metric,
                    score=aggregate.score,
                    mmlu_score=aggregate.mmlu_score,
                    parameter_count=parameter_count,
                    token_count=token_count,
                )
            )
    return tuple(scores), tuple(missing_inputs)


def select_canonical_final_checkpoints(
    scores: Iterable[OlmesAggregateScore], policy: NormalizedOlmesPolicy
) -> CanonicalFinalSelection:
    """Select explicitly mapped final checkpoints and expose absent groups."""
    score_lookup = {
        (score.model_size, score.recipe, score.seed, score.step, score.metric): score
        for score in scores
    }
    selected: list[OlmesAggregateScore] = []
    missing: list[MissingInput] = []
    for checkpoint in sorted(
        policy.final_checkpoints, key=lambda value: (value.model_size, value.step)
    ):
        seeds = (
            policy.target_seeds
            if checkpoint.model_size == policy.target_size
            else policy.prediction_seeds
        )
        for recipe in sorted(policy.recipes):
            for seed in sorted(seeds):
                for metric in policy.metric_columns:
                    key = (
                        checkpoint.model_size,
                        recipe,
                        seed,
                        checkpoint.step,
                        metric,
                    )
                    score = score_lookup.get(key)
                    if score is None:
                        missing.append(
                            _missing_or_raise(
                                MissingInput(
                                    stage="canonical_final",
                                    model_size=checkpoint.model_size,
                                    step=checkpoint.step,
                                    metric=metric,
                                    recipe=recipe,
                                    seed=seed,
                                ),
                                behavior=policy.missing_data_behavior,
                            )
                        )
                    else:
                        selected.append(score)
    return CanonicalFinalSelection(scores=tuple(selected), missing=tuple(missing))


def compute_target_ranking(
    final_scores: Iterable[OlmesAggregateScore], policy: NormalizedOlmesPolicy
) -> tuple[TargetRanking | None, tuple[MissingInput, ...]]:
    """Compute the target recipe ranking from the explicit target seed set."""
    step = policy.final_step_by_size[policy.target_size]
    lookup = {
        (score.recipe, score.seed): score.score
        for score in final_scores
        if score.model_size == policy.target_size
        and score.step == step
        and score.metric == policy.target_metric_column
    }
    missing: list[MissingInput] = []
    for recipe in sorted(policy.recipes):
        absent_seeds = tuple(
            seed for seed in sorted(policy.target_seeds) if (recipe, seed) not in lookup
        )
        for seed in absent_seeds:
            missing.append(
                _missing_or_raise(
                    MissingInput(
                        stage="target_mean",
                        model_size=policy.target_size,
                        step=step,
                        metric=policy.target_metric_column,
                        recipe=recipe,
                        seed=seed,
                    ),
                    behavior=policy.missing_data_behavior,
                )
            )
    if missing:
        return None, tuple(missing)

    recipe_means = tuple(
        RecipeMean(
            recipe=recipe,
            score=math.fsum(
                sorted(lookup[(recipe, seed)] for seed in policy.target_seeds)
            )
            / len(policy.target_seeds),
        )
        for recipe in policy.recipes
    )
    ranking = tuple(
        sorted(recipe_means, key=lambda value: (-value.score, value.recipe))
    )
    return (
        TargetRanking(
            metric=policy.target_metric_column,
            model_size=policy.target_size,
            step=step,
            seed_count=len(policy.target_seeds),
            scores=ranking,
        ),
        (),
    )


def _pair_decisions(
    target_scores: tuple[tuple[str, float], ...],
    predicted_scores: tuple[tuple[str, float], ...],
    *,
    tie_policy: TiePolicy,
) -> tuple[PairDecision, ...]:
    target = dict(target_scores)
    predicted = dict(predicted_scores)
    recipes = tuple(sorted(target))
    result: list[PairDecision] = []
    for index, recipe_a in enumerate(recipes):
        for recipe_b in recipes[index + 1 :]:
            target_sign = (target[recipe_a] > target[recipe_b]) - (
                target[recipe_a] < target[recipe_b]
            )
            predicted_sign = (predicted[recipe_a] > predicted[recipe_b]) - (
                predicted[recipe_a] < predicted[recipe_b]
            )
            target_tie = target_sign == 0
            predicted_tie = predicted_sign == 0
            excluded = target_tie or (tie_policy is TiePolicy.EXCLUDE and predicted_tie)
            result.append(
                PairDecision(
                    recipe_a=recipe_a,
                    recipe_b=recipe_b,
                    target_sign=target_sign,
                    predicted_sign=predicted_sign,
                    correct=not excluded and target_sign == predicted_sign,
                    excluded=excluded,
                    target_tie=target_tie,
                    predicted_tie=predicted_tie,
                )
            )
    return tuple(result)


def compute_single_scale_decisions(
    scores: Iterable[OlmesAggregateScore],
    target_ranking: TargetRanking,
    policy: NormalizedOlmesPolicy,
) -> tuple[
    tuple[SeedDecisionAccuracy, ...],
    tuple[CheckpointDecisionSummary, ...],
    tuple[MissingInput, ...],
]:
    """Calculate seed attempts and mean/SD for every observed single checkpoint."""
    target_scores = tuple(
        (score.recipe, score.score) for score in target_ranking.scores
    )
    score_values = tuple(scores)
    checkpoint_keys = sorted(
        {
            (score.model_size, score.step, score.metric)
            for score in score_values
            if score.model_size != policy.target_size
            and score.seed in policy.prediction_seeds
            and score.metric in policy.metric_columns
        }
    )
    lookup = {
        (score.model_size, score.step, score.metric, score.seed, score.recipe): score
        for score in score_values
    }
    attempts: list[SeedDecisionAccuracy] = []
    summaries: list[CheckpointDecisionSummary] = []
    missing: list[MissingInput] = []
    for model_size, step, metric in checkpoint_keys:
        checkpoint_attempts: list[SeedDecisionAccuracy] = []
        for seed in sorted(policy.prediction_seeds):
            absent_recipes = tuple(
                recipe
                for recipe in sorted(policy.recipes)
                if (model_size, step, metric, seed, recipe) not in lookup
            )
            if absent_recipes:
                missing.append(
                    _missing_or_raise(
                        MissingInput(
                            stage="single_scale_seed",
                            model_size=model_size,
                            step=step,
                            metric=metric,
                            recipe=None,
                            seed=seed,
                            missing_recipes=absent_recipes,
                        ),
                        behavior=policy.missing_data_behavior,
                    )
                )
                continue
            prediction_rows = tuple(
                lookup[(model_size, step, metric, seed, recipe)]
                for recipe in sorted(policy.recipes)
            )
            parameter_token_pairs = {
                (row.parameter_count, row.token_count) for row in prediction_rows
            }
            if len(parameter_token_pairs) != 1:
                raise ValueError(
                    "single-scale parameter/token evidence differs across recipes: "
                    f"{(model_size, step, metric, seed)!r}"
                )
            parameter_count, token_count = next(iter(parameter_token_pairs))
            compute = theoretical_training_flops(parameter_count, token_count)
            budget = percent_target_compute(
                (compute,), target_compute=policy.target_compute_denominator
            )
            predicted_scores = tuple((row.recipe, row.score) for row in prediction_rows)
            accuracy = decision_accuracy(
                target_scores,
                predicted_scores,
                tie_policy=policy.tie_policy,
            )
            attempt = SeedDecisionAccuracy(
                model_size=model_size,
                step=step,
                metric=metric,
                seed=seed,
                accuracy=accuracy,
                pairs=_pair_decisions(
                    target_scores,
                    predicted_scores,
                    tie_policy=policy.tie_policy,
                ),
                percent_target_compute=budget.percent,
                included_compute=budget.included_compute,
            )
            checkpoint_attempts.append(attempt)
            attempts.append(attempt)
        if len(checkpoint_attempts) != len(policy.prediction_seeds):
            continue
        compute_values = {
            attempt.percent_target_compute for attempt in checkpoint_attempts
        }
        if len(compute_values) != 1:
            raise ValueError(
                "single-scale compute differs across prediction seeds: "
                f"{(model_size, step, metric)!r}"
            )
        summary = summarize_prediction_attempts(
            (attempt.accuracy.accuracy for attempt in checkpoint_attempts),
            ddof=policy.attempt_ddof,
        )
        summaries.append(
            CheckpointDecisionSummary(
                model_size=model_size,
                step=step,
                metric=metric,
                mean_accuracy=summary.mean,
                sd_accuracy=summary.sd,
                seed_count=summary.count,
                ddof=summary.ddof,
                sd_denominator=summary.denominator,
                percent_target_compute=next(iter(compute_values)),
                attempts=tuple(checkpoint_attempts),
            )
        )
    return tuple(attempts), tuple(summaries), tuple(missing)


def compute_task_metric_noise_spread(
    frame: pd.DataFrame, policy: NormalizedOlmesPolicy
) -> tuple[tuple[TaskMetricNoiseSpread, ...], tuple[MissingInput, ...]]:
    """Calculate final-checkpoint 150M-style per-task/metric noise and spread."""
    validate_normalized_olmes_frame(frame, policy)
    model_size = policy.noise_size
    step = policy.final_step_by_size[model_size]
    subset = frame.loc[
        (frame["params"] == model_size)
        & (frame["step"] == step)
        & frame["data"].isin(policy.recipes)
        & frame["seed"].isin(policy.prediction_seeds)
    ]
    key_to_row = {
        (str(row[0]), str(row[1]), str(row[2])): row
        for row in subset.loc[
            :, ["data", "seed", "task", *policy.metric_columns]
        ].itertuples(index=False, name=None)
    }
    results: list[TaskMetricNoiseSpread] = []
    missing: list[MissingInput] = []
    logical_tasks = (
        *policy.task_grouping.non_mmlu_tasks,
        policy.task_grouping.mmlu_task_name,
    )
    for metric_index, metric in enumerate(policy.metric_columns, start=3):
        for logical_task in logical_tasks:
            observations: list[tuple[str, str, float]] = []
            for recipe in sorted(policy.recipes):
                for seed in sorted(policy.prediction_seeds):
                    if logical_task == policy.task_grouping.mmlu_task_name:
                        source_tasks = policy.task_grouping.mmlu_subjects
                    else:
                        source_tasks = (logical_task,)
                    absent_tasks = tuple(
                        task
                        for task in source_tasks
                        if (recipe, seed, task) not in key_to_row
                        or pd.isna(key_to_row[(recipe, seed, task)][metric_index])
                    )
                    if absent_tasks:
                        missing.append(
                            _missing_or_raise(
                                MissingInput(
                                    stage="noise_spread",
                                    model_size=model_size,
                                    step=step,
                                    metric=metric,
                                    recipe=recipe,
                                    seed=seed,
                                    missing_tasks=absent_tasks,
                                ),
                                behavior=policy.missing_data_behavior,
                            )
                        )
                        continue
                    values = tuple(
                        _finite_number(
                            key_to_row[(recipe, seed, task)][metric_index],
                            label=f"{metric!r} for {(recipe, seed, task)!r}",
                        )
                        for task in source_tasks
                    )
                    observations.append(
                        (recipe, seed, math.fsum(sorted(values)) / len(values))
                    )
            if len(observations) != len(policy.recipes) * len(policy.prediction_seeds):
                continue
            result = noise_and_spread(
                observations,
                expected_recipes=policy.recipes,
                expected_seeds=policy.prediction_seeds,
                within_recipe_ddof=policy.within_recipe_ddof,
                spread_ddof=policy.spread_ddof,
            )
            results.append(
                TaskMetricNoiseSpread(
                    model_size=model_size,
                    step=step,
                    task=logical_task,
                    metric=metric,
                    result=result,
                )
            )
    return tuple(results), tuple(missing)


def _complete_facts(
    target_ranking: TargetRanking | None,
    attempts: tuple[SeedDecisionAccuracy, ...],
    summaries: tuple[CheckpointDecisionSummary, ...],
    noise_spread_results: tuple[TaskMetricNoiseSpread, ...],
    missing: tuple[MissingInput, ...],
) -> tuple[FactRow, ...]:
    facts: list[FactRow] = []
    if target_ranking is not None:
        for recipe_score in target_ranking.scores:
            facts.append(
                FactRow(
                    fact="target_recipe_mean",
                    status=FactStatus.COMPLETE,
                    dimensions=(
                        ("model_size", target_ranking.model_size),
                        ("step", str(target_ranking.step)),
                        ("metric", target_ranking.metric),
                        ("recipe", recipe_score.recipe),
                    ),
                    value=recipe_score.score,
                    denominator=target_ranking.seed_count,
                    exclusions=0,
                    target_ties=0,
                    predicted_ties=0,
                    seed_count=target_ranking.seed_count,
                    input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
                )
            )
    for attempt in attempts:
        accuracy = attempt.accuracy
        facts.append(
            FactRow(
                fact="single_scale_seed_decision_accuracy",
                status=FactStatus.COMPLETE,
                dimensions=(
                    ("model_size", attempt.model_size),
                    ("step", str(attempt.step)),
                    ("metric", attempt.metric),
                    ("seed", attempt.seed),
                    ("percent_target_compute", str(attempt.percent_target_compute)),
                ),
                value=accuracy.accuracy,
                denominator=accuracy.denominator,
                exclusions=accuracy.excluded_pairs,
                target_ties=accuracy.target_ties,
                predicted_ties=accuracy.predicted_ties,
                seed_count=1,
                input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
            )
        )
    for summary in summaries:
        exclusions = sum(
            attempt.accuracy.excluded_pairs for attempt in summary.attempts
        )
        target_ties = sum(attempt.accuracy.target_ties for attempt in summary.attempts)
        predicted_ties = sum(
            attempt.accuracy.predicted_ties for attempt in summary.attempts
        )
        dimensions = (
            ("model_size", summary.model_size),
            ("step", str(summary.step)),
            ("metric", summary.metric),
            ("percent_target_compute", str(summary.percent_target_compute)),
        )
        facts.extend(
            (
                FactRow(
                    fact="single_scale_mean_decision_accuracy",
                    status=FactStatus.COMPLETE,
                    dimensions=dimensions,
                    value=summary.mean_accuracy,
                    denominator=summary.seed_count,
                    exclusions=exclusions,
                    target_ties=target_ties,
                    predicted_ties=predicted_ties,
                    seed_count=summary.seed_count,
                    input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
                ),
                FactRow(
                    fact="single_scale_sd_decision_accuracy",
                    status=FactStatus.COMPLETE,
                    dimensions=dimensions,
                    value=summary.sd_accuracy,
                    denominator=summary.sd_denominator,
                    exclusions=exclusions,
                    target_ties=target_ties,
                    predicted_ties=predicted_ties,
                    seed_count=summary.seed_count,
                    input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
                ),
            )
        )
    for item in noise_spread_results:
        dimensions = (
            ("model_size", item.model_size),
            ("step", str(item.step)),
            ("task", item.task),
            ("metric", item.metric),
        )
        facts.extend(
            (
                FactRow(
                    fact="task_metric_noise",
                    status=FactStatus.COMPLETE,
                    dimensions=dimensions,
                    value=item.result.noise,
                    denominator=item.result.within_recipe_denominator,
                    exclusions=0,
                    target_ties=0,
                    predicted_ties=0,
                    seed_count=item.result.seed_count,
                    input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
                ),
                FactRow(
                    fact="task_metric_spread",
                    status=FactStatus.COMPLETE,
                    dimensions=dimensions,
                    value=item.result.spread,
                    denominator=item.result.spread_denominator,
                    exclusions=0,
                    target_ties=0,
                    predicted_ties=0,
                    seed_count=item.result.seed_count,
                    input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
                ),
            )
        )
    for item in missing:
        facts.append(
            FactRow(
                fact="missing_input",
                status=FactStatus.MISSING,
                dimensions=(
                    ("stage", item.stage),
                    ("model_size", item.model_size),
                    ("step", str(item.step)),
                    ("metric", item.metric),
                    ("recipe", item.recipe or ""),
                    ("seed", item.seed or ""),
                    ("missing_tasks", ",".join(item.missing_tasks)),
                    ("missing_recipes", ",".join(item.missing_recipes)),
                ),
                value=None,
                denominator=0,
                exclusions=0,
                target_ties=0,
                predicted_ties=0,
                seed_count=0,
                input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
            )
        )
    return tuple(facts)


def verify_normalized_olmes(
    frame: pd.DataFrame, policy: NormalizedOlmesPolicy
) -> NormalizedOlmesVerification:
    """Verify answerable OLMES aggregate-evaluation claim families."""
    scores, aggregation_missing = aggregate_olmes_scores(frame, policy)
    canonical_finals = select_canonical_final_checkpoints(scores, policy)
    target_ranking, target_missing = compute_target_ranking(
        canonical_finals.scores, policy
    )
    if target_ranking is None:
        seed_decisions: tuple[SeedDecisionAccuracy, ...] = ()
        summaries: tuple[CheckpointDecisionSummary, ...] = ()
        decision_missing: tuple[MissingInput, ...] = ()
    else:
        seed_decisions, summaries, decision_missing = compute_single_scale_decisions(
            scores, target_ranking, policy
        )
    noise_spread_results, noise_missing = compute_task_metric_noise_spread(
        frame, policy
    )
    missing = (
        *aggregation_missing,
        *canonical_finals.missing,
        *target_missing,
        *decision_missing,
        *noise_missing,
    )
    facts = _complete_facts(
        target_ranking,
        seed_decisions,
        summaries,
        noise_spread_results,
        missing,
    )
    return NormalizedOlmesVerification(
        canonical_finals=canonical_finals,
        target_ranking=target_ranking,
        seed_decisions=seed_decisions,
        checkpoint_summaries=summaries,
        noise_spread=noise_spread_results,
        missing=missing,
        facts=facts,
    )


def verify_normalized_olmes_parquet(
    path: str | Path, policy: NormalizedOlmesPolicy
) -> NormalizedOlmesVerification:
    """Read the normalized aggregate parquet and run the deterministic adapter."""
    return verify_normalized_olmes(pd.read_parquet(path), policy)


__all__ = [
    "CanonicalFinalSelection",
    "CheckpointDecisionSummary",
    "FactRow",
    "FactStatus",
    "FinalCheckpoint",
    "MissingDataBehavior",
    "MissingInput",
    "NORMALIZED_KEY_COLUMNS",
    "NormalizedOlmesPolicy",
    "NormalizedOlmesVerification",
    "OlmesAggregateScore",
    "OlmesTaskGrouping",
    "PairDecision",
    "RecipeMean",
    "SeedDecisionAccuracy",
    "TargetRanking",
    "TaskMetricNoiseSpread",
    "aggregate_olmes_scores",
    "compute_single_scale_decisions",
    "compute_target_ranking",
    "compute_task_metric_noise_spread",
    "select_canonical_final_checkpoints",
    "validate_normalized_olmes_frame",
    "verify_normalized_olmes",
    "verify_normalized_olmes_parquet",
]
