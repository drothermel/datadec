from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import pandas as pd

from datadec.data.ingest.enums import MMLU_SUBJECT_TASKS

OLMES_NON_MMLU_TASKS: tuple[str, ...] = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
MMLU_SUBJECTS: tuple[str, ...] = tuple(task.value for task in MMLU_SUBJECT_TASKS)


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string: {value!r}")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _identifiers(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} values must be supplied as a tuple")
    if not values:
        raise ValueError(f"{label} values must not be empty")
    result = tuple(_identifier(value, label=label) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} values must be unique")
    return result


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer: {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative: {result}")
    return result


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite: {value!r}")
    return result


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("at least one value is required")
    return math.fsum(ordered) / len(ordered)


def _sample_sd(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if len(ordered) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    mean = _mean(ordered)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in ordered) / (len(ordered) - 1)
    )


@dataclass(frozen=True, slots=True)
class OlmesTaskGrouping:
    non_mmlu_tasks: tuple[str, ...]
    mmlu_subjects: tuple[str, ...]

    def __post_init__(self) -> None:
        non_mmlu = _identifiers(self.non_mmlu_tasks, label="non-MMLU task")
        subjects = _identifiers(self.mmlu_subjects, label="MMLU subject")
        overlap = set(non_mmlu).intersection(subjects)
        if overlap:
            raise ValueError(f"OLMES task groups overlap: {tuple(sorted(overlap))!r}")

    @property
    def source_tasks(self) -> tuple[str, ...]:
        return (*self.non_mmlu_tasks, *self.mmlu_subjects)


DEFAULT_TASK_GROUPING = OlmesTaskGrouping(
    non_mmlu_tasks=OLMES_NON_MMLU_TASKS,
    mmlu_subjects=MMLU_SUBJECTS,
)


@dataclass(frozen=True, slots=True)
class SingleScaleUniverse:
    model_size: str
    recipes: tuple[str, ...]
    seeds: tuple[str, ...]
    source_tasks: tuple[str, ...]
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.model_size, label="model size")
        _identifiers(self.recipes, label="recipe")
        _identifiers(self.seeds, label="seed")
        _identifiers(self.source_tasks, label="source task")
        _identifiers(self.metrics, label="metric")

    @property
    def expected_raw_row_count(self) -> int:
        return len(self.recipes) * len(self.seeds) * len(self.source_tasks)

    @property
    def expected_observation_count(self) -> int:
        return self.expected_raw_row_count * len(self.metrics)


@dataclass(frozen=True, slots=True)
class MetricObservation:
    model_size: str
    recipe: str
    seed: str
    step: int
    source_task: str
    metric: str
    score: float
    compute: float


@dataclass(frozen=True, slots=True)
class CheckpointRows:
    universe: SingleScaleUniverse
    step: int
    observations: tuple[MetricObservation, ...]
    raw_row_count: int
    selected_observation_count: int
    expected_observation_count: int
    actual_compute: float


@dataclass(frozen=True, slots=True)
class CommonCompleteCheckpoints:
    default: CheckpointRows
    preceding: tuple[CheckpointRows, ...]
    complete_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AggregateScore:
    model_size: str
    recipe: str
    seed: str
    step: int
    metric: str
    score: float
    mmlu_score: float
    source_task_count: int
    logical_task_count: int


@dataclass(frozen=True, slots=True)
class RankedRecipe:
    recipe: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class TargetRanking:
    model_size: str
    step: int
    metric: str
    seed_count: int
    scores: tuple[RankedRecipe, ...]


@dataclass(frozen=True, slots=True)
class PredictionRanking:
    model_size: str
    step: int
    metric: str
    seed: str
    scores: tuple[RankedRecipe, ...]


@dataclass(frozen=True, slots=True)
class PairDecision:
    model_size: str
    step: int
    metric: str
    seed: str
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
    accuracy: float
    correct: int
    denominator: int
    total_pairs: int
    target_ties: int
    predicted_ties: int
    actual_compute: float
    percent_target_compute: float
    pairs: tuple[PairDecision, ...]


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    model_size: str
    step: int
    metric: str
    mean_accuracy: float
    sample_sd_accuracy: float
    seed_count: int
    ddof: int
    sd_denominator: int
    denominator_per_seed: int
    total_pairs_per_seed: int
    correct_counts: tuple[int, ...]
    target_ties: int
    predicted_ties: int
    actual_compute: float
    percent_target_compute: float


@dataclass(frozen=True, slots=True)
class NoiseSpread:
    model_size: str
    step: int
    metric: str
    noise: float
    spread: float
    recipe_count: int
    seed_count: int
    within_recipe_ddof: int
    spread_ddof: int


@dataclass(frozen=True, slots=True)
class PredictionCheckpointResult:
    checkpoint: CheckpointRows
    aggregate_scores: tuple[AggregateScore, ...]
    rankings: tuple[PredictionRanking, ...]
    seed_decisions: tuple[SeedDecisionAccuracy, ...]
    summaries: tuple[CheckpointSummary, ...]
    noise_spread: tuple[NoiseSpread, ...]


@dataclass(frozen=True, slots=True)
class SingleScaleAnalysis:
    target_checkpoints: CommonCompleteCheckpoints
    prediction_checkpoints: CommonCompleteCheckpoints
    target_aggregate_scores: tuple[AggregateScore, ...]
    target_ranking: TargetRanking
    predictions: tuple[PredictionCheckpointResult, ...]


def observations_from_olmes_frame(
    frame: pd.DataFrame,
    *,
    metric_columns: tuple[str, ...],
) -> tuple[MetricObservation, ...]:
    """Parse normalized wide OLMES rows into finite metric observations."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    metrics = _identifiers(metric_columns, label="metric column")
    key_columns = ("params", "data", "seed", "step", "task")
    required = {*key_columns, "compute", *metrics}
    missing_columns = tuple(sorted(required.difference(frame.columns)))
    if missing_columns:
        raise ValueError(
            f"normalized OLMES frame is missing columns: {missing_columns!r}"
        )
    if any(frame[column].isna().any() for column in key_columns):
        null_columns = tuple(
            column for column in key_columns if frame[column].isna().any()
        )
        raise ValueError(f"normalized OLMES keys contain nulls: {null_columns!r}")
    duplicate_mask = frame.duplicated(list(key_columns), keep=False)
    if duplicate_mask.any():
        raise ValueError("normalized OLMES input contains duplicate source-row keys")

    columns = (*key_columns, "compute", *metrics)
    observations: list[MetricObservation] = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        model_size = _identifier(row[0], label="model size")
        recipe = _identifier(row[1], label="recipe")
        seed = _identifier(row[2], label="seed")
        step = _non_negative_int(row[3], label="step")
        source_task = _identifier(row[4], label="source task")
        compute = _finite_number(row[5], label="compute")
        if compute < 0:
            raise ValueError(f"compute must be non-negative: {compute}")
        for index, metric in enumerate(metrics, start=6):
            raw_score = row[index]
            if pd.isna(raw_score):
                continue
            observations.append(
                MetricObservation(
                    model_size=model_size,
                    recipe=recipe,
                    seed=seed,
                    step=step,
                    source_task=source_task,
                    metric=metric,
                    score=_finite_number(raw_score, label=f"{metric} score"),
                    compute=compute,
                )
            )
    return tuple(
        sorted(
            observations,
            key=lambda value: (
                value.model_size,
                value.step,
                value.recipe,
                value.seed,
                value.source_task,
                value.metric,
            ),
        )
    )


def _expected_keys(
    universe: SingleScaleUniverse,
) -> set[tuple[str, str, str, str]]:
    return {
        (recipe, seed, source_task, metric)
        for recipe in universe.recipes
        for seed in universe.seeds
        for source_task in universe.source_tasks
        for metric in universe.metrics
    }


def _observations_at_step(
    observations: Iterable[MetricObservation],
    universe: SingleScaleUniverse,
    step: int,
) -> tuple[MetricObservation, ...]:
    recipe_set = set(universe.recipes)
    seed_set = set(universe.seeds)
    task_set = set(universe.source_tasks)
    metric_set = set(universe.metrics)
    return tuple(
        value
        for value in observations
        if value.model_size == universe.model_size
        and value.step == step
        and value.recipe in recipe_set
        and value.seed in seed_set
        and value.source_task in task_set
        and value.metric in metric_set
    )


def select_exact_common_complete_checkpoint(
    observations: Iterable[MetricObservation],
    universe: SingleScaleUniverse,
    *,
    step: int,
) -> CheckpointRows:
    """Select one step only after proving the entire declared grid is complete."""
    selected_step = _non_negative_int(step, label="checkpoint step")
    selected = _observations_at_step(observations, universe, selected_step)
    expected = _expected_keys(universe)
    actual = {
        (value.recipe, value.seed, value.source_task, value.metric)
        for value in selected
    }
    missing = tuple(sorted(expected.difference(actual)))
    unexpected = tuple(sorted(actual.difference(expected)))
    if missing or unexpected or len(selected) != len(actual):
        raise ValueError(
            "checkpoint grid is incomplete: "
            f"model_size={universe.model_size!r}, step={selected_step}, "
            f"selected={len(actual)}, expected={len(expected)}, "
            f"missing_count={len(missing)}, unexpected_count={len(unexpected)}, "
            f"missing_sample={missing[:5]!r}"
        )
    computes = {value.compute for value in selected}
    if len(computes) != 1:
        raise ValueError(
            "checkpoint compute differs across the declared grid: "
            f"model_size={universe.model_size!r}, step={selected_step}, "
            f"values={tuple(sorted(computes))!r}"
        )
    raw_keys = {(value.recipe, value.seed, value.source_task) for value in selected}
    if len(raw_keys) != universe.expected_raw_row_count:
        raise ValueError(
            "checkpoint raw-row grid is incomplete: "
            f"selected={len(raw_keys)}, expected={universe.expected_raw_row_count}"
        )
    return CheckpointRows(
        universe=universe,
        step=selected_step,
        observations=tuple(
            sorted(
                selected,
                key=lambda value: (
                    value.recipe,
                    value.seed,
                    value.source_task,
                    value.metric,
                ),
            )
        ),
        raw_row_count=len(raw_keys),
        selected_observation_count=len(selected),
        expected_observation_count=universe.expected_observation_count,
        actual_compute=next(iter(computes)),
    )


def select_common_complete_checkpoints(
    observations: Iterable[MetricObservation],
    universe: SingleScaleUniverse,
    *,
    preceding_count: int = 2,
) -> CommonCompleteCheckpoints:
    """Select the latest and fixed preceding full-grid checkpoints."""
    preceding = _non_negative_int(preceding_count, label="preceding checkpoint count")
    values = tuple(observations)
    candidate_steps = tuple(
        sorted(
            {
                value.step
                for value in values
                if value.model_size == universe.model_size
                and value.recipe in set(universe.recipes)
                and value.seed in set(universe.seeds)
                and value.source_task in set(universe.source_tasks)
                and value.metric in set(universe.metrics)
            }
        )
    )
    expected = _expected_keys(universe)
    complete_steps = tuple(
        step
        for step in candidate_steps
        if {
            (value.recipe, value.seed, value.source_task, value.metric)
            for value in _observations_at_step(values, universe, step)
        }
        == expected
    )
    if not complete_steps:
        counts = tuple(
            (
                step,
                len(
                    {
                        (value.recipe, value.seed, value.source_task, value.metric)
                        for value in _observations_at_step(values, universe, step)
                    }
                ),
            )
            for step in candidate_steps
        )
        raise ValueError(
            "no common complete checkpoint exists for the declared grid: "
            f"model_size={universe.model_size!r}, expected={len(expected)}, "
            f"step_counts={counts!r}"
        )
    selected_steps = tuple(reversed(complete_steps[-(preceding + 1) :]))
    checkpoints = tuple(
        select_exact_common_complete_checkpoint(values, universe, step=step)
        for step in selected_steps
    )
    return CommonCompleteCheckpoints(
        default=checkpoints[0],
        preceding=checkpoints[1:],
        complete_steps=complete_steps,
    )


def aggregate_checkpoint(
    checkpoint: CheckpointRows,
    *,
    task_grouping: OlmesTaskGrouping = DEFAULT_TASK_GROUPING,
) -> tuple[AggregateScore, ...]:
    """Average MMLU subjects first, then equally weight all logical tasks."""
    if set(checkpoint.universe.source_tasks) != set(task_grouping.source_tasks):
        raise ValueError(
            "checkpoint source-task universe differs from the OLMES grouping"
        )
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for observation in checkpoint.observations:
        key = (observation.recipe, observation.seed, observation.metric)
        task_scores = grouped.setdefault(key, {})
        if observation.source_task in task_scores:
            raise ValueError(
                f"duplicate source task in aggregate group: {(*key, observation.source_task)!r}"
            )
        task_scores[observation.source_task] = observation.score

    expected_tasks = set(task_grouping.source_tasks)
    results: list[AggregateScore] = []
    for key in sorted(grouped):
        recipe, seed, metric = key
        task_scores = grouped[key]
        if set(task_scores) != expected_tasks:
            missing = tuple(sorted(expected_tasks.difference(task_scores)))
            raise ValueError(
                f"aggregate group has incomplete source tasks: key={key!r}, missing={missing!r}"
            )
        mmlu_score = _mean(
            task_scores[subject] for subject in task_grouping.mmlu_subjects
        )
        logical_scores = [task_scores[task] for task in task_grouping.non_mmlu_tasks]
        logical_scores.append(mmlu_score)
        results.append(
            AggregateScore(
                model_size=checkpoint.universe.model_size,
                recipe=recipe,
                seed=seed,
                step=checkpoint.step,
                metric=metric,
                score=_mean(logical_scores),
                mmlu_score=mmlu_score,
                source_task_count=len(task_scores),
                logical_task_count=len(logical_scores),
            )
        )
    return tuple(results)


def _ranked_scores(scores: dict[str, float]) -> tuple[RankedRecipe, ...]:
    return tuple(
        RankedRecipe(recipe=recipe, rank=index, score=score)
        for index, (recipe, score) in enumerate(
            sorted(scores.items(), key=lambda value: (-value[1], value[0])),
            start=1,
        )
    )


def build_target_ranking(
    checkpoint: CheckpointRows,
    aggregate_scores: Iterable[AggregateScore],
    *,
    metric: str,
) -> TargetRanking:
    target_metric = _identifier(metric, label="target metric")
    if target_metric not in checkpoint.universe.metrics:
        raise ValueError(
            f"target metric is outside the declared universe: {target_metric!r}"
        )
    lookup = {
        (score.recipe, score.seed): score.score
        for score in aggregate_scores
        if score.metric == target_metric
    }
    expected = {
        (recipe, seed)
        for recipe in checkpoint.universe.recipes
        for seed in checkpoint.universe.seeds
    }
    if set(lookup) != expected:
        raise ValueError("target aggregate-score grid is incomplete")
    recipe_means = {
        recipe: _mean(lookup[(recipe, seed)] for seed in checkpoint.universe.seeds)
        for recipe in checkpoint.universe.recipes
    }
    return TargetRanking(
        model_size=checkpoint.universe.model_size,
        step=checkpoint.step,
        metric=target_metric,
        seed_count=len(checkpoint.universe.seeds),
        scores=_ranked_scores(recipe_means),
    )


def build_prediction_rankings(
    checkpoint: CheckpointRows,
    aggregate_scores: Iterable[AggregateScore],
) -> tuple[PredictionRanking, ...]:
    lookup = {
        (score.metric, score.seed, score.recipe): score.score
        for score in aggregate_scores
    }
    results: list[PredictionRanking] = []
    for metric in sorted(checkpoint.universe.metrics):
        for seed in sorted(checkpoint.universe.seeds):
            expected = {
                (metric, seed, recipe) for recipe in checkpoint.universe.recipes
            }
            if not expected.issubset(lookup):
                raise ValueError(
                    f"prediction aggregate-score grid is incomplete for {(metric, seed)!r}"
                )
            scores = {
                recipe: lookup[(metric, seed, recipe)]
                for recipe in checkpoint.universe.recipes
            }
            results.append(
                PredictionRanking(
                    model_size=checkpoint.universe.model_size,
                    step=checkpoint.step,
                    metric=metric,
                    seed=seed,
                    scores=_ranked_scores(scores),
                )
            )
    return tuple(results)


def compare_rankings(
    target: TargetRanking,
    predicted: PredictionRanking,
    *,
    actual_compute: float,
    target_compute: float,
) -> SeedDecisionAccuracy:
    """Compare every unordered pair; exclude target ties and miss predicted ties."""
    compute = _finite_number(actual_compute, label="actual compute")
    target_budget = _finite_number(target_compute, label="target compute")
    if compute < 0 or target_budget <= 0:
        raise ValueError(
            "actual compute must be non-negative and target compute positive"
        )
    target_scores = {value.recipe: value.score for value in target.scores}
    predicted_scores = {value.recipe: value.score for value in predicted.scores}
    if set(target_scores) != set(predicted_scores):
        raise ValueError("target and prediction recipe universes differ")
    recipes = tuple(sorted(target_scores))
    pairs: list[PairDecision] = []
    correct = 0
    denominator = 0
    target_ties = 0
    predicted_ties = 0
    for index, recipe_a in enumerate(recipes):
        for recipe_b in recipes[index + 1 :]:
            target_sign = (target_scores[recipe_a] > target_scores[recipe_b]) - (
                target_scores[recipe_a] < target_scores[recipe_b]
            )
            predicted_sign = (
                predicted_scores[recipe_a] > predicted_scores[recipe_b]
            ) - (predicted_scores[recipe_a] < predicted_scores[recipe_b])
            target_tie = target_sign == 0
            predicted_tie = predicted_sign == 0
            target_ties += target_tie
            predicted_ties += predicted_tie
            excluded = target_tie
            is_correct = (
                not excluded and not predicted_tie and target_sign == predicted_sign
            )
            denominator += not excluded
            correct += is_correct
            pairs.append(
                PairDecision(
                    model_size=predicted.model_size,
                    step=predicted.step,
                    metric=predicted.metric,
                    seed=predicted.seed,
                    recipe_a=recipe_a,
                    recipe_b=recipe_b,
                    target_sign=target_sign,
                    predicted_sign=predicted_sign,
                    correct=is_correct,
                    excluded=excluded,
                    target_tie=target_tie,
                    predicted_tie=predicted_tie,
                )
            )
    if denominator == 0:
        raise ValueError("target ties exclude every recipe pair")
    return SeedDecisionAccuracy(
        model_size=predicted.model_size,
        step=predicted.step,
        metric=predicted.metric,
        seed=predicted.seed,
        accuracy=correct / denominator,
        correct=correct,
        denominator=denominator,
        total_pairs=len(pairs),
        target_ties=target_ties,
        predicted_ties=predicted_ties,
        actual_compute=compute,
        percent_target_compute=compute / target_budget * 100,
        pairs=tuple(pairs),
    )


def _noise_spread_rows(
    checkpoint: CheckpointRows,
    aggregate_scores: Iterable[AggregateScore],
) -> tuple[NoiseSpread, ...]:
    lookup = {
        (score.metric, score.recipe, score.seed): score.score
        for score in aggregate_scores
    }
    results: list[NoiseSpread] = []
    for metric in sorted(checkpoint.universe.metrics):
        recipe_means: list[float] = []
        recipe_sds: list[float] = []
        for recipe in sorted(checkpoint.universe.recipes):
            values = tuple(
                lookup[(metric, recipe, seed)] for seed in checkpoint.universe.seeds
            )
            recipe_means.append(_mean(values))
            recipe_sds.append(_sample_sd(values))
        results.append(
            NoiseSpread(
                model_size=checkpoint.universe.model_size,
                step=checkpoint.step,
                metric=metric,
                noise=_mean(recipe_sds),
                spread=_sample_sd(recipe_means),
                recipe_count=len(checkpoint.universe.recipes),
                seed_count=len(checkpoint.universe.seeds),
                within_recipe_ddof=1,
                spread_ddof=1,
            )
        )
    return tuple(results)


def analyze_prediction_checkpoint(
    checkpoint: CheckpointRows,
    target_ranking: TargetRanking,
    *,
    target_compute: float,
    task_grouping: OlmesTaskGrouping = DEFAULT_TASK_GROUPING,
) -> PredictionCheckpointResult:
    aggregate_scores = aggregate_checkpoint(checkpoint, task_grouping=task_grouping)
    rankings = build_prediction_rankings(checkpoint, aggregate_scores)
    attempts = tuple(
        compare_rankings(
            target_ranking,
            ranking,
            actual_compute=checkpoint.actual_compute,
            target_compute=target_compute,
        )
        for ranking in rankings
    )
    summaries: list[CheckpointSummary] = []
    for metric in sorted(checkpoint.universe.metrics):
        metric_attempts = tuple(
            attempt for attempt in attempts if attempt.metric == metric
        )
        accuracies = tuple(attempt.accuracy for attempt in metric_attempts)
        denominators = {attempt.denominator for attempt in metric_attempts}
        total_pairs = {attempt.total_pairs for attempt in metric_attempts}
        if len(denominators) != 1 or len(total_pairs) != 1:
            raise ValueError("pair denominators differ across prediction seeds")
        summaries.append(
            CheckpointSummary(
                model_size=checkpoint.universe.model_size,
                step=checkpoint.step,
                metric=metric,
                mean_accuracy=_mean(accuracies),
                sample_sd_accuracy=_sample_sd(accuracies),
                seed_count=len(metric_attempts),
                ddof=1,
                sd_denominator=len(metric_attempts) - 1,
                denominator_per_seed=next(iter(denominators)),
                total_pairs_per_seed=next(iter(total_pairs)),
                correct_counts=tuple(attempt.correct for attempt in metric_attempts),
                target_ties=sum(attempt.target_ties for attempt in metric_attempts),
                predicted_ties=sum(
                    attempt.predicted_ties for attempt in metric_attempts
                ),
                actual_compute=checkpoint.actual_compute,
                percent_target_compute=metric_attempts[0].percent_target_compute,
            )
        )
    return PredictionCheckpointResult(
        checkpoint=checkpoint,
        aggregate_scores=aggregate_scores,
        rankings=rankings,
        seed_decisions=attempts,
        summaries=tuple(summaries),
        noise_spread=_noise_spread_rows(checkpoint, aggregate_scores),
    )


def analyze_single_scale(
    observations: Iterable[MetricObservation],
    *,
    target_universe: SingleScaleUniverse,
    prediction_universe: SingleScaleUniverse,
    target_metric: str,
    preceding_count: int = 2,
    task_grouping: OlmesTaskGrouping = DEFAULT_TASK_GROUPING,
) -> SingleScaleAnalysis:
    """Run the complete model-independent target and prediction calculation."""
    values = tuple(observations)
    target_checkpoints = select_common_complete_checkpoints(
        values, target_universe, preceding_count=preceding_count
    )
    prediction_checkpoints = select_common_complete_checkpoints(
        values, prediction_universe, preceding_count=preceding_count
    )
    target_scores = aggregate_checkpoint(
        target_checkpoints.default, task_grouping=task_grouping
    )
    target_ranking = build_target_ranking(
        target_checkpoints.default, target_scores, metric=target_metric
    )
    selected_predictions = (
        prediction_checkpoints.default,
        *prediction_checkpoints.preceding,
    )
    predictions = tuple(
        analyze_prediction_checkpoint(
            checkpoint,
            target_ranking,
            target_compute=target_checkpoints.default.actual_compute,
            task_grouping=task_grouping,
        )
        for checkpoint in selected_predictions
    )
    return SingleScaleAnalysis(
        target_checkpoints=target_checkpoints,
        prediction_checkpoints=prediction_checkpoints,
        target_aggregate_scores=target_scores,
        target_ranking=target_ranking,
        predictions=predictions,
    )


__all__ = [
    "DEFAULT_TASK_GROUPING",
    "MMLU_SUBJECTS",
    "OLMES_NON_MMLU_TASKS",
    "AggregateScore",
    "CheckpointRows",
    "CheckpointSummary",
    "CommonCompleteCheckpoints",
    "MetricObservation",
    "NoiseSpread",
    "OlmesTaskGrouping",
    "PairDecision",
    "PredictionCheckpointResult",
    "PredictionRanking",
    "RankedRecipe",
    "SeedDecisionAccuracy",
    "SingleScaleAnalysis",
    "SingleScaleUniverse",
    "TargetRanking",
    "aggregate_checkpoint",
    "analyze_prediction_checkpoint",
    "analyze_single_scale",
    "build_prediction_rankings",
    "build_target_ranking",
    "compare_rankings",
    "observations_from_olmes_frame",
    "select_common_complete_checkpoints",
    "select_exact_common_complete_checkpoint",
]
