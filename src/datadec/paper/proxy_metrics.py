from __future__ import annotations

import math
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from typing import Iterable

type InstanceId = int | str


@verify(UNIQUE)
class LikelihoodNormalization(StrEnum):
    RAW = "raw"
    PER_TOKEN = "per_token"
    PER_CHARACTER = "per_char"


@verify(UNIQUE)
class PredictedTiePolicy(StrEnum):
    COUNT_AS_INCORRECT = "count_as_incorrect"
    EXCLUDE = "exclude"


@verify(UNIQUE)
class CrossoverTiePolicy(StrEnum):
    ADJACENT_NON_TIED = "adjacent_non_tied"
    BRIDGE_TIED_POINTS = "bridge_tied_points"


@verify(UNIQUE)
class ThresholdComparison(StrEnum):
    GREATER_THAN = "greater_than"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    LESS_THAN = "less_than"


@verify(UNIQUE)
class InstanceExclusionReason(StrEnum):
    MISSING_CHOICES = "missing_choices"
    MISSING_CORRECT_CHOICE = "missing_correct_choice"
    NO_INCORRECT_CHOICE = "no_incorrect_choice"


@dataclass(frozen=True, slots=True)
class ChoiceEvidence:
    instance_id: InstanceId
    choice_index: int
    log_probability: float
    token_count: int
    character_count: int

    def __post_init__(self) -> None:
        _validate_instance_id(self.instance_id)
        _non_negative_int(self.choice_index, label="choice index")
        log_probability = _finite_number(
            self.log_probability, label="choice log probability"
        )
        if log_probability > 0:
            raise ValueError("choice log probability must not be positive")
        _positive_int(self.token_count, label="choice token count")
        _positive_int(self.character_count, label="choice character count")


@dataclass(frozen=True, slots=True)
class InstanceEvidence:
    instance_id: InstanceId
    correct_choice_index: int

    def __post_init__(self) -> None:
        _validate_instance_id(self.instance_id)
        _non_negative_int(self.correct_choice_index, label="correct choice index")


@dataclass(frozen=True, slots=True)
class InstanceExclusion:
    instance_id: InstanceId
    reason: InstanceExclusionReason


@dataclass(frozen=True, slots=True)
class InstanceProxyMetrics:
    instance_id: InstanceId
    correct_probability: float
    margin: float
    normalized_correct_probability: float
    total_probability: float
    accuracy: float
    top_tie: bool
    choice_count: int


@dataclass(frozen=True, slots=True)
class ProxyMetricSummary:
    normalization: LikelihoodNormalization
    correct_probability: float | None
    margin: float | None
    normalized_correct_probability: float | None
    total_probability: float | None
    accuracy: float | None
    total_instances: int
    denominator: int
    excluded_instances: tuple[InstanceExclusion, ...]
    top_ties: int
    instances: tuple[InstanceProxyMetrics, ...]

    @property
    def exclusion_count(self) -> int:
        return len(self.excluded_instances)


@dataclass(frozen=True, slots=True)
class LogicalTaskScore:
    task: str
    recipe: str
    score: float

    def __post_init__(self) -> None:
        _identifier(self.task, label="logical task")
        _identifier(self.recipe, label="recipe")
        _finite_number(self.score, label="logical task score")


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
class PairDecisionSummary:
    accuracy: float
    correct: int
    denominator: int
    total_pairs: int
    excluded_pairs: int
    target_ties: int
    predicted_ties: int
    pairs: tuple[PairDecision, ...]


@dataclass(frozen=True, slots=True)
class TaskDecisionSummary:
    task: str
    result: PairDecisionSummary


@dataclass(frozen=True, slots=True)
class LogicalTaskDecisionSummary:
    macro_accuracy: float
    task_denominator: int
    tasks: tuple[TaskDecisionSummary, ...]


@dataclass(frozen=True, slots=True)
class CheckpointTaskScore:
    model_size: str
    step: int
    task: str
    metric: str
    recipe: str
    seed: str
    score: float

    def __post_init__(self) -> None:
        _identifier(self.model_size, label="model size")
        _non_negative_int(self.step, label="checkpoint step")
        _identifier(self.task, label="logical task")
        _identifier(self.metric, label="metric")
        _identifier(self.recipe, label="recipe")
        _identifier(self.seed, label="seed")
        _finite_number(self.score, label="checkpoint task score")


@dataclass(frozen=True, slots=True)
class TaskMetricNoiseSpread:
    task: str
    metric: str
    noise: float
    spread: float
    recipe_count: int
    seed_count: int
    within_recipe_ddof: int
    spread_ddof: int
    within_recipe_denominator: int
    spread_denominator: int


@dataclass(frozen=True, slots=True)
class LatestCommonCompleteNoiseSpread:
    model_size: str
    step: int
    expected_group_count: int
    selected_group_count: int
    complete_steps: tuple[int, ...]
    preceding_complete_steps: tuple[int, ...]
    results: tuple[TaskMetricNoiseSpread, ...]


@dataclass(frozen=True, slots=True)
class ScaleRecipeScore:
    scale: float
    recipe: str
    score: float

    def __post_init__(self) -> None:
        _finite_number(self.scale, label="scale")
        _identifier(self.recipe, label="recipe")
        _finite_number(self.score, label="scale recipe score")


@dataclass(frozen=True, slots=True)
class CrossoverEvent:
    recipe_a: str
    recipe_b: str
    scale_before: float
    scale_after: float
    sign_before: int
    sign_after: int


@dataclass(frozen=True, slots=True)
class CrossoverSummary:
    tie_policy: CrossoverTiePolicy
    scale_count: int
    recipe_count: int
    pair_count: int
    eligible_transitions: int
    excluded_tied_transitions: int
    crossover_count: int
    pairs_with_crossover: int
    endpoint_reversals: int
    events: tuple[CrossoverEvent, ...]


@dataclass(frozen=True, slots=True)
class ThresholdEvaluation:
    threshold: float
    satisfied: bool


@dataclass(frozen=True, slots=True)
class ThresholdSensitivity:
    value: float
    comparison: ThresholdComparison
    evaluations: tuple[ThresholdEvaluation, ...]


@dataclass(frozen=True, slots=True)
class NoiseSpreadChange:
    noise_reduction: float
    spread_increase: float
    minimum_change: float
    noise_improved: bool
    spread_improved: bool
    either_improved: bool


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string: {value!r}")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _validate_instance_id(value: object) -> InstanceId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"instance ID must be an integer or string: {value!r}")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"integer instance ID must be non-negative: {value}")
    if isinstance(value, str) and not value:
        raise ValueError("string instance ID must not be empty")
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


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer: {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative: {value}")
    return value


def _positive_int(value: object, *, label: str) -> int:
    integer = _non_negative_int(value, label=label)
    if integer == 0:
        raise ValueError(f"{label} must be positive")
    return integer


def _unique_identifiers(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    result: set[str] = set()
    for value in values:
        identifier = _identifier(value, label=label)
        if identifier in result:
            raise ValueError(f"duplicate {label}: {identifier!r}")
        result.add(identifier)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return tuple(sorted(result))


def _instance_sort_key(value: InstanceId) -> tuple[int, int | str]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("at least one value is required")
    return math.fsum(ordered) / len(ordered)


def _sd(values: Iterable[float], *, ddof: int, label: str) -> float:
    ordered = tuple(sorted(values))
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError(f"{label} DDOF must be an integer: {ddof!r}")
    if ddof < 0 or ddof >= len(ordered):
        raise ValueError(
            f"{label} DDOF must be non-negative and less than "
            f"observation count {len(ordered)}: {ddof}"
        )
    mean = _mean(ordered)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in ordered) / (len(ordered) - ddof)
    )


def _choice_log_score(
    choice: ChoiceEvidence, normalization: LikelihoodNormalization
) -> float:
    if normalization is LikelihoodNormalization.RAW:
        return choice.log_probability
    if normalization is LikelihoodNormalization.PER_TOKEN:
        return choice.log_probability / choice.token_count
    if normalization is LikelihoodNormalization.PER_CHARACTER:
        return choice.log_probability / choice.character_count
    raise TypeError(
        f"normalization must be a LikelihoodNormalization: {normalization!r}"
    )


def _logsumexp(values: Iterable[float]) -> float:
    scores = tuple(values)
    maximum = max(scores)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in scores))


def calculate_proxy_metrics(
    instances: Iterable[InstanceEvidence],
    choices: Iterable[ChoiceEvidence],
    *,
    normalization: LikelihoodNormalization,
) -> ProxyMetricSummary:
    """Compute paper proxy formulas item-first, then macro-average over items."""
    if not isinstance(normalization, LikelihoodNormalization):
        raise TypeError(
            f"normalization must be a LikelihoodNormalization: {normalization!r}"
        )
    instance_lookup: dict[InstanceId, InstanceEvidence] = {}
    for instance in instances:
        if not isinstance(instance, InstanceEvidence):
            raise TypeError("instances must contain InstanceEvidence values")
        if instance.instance_id in instance_lookup:
            raise ValueError(f"duplicate instance evidence: {instance.instance_id!r}")
        instance_lookup[instance.instance_id] = instance
    if not instance_lookup:
        raise ValueError("instance evidence must not be empty")

    choice_lookup: dict[InstanceId, dict[int, ChoiceEvidence]] = {}
    for choice in choices:
        if not isinstance(choice, ChoiceEvidence):
            raise TypeError("choices must contain ChoiceEvidence values")
        if choice.instance_id not in instance_lookup:
            raise ValueError(
                f"choice references unknown instance: {choice.instance_id!r}"
            )
        instance_choices = choice_lookup.setdefault(choice.instance_id, {})
        if choice.choice_index in instance_choices:
            raise ValueError(
                "duplicate choice evidence: "
                f"{(choice.instance_id, choice.choice_index)!r}"
            )
        instance_choices[choice.choice_index] = choice

    results: list[InstanceProxyMetrics] = []
    exclusions: list[InstanceExclusion] = []
    for instance_id in sorted(instance_lookup, key=_instance_sort_key):
        instance = instance_lookup[instance_id]
        instance_choices = choice_lookup.get(instance_id)
        if not instance_choices:
            exclusions.append(
                InstanceExclusion(instance_id, InstanceExclusionReason.MISSING_CHOICES)
            )
            continue
        choice_indices = tuple(sorted(instance_choices))
        if choice_indices != tuple(range(len(choice_indices))):
            raise ValueError(
                f"choice indices must be contiguous from zero for {instance_id!r}: "
                f"{choice_indices!r}"
            )
        correct_choice = instance_choices.get(instance.correct_choice_index)
        if correct_choice is None:
            exclusions.append(
                InstanceExclusion(
                    instance_id, InstanceExclusionReason.MISSING_CORRECT_CHOICE
                )
            )
            continue
        incorrect_choices = tuple(
            choice
            for index, choice in sorted(instance_choices.items())
            if index != instance.correct_choice_index
        )
        if not incorrect_choices:
            exclusions.append(
                InstanceExclusion(
                    instance_id, InstanceExclusionReason.NO_INCORRECT_CHOICE
                )
            )
            continue

        choice_scores = tuple(
            _choice_log_score(choice, normalization)
            for _, choice in sorted(instance_choices.items())
        )
        correct_score = _choice_log_score(correct_choice, normalization)
        incorrect_scores = tuple(
            _choice_log_score(choice, normalization) for choice in incorrect_choices
        )
        probabilities = tuple(math.exp(score) for score in choice_scores)
        correct_probability = math.exp(correct_score)
        most_likely_incorrect = max(incorrect_scores)
        top_score = max(choice_scores)
        top_tie = sum(score == top_score for score in choice_scores) > 1
        results.append(
            InstanceProxyMetrics(
                instance_id=instance_id,
                correct_probability=correct_probability,
                margin=correct_probability - math.exp(most_likely_incorrect),
                normalized_correct_probability=math.exp(
                    correct_score - _logsumexp(choice_scores)
                ),
                total_probability=math.fsum(probabilities),
                accuracy=float(correct_score == top_score and not top_tie),
                top_tie=top_tie,
                choice_count=len(choice_scores),
            )
        )

    denominator = len(results)
    aggregate = None if not results else results
    return ProxyMetricSummary(
        normalization=normalization,
        correct_probability=(
            None
            if aggregate is None
            else _mean(row.correct_probability for row in aggregate)
        ),
        margin=None if aggregate is None else _mean(row.margin for row in aggregate),
        normalized_correct_probability=(
            None
            if aggregate is None
            else _mean(row.normalized_correct_probability for row in aggregate)
        ),
        total_probability=(
            None
            if aggregate is None
            else _mean(row.total_probability for row in aggregate)
        ),
        accuracy=None
        if aggregate is None
        else _mean(row.accuracy for row in aggregate),
        total_instances=len(instance_lookup),
        denominator=denominator,
        excluded_instances=tuple(exclusions),
        top_ties=sum(row.top_tie for row in results),
        instances=tuple(results),
    )


def calculate_all_proxy_metrics(
    instances: Iterable[InstanceEvidence], choices: Iterable[ChoiceEvidence]
) -> tuple[ProxyMetricSummary, ...]:
    """Compute raw, per-token, and per-character variants from one evidence set."""
    instance_values = tuple(instances)
    choice_values = tuple(choices)
    return tuple(
        calculate_proxy_metrics(
            instance_values, choice_values, normalization=normalization
        )
        for normalization in LikelihoodNormalization
    )


def _score_lookup(
    scores: Iterable[LogicalTaskScore], *, label: str
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for score in scores:
        if not isinstance(score, LogicalTaskScore):
            raise TypeError(f"{label} scores must contain LogicalTaskScore values")
        key = (score.task, score.recipe)
        if key in result:
            raise ValueError(f"duplicate {label} logical-task score: {key!r}")
        result[key] = score.score
    return result


def _pair_decision_summary(
    target: dict[str, float],
    predicted: dict[str, float],
    *,
    predicted_tie_policy: PredictedTiePolicy,
) -> PairDecisionSummary:
    if set(target) != set(predicted):
        raise ValueError("target and predicted recipe universes differ")
    recipes = tuple(sorted(target))
    if len(recipes) < 2:
        raise ValueError("at least two recipes are required")
    pairs: list[PairDecision] = []
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
            excluded = target_tie or (
                predicted_tie and predicted_tie_policy is PredictedTiePolicy.EXCLUDE
            )
            pairs.append(
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
    denominator = sum(not pair.excluded for pair in pairs)
    if denominator == 0:
        raise ValueError("tie policy excluded every recipe pair")
    correct = sum(pair.correct for pair in pairs)
    return PairDecisionSummary(
        accuracy=correct / denominator,
        correct=correct,
        denominator=denominator,
        total_pairs=len(pairs),
        excluded_pairs=sum(pair.excluded for pair in pairs),
        target_ties=sum(pair.target_tie for pair in pairs),
        predicted_ties=sum(pair.predicted_tie for pair in pairs),
        pairs=tuple(pairs),
    )


def compare_logical_task_scores(
    primary_scores: Iterable[LogicalTaskScore],
    proxy_scores: Iterable[LogicalTaskScore],
    *,
    logical_tasks: Iterable[str],
    predicted_tie_policy: PredictedTiePolicy,
) -> LogicalTaskDecisionSummary:
    """Compare proxy and primary recipe decisions, macro-averaging logical tasks."""
    if not isinstance(predicted_tie_policy, PredictedTiePolicy):
        raise TypeError(
            "predicted_tie_policy must be a PredictedTiePolicy: "
            f"{predicted_tie_policy!r}"
        )
    tasks = _unique_identifiers(logical_tasks, label="logical task")
    invalid_mmlu = tuple(task for task in tasks if task.startswith("mmlu_"))
    if invalid_mmlu:
        raise ValueError(
            "MMLU subjects must be macro-averaged into the logical 'mmlu' task: "
            f"{invalid_mmlu!r}"
        )
    primary = _score_lookup(primary_scores, label="primary")
    proxy = _score_lookup(proxy_scores, label="proxy")
    expected_tasks = set(tasks)
    observed_tasks = {task for task, _ in primary} | {task for task, _ in proxy}
    if observed_tasks != expected_tasks:
        raise ValueError(
            "logical-task universe differs from declared tasks: "
            f"missing={tuple(sorted(expected_tasks - observed_tasks))!r}, "
            f"unexpected={tuple(sorted(observed_tasks - expected_tasks))!r}"
        )

    summaries: list[TaskDecisionSummary] = []
    recipe_universe: set[str] | None = None
    for task in tasks:
        primary_task = {
            recipe: score
            for (score_task, recipe), score in primary.items()
            if score_task == task
        }
        proxy_task = {
            recipe: score
            for (score_task, recipe), score in proxy.items()
            if score_task == task
        }
        task_recipes = set(primary_task)
        if recipe_universe is None:
            recipe_universe = task_recipes
        elif task_recipes != recipe_universe:
            raise ValueError("primary recipe universe differs across logical tasks")
        result = _pair_decision_summary(
            primary_task,
            proxy_task,
            predicted_tie_policy=predicted_tie_policy,
        )
        summaries.append(TaskDecisionSummary(task=task, result=result))
    return LogicalTaskDecisionSummary(
        macro_accuracy=_mean(summary.result.accuracy for summary in summaries),
        task_denominator=len(summaries),
        tasks=tuple(summaries),
    )


def latest_common_complete_noise_spread(
    scores: Iterable[CheckpointTaskScore],
    *,
    model_size: str,
    expected_recipes: Iterable[str],
    expected_seeds: Iterable[str],
    expected_tasks: Iterable[str],
    expected_metrics: Iterable[str],
    within_recipe_ddof: int = 1,
    spread_ddof: int = 1,
) -> LatestCommonCompleteNoiseSpread:
    """Select one latest complete step, then compute paper noise and spread."""
    selected_size = _identifier(model_size, label="model size")
    recipes = _unique_identifiers(expected_recipes, label="expected recipe")
    seeds = _unique_identifiers(expected_seeds, label="expected seed")
    tasks = _unique_identifiers(expected_tasks, label="expected logical task")
    metrics = _unique_identifiers(expected_metrics, label="expected metric")
    invalid_mmlu = tuple(task for task in tasks if task.startswith("mmlu_"))
    if invalid_mmlu:
        raise ValueError(
            "MMLU subjects must be macro-averaged before noise/spread: "
            f"{invalid_mmlu!r}"
        )
    _sd((0.0 for _ in seeds), ddof=within_recipe_ddof, label="within-recipe")
    _sd((0.0 for _ in recipes), ddof=spread_ddof, label="spread")

    recipe_set = set(recipes)
    seed_set = set(seeds)
    task_set = set(tasks)
    metric_set = set(metrics)
    values: dict[tuple[int, str, str, str, str], float] = {}
    for score in scores:
        if not isinstance(score, CheckpointTaskScore):
            raise TypeError("scores must contain CheckpointTaskScore values")
        if score.model_size != selected_size:
            continue
        if (
            score.recipe not in recipe_set
            or score.seed not in seed_set
            or score.task not in task_set
            or score.metric not in metric_set
        ):
            continue
        key = (score.step, score.task, score.metric, score.recipe, score.seed)
        if key in values:
            raise ValueError(f"duplicate checkpoint task score: {key!r}")
        values[key] = score.score

    expected_keys = {
        (task, metric, recipe, seed)
        for task in tasks
        for metric in metrics
        for recipe in recipes
        for seed in seeds
    }
    observed_steps = tuple(sorted({key[0] for key in values}))
    complete_steps = tuple(
        step
        for step in observed_steps
        if {
            (task, metric, recipe, seed)
            for observed_step, task, metric, recipe, seed in values
            if observed_step == step
        }
        == expected_keys
    )
    if not complete_steps:
        counts = tuple(
            (
                step,
                sum(observed_step == step for observed_step, *_ in values),
            )
            for step in observed_steps
        )
        raise ValueError(
            "no common complete checkpoint for declared noise/spread universe: "
            f"expected_groups={len(expected_keys)}, observed_counts={counts!r}"
        )
    step = complete_steps[-1]
    results: list[TaskMetricNoiseSpread] = []
    for task in tasks:
        for metric in metrics:
            recipe_means: list[float] = []
            recipe_sds: list[float] = []
            for recipe in recipes:
                seed_values = tuple(
                    values[(step, task, metric, recipe, seed)] for seed in seeds
                )
                recipe_means.append(_mean(seed_values))
                recipe_sds.append(
                    _sd(
                        seed_values,
                        ddof=within_recipe_ddof,
                        label="within-recipe",
                    )
                )
            results.append(
                TaskMetricNoiseSpread(
                    task=task,
                    metric=metric,
                    noise=_mean(recipe_sds),
                    spread=_sd(recipe_means, ddof=spread_ddof, label="spread"),
                    recipe_count=len(recipes),
                    seed_count=len(seeds),
                    within_recipe_ddof=within_recipe_ddof,
                    spread_ddof=spread_ddof,
                    within_recipe_denominator=len(seeds) - within_recipe_ddof,
                    spread_denominator=len(recipes) - spread_ddof,
                )
            )
    return LatestCommonCompleteNoiseSpread(
        model_size=selected_size,
        step=step,
        expected_group_count=len(expected_keys),
        selected_group_count=len(expected_keys),
        complete_steps=complete_steps,
        preceding_complete_steps=tuple(reversed(complete_steps[-3:-1])),
        results=tuple(results),
    )


def summarize_crossovers(
    scores: Iterable[ScaleRecipeScore],
    *,
    expected_recipes: Iterable[str],
    tie_policy: CrossoverTiePolicy,
) -> CrossoverSummary:
    """Count pair-order reversals across an explicitly ordered scale grid."""
    if not isinstance(tie_policy, CrossoverTiePolicy):
        raise TypeError(f"tie_policy must be a CrossoverTiePolicy: {tie_policy!r}")
    recipes = _unique_identifiers(expected_recipes, label="expected recipe")
    values: dict[tuple[float, str], float] = {}
    for observation in scores:
        if not isinstance(observation, ScaleRecipeScore):
            raise TypeError("scores must contain ScaleRecipeScore values")
        if observation.recipe not in set(recipes):
            raise ValueError(f"unexpected recipe: {observation.recipe!r}")
        key = (observation.scale, observation.recipe)
        if key in values:
            raise ValueError(f"duplicate scale/recipe score: {key!r}")
        values[key] = observation.score
    scales = tuple(sorted({scale for scale, _ in values}))
    if len(scales) < 2:
        raise ValueError("at least two scales are required")
    missing = tuple(
        (scale, recipe)
        for scale in scales
        for recipe in recipes
        if (scale, recipe) not in values
    )
    if missing:
        raise ValueError(f"incomplete scale/recipe grid: missing={missing!r}")

    events: list[CrossoverEvent] = []
    eligible_transitions = 0
    excluded_tied_transitions = 0
    endpoint_reversals = 0
    crossing_pairs: set[tuple[str, str]] = set()
    for index, recipe_a in enumerate(recipes):
        for recipe_b in recipes[index + 1 :]:
            signs = tuple(
                (
                    scale,
                    (values[(scale, recipe_a)] > values[(scale, recipe_b)])
                    - (values[(scale, recipe_a)] < values[(scale, recipe_b)]),
                )
                for scale in scales
            )
            non_tied = tuple(item for item in signs if item[1] != 0)
            if len(non_tied) >= 2 and non_tied[0][1] != non_tied[-1][1]:
                endpoint_reversals += 1
            excluded_tied_transitions += sum(
                before[1] == 0 or after[1] == 0
                for before, after in zip(signs, signs[1:], strict=False)
            )
            transitions = (
                zip(signs, signs[1:], strict=False)
                if tie_policy is CrossoverTiePolicy.ADJACENT_NON_TIED
                else zip(non_tied, non_tied[1:], strict=False)
            )
            for before, after in transitions:
                if before[1] == 0 or after[1] == 0:
                    continue
                eligible_transitions += 1
                if before[1] == after[1]:
                    continue
                crossing_pairs.add((recipe_a, recipe_b))
                events.append(
                    CrossoverEvent(
                        recipe_a=recipe_a,
                        recipe_b=recipe_b,
                        scale_before=before[0],
                        scale_after=after[0],
                        sign_before=before[1],
                        sign_after=after[1],
                    )
                )
    pair_count = len(recipes) * (len(recipes) - 1) // 2
    return CrossoverSummary(
        tie_policy=tie_policy,
        scale_count=len(scales),
        recipe_count=len(recipes),
        pair_count=pair_count,
        eligible_transitions=eligible_transitions,
        excluded_tied_transitions=excluded_tied_transitions,
        crossover_count=len(events),
        pairs_with_crossover=len(crossing_pairs),
        endpoint_reversals=endpoint_reversals,
        events=tuple(events),
    )


def evaluate_threshold_sensitivity(
    value: float,
    *,
    thresholds: Iterable[float],
    comparison: ThresholdComparison,
) -> ThresholdSensitivity:
    """Evaluate only the caller's predeclared threshold grid, in supplied order."""
    observed = _finite_number(value, label="predicate value")
    if not isinstance(comparison, ThresholdComparison):
        raise TypeError(f"comparison must be a ThresholdComparison: {comparison!r}")
    threshold_values = tuple(
        _finite_number(threshold, label="predicate threshold")
        for threshold in thresholds
    )
    if not threshold_values:
        raise ValueError("predicate threshold grid must not be empty")
    if len(threshold_values) != len(set(threshold_values)):
        raise ValueError("predicate threshold grid must not contain duplicates")

    def satisfied(threshold: float) -> bool:
        if comparison is ThresholdComparison.GREATER_THAN:
            return observed > threshold
        if comparison is ThresholdComparison.AT_LEAST:
            return observed >= threshold
        if comparison is ThresholdComparison.AT_MOST:
            return observed <= threshold
        if comparison is ThresholdComparison.LESS_THAN:
            return observed < threshold
        raise AssertionError("unreachable threshold comparison")

    return ThresholdSensitivity(
        value=observed,
        comparison=comparison,
        evaluations=tuple(
            ThresholdEvaluation(threshold, satisfied(threshold))
            for threshold in threshold_values
        ),
    )


def metric_at_least_as_good(
    proxy_value: float, primary_value: float, *, tolerance: float
) -> bool:
    proxy = _finite_number(proxy_value, label="proxy metric value")
    primary = _finite_number(primary_value, label="primary metric value")
    allowed = _finite_number(tolerance, label="comparison tolerance")
    if allowed < 0:
        raise ValueError("comparison tolerance must be non-negative")
    return proxy >= primary - allowed


def noise_spread_change(
    *,
    primary_noise: float,
    primary_spread: float,
    proxy_noise: float,
    proxy_spread: float,
    minimum_change: float,
) -> NoiseSpreadChange:
    """Apply a fixed minimum change to lower-noise or wider-spread claims."""
    baseline_noise = _finite_number(primary_noise, label="primary noise")
    baseline_spread = _finite_number(primary_spread, label="primary spread")
    candidate_noise = _finite_number(proxy_noise, label="proxy noise")
    candidate_spread = _finite_number(proxy_spread, label="proxy spread")
    threshold = _finite_number(minimum_change, label="minimum change")
    if threshold < 0:
        raise ValueError("minimum change must be non-negative")
    noise_reduction = baseline_noise - candidate_noise
    spread_increase = candidate_spread - baseline_spread
    noise_improved = noise_reduction >= threshold
    spread_improved = spread_increase >= threshold
    return NoiseSpreadChange(
        noise_reduction=noise_reduction,
        spread_increase=spread_increase,
        minimum_change=threshold,
        noise_improved=noise_improved,
        spread_improved=spread_improved,
        either_improved=noise_improved or spread_improved,
    )
