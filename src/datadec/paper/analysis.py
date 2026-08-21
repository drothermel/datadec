from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, UNIQUE, verify
from typing import Iterable

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


@dataclass(frozen=True, slots=True)
class OlmesMacroAverage:
    score: float
    mmlu_score: float
    task_denominator: int
    mmlu_subject_denominator: int


@dataclass(frozen=True, slots=True)
class RecipePair:
    recipe_a: str
    recipe_b: str
    score_a: float
    score_b: float


@verify(UNIQUE)
class TiePolicy(StrEnum):
    ERROR = "error"
    EXCLUDE = "exclude"
    COUNT_AS_INCORRECT = "count_as_incorrect"


@dataclass(frozen=True, slots=True)
class DecisionAccuracy:
    accuracy: float
    correct: int
    denominator: int
    total_pairs: int
    excluded_pairs: int
    target_ties: int
    predicted_ties: int


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    mean: float
    sd: float
    count: int
    ddof: int
    denominator: int


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    percent: float
    included_compute: float
    target_compute: float
    included_cost_count: int


@dataclass(frozen=True, slots=True)
class NoiseSpread:
    noise: float
    spread: float
    recipe_count: int
    seed_count: int
    within_recipe_ddof: int
    spread_ddof: int
    within_recipe_denominator: int
    spread_denominator: int


def _finite_number(value: float | int, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite: {value!r}")
    return number


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string: {value!r}")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _unique_identifiers(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    result: set[str] = set()
    for raw_value in values:
        value = _identifier(raw_value, label=label)
        if value in result:
            raise ValueError(f"duplicate {label}: {value!r}")
        result.add(value)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return tuple(sorted(result))


def _unique_scores(
    scores: Iterable[tuple[str, float]], *, label: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_name, raw_score in scores:
        name = _identifier(raw_name, label=label)
        if name in result:
            raise ValueError(f"duplicate {label}: {name!r}")
        result[name] = _finite_number(raw_score, label=f"score for {name!r}")
    return result


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("at least one value is required")
    return math.fsum(ordered) / len(ordered)


def _validate_ddof(ddof: int, *, count: int, label: str) -> None:
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError(f"{label} must be an integer: {ddof!r}")
    if ddof < 0:
        raise ValueError(f"{label} must be non-negative: {ddof}")
    if ddof >= count:
        raise ValueError(f"{label} must be less than observation count {count}: {ddof}")


def _mean_and_sd(values: Iterable[float], *, ddof: int) -> AttemptSummary:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("at least one value is required")
    _validate_ddof(ddof, count=len(ordered), label="ddof")
    mean = _mean(ordered)
    variance = math.fsum((value - mean) ** 2 for value in ordered) / (
        len(ordered) - ddof
    )
    sd = math.sqrt(variance)
    if not math.isfinite(mean) or not math.isfinite(sd):
        raise ValueError("mean or standard deviation is not finite")
    return AttemptSummary(
        mean=mean,
        sd=sd,
        count=len(ordered),
        ddof=ddof,
        denominator=len(ordered) - ddof,
    )


def macro_average_olmes(
    task_scores: Iterable[tuple[str, float]],
    *,
    expected_mmlu_subjects: Iterable[str] = MMLU_SUBJECTS,
) -> OlmesMacroAverage:
    """Aggregate MMLU subjects, then macro-average the ten OLMES tasks."""
    expected_subjects = _unique_identifiers(
        expected_mmlu_subjects, label="expected MMLU subject"
    )
    invalid_subjects = tuple(
        subject for subject in expected_subjects if not subject.startswith("mmlu_")
    )
    if invalid_subjects:
        raise ValueError(f"invalid expected MMLU subjects: {invalid_subjects!r}")

    scores = _unique_scores(task_scores, label="OLMES task")
    expected = set(OLMES_NON_MMLU_TASKS) | set(expected_subjects)
    missing = tuple(sorted(expected - scores.keys()))
    unexpected = tuple(sorted(scores.keys() - expected))
    if missing or unexpected:
        raise ValueError(
            f"incomplete OLMES task scores: missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )

    mmlu_score = _mean(scores[subject] for subject in expected_subjects)
    ten_task_scores = [scores[task] for task in OLMES_NON_MMLU_TASKS]
    ten_task_scores.append(mmlu_score)
    return OlmesMacroAverage(
        score=_mean(ten_task_scores),
        mmlu_score=mmlu_score,
        task_denominator=len(ten_task_scores),
        mmlu_subject_denominator=len(expected_subjects),
    )


def construct_recipe_pairs(
    recipe_scores: Iterable[tuple[str, float]],
) -> tuple[RecipePair, ...]:
    """Construct every unordered pair in deterministic recipe-name order."""
    scores = _unique_scores(recipe_scores, label="recipe")
    recipes = tuple(sorted(scores))
    if len(recipes) < 2:
        raise ValueError("at least two recipes are required")
    return tuple(
        RecipePair(
            recipe_a=recipes[index],
            recipe_b=recipes[other_index],
            score_a=scores[recipes[index]],
            score_b=scores[recipes[other_index]],
        )
        for index in range(len(recipes))
        for other_index in range(index + 1, len(recipes))
    )


def _comparison_sign(left: float, right: float) -> int:
    return (left > right) - (left < right)


def decision_accuracy(
    target_scores: Iterable[tuple[str, float]],
    predicted_scores: Iterable[tuple[str, float]],
    *,
    tie_policy: TiePolicy,
) -> DecisionAccuracy:
    """Calculate the paper's two-class pairwise sign-agreement accuracy."""
    if not isinstance(tie_policy, TiePolicy):
        raise TypeError(f"tie_policy must be a TiePolicy: {tie_policy!r}")
    target_pairs = construct_recipe_pairs(target_scores)
    predicted_pairs = construct_recipe_pairs(predicted_scores)
    target_recipes = {(pair.recipe_a, pair.recipe_b) for pair in target_pairs}
    predicted_recipes = {(pair.recipe_a, pair.recipe_b) for pair in predicted_pairs}
    if target_recipes != predicted_recipes:
        raise ValueError(
            "target and predicted recipe universes differ: "
            f"target={tuple(sorted(target_recipes))!r}, "
            f"predicted={tuple(sorted(predicted_recipes))!r}"
        )

    correct = 0
    denominator = 0
    excluded = 0
    target_ties = 0
    predicted_ties = 0
    for target, predicted in zip(target_pairs, predicted_pairs, strict=True):
        target_sign = _comparison_sign(target.score_a, target.score_b)
        predicted_sign = _comparison_sign(predicted.score_a, predicted.score_b)
        target_is_tie = target_sign == 0
        predicted_is_tie = predicted_sign == 0
        target_ties += target_is_tie
        predicted_ties += predicted_is_tie

        if tie_policy is TiePolicy.ERROR and (target_is_tie or predicted_is_tie):
            raise ValueError(
                "tie encountered for recipe pair "
                f"{(target.recipe_a, target.recipe_b)!r}"
            )
        exclude = target_is_tie or (
            tie_policy is TiePolicy.EXCLUDE and predicted_is_tie
        )
        if exclude:
            excluded += 1
            continue
        denominator += 1
        correct += target_sign == predicted_sign

    if denominator == 0:
        raise ValueError("tie policy excluded every recipe pair")
    return DecisionAccuracy(
        accuracy=correct / denominator,
        correct=correct,
        denominator=denominator,
        total_pairs=len(target_pairs),
        excluded_pairs=excluded,
        target_ties=target_ties,
        predicted_ties=predicted_ties,
    )


def summarize_prediction_attempts(
    attempts: Iterable[float], *, ddof: int
) -> AttemptSummary:
    values = tuple(
        _finite_number(value, label="prediction attempt") for value in attempts
    )
    return _mean_and_sd(values, ddof=ddof)


def theoretical_training_flops(parameter_count: float, token_count: float) -> float:
    parameters = _finite_number(parameter_count, label="parameter_count")
    tokens = _finite_number(token_count, label="token_count")
    if parameters <= 0 or tokens <= 0:
        raise ValueError("parameter_count and token_count must both be positive")
    flops = 6 * parameters * tokens
    if not math.isfinite(flops):
        raise ValueError("theoretical training FLOPs are not finite")
    return flops


def percent_target_compute(
    included_costs: Iterable[float], *, target_compute: float
) -> ComputeBudget:
    costs = tuple(
        _finite_number(cost, label="included compute cost") for cost in included_costs
    )
    if not costs:
        raise ValueError("at least one included compute cost is required")
    if any(cost < 0 for cost in costs):
        raise ValueError("included compute costs must be non-negative")
    target = _finite_number(target_compute, label="target_compute")
    if target <= 0:
        raise ValueError("target_compute must be positive")
    included = math.fsum(sorted(costs))
    if not math.isfinite(included):
        raise ValueError("included compute is not finite")
    percent = included / target * 100
    if not math.isfinite(percent):
        raise ValueError("percent of target compute is not finite")
    return ComputeBudget(
        percent=percent,
        included_compute=included,
        target_compute=target,
        included_cost_count=len(costs),
    )


def noise_and_spread(
    observations: Iterable[tuple[str, str, float]],
    *,
    expected_recipes: Iterable[str],
    expected_seeds: Iterable[str],
    within_recipe_ddof: int,
    spread_ddof: int,
) -> NoiseSpread:
    """Calculate mean within-recipe seed SD and SD across recipe seed means."""
    recipes = _unique_identifiers(expected_recipes, label="expected recipe")
    seeds = _unique_identifiers(expected_seeds, label="expected seed")
    values: dict[tuple[str, str], float] = {}
    for raw_recipe, raw_seed, raw_score in observations:
        recipe = _identifier(raw_recipe, label="recipe")
        seed = _identifier(raw_seed, label="seed")
        key = (recipe, seed)
        if key in values:
            raise ValueError(f"duplicate recipe/seed observation: {key!r}")
        values[key] = _finite_number(raw_score, label=f"score for {key!r}")

    expected_grid = {(recipe, seed) for recipe in recipes for seed in seeds}
    missing = tuple(sorted(expected_grid - values.keys()))
    unexpected = tuple(sorted(values.keys() - expected_grid))
    if missing or unexpected:
        raise ValueError(
            f"incomplete recipe/seed grid: missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )

    recipe_summaries = tuple(
        _mean_and_sd(
            (values[(recipe, seed)] for seed in seeds), ddof=within_recipe_ddof
        )
        for recipe in recipes
    )
    spread_summary = _mean_and_sd(
        (summary.mean for summary in recipe_summaries), ddof=spread_ddof
    )
    return NoiseSpread(
        noise=_mean(summary.sd for summary in recipe_summaries),
        spread=spread_summary.sd,
        recipe_count=len(recipes),
        seed_count=len(seeds),
        within_recipe_ddof=within_recipe_ddof,
        spread_ddof=spread_ddof,
        within_recipe_denominator=len(seeds) - within_recipe_ddof,
        spread_denominator=len(recipes) - spread_ddof,
    )
